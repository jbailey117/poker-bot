"""
iterate.py - automated policy-iteration rounds: generate -> train -> gate -> promote.

    python tools/iterate.py --rounds 4 --hands 2000000
    python tools/iterate.py --rounds 1 --skip-gen data/raw/run3     # train/gate on existing data

Each round:
  1. GENERATE: play --hands hands with agent.py (= the current champion
     model) as the learner, against the league pool (baselines, rule
     variants, two style bots, every frozen model in models/v*/).
  2. TRAIN: residual XGBoost (tools/train_xgb.py --residual) on that round's
     data -> models/sweep/round_<n>/.
  3. GATE: the candidate is promoted only if
       a. it beats the champion head-to-head (pooled 95% CI above 0), and
       b. on the HELD-OUT style bots (never in the training pool): the
          average result doesn't drop, every style stays >= --heldout-min
          bb/deal (a 1000-deal match is still a likely win), and no single
          style drops by more than --heldout-max-drop, and
       c. its thinking time stays under --max-think seconds per half.
  4. PROMOTE: copy to models/v<k>/, make it the active model
     (models/q_trees.npz), commit and tag v<k>-xgb.
All results are appended to models/iterations.jsonl.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable          # run everything with the training env (Python 3.12, like the tournament)
HELDOUT = ["balanced", "trapper", "guide"]
LOG = os.path.join(ROOT, "models", "iterations.jsonl")


def sh(cmd, **kw):
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True,
                          shell=isinstance(cmd, str), **kw).stdout


def pooled(text):
    m = re.search(r"B ([+-][\d.]+) bb/deal, 95% CI \[([+-][\d.]+), ([+-][\d.]+)\].*B ([\d.]+)s", text)
    if not m:
        raise RuntimeError(f"could not parse selfplay output:\n{text[-2000:]}")
    return {"mean": float(m[1]), "lo": float(m[2]), "hi": float(m[3]), "think_b": float(m[4])}


def versions():
    vs = [int(m[1]) for d in glob.glob(os.path.join(ROOT, "models", "v*"))
          if (m := re.search(r"v(\d+)$", d))]
    return sorted(vs)


def heldout(model_env, seeds, port):
    """Pooled result per held-out style for a given model env string."""
    out = {}
    procs = []
    for i, st in enumerate(HELDOUT):
        p1 = port + 2 * i
        procs.append((st, subprocess.Popen(
            [PY, "tools/selfplay.py", "--a-file", "opponents/style_bot.py", "--a", f"STYLE={st}",
             "--b", model_env, "--seeds", seeds, "--ports", f"{p1},{p1 + 1}"],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)))
    for st, pr in procs:
        text, _ = pr.communicate()
        out[st] = pooled(text)
    return out


def champion_heldout(vdir, seeds, port):
    cache = os.path.join(vdir, "heldout.json")
    if os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        if data.get("seeds") == seeds:
            return data["results"]
    res = heldout(f"MODEL_PATH={os.path.relpath(os.path.join(vdir, 'q_trees.npz'), ROOT)}", seeds, port)
    with open(cache, "w") as f:
        json.dump({"seeds": seeds, "results": res}, f, indent=1)
    return res


def log(entry):
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--hands", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--skip-gen", default=None, help="use this existing data dir for the first round")
    ap.add_argument("--h2h-seeds", default="1-6")
    ap.add_argument("--heldout-seeds", default="1-4")
    ap.add_argument("--heldout-min", type=float, default=0.75)
    ap.add_argument("--heldout-max-drop", type=float, default=1.0)
    ap.add_argument("--skip-train", default=None, help="use this existing candidate dir for the first round")
    ap.add_argument("--max-think", type=float, default=4.0)
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args()

    for r in range(args.rounds):
        t0 = time.time()
        vs = versions()
        champ = vs[-1]
        champ_dir = os.path.join(ROOT, "models", f"v{champ}")
        new_v = champ + 1
        tag = f"round_v{new_v}"
        entry = {"round": tag, "champion": f"v{champ}", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        print(f"\n===== {tag}: champion v{champ} =====", flush=True)

        # 1. generate
        if r == 0 and args.skip_gen:
            data = args.skip_gen
        else:
            data = f"data/raw/{tag}"
            seed = int(time.time()) % 100_000
            sh([PY, "tools/gen_data.py", "--hands", str(args.hands), "--hands-per-task", "5000",
                "--workers", str(args.workers), "--eps", str(args.eps), "--seed", str(seed), "--out", data])
        entry["data"] = data

        # 2. train
        if r == 0 and args.skip_train:
            cand_dir = args.skip_train
            entry["train"] = "skipped (existing candidate)"
        else:
            cand_dir = f"models/sweep/{tag}"
            out = sh([PY, "tools/train_xgb.py", "--data", data, "--residual", "--patience", "600",
                      "--minutes", "10", "--out", cand_dir], env={**os.environ, "PYTHONWARNINGS": "ignore"})
            m = re.search(r"best val rmse ([\d.]+) at round (\d+)", out)
            entry["train"] = {"best_val_rmse": float(m[1]), "best_round": int(m[2])} if m else out[-500:]
        cand_env = f"MODEL_PATH={cand_dir}/q_trees.npz"
        champ_env = f"MODEL_PATH={os.path.relpath(champ_dir, ROOT)}/q_trees.npz"

        # 3. gates
        h2h = pooled(sh([PY, "tools/selfplay.py", "--a", champ_env, "--b", cand_env,
                         "--seeds", args.h2h_seeds, "--ports", "6001,6002"]))
        entry["h2h"] = h2h
        print(f"head-to-head vs v{champ}: {h2h}", flush=True)
        champ_ho = champion_heldout(champ_dir, args.heldout_seeds, 6010)
        cand_ho = heldout(cand_env, args.heldout_seeds, 6030)
        entry["heldout"] = {"champion": champ_ho, "candidate": cand_ho}
        for st in HELDOUT:
            print(f"held-out {st:9s} champion {champ_ho[st]['mean']:+.3f}  candidate {cand_ho[st]['mean']:+.3f}",
                  flush=True)
        reasons = []
        if h2h["lo"] <= 0:
            reasons.append(f"head-to-head CI lower bound {h2h['lo']:+.3f} <= 0")
        champ_avg = sum(champ_ho[st]["mean"] for st in HELDOUT) / len(HELDOUT)
        cand_avg = sum(cand_ho[st]["mean"] for st in HELDOUT) / len(HELDOUT)
        entry["heldout_avg"] = {"champion": champ_avg, "candidate": cand_avg}
        print(f"held-out average: champion {champ_avg:+.3f}  candidate {cand_avg:+.3f}", flush=True)
        if cand_avg < champ_avg:
            reasons.append(f"held-out average {cand_avg:+.3f} < champion {champ_avg:+.3f}")
        for st in HELDOUT:
            if cand_ho[st]["mean"] < args.heldout_min:
                reasons.append(f"held-out {st}: {cand_ho[st]['mean']:+.3f} < {args.heldout_min}")
            if cand_ho[st]["mean"] < champ_ho[st]["mean"] - args.heldout_max_drop:
                reasons.append(f"held-out {st}: dropped {champ_ho[st]['mean']:+.3f} -> {cand_ho[st]['mean']:+.3f}")
        think = max([h2h["think_b"]] + [v["think_b"] for v in cand_ho.values()])
        entry["max_think"] = think
        if think > args.max_think:
            reasons.append(f"thinking time {think:.2f}s > {args.max_think}s")
        entry["promoted"] = not reasons
        entry["reasons"] = reasons
        entry["minutes"] = round((time.time() - t0) / 60, 1)

        # 4. promote
        if not reasons:
            vdir = os.path.join(ROOT, "models", f"v{new_v}")
            os.makedirs(vdir, exist_ok=True)
            for f in ("q_trees.npz", "q_xgb_meta.json", "a_trees.npz", "v_trees.npz"):
                if os.path.exists(os.path.join(ROOT, cand_dir, f)):
                    shutil.copy(os.path.join(ROOT, cand_dir, f), vdir)
            with open(os.path.join(vdir, "heldout.json"), "w") as f:
                json.dump({"seeds": args.heldout_seeds, "results": cand_ho}, f, indent=1)
            for f in ("q_trees.npz", "q_xgb_meta.json", "a_trees.npz", "v_trees.npz", "q_xgb.ubj",
                      "v_xgb.ubj", "train_log.csv", "training_loss.png"):
                src = os.path.join(ROOT, cand_dir, f)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(ROOT, "models", f))
            print(f"PROMOTED v{new_v}", flush=True)
        else:
            print(f"NOT promoted: {reasons}", flush=True)
        log(entry)

        if not reasons and not args.no_commit:
            sh(["git", "add", "models", "tools", "agent.py", "features.py", "fast_eval.py", "tree_model.py",
                "preflop_table.py", "opponents"])
            msg = (f"v{new_v}: promoted by tools/iterate.py\n\n"
                   f"Head-to-head vs v{champ}: {h2h['mean']:+.3f} bb/deal (CI {h2h['lo']:+.3f}..{h2h['hi']:+.3f}).\n"
                   + "".join(f"Held-out {st}: {cand_ho[st]['mean']:+.3f} (v{champ}: {champ_ho[st]['mean']:+.3f})\n"
                             for st in HELDOUT)
                   + f"Data: {data}; max thinking {think:.2f}s/half.\n\n"
                   "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
            sh(["git", "commit", "-q", "-m", msg])
            sh(["git", "tag", f"v{new_v}-xgb"])
            print(f"committed and tagged v{new_v}-xgb", flush=True)


if __name__ == "__main__":
    main()
