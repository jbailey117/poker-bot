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
  3. GATE: two candidates, the new model alone and the mean of the
     champion's member models plus the new one (an ensemble, at most
     --max-members). The best passing one is promoted. A candidate passes only if
       a. it beats the champion head-to-head (pooled 95% CI above 0), and
       b. on the HELD-OUT style bots (never in the training pool): the
          average result doesn't drop, every style stays >= --heldout-min
          bb/deal (a 1000-deal match is still a likely win), and no single
          style drops by more than --heldout-max-drop, and
       c. its thinking time stays under --max-think s/half, or under
          --think-ratio x the champion's in the same match (and --think-hard).
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


def members_of(vdir):
    """Component model dirs of a (possibly ensemble) version dir."""
    with open(os.path.join(vdir, "q_xgb_meta.json")) as f:
        meta = json.load(f)
    return [os.path.join(ROOT, m) for m in meta.get("ensemble_of", [os.path.relpath(vdir, ROOT)])]


def model_env(dirs):
    return "MODEL_PATH=" + ",".join(os.path.relpath(os.path.join(d, "q_trees.npz"), ROOT) for d in dirs)


def gate(name, dirs, champ_env, champ_ho, args, port):
    """Run head-to-head + held-out for one candidate; return (result dict, failure reasons)."""
    env = model_env(dirs)
    text = sh([PY, "tools/selfplay.py", "--a", champ_env, "--b", env,
               "--seeds", args.h2h_seeds, "--ports", f"{port},{port + 1}"])
    h2h = pooled(text)
    m = re.search(r"max time/half A ([\d.]+)s", text)
    h2h["think_a"] = float(m[1]) if m else None
    print(f"[{name}] head-to-head: {h2h}", flush=True)
    ho = heldout(env, args.heldout_seeds, port + 20)
    res = {"members": [os.path.relpath(d, ROOT) for d in dirs], "h2h": h2h, "heldout": ho}
    reasons = []
    if h2h["lo"] <= 0:
        reasons.append(f"head-to-head CI lower bound {h2h['lo']:+.3f} <= 0")
    champ_avg = sum(champ_ho[st]["mean"] for st in HELDOUT) / len(HELDOUT)
    cand_avg = sum(ho[st]["mean"] for st in HELDOUT) / len(HELDOUT)
    res["heldout_avg"] = cand_avg
    for st in HELDOUT:
        print(f"[{name}] held-out {st:9s} champion {champ_ho[st]['mean']:+.3f}  candidate {ho[st]['mean']:+.3f}",
              flush=True)
    print(f"[{name}] held-out average: champion {champ_avg:+.3f}  candidate {cand_avg:+.3f}", flush=True)
    if cand_avg < champ_avg:
        reasons.append(f"held-out average {cand_avg:+.3f} < champion {champ_avg:+.3f}")
    for st in HELDOUT:
        if ho[st]["mean"] < args.heldout_min:
            reasons.append(f"held-out {st}: {ho[st]['mean']:+.3f} < {args.heldout_min}")
        if ho[st]["mean"] < champ_ho[st]["mean"] - args.heldout_max_drop:
            reasons.append(f"held-out {st}: dropped {champ_ho[st]['mean']:+.3f} -> {ho[st]['mean']:+.3f}")
    # Thinking time: measured in the head-to-head, where champion and candidate
    # share the machine, so the comparison holds even if the machine is slow.
    limit = max(args.max_think, args.think_ratio * (h2h["think_a"] or 0))
    if h2h["think_b"] > limit:
        reasons.append(f"thinking time {h2h['think_b']:.2f}s > {limit:.2f}s")
    if h2h["think_b"] > args.think_hard:
        reasons.append(f"thinking time {h2h['think_b']:.2f}s > hard limit {args.think_hard}s")
    res["reasons"] = reasons
    return res, reasons


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
    ap.add_argument("--max-think", type=float, default=4.0, help="always-OK thinking time (s/half)")
    ap.add_argument("--think-ratio", type=float, default=1.35,
                    help="else allow up to this x the champion's thinking time in the same match")
    ap.add_argument("--think-hard", type=float, default=6.5, help="never above this (s/half)")
    ap.add_argument("--max-members", type=int, default=3, help="max models in an ensemble candidate")
    ap.add_argument("--no-ensemble", action="store_true", help="only gate the new model alone")
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args()

    for r in range(args.rounds):
        t0 = time.time()
        vs = versions()
        champ = vs[-1]
        champ_dir = os.path.join(ROOT, "models", f"v{champ}")
        new_v = champ + 1
        tag = f"round_v{new_v}"
        for suffix in "bcdefghij":     # never reuse a round's data/model dirs
            if not any(os.path.exists(os.path.join(ROOT, p, tag))
                       for p in ("data/raw", "models/sweep", "models/candidates")):
                break
            tag = f"round_v{new_v}{suffix}"
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
        # Keep every trained model (small) as a possible ensemble member.
        keep = os.path.join(ROOT, "models", "candidates", tag)
        os.makedirs(keep, exist_ok=True)
        for f in ("q_trees.npz", "q_xgb_meta.json", "a_trees.npz", "v_trees.npz", "training_loss.png"):
            if os.path.exists(os.path.join(ROOT, cand_dir, f)):
                shutil.copy(os.path.join(ROOT, cand_dir, f), keep)

        # 3. gates: the new model alone, and averaged with the champion's members
        champ_env = model_env(members_of(champ_dir))
        champ_ho = champion_heldout(champ_dir, args.heldout_seeds, 6010)
        cands = {"alone": [keep]}
        if not args.no_ensemble:
            cands["ensemble"] = (members_of(champ_dir) + [keep])[-args.max_members:]
        results = {}
        for i, (name, dirs) in enumerate(cands.items()):
            results[name], _ = gate(name, dirs, champ_env, champ_ho, args, 6100 + 100 * i)
        entry["candidates"] = results
        passing = [n for n, res in results.items() if not res["reasons"]]
        best = max(passing, key=lambda n: results[n]["h2h"]["mean"]) if passing else None
        entry["promoted"] = best
        entry["minutes"] = round((time.time() - t0) / 60, 1)

        # 4. promote
        if best:
            from tree_model import TreeModel
            dirs = cands[best]
            vdir = os.path.join(ROOT, "models", f"v{new_v}")
            os.makedirs(vdir, exist_ok=True)
            for f in ("q_trees.npz", "a_trees.npz", "v_trees.npz"):
                ms = [TreeModel.load(os.path.join(d, f)) for d in dirs]
                (ms[0] if len(ms) == 1 else TreeModel.blend(ms)).save(os.path.join(vdir, f))
            with open(os.path.join(keep, "q_xgb_meta.json")) as f:
                meta = json.load(f)
            meta["ensemble_of"] = [os.path.relpath(d, ROOT) for d in dirs]
            with open(os.path.join(vdir, "q_xgb_meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            with open(os.path.join(vdir, "heldout.json"), "w") as f:
                json.dump({"seeds": args.heldout_seeds, "results": results[best]["heldout"]}, f, indent=1)
            for f in ("q_trees.npz", "a_trees.npz", "v_trees.npz", "q_xgb_meta.json"):
                shutil.copy(os.path.join(vdir, f), os.path.join(ROOT, "models", f))
            print(f"PROMOTED v{new_v} ({best}: {meta['ensemble_of']})", flush=True)
        else:
            print(f"NOT promoted: { {n: res['reasons'] for n, res in results.items()} }", flush=True)
        with open(os.path.join(keep, "gate.json"), "w") as f:
            json.dump(entry, f, indent=1)
        log(entry)

        if args.no_commit:
            continue
        if best:
            h2h, ho = results[best]["h2h"], results[best]["heldout"]
            sh(["git", "add", "models", "tools", "agent.py", "features.py", "fast_eval.py", "tree_model.py",
                "preflop_table.py", "opponents"])
            msg = (f"v{new_v}: promoted by tools/iterate.py ({best})\n\n"
                   f"Members: {', '.join(results[best]['members'])}\n"
                   f"Head-to-head vs v{champ}: {h2h['mean']:+.3f} bb/deal (CI {h2h['lo']:+.3f}..{h2h['hi']:+.3f}).\n"
                   + "".join(f"Held-out {st}: {ho[st]['mean']:+.3f} (v{champ}: {champ_ho[st]['mean']:+.3f})\n"
                             for st in HELDOUT)
                   + f"Data: {data}; thinking {h2h['think_b']:.2f}s/half (v{champ}: {h2h['think_a']}s).\n\n"
                   "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
            sh(["git", "commit", "-q", "-m", msg])
            sh(["git", "tag", f"v{new_v}-xgb"])
            print(f"committed and tagged v{new_v}-xgb", flush=True)
        else:
            sh(["git", "add", "models/candidates", "models/iterations.jsonl"])
            sh(["git", "commit", "-q", "-m",
                f"{tag} candidate (not promoted, champion stays v{champ})\n\n"
                + "".join(f"{n}: h2h {res['h2h']['mean']:+.3f} (CI {res['h2h']['lo']:+.3f}..{res['h2h']['hi']:+.3f}); "
                          f"fails: {'; '.join(res['reasons'])}\n" for n, res in results.items())
                + "\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"])
            print(f"committed candidate {tag} (not promoted)", flush=True)


if __name__ == "__main__":
    main()
