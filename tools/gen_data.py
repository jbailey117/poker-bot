"""
gen_data.py - generate training data by playing hands in-process (no HTTP).

The LEARNER is agent.py as it currently is: if models/q_trees.npz exists it
plays with the learned model (that's how each training round improves on the
last); run with --learner-env "USE_MODEL=0" to collect data from the rules only.
At every one of its decisions we record:
    X        features.extract(...) of the decision point     (float32[N_FEATURES])
    action   the action actually taken (0 FOLD, 1 CHECK_CALL, 2 RAISE_HALF, 3 RAISE_POT)
    greedy   the action agent.py wanted (differs when we explored)
    legal    which of the 4 actions were legal                 (bool[4])
    reward   chips won FROM THIS POINT ON, in big blinds:
             (final stack - stack at the decision) / big blind
             = what we get back from the pot minus what we still put in.
             Chips already in the pot are sunk, so FOLD is always exactly 0;
             that's why exploration never folds (its value is known).
    explore  1 if the action was a random exploration step
    street, opp (index into OPPONENTS), hand (global hand id), seat info

With probability --eps the learner replaces its action with a uniformly random
legal NON-fold action, so the data contains every action in every kind of
spot (needed to learn "what if I had raised here?").

Opponents come from a pool (baselines, the frozen v1, and agent.py with
different parameter styles) so models don't overfit to one opponent.

Usage:
    python tools/gen_data.py --hands 20000 --out data/raw/run1          # all cores - 2
    python tools/gen_data.py --hands 2000 --workers 4 --eps 0.2 --out data/raw/test
    python tools/gen_data.py --list-opponents

Output: one .npz shard per task in --out, plus meta.json describing the run.
Speed: ~100-200 hands/s per worker against equity bots, faster vs baselines.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from game import BIG_BLIND, STARTING_STACK, Action, AgentConfig, Game, Street, make_deal  # noqa: E402

# name: (file, env overrides, sampling weight)
OPPONENTS = {
    "random":          ("baseline-bots/baseline_random.py", {}, 1.0),
    "calling_station": ("baseline-bots/baseline_calling_station.py", {}, 2.0),
    "always_half":     ("baseline-bots/baseline_always_half.py", {}, 1.5),
    "always_pot":      ("baseline-bots/baseline_always_pot.py", {}, 1.0),
    "v1":              ("opponents/equitybot_v1.py", {}, 1.5),
    # agent.py's hand-written rules (USE_MODEL=0) in several styles...
    "self":            ("agent.py", {"USE_MODEL": "0"}, 1.5),
    "self_tight":      ("agent.py", {"USE_MODEL": "0", "PRE_OPEN_RAISE": "0.56", "PRE_OPEN_CALL": "0.47",
                                     "POST_CALL_MARGIN": "0.08", "BLUFF_FREQ": "0.1"}, 1.5),
    "self_loose":      ("agent.py", {"USE_MODEL": "0", "PRE_OPEN_CALL": "0.40", "PRE_CALL_MARGIN": "0.04",
                                     "POST_CALL_MARGIN": "-0.03", "BLUFF_FREQ": "0.4"}, 1.5),
    "self_aggro":      ("agent.py", {"USE_MODEL": "0", "PRE_OPEN_RAISE": "0.50", "PRE_3BET": "0.55",
                                     "POST_BET_HALF": "0.52", "POST_BET_POT": "0.62",
                                     "POST_RAISE": "0.62", "BLUFF_FREQ": "0.5"}, 1.5),
    # ...and agent.py with whatever learned model is in models/ (if any).
    "model":           ("agent.py", {}, 2.0),
    # frozen earlier models (league-style: every past version stays in the pool)
    "model_v3":        ("agent.py", {"MODEL_PATH": "models/v3/q_trees.npz"}, 1.5),
}
OPP_NAMES = list(OPPONENTS)
HALF_LENGTH = 1000   # hands between new_half() calls, like the tournament

_MODULES = {}        # per-process cache: (file, env) -> loaded module


def load_bot_module(path, env):
    """Load a bot file as a fresh module with env overrides applied at import
    time (bot parameters are read when the module loads)."""
    key = (path, tuple(sorted(env.items())))
    if key in _MODULES:
        return _MODULES[key]
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        name = f"bot_{len(_MODULES)}_{os.path.basename(path)[:-3]}"
        spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _MODULES[key] = mod
    return mod


class Bot:
    def __init__(self, path, env=None):
        self.mod = load_bot_module(path, env or {})
        self.mem = {}

    def new_half(self):
        self.mem = {}
        self.mod.new_half(AgentConfig(), self.mem)

    def act(self, obs):
        return self.mod.choose_action(obs, self.mem)

    def hand_end(self, result):
        self.mod.on_hand_end(result, self.mem)


def run_task(task):
    """Play `hands` hands of the learner vs one opponent; return a shard path."""
    import features  # imported here so each worker process builds its own tables

    opp_name, n_hands, seed, eps, out_dir, task_id, learner_env = task
    rng = random.Random(seed)
    learner = Bot("agent.py", learner_env)
    lmod = learner.mod
    opp_file, opp_env, _ = OPPONENTS[opp_name]
    opp = Bot(opp_file, opp_env)
    deal_seed = f"gen-{seed}"

    rows = {k: [] for k in ("X", "action", "greedy", "legal", "street", "explore", "hand", "stack")}
    hand_of_row = []
    rewards_by_hand = {}
    t0 = time.time()
    for h in range(n_hands):
        if h % HALF_LENGTH == 0:
            learner.new_half()
            opp.new_half()
        learner_seat = 0
        game = Game(make_deal(deal_seed, h), button_seat=h % 2, hand_index=h % HALF_LENGTH)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat != learner_seat:
                game.apply(opp.act(obs))
                continue
            greedy = learner.act(obs)
            action, explored = greedy, 0
            if rng.random() < eps:
                options = [a for a in obs.legal_actions if a != Action.FOLD]
                action = rng.choice(options)
                explored = 1
            mem = learner.mem
            base = lmod.SAMPLES_PREFLOP if obs.street == Street.PREFLOP else lmod.SAMPLES_POSTFLOP
            x = features.extract(obs, lambda k: lmod._equity(obs, mem, k, base), lmod.opp_stats(mem))
            mem["time_used"] = 0.0   # no time bank offline: keep full sample counts
            rows["X"].append(x)
            rows["action"].append(int(action))
            rows["greedy"].append(int(greedy))
            rows["legal"].append([a in obs.legal_actions for a in Action])
            rows["street"].append(int(obs.street))
            rows["explore"].append(explored)
            rows["hand"].append(task_id * 10_000_000 + h)
            rows["stack"].append(obs.my_stack)
            hand_of_row.append(h)
            game.apply(action)
        learner.hand_end(game.hand_result(learner_seat))
        opp.hand_end(game.hand_result(1 - learner_seat))
        final_stack = STARTING_STACK + game.hand_result(learner_seat).my_net
        rewards_by_hand[h] = final_stack

    stack = np.asarray(rows.pop("stack"), dtype=np.float32)
    final = np.asarray([rewards_by_hand[h] for h in hand_of_row], dtype=np.float32)
    reward = (final - stack) / BIG_BLIND
    out = os.path.join(out_dir, f"shard_{task_id:05d}_{opp_name}.npz")
    np.savez_compressed(
        out,
        X=np.asarray(rows["X"], dtype=np.float32).reshape(-1, features.N_FEATURES),
        action=np.asarray(rows["action"], dtype=np.int8),
        greedy=np.asarray(rows["greedy"], dtype=np.int8),
        legal=np.asarray(rows["legal"], dtype=bool).reshape(-1, 4),
        street=np.asarray(rows["street"], dtype=np.int8),
        explore=np.asarray(rows["explore"], dtype=np.int8),
        hand=np.asarray(rows["hand"], dtype=np.int64),
        reward=reward.astype(np.float32),
        opp=np.full(len(reward), OPP_NAMES.index(opp_name), dtype=np.int8),
    )
    return out, opp_name, n_hands, len(reward), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=20000, help="total hands to play")
    ap.add_argument("--hands-per-task", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--eps", type=float, default=0.15, help="exploration probability per decision")
    ap.add_argument("--opponents", default="all", help="comma-separated names, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw/run")
    ap.add_argument("--learner-env", default="", help='e.g. "USE_MODEL=0 MODEL_MARGIN=1"')
    ap.add_argument("--list-opponents", action="store_true")
    args = ap.parse_args()

    if args.list_opponents:
        for k, (f, env, w) in OPPONENTS.items():
            print(f"{k:16s} weight {w:<4} {f} {env or ''}")
        return

    names = OPP_NAMES if args.opponents == "all" else args.opponents.split(",")
    for n in names:
        if n not in OPPONENTS:
            sys.exit(f"unknown opponent {n!r}; see --list-opponents")
    out_dir = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(out_dir, exist_ok=True)

    import features
    learner_env = dict(kv.split("=", 1) for kv in args.learner_env.split()) if args.learner_env else {}
    n_tasks = max(1, args.hands // args.hands_per_task)
    rng = random.Random(args.seed)
    weights = [OPPONENTS[n][2] for n in names]
    tasks = [(rng.choices(names, weights)[0], args.hands_per_task, args.seed * 1_000_003 + i,
              args.eps, out_dir, args.seed * 100_000 + i, learner_env) for i in range(n_tasks)]
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"feature_names": features.FEATURE_NAMES, "feature_version": features.FEATURE_VERSION,
                   "opponents": OPP_NAMES, "eps": args.eps, "hands": n_tasks * args.hands_per_task,
                   "seed": args.seed, "learner_env": learner_env, "actions": [a.name for a in Action]}, f, indent=1)

    print(f"{n_tasks} tasks x {args.hands_per_task} hands on {args.workers} workers -> {out_dir}")
    t0 = time.time()
    done_hands = rows = 0
    with Pool(args.workers) as pool:
        for i, (path, opp, nh, nr, secs) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            done_hands += nh
            rows += nr
            el = time.time() - t0
            print(f"[{i}/{n_tasks}] {opp:16s} {nh} hands, {nr} decisions in {secs:.0f}s | "
                  f"total {done_hands} hands, {rows} rows, {done_hands/el:.0f} hands/s", flush=True)
    print(f"done: {done_hands} hands, {rows} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
