"""
dataset.py - load shards written by tools/gen_data.py.

    from tools.dataset import load
    d = load("data/raw/run1")            # or several: load(["data/raw/a", "data/raw/b"])
    d["X"], d["action"], d["reward"], d["legal"], ...
    d["meta"]["feature_names"]

Also: python tools/dataset.py data/raw/run1   -> prints a summary (sanity checks).
"""

import glob
import json
import os
import sys

import numpy as np

ACTIONS = ["FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT"]
STREETS = ["PREFLOP", "FLOP", "TURN", "RIVER"]


def load(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]
    parts, meta = [], None
    for d in dirs:
        with open(os.path.join(d, "meta.json")) as f:
            m = json.load(f)
        if meta and m["feature_names"] != meta["feature_names"]:
            raise ValueError(f"{d}: features differ from {dirs[0]}; regenerate the data")
        meta = meta or m
        for path in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
            with np.load(path) as z:
                parts.append({k: z[k] for k in z.files})
    if not parts:
        raise FileNotFoundError(f"no shards in {dirs}")
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["meta"] = meta
    return data


def summary(d):
    n = len(d["reward"])
    names = d["meta"]["feature_names"]
    print(f"{n} decisions from {len(np.unique(d['hand']))} hands, {len(names)} features, "
          f"{d['explore'].mean():.1%} exploratory")
    bad = ~np.isfinite(d["X"]).all(axis=1)
    print(f"rows with NaN/inf features: {bad.sum()}")

    print("\nopponents:", {d["meta"]["opponents"][i]: int((d["opp"] == i).sum())
                           for i in np.unique(d["opp"])})

    print("\nMean reward (bb from this point) by street x action, exploratory rows only")
    print("(these are the unbiased 'what if I took a random action' values; FOLD is 0 by definition)")
    ex = d["explore"] == 1
    print(f"{'':10s}" + "".join(f"{a:>22s}" for a in ACTIONS[1:]))
    for s, sn in enumerate(STREETS):
        cells = []
        for a in (1, 2, 3):
            m = ex & (d["street"] == s) & (d["action"] == a)
            r = d["reward"][m]
            cells.append(f"{r.mean():+7.2f} ±{1.96 * r.std() / max(np.sqrt(len(r)), 1):5.2f} (n={len(r):>6})"
                         if len(r) > 1 else f"{'-':>22s}")
        print(f"{sn:10s}" + "".join(f"{c:>22s}" for c in cells))

    g = ~ex
    print("\nGreedy (agent.py's own) action mix by street:")
    for s, sn in enumerate(STREETS):
        m = g & (d["street"] == s)
        mix = np.bincount(d["action"][m], minlength=4) / max(m.sum(), 1)
        print(f"  {sn:8s} " + "  ".join(f"{ACTIONS[a]} {mix[a]:5.1%}" for a in range(4)))

    print("\nFeature ranges:")
    X = d["X"]
    for i, nm in enumerate(names):
        print(f"  {nm:24s} min {X[:, i].min():8.3f}  mean {X[:, i].mean():8.3f}  max {X[:, i].max():8.3f}")


if __name__ == "__main__":
    summary(load(sys.argv[1:] or "data/raw/run"))
