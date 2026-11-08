"""
analyze_model.py - how well is the action-value model learning?

    python tools/analyze_model.py --data data/raw/run1 --model models/v3/q_trees.npz

Uses the same hand-level validation split as train_xgb.py (rows it never
trained on) and reports:

1. Calibration: mean predicted vs mean actual reward in prediction deciles.
2. Off-policy value estimates from the EXPLORATORY rows. Exploration picks a
   uniformly random legal non-fold action, so for any policy pi:
       V(pi) ~= mean over explore rows of  r * 1[a == pi(s)] * n_options(s)
   (plus 0 when pi folds, since folding is worth exactly 0). This gives an
   unbiased estimate of "bb per decision if we played pi at this decision
   and the recorded policy afterwards" for both the model and the rules, and
   their difference with a confidence interval, by street.
3. Overrides: how often the model's argmax differs from the rule action.
4. Plot: average predicted value of each action vs equity, by street/spot.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from tools.dataset import ACTIONS, STREETS, load  # noqa: E402
from tree_model import TreeModel  # noqa: E402


def val_mask(hands, frac=0.1):
    """Exactly train_xgb.py's split (it drops folds first, but the RNG draws
    over unique hands of the non-fold rows; we reproduce that)."""
    rng = np.random.default_rng(0)
    uniq = np.unique(hands)
    val = set(rng.choice(uniq, int(len(uniq) * frac), replace=False).tolist())
    return val


def q_all(model, X):
    """Predicted value for FOLD(0), CHECK_CALL, RAISE_HALF, RAISE_POT."""
    n, f = X.shape
    out = np.zeros((n, 4))
    for a in (1, 2, 3):
        Z = np.zeros((n, f + 3), dtype=np.float32)
        Z[:, :f] = X
        Z[:, f + a - 1] = 1.0
        out[:, a] = model.predict(Z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/raw/run1"])
    ap.add_argument("--model", default="models/q_trees.npz")
    ap.add_argument("--plot", default="models/analysis_q_vs_equity.png")
    ap.add_argument("--all-rows", action="store_true", help="don't restrict to the validation split")
    args = ap.parse_args()

    d = load([os.path.join(ROOT, p) for p in args.data])
    model = TreeModel.load(os.path.join(ROOT, args.model))
    names = d["meta"]["feature_names"]
    if args.all_rows:
        m = np.ones(len(d["reward"]), dtype=bool)
    else:
        nonfold = d["action"] != 0
        val = val_mask(d["hand"][nonfold])
        m = np.fromiter((h in val for h in d["hand"]), dtype=bool, count=len(d["hand"]))
    X, a, r, legal = d["X"][m], d["action"][m], d["reward"][m], d["legal"][m]
    street, explore, greedy = d["street"][m], d["explore"][m] == 1, d["greedy"][m]
    print(f"{m.sum()} validation decisions ({explore.sum()} exploratory), model {args.model} "
          f"({model.n_trees} trees)")

    Q = q_all(model, X)
    Qm = np.where(legal, Q, -np.inf)
    pi_model = Qm.argmax(axis=1)

    # 1. calibration (non-fold rows, the taken action's prediction)
    nf = a != 0
    pred = Q[nf, a[nf]]
    act = r[nf]
    edges = np.quantile(pred, np.linspace(0, 1, 11))
    print("\n1. Calibration (taken action): predicted vs actual mean reward, by prediction decile")
    for i in range(10):
        b = (pred >= edges[i]) & (pred <= edges[i + 1])
        print(f"   decile {i + 1:2d}: predicted {pred[b].mean():+7.2f}   actual {act[b].mean():+7.2f}"
              f"  ±{1.96 * act[b].std() / np.sqrt(b.sum()):.2f}")

    # 2. off-policy estimates on exploratory rows
    n_opt = legal[:, 1:].sum(axis=1)            # non-fold legal options
    print("\n2. Off-policy value per decision (bb), exploratory rows: model vs rules")
    print(f"   {'street':8s} {'rules':>16s} {'model':>16s} {'model - rules':>22s}")
    for s in list(range(4)) + [None]:
        sel = explore & ((street == s) if s is not None else True)
        w_r = (a[sel] == greedy[sel]) * n_opt[sel]
        w_m = (a[sel] == pi_model[sel]) * n_opt[sel]
        v_r = r[sel] * w_r
        v_m = r[sel] * w_m
        diff = v_m - v_r
        se = lambda v: 1.96 * v.std() / np.sqrt(len(v))
        label = STREETS[s] if s is not None else "ALL"
        print(f"   {label:8s} {v_r.mean():+8.3f} ±{se(v_r):5.3f} {v_m.mean():+8.3f} ±{se(v_m):5.3f}"
              f"   {diff.mean():+8.3f} ±{se(diff):5.3f}")

    # 3. overrides
    print("\n3. Model choice vs rule choice (non-exploratory rows), share of decisions per street")
    g = ~explore
    for s in range(4):
        sel = g & (street == s)
        ov = (pi_model[sel] != greedy[sel]).mean()
        print(f"   {STREETS[s]:8s} overrides {ov:5.1%}")
        M = np.zeros((4, 4))
        for i, j in zip(greedy[sel], pi_model[sel]):
            M[i, j] += 1
        M /= max(sel.sum(), 1)
        for i in range(4):
            cells = "  ".join(f"{ACTIONS[j][:10]:>10s} {M[i, j]:5.1%}" for j in range(4) if M[i, j] >= 0.005 and i != j)
            if cells:
                print(f"      rules {ACTIONS[i]:10s} -> {cells}")

    # 4. plot predicted Q vs equity
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipping plot)")
        return
    eq = X[:, names.index("eq_strong")]
    facing = X[:, names.index("to_call_frac")] > 0
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    bins = np.linspace(0, 1, 21)
    colors = {0: "#868e96", 1: "#2a6fdb", 2: "#f59f00", 3: "#e8590c"}
    for row, face in enumerate((False, True)):
        for s in range(4):
            ax = axes[row, s]
            sel = (street == s) & (facing == face)
            idx = np.digitize(eq[sel], bins) - 1
            xs = (bins[:-1] + bins[1:]) / 2
            for act_i in (0, 1, 2, 3):
                if act_i == 0 and not face:
                    continue
                ys = []
                for k in range(len(xs)):
                    v = Qm[sel][idx == k, act_i]
                    v = v[np.isfinite(v)]          # only rows where this action was legal
                    ys.append(v.mean() if len(v) > 30 else np.nan)
                ax.plot(xs, ys, color=colors[act_i], label=ACTIONS[act_i], lw=1.6)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{STREETS[s]} - {'facing a bet' if face else 'checked to / first'}", fontsize=9)
            ax.grid(alpha=0.3)
            if s == 0:
                ax.set_ylabel("predicted bb from here")
            if row == 1:
                ax.set_xlabel("equity vs strong range (eq_strong)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"What the model learned: predicted value of each action vs hand strength ({args.model})")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, args.plot), dpi=110)
    print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    main()
