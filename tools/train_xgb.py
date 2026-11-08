"""
train_xgb.py - train the XGBoost action-value model Q(state, action).

Q predicts "chips (in bb) won from this decision on" for CHECK_CALL,
RAISE_HALF and RAISE_POT. FOLD is exactly 0 by construction, so it needs no
model. The bot then plays argmax over the legal actions.

One model, with the action one-hot encoded as 3 extra input columns, so the
trees can share what they learn about the state across actions.

Run in the training env (Python 3.12, GPU):
    conda activate pokerbot-train
    python tools/train_xgb.py --data data/raw/run1 --minutes 10
    python tools/train_xgb.py --data data/raw/run1 data/raw/run2 --device cpu --minutes 2

Writes:
    models/q_xgb.ubj            the xgboost model (best validation iteration)
    models/q_trees.npz          the same trees as numpy arrays: what agent.py loads
    models/q_xgb_meta.json      feature names, version, best iteration, metrics
    models/train_log.csv        train/val loss after every boosting round
    models/training_loss.png    loss curve
"""

import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402

from tools.dataset import load  # noqa: E402

MODEL_ACTIONS = (1, 2, 3)   # CHECK_CALL, RAISE_HALF, RAISE_POT (FOLD = 0 is never modelled)
ACTION_COLS = ["act_check_call", "act_raise_half", "act_raise_pot"]


def with_action(X, actions):
    """Append the 3 one-hot action columns."""
    oh = np.zeros((len(X), 3), dtype=np.float32)
    oh[np.arange(len(X)), np.asarray(actions) - 1] = 1.0
    return np.hstack([X, oh])


class Logger(xgb.callback.TrainingCallback):
    """Records losses each round, prints a line every `every` rounds ("sets"),
    and stops when the time budget is used up."""

    def __init__(self, budget_s, every, baseline):
        self.budget_s, self.every, self.baseline = budget_s, every, baseline
        self.t0 = time.time()
        self.rows = []

    def after_iteration(self, model, epoch, evals_log):
        tr = evals_log["train"]["rmse"][-1]
        va = evals_log["val"]["rmse"][-1]
        el = time.time() - self.t0
        self.rows.append((epoch + 1, tr, va, el))
        if (epoch + 1) % self.every == 0:
            best = min(r[2] for r in self.rows)
            print(f"set {(epoch + 1) // self.every:4d} | round {epoch + 1:5d} | train rmse {tr:.4f} | "
                  f"val rmse {va:.4f} (best {best:.4f}, baseline {self.baseline:.4f}) | {el:6.1f}s",
                  flush=True)
        return el > self.budget_s   # True = stop


def plot(rows, baseline, best_iter, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, zoom in ((ax1, False), (ax2, True)):
        ax.plot(it, [r[1] for r in rows], label="train RMSE", color="#2a6fdb")
        ax.plot(it, [r[2] for r in rows], label="validation RMSE", color="#e8590c")
        ax.axhline(baseline, ls="--", color="gray", label="baseline (predict the mean)")
        ax.axvline(best_iter, ls=":", color="black", label=f"best round {best_iter}")
        ax.set_xlabel("boosting round")
        ax.set_ylabel("RMSE (bb)")
        ax.grid(alpha=0.3)
        if zoom:
            lo = min(min(r[1] for r in rows), min(r[2] for r in rows))
            hi = baseline
            pad = (hi - lo) * 0.1
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title("zoomed")
        else:
            ax.set_title(title)
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/raw/run1"])
    ap.add_argument("--minutes", type=float, default=10.0, help="training time budget")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eta", type=float, default=0.02)
    ap.add_argument("--max-depth", type=int, default=9)
    ap.add_argument("--min-child-weight", type=float, default=300)
    ap.add_argument("--patience", type=int, default=1000, help="early-stopping rounds on val loss")
    ap.add_argument("--set-size", type=int, default=100, help="rounds per logged 'set'")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    d = load([os.path.join(ROOT, p) if not os.path.isabs(p) else p for p in args.data])
    keep = d["action"] != 0                     # drop folds: their value is exactly 0
    X = with_action(d["X"][keep], d["action"][keep])
    y = d["reward"][keep]
    hands = d["hand"][keep]
    # Split by HAND so decisions from one hand never sit in both train and val.
    rng = np.random.default_rng(0)
    uniq = np.unique(hands)
    val_hands = set(rng.choice(uniq, int(len(uniq) * args.val_frac), replace=False).tolist())
    is_val = np.fromiter((h in val_hands for h in hands), dtype=bool, count=len(hands))
    names = d["meta"]["feature_names"] + ACTION_COLS
    dtr = xgb.DMatrix(X[~is_val], y[~is_val], feature_names=names)
    dva = xgb.DMatrix(X[is_val], y[is_val], feature_names=names)
    baseline = float(np.sqrt(np.mean((y[is_val] - y[~is_val].mean()) ** 2)))
    print(f"train {(~is_val).sum()} rows, val {is_val.sum()} rows, {X.shape[1]} inputs, "
          f"baseline val rmse {baseline:.4f}, device {args.device}")

    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "device": args.device, "tree_method": "hist",
        "eta": args.eta, "max_depth": args.max_depth, "min_child_weight": args.min_child_weight,
        "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 10.0,
    }
    logger = Logger(args.minutes * 60, args.set_size, baseline)
    stopper = xgb.callback.EarlyStopping(rounds=args.patience, save_best=True, data_name="val",
                                         metric_name="rmse")
    bst = xgb.train(params, dtr, num_boost_round=1_000_000, evals=[(dtr, "train"), (dva, "val")],
                    callbacks=[logger, stopper], verbose_eval=False)

    rows = logger.rows
    best_iter = int(np.argmin([r[2] for r in rows])) + 1
    best_val = min(r[2] for r in rows)
    stopped = "time budget" if rows[-1][3] > args.minutes * 60 else "early stopping"
    print(f"stopped by {stopped} after {len(rows)} rounds / {rows[-1][3]:.0f}s; "
          f"best val rmse {best_val:.4f} at round {best_iter} (baseline {baseline:.4f}, "
          f"R^2 {1 - (best_val / baseline) ** 2:.4f})")

    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    bst.set_param({"device": "cpu"})            # the tournament has no GPU
    bst.save_model(os.path.join(out, "q_xgb.ubj"))
    # numpy-only copy that agent.py actually loads (fast, no xgboost needed)
    import tree_model
    npz = os.path.join(out, "q_trees.npz")
    tree_model.export(bst, npz)
    check = X[is_val][:5000]
    diff = np.abs(tree_model.TreeModel.load(npz).predict(check) - bst.inplace_predict(check)).max()
    print(f"numpy export -> {npz} (max diff vs xgboost {diff:.2e})")
    if diff > 1e-3:
        raise RuntimeError("numpy tree export doesn't match xgboost")
    with open(os.path.join(out, "train_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "train_rmse", "val_rmse", "seconds"])
        w.writerows(rows)
    meta = {
        "feature_names": d["meta"]["feature_names"], "feature_version": d["meta"]["feature_version"],
        "action_cols": ACTION_COLS, "model_actions": list(MODEL_ACTIONS),
        "best_iteration": best_iter, "rounds_trained": len(rows), "stopped_by": stopped,
        "best_val_rmse": best_val, "baseline_val_rmse": baseline, "params": params,
        "data": args.data, "train_rows": int((~is_val).sum()), "val_rows": int(is_val.sum()),
    }
    with open(os.path.join(out, "q_xgb_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    plot(rows, baseline, best_iter, os.path.join(out, "training_loss.png"),
         f"XGBoost Q-model on {', '.join(args.data)}")

    # Feature importance (gain), top 15.
    gain = bst.get_score(importance_type="gain")
    print("top features by gain:")
    for k, v in sorted(gain.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {k:24s} {v:10.1f}")


if __name__ == "__main__":
    main()
