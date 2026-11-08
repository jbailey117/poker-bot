"""
plot_sweep.py - overlay validation loss curves from several train_xgb.py runs.

    python tools/plot_sweep.py models/sweep            # every subfolder with a train_log.csv
    python tools/plot_sweep.py models/sweep --out models/sweep_loss.png
"""

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = sorted(glob.glob(os.path.join(args.folder, "*", "train_log.csv")))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    baseline = None
    for path in runs:
        name = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            rows = list(csv.DictReader(f))
        meta = json.load(open(os.path.join(os.path.dirname(path), "q_xgb_meta.json")))
        baseline = meta["baseline_val_rmse"]
        it = [int(r["round"]) for r in rows]
        va = [float(r["val_rmse"]) for r in rows]
        ax.plot(it, va, label=f"{name}  (best {min(va):.4f} @ {meta['best_iteration']})", lw=1.3)
    ax.set_xscale("log")
    ax.set_ylim(18.08, 18.6)
    ax.set_xlabel("boosting round (log scale)")
    ax.set_ylabel("validation RMSE (bb)")
    ax.set_title(f"Hyperparameter sweep: validation loss (predict-the-mean baseline = {baseline:.2f}, off the chart)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(args.folder, "sweep_loss.png")
    fig.savefig(out, dpi=120)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
