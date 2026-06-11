"""
analyze_waterbirds_results.py
=============================

Reads all waterbirds_results_<sample_size>_<timestamp>.csv files from a results
directory and plots per-group accuracy as a function of fine-tuning sample size.

Files without a numeric sample_size in their name are skipped (old runs).
When multiple files share the same sample_size the most recent timestamp wins.

Usage:
    python analyze_waterbirds_results.py
    python analyze_waterbirds_results.py --results_dir ./results/waterbirds
    python analyze_waterbirds_results.py --output plot_ft_curve.png
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


GROUP_NAMES = [
    "landbird / land bg",
    "landbird / water bg",
    "waterbird / land bg",
    "waterbird / water bg",
]
GROUP_COLORS = {
    0: "#59a14f",   # green  — spurious-aligned
    1: "#e15759",   # red    — counter-spurious
    2: "#f28e2b",   # orange — counter-spurious
    3: "#4e79a7",   # blue   — spurious-aligned
}
GROUP_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="./results/waterbirds")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: <results_dir>/ft_curve_<ts>.png)")
    return parser.parse_args()


def load_csvs(results_dir):
    pattern = re.compile(r"^waterbirds_results_(\d+)_(\d{8}_\d{6})\.csv$")

    # sample_size -> {timestamp_str -> filepath}
    candidates = defaultdict(dict)
    for fname in os.listdir(results_dir):
        m = pattern.match(fname)
        if not m:
            continue
        size, ts = int(m.group(1)), m.group(2)
        candidates[size][ts] = os.path.join(results_dir, fname)

    records = []
    zs_rows = None

    for size in sorted(candidates):
        latest_ts  = max(candidates[size])
        fpath      = candidates[size][latest_ts]
        df         = pd.read_csv(fpath)

        if zs_rows is None:
            zs_rows = df[["group_id", "zs_acc"]].copy()

        for _, row in df.iterrows():
            records.append({
                "sample_size": size,
                "group_id":    int(row["group_id"]),
                "ft_acc":      float(row["ft_acc"]),
            })

    if not records:
        raise RuntimeError(
            f"No waterbirds_results_<size>_<ts>.csv files found in {results_dir}"
        )

    return pd.DataFrame(records), zs_rows


def plot(data, zs_rows, output_path):
    sizes  = sorted(data["sample_size"].unique())
    # x=0 is the zero-shot point; FT points follow at their actual sample sizes
    x_ft   = np.array(sizes)
    x_all  = np.concatenate([[0], x_ft])

    fig, ax = plt.subplots(figsize=(10, 6))

    for gid, gname in enumerate(GROUP_NAMES):
        color  = GROUP_COLORS[gid]
        marker = GROUP_MARKERS[gid]
        label_suffix = "aligned" if gid in {0, 3} else "counter-spurious"

        zs_val = None
        if zs_rows is not None:
            zs_arr = zs_rows.loc[zs_rows["group_id"] == gid, "zs_acc"].values
            if len(zs_arr):
                zs_val = zs_arr[0]

        ft_accs = [
            data.loc[(data["sample_size"] == s) & (data["group_id"] == gid),
                     "ft_acc"].values[0]
            for s in sizes
        ]

        # Full curve starting from zero-shot point at x=0
        y_all = ([zs_val] + ft_accs) if zs_val is not None else ft_accs
        x_plot = x_all if zs_val is not None else x_ft
        ax.plot(x_plot, y_all, color=color, marker=marker, linewidth=2,
                markersize=7, label=f"{gname}  [{label_suffix}]")

        # ZS horizontal dashed reference line
        if zs_val is not None:
            ax.axhline(zs_val, color=color, linestyle="--",
                       linewidth=1.0, alpha=0.4)

    # Dummy handle for legend entry
    ax.plot([], [], color="gray", linestyle="--", linewidth=1.0,
            alpha=0.5, label="── zero-shot level (dashed)")

    ax.set_xlabel("Fine-tuning sample size (per class)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Per-Group Accuracy vs Fine-Tuning Size\n"
        "x=0 is zero-shot  ·  dashed = zero-shot reference level",
        fontsize=13, fontweight="bold",
    )
    tick_positions = x_all
    tick_labels    = ["ZS"] + [str(s) for s in sizes]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    data, zs_rows = load_csvs(args.results_dir)

    sizes = sorted(data["sample_size"].unique())
    print(f"Found {len(sizes)} sample sizes: {sizes}")

    if args.output:
        out = args.output
    else:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.results_dir, f"ft_curve_{ts}.png")

    plot(data, zs_rows, out)


if __name__ == "__main__":
    main()
