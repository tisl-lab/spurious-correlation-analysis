"""
evaluate_cifar.py
=================
Reads all CSVs from results/all_experiments_cifar/ and produces a single
multi-panel comparison figure comparing zero-shot vs random fine-tuning
on CIFAR-10.

Usage:
    python evaluate_cifar.py
    python evaluate_cifar.py --results_dir results/all_experiments_cifar --output_dir results/evaluation
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
RANDOM_PCTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]


# ── helpers ──────────────────────────────────────────────────────────────────

def _first_csv(directory, prefix):
    pattern = os.path.join(directory, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None


def overall_accuracy(per_class_df):
    return per_class_df["correct"].sum() / per_class_df["total"].sum() * 100


def load_run(directory):
    result = {}
    for prefix in ("per_class", "confusion_matrix"):
        f = _first_csv(directory, prefix)
        if f:
            result[prefix] = pd.read_csv(f)
    return result


# ── data loading ─────────────────────────────────────────────────────────────

def load_all(results_dir):
    runs = {}

    zs_dir = os.path.join(results_dir, "zeroshot")
    if os.path.isdir(zs_dir):
        runs["zeroshot"] = load_run(zs_dir)

    for pct in RANDOM_PCTS:
        d = os.path.join(results_dir, "random_ft", f"pct_{pct}")
        if os.path.isdir(d):
            runs[f"random_{pct}"] = load_run(d)

    return runs


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_all(runs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("CLIP on CIFAR-10 — Experiment Comparison", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :])   # Overall accuracy — full width
    ax2 = fig.add_subplot(gs[1, 0])   # Per-class accuracy heatmap
    ax3 = fig.add_subplot(gs[1, 1])   # Per-class delta vs zero-shot
    ax4 = fig.add_subplot(gs[2, 0])   # Confusion matrix (zero-shot)
    ax5 = fig.add_subplot(gs[2, 1])   # Confusion matrix (best FT)

    _panel1_overall(ax1, runs)
    _panel2_perclass_heatmap(ax2, runs)
    _panel3_delta(ax3, runs)
    _panel4_confusion(ax4, runs, "zeroshot", "Panel 4 — Confusion Matrix: Zero-shot")
    best_key = _best_run_key(runs)
    label = f"Panel 5 — Confusion Matrix: {best_key.replace('random_', 'Rand FT ')}%"
    _panel4_confusion(ax5, runs, best_key, label)

    out_path = os.path.join(output_dir, "cifar10_evaluation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _best_run_key(runs):
    best_key, best_acc = "zeroshot", -1
    for pct in RANDOM_PCTS:
        key = f"random_{pct}"
        if key in runs and "per_class" in runs[key]:
            acc = overall_accuracy(runs[key]["per_class"])
            if acc > best_acc:
                best_acc, best_key = acc, key
    return best_key


# ── Panel 1: Overall accuracy line chart ─────────────────────────────────────

def _panel1_overall(ax, runs):
    pcts, accs = [], []
    for pct in RANDOM_PCTS:
        key = f"random_{pct}"
        if key in runs and "per_class" in runs[key]:
            pcts.append(pct)
            accs.append(overall_accuracy(runs[key]["per_class"]))

    zs_acc = None
    if "zeroshot" in runs and "per_class" in runs["zeroshot"]:
        zs_acc = overall_accuracy(runs["zeroshot"]["per_class"])
        ax.axhline(zs_acc, color="#888888", linestyle="--", linewidth=1.5,
                   label=f"Zero-shot ({zs_acc:.1f}%)")

    if pcts:
        ax.plot(pcts, accs, marker="o", color="#4C72B0", linewidth=2, markersize=6, label="Random FT")
        for x, y in zip(pcts, accs):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)

    ax.set_xlabel("Fine-tuning data (%)")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Panel 1 — Overall Accuracy vs Fine-Tuning Data Size")
    ax.set_xticks(RANDOM_PCTS)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)


# ── Panel 2: Per-class accuracy heatmap ──────────────────────────────────────

def _panel2_perclass_heatmap(ax, runs):
    ordered_keys = ["zeroshot"] + [f"random_{p}" for p in RANDOM_PCTS if f"random_{p}" in runs]
    available = [k for k in ordered_keys if k in runs and "per_class" in runs[k]]

    if not available:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    n_classes = len(CIFAR10_CLASSES)
    matrix = np.full((n_classes, len(available)), np.nan)

    for col_idx, key in enumerate(available):
        df = runs[key]["per_class"].set_index("class_name")
        for row_idx, cls in enumerate(CIFAR10_CLASSES):
            if cls in df.index:
                matrix[row_idx, col_idx] = df.loc[cls, "accuracy"]

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")

    col_labels = ["ZS" if k == "zeroshot" else f"{k.split('_')[1]}%" for k in available]
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax.set_xlabel("Setting")
    ax.set_title("Panel 2 — Per-Class Accuracy Heatmap\n(columns = settings, rows = classes)")

    for row in range(n_classes):
        for col in range(len(available)):
            val = matrix[row, col]
            if not np.isnan(val):
                ax.text(col, row, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color="black" if 30 < val < 80 else "white")


# ── Panel 3: Per-class accuracy delta vs zero-shot ───────────────────────────

def _panel3_delta(ax, runs):
    if "zeroshot" not in runs or "per_class" not in runs["zeroshot"]:
        ax.text(0.5, 0.5, "No zero-shot data", ha="center", va="center", transform=ax.transAxes)
        return

    zs_df = runs["zeroshot"]["per_class"].set_index("class_name")
    best_key = _best_run_key(runs)

    if best_key == "zeroshot" or "per_class" not in runs.get(best_key, {}):
        ax.text(0.5, 0.5, "No FT data to compare", ha="center", va="center", transform=ax.transAxes)
        return

    ft_df = runs[best_key]["per_class"].set_index("class_name")
    deltas = []
    for cls in CIFAR10_CLASSES:
        zs = zs_df.loc[cls, "accuracy"] if cls in zs_df.index else np.nan
        ft = ft_df.loc[cls, "accuracy"] if cls in ft_df.index else np.nan
        deltas.append(ft - zs if not (np.isnan(zs) or np.isnan(ft)) else 0.0)

    bar_colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]
    x = np.arange(len(CIFAR10_CLASSES))
    ax.barh(x, deltas, color=bar_colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax.set_xlabel("Accuracy Change (%)")
    pct_label = best_key.replace("random_", "") + "%"
    ax.set_title(f"Panel 3 — Per-Class Accuracy Δ vs Zero-shot\n(best FT = {pct_label}, green = improved)")

    for xi, d in zip(x, deltas):
        ax.text(d + (0.3 if d >= 0 else -0.3), xi, f"{d:+.1f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=7)


# ── Panels 4 & 5: Confusion matrix heatmap ───────────────────────────────────

def _panel4_confusion(ax, runs, key, title):
    if key not in runs or "confusion_matrix" not in runs[key]:
        ax.text(0.5, 0.5, f"No data for {key}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    df = runs[key]["confusion_matrix"]
    n = len(CIFAR10_CLASSES)
    matrix = np.zeros((n, n))
    cls_to_idx = {c: i for i, c in enumerate(CIFAR10_CLASSES)}

    for _, row in df.iterrows():
        ti = cls_to_idx.get(row["true_class"])
        pi = cls_to_idx.get(row["pred_class"])
        if ti is not None and pi is not None:
            matrix[ti, pi] = row["count"]

    # Row-normalise
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = matrix / row_sums * 100

    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, shrink=0.8, label="% of true class")

    short = [c[:4] for c in CIFAR10_CLASSES]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            val = norm_matrix[i, j]
            if val > 5:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 60 else "black")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/all_experiments_cifar")
    parser.add_argument("--output_dir", default="results/evaluation")
    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    runs = load_all(args.results_dir)
    print(f"Found {len(runs)} run(s): {list(runs.keys())}")

    if not runs:
        print("No results found. Run the experiment suite first.")
        return

    plot_all(runs, args.output_dir)

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Setting':<25} {'Overall Acc':>12}")
    print("-" * 55)
    for key in (["zeroshot"] + [f"random_{p}" for p in RANDOM_PCTS]):
        if key not in runs or "per_class" not in runs[key]:
            continue
        acc = overall_accuracy(runs[key]["per_class"])
        label = key.replace("random_", "Random FT ").replace("zeroshot", "Zero-shot")
        if "random" in key:
            label += "%"
        print(f"{label:<25} {acc:>11.2f}%")
    print("=" * 55)

    # Per-class summary for best FT vs zero-shot
    best_key = _best_run_key(runs)
    if best_key != "zeroshot" and "per_class" in runs.get(best_key, {}):
        zs_df = runs["zeroshot"]["per_class"].set_index("class_name") if "per_class" in runs.get("zeroshot", {}) else None
        ft_df = runs[best_key]["per_class"].set_index("class_name")
        pct = best_key.replace("random_", "")
        print(f"\nPer-class: Zero-shot vs Best FT ({pct}%)")
        print(f"{'Class':<14} {'Zero-shot':>10} {'Best FT':>10} {'Δ':>8}")
        print("-" * 48)
        for cls in CIFAR10_CLASSES:
            zs = zs_df.loc[cls, "accuracy"] if zs_df is not None and cls in zs_df.index else float("nan")
            ft = ft_df.loc[cls, "accuracy"] if cls in ft_df.index else float("nan")
            delta = ft - zs if not (np.isnan(zs) or np.isnan(ft)) else float("nan")
            print(f"{cls:<14} {zs:>9.1f}% {ft:>9.1f}% {delta:>+7.1f}%")


if __name__ == "__main__":
    main()
