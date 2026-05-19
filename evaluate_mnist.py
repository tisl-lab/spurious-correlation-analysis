"""
evaluate_mnist.py
=================
Reads all CSVs from results/all_experiments/ and produces a single
multi-panel comparison figure showing how zero-shot, random fine-tuning,
and color-biased fine-tuning differ on Colored MNIST.

Usage:
    python evaluate_mnist.py
    python evaluate_mnist.py --results_dir results/all_experiments --output_dir results/evaluation
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _first_csv(directory, prefix):
    """Return the first CSV matching <directory>/<prefix>_*.csv, or None."""
    pattern = os.path.join(directory, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None


def overall_accuracy(per_digit_df):
    """Compute overall accuracy from per_digit DataFrame."""
    return per_digit_df["correct"].sum() / per_digit_df["total"].sum() * 100


def spurious_strength(per_digit_per_color_df):
    """
    For each digit, compute max_color_acc - min_color_acc.
    Return the mean across digits (higher = more spurious correlation).
    """
    gaps = []
    for digit, grp in per_digit_per_color_df.groupby("digit"):
        if len(grp) >= 2:
            gaps.append(grp["accuracy"].max() - grp["accuracy"].min())
    return np.mean(gaps) if gaps else 0.0


def load_run(directory):
    """Load all CSVs from one run directory. Returns dict of DataFrames."""
    result = {}
    for prefix in ("per_digit", "per_color", "per_digit_per_color"):
        f = _first_csv(directory, prefix)
        if f:
            result[prefix] = pd.read_csv(f)
    return result


# ── data loading ─────────────────────────────────────────────────────────────

def load_all(results_dir):
    runs = {}

    # Zero-shot
    zs_dir = os.path.join(results_dir, "zeroshot")
    if os.path.isdir(zs_dir):
        runs["zeroshot"] = load_run(zs_dir)

    # Random FT
    for pct in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        d = os.path.join(results_dir, "random_ft", f"pct_{pct}")
        if os.path.isdir(d):
            runs[f"random_{pct}"] = load_run(d)

    # Color-biased FT
    for pd_val in [10, 50, 100, 150]:
        d = os.path.join(results_dir, "color_biased", f"pd_{pd_val}")
        if os.path.isdir(d):
            runs[f"colorbiased_{pd_val}"] = load_run(d)

    return runs


# ── plotting ─────────────────────────────────────────────────────────────────

COLORS = {"blue": "#4C72B0", "green": "#55A868", "red": "#C44E52"}
DIGITS = list(range(10))
RANDOM_PCTS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
CB_PDS = [10, 50, 100, 150]


def plot_all(runs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle("CLIP on Colored MNIST — Experiment Comparison", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # Overall accuracy — full width
    ax2 = fig.add_subplot(gs[1, 0])   # Per-color accuracy across random FT
    ax3 = fig.add_subplot(gs[1, 1])   # Spurious correlation strength
    ax4 = fig.add_subplot(gs[2, 0])   # Color-biased per-color accuracy
    ax5 = fig.add_subplot(gs[2, 1])   # Per-digit accuracy: zero-shot vs best FT

    _panel1_overall(ax1, runs)
    _panel2_per_color_random(ax2, runs)
    _panel3_spurious(ax3, runs)
    _panel4_colorbiased(ax4, runs)
    _panel5_per_digit(ax5, runs)

    out_path = os.path.join(output_dir, "mnist_evaluation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Panel 1: Overall accuracy bar chart ──────────────────────────────────────

def _panel1_overall(ax, runs):
    labels, accs, bar_colors = [], [], []

    # Zero-shot
    if "zeroshot" in runs and "per_digit" in runs["zeroshot"]:
        labels.append("Zero-shot")
        accs.append(overall_accuracy(runs["zeroshot"]["per_digit"]))
        bar_colors.append("#888888")

    # Random FT
    for pct in RANDOM_PCTS:
        key = f"random_{pct}"
        if key in runs and "per_digit" in runs[key]:
            labels.append(f"Rand FT {pct}%")
            accs.append(overall_accuracy(runs[key]["per_digit"]))
            bar_colors.append("#4C72B0")

    # Color-biased FT
    for pd_val in CB_PDS:
        key = f"colorbiased_{pd_val}"
        if key in runs and "per_digit" in runs[key]:
            labels.append(f"CB FT {pd_val}pd")
            accs.append(overall_accuracy(runs[key]["per_digit"]))
            bar_colors.append("#C44E52")

    x = np.arange(len(labels))
    bars = ax.bar(x, accs, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Panel 1 — Overall Accuracy Across All Settings")
    ax.set_ylim(0, 105)
    ax.axhline(accs[0] if accs else 0, color="#888888", linestyle="--", linewidth=1, alpha=0.6, label="Zero-shot baseline")
    ax.legend(fontsize=8)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", label="Zero-shot"),
        Patch(facecolor="#4C72B0", label="Random FT"),
        Patch(facecolor="#C44E52", label="Color-biased FT"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")


# ── Panel 2: Per-color accuracy across random FT sweep ───────────────────────

def _panel2_per_color_random(ax, runs):
    x_vals = []
    color_accs = {"blue": [], "green": [], "red": []}

    for pct in RANDOM_PCTS:
        key = f"random_{pct}"
        if key in runs and "per_color" in runs[key]:
            df = runs[key]["per_color"]
            x_vals.append(pct)
            for c in ("blue", "green", "red"):
                row = df[df["color"] == c]
                color_accs[c].append(row["accuracy"].values[0] if len(row) else np.nan)

    if not x_vals:
        ax.text(0.5, 0.5, "No random FT data", ha="center", va="center", transform=ax.transAxes)
        return

    for c in ("blue", "green", "red"):
        ax.plot(x_vals, color_accs[c], marker="o", color=COLORS[c], label=f"{c.capitalize()} bg", linewidth=2)

    # Add zero-shot reference lines
    if "zeroshot" in runs and "per_color" in runs["zeroshot"]:
        df_zs = runs["zeroshot"]["per_color"]
        for c in ("blue", "green", "red"):
            row = df_zs[df_zs["color"] == c]
            if len(row):
                ax.axhline(row["accuracy"].values[0], color=COLORS[c], linestyle=":", linewidth=1, alpha=0.7)

    ax.set_xlabel("Fine-tuning data (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Panel 2 — Per-Color Accuracy vs FT Data Size\n(dotted = zero-shot baseline)")
    ax.legend(fontsize=8)
    ax.set_xticks(RANDOM_PCTS)
    ax.set_ylim(0, 105)


# ── Panel 3: Spurious correlation strength ───────────────────────────────────

def _panel3_spurious(ax, runs):
    labels, strengths, bar_colors = [], [], []

    if "zeroshot" in runs and "per_digit_per_color" in runs["zeroshot"]:
        labels.append("Zero-shot")
        strengths.append(spurious_strength(runs["zeroshot"]["per_digit_per_color"]))
        bar_colors.append("#888888")

    for pct in RANDOM_PCTS:
        key = f"random_{pct}"
        if key in runs and "per_digit_per_color" in runs[key]:
            labels.append(f"Rand {pct}%")
            strengths.append(spurious_strength(runs[key]["per_digit_per_color"]))
            bar_colors.append("#4C72B0")

    for pd_val in CB_PDS:
        key = f"colorbiased_{pd_val}"
        if key in runs and "per_digit_per_color" in runs[key]:
            labels.append(f"CB {pd_val}pd")
            strengths.append(spurious_strength(runs[key]["per_digit_per_color"]))
            bar_colors.append("#C44E52")

    x = np.arange(len(labels))
    ax.bar(x, strengths, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Avg. Accuracy Gap (max−min across colors) %")
    ax.set_title("Panel 3 — Spurious Correlation Strength\n(higher = stronger color bias in predictions)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", label="Zero-shot"),
        Patch(facecolor="#4C72B0", label="Random FT"),
        Patch(facecolor="#C44E52", label="Color-biased FT"),
    ]
    ax.legend(handles=legend_elements, fontsize=7)


# ── Panel 4: Color-biased FT — per-color accuracy ────────────────────────────

def _panel4_colorbiased(ax, runs):
    available_pds = [pd_val for pd_val in CB_PDS if f"colorbiased_{pd_val}" in runs
                     and "per_color" in runs[f"colorbiased_{pd_val}"]]

    if not available_pds:
        ax.text(0.5, 0.5, "No color-biased data", ha="center", va="center", transform=ax.transAxes)
        return

    color_accs = {"blue": [], "green": [], "red": []}
    for pd_val in available_pds:
        df = runs[f"colorbiased_{pd_val}"]["per_color"]
        for c in ("blue", "green", "red"):
            row = df[df["color"] == c]
            color_accs[c].append(row["accuracy"].values[0] if len(row) else np.nan)

    x = np.arange(len(available_pds))
    width = 0.25
    for i, c in enumerate(("blue", "green", "red")):
        ax.bar(x + (i - 1) * width, color_accs[c], width, color=COLORS[c],
               label=f"{c.capitalize()} bg", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{pd_val} per-digit" for pd_val in available_pds], fontsize=8)
    ax.set_xlabel("Training images per digit (color-biased)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Panel 4 — Color-Biased FT: Per-Color Accuracy\n(digits 0-2→blue, 3-5→green, 6-9→red in training)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)


# ── Panel 5: Per-digit accuracy comparison ───────────────────────────────────

def _panel5_per_digit(ax, runs):
    settings = {}

    if "zeroshot" in runs and "per_digit" in runs["zeroshot"]:
        settings["Zero-shot"] = runs["zeroshot"]["per_digit"].set_index("digit")["accuracy"]

    # Best random FT by overall accuracy
    best_key, best_acc = None, -1
    for pct in RANDOM_PCTS:
        key = f"random_{pct}"
        if key in runs and "per_digit" in runs[key]:
            acc = overall_accuracy(runs[key]["per_digit"])
            if acc > best_acc:
                best_acc, best_key = acc, key
    if best_key:
        pct = best_key.split("_")[1]
        settings[f"Best Rand FT ({pct}%)"] = runs[best_key]["per_digit"].set_index("digit")["accuracy"]

    # Best color-biased FT
    best_cb_key, best_cb_acc = None, -1
    for pd_val in CB_PDS:
        key = f"colorbiased_{pd_val}"
        if key in runs and "per_digit" in runs[key]:
            acc = overall_accuracy(runs[key]["per_digit"])
            if acc > best_cb_acc:
                best_cb_acc, best_cb_key = acc, key
    if best_cb_key:
        pd_val = best_cb_key.split("_")[1]
        settings[f"Best CB FT ({pd_val}pd)"] = runs[best_cb_key]["per_digit"].set_index("digit")["accuracy"]

    if not settings:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    digits = DIGITS
    x = np.arange(len(digits))
    n = len(settings)
    width = 0.8 / n
    palette = ["#888888", "#4C72B0", "#C44E52"]

    for i, (label, series) in enumerate(settings.items()):
        accs = []
        for d in digits:
            val = series.get(d, np.nan)
            # Series.get() may return a sub-Series if index has duplicates
            accs.append(float(val.iloc[0]) if hasattr(val, "iloc") else float(val))
        ax.bar(x + (i - (n - 1) / 2) * width, accs, width, label=label,
               color=palette[i % len(palette)], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits])
    ax.set_xlabel("Digit")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Panel 5 — Per-Digit Accuracy\n(zero-shot vs best random FT vs best color-biased FT)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/all_experiments")
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
    print("\n" + "=" * 60)
    print(f"{'Setting':<30} {'Overall Acc':>12} {'Spurious Gap':>14}")
    print("-" * 60)
    for key in (["zeroshot"]
                + [f"random_{p}" for p in RANDOM_PCTS]
                + [f"colorbiased_{pd}" for pd in CB_PDS]):
        if key not in runs:
            continue
        run = runs[key]
        acc = overall_accuracy(run["per_digit"]) if "per_digit" in run else float("nan")
        gap = spurious_strength(run["per_digit_per_color"]) if "per_digit_per_color" in run else float("nan")
        label = key.replace("random_", "Random FT ").replace("colorbiased_", "Color-biased ").replace("zeroshot", "Zero-shot")
        print(f"{label:<30} {acc:>11.2f}% {gap:>13.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
