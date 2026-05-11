"""
visualize_sweep.py
==================
Reads per_digit / per_color CSVs produced by the fine-tuning sweep
(run_sweep.sh) and generates a visual summary.

Outputs saved to the same results directory:
  sweep_summary.png   — 4-panel figure
  sweep_summary.csv   — flat table with one row per fine-tune %
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sweep_results(results_dir: str) -> list[dict]:
    """
    Walk pct_0, pct_10, …, pct_90 subdirectories and load CSVs.
    Returns a list of dicts sorted by fine-tune percentage.
    """
    entries = []
    for name in os.listdir(results_dir):
        if not name.startswith("pct_"):
            continue
        dirpath = os.path.join(results_dir, name)
        if not os.path.isdir(dirpath):
            continue
        try:
            pct = int(name.split("pct_")[1])
        except ValueError:
            continue

        digit_csvs       = sorted(glob.glob(os.path.join(dirpath, "per_digit_*.csv")))
        color_csvs       = sorted(glob.glob(os.path.join(dirpath, "per_color_*.csv")))
        digit_color_csvs = sorted(glob.glob(os.path.join(dirpath, "per_digit_per_color_*.csv")))

        if not digit_csvs:
            print(f"  Warning: no per_digit CSV in {dirpath} — skipping")
            continue

        digit_df       = pd.read_csv(digit_csvs[-1])
        color_df       = pd.read_csv(color_csvs[-1])       if color_csvs       else None
        digit_color_df = pd.read_csv(digit_color_csvs[-1]) if digit_color_csvs else None

        overall_acc = digit_df["correct"].sum() / digit_df["total"].sum() * 100

        entries.append({
            "pct":               pct,
            "overall_acc":       overall_acc,
            "per_digit":         digit_df,
            "per_color":         color_df,
            "per_digit_per_color": digit_color_df,
        })

    return sorted(entries, key=lambda x: x["pct"])


# ── Plotting ──────────────────────────────────────────────────────────────────

COLOR_PALETTE = {
    "red":   "tomato",
    "green": "mediumseagreen",
    "blue":  "cornflowerblue",
}

def _annotate(ax, xs, ys, fontsize=7.5):
    for x, y in zip(xs, ys):
        if not np.isnan(y):
            ax.annotate(f"{y:.1f}", (x, y),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=fontsize)


def plot_overall(ax, records):
    pcts = [r["pct"] for r in records]
    accs = [r["overall_acc"] for r in records]

    ax.plot(pcts, accs, "o-", color="steelblue", linewidth=2.5, markersize=9, zorder=3)
    ax.fill_between(pcts, accs, alpha=0.12, color="steelblue")
    _annotate(ax, pcts, accs)

    ax.set_title("Overall Accuracy vs Fine-Tune %", fontweight="bold")
    ax.set_xlabel("Fine-Tune Percentage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(pcts)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)


def plot_per_digit(ax, records):
    pcts    = [r["pct"] for r in records]
    palette = cm.tab10(np.linspace(0, 1, 10))

    for digit in range(10):
        accs = []
        for r in records:
            row = r["per_digit"][r["per_digit"]["digit"] == digit]
            accs.append(row["accuracy"].values[0] if len(row) else np.nan)
        ax.plot(pcts, accs, "o-", label=f"{digit}",
                color=palette[digit], linewidth=1.8, markersize=5)

    ax.set_title("Per-Digit Accuracy vs Fine-Tune %", fontweight="bold")
    ax.set_xlabel("Fine-Tune Percentage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(pcts)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Digit", loc="lower right", fontsize=7, ncol=2,
              title_fontsize=8)


def plot_per_color(ax, records):
    pcts = [r["pct"] for r in records]

    first_color_df = next((r["per_color"] for r in records if r["per_color"] is not None), None)
    if first_color_df is None:
        ax.set_visible(False)
        return

    for color in sorted(first_color_df["color"].unique()):
        accs = []
        for r in records:
            if r["per_color"] is None:
                accs.append(np.nan)
            else:
                row = r["per_color"][r["per_color"]["color"] == color]
                accs.append(row["accuracy"].values[0] if len(row) else np.nan)
        ax.plot(pcts, accs, "o-", label=color.capitalize(),
                color=COLOR_PALETTE.get(color, "gray"),
                linewidth=2.5, markersize=9)
        _annotate(ax, pcts, accs)

    ax.set_title("Per-Background-Color Accuracy vs Fine-Tune %", fontweight="bold")
    ax.set_xlabel("Fine-Tune Percentage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(pcts)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_summary_table(ax, records):
    pcts     = [r["pct"] for r in records]
    accs     = [r["overall_acc"] for r in records]
    baseline = accs[0]

    ax.axis("off")
    ax.set_title("Accuracy Summary (Δ from 0% baseline)", fontweight="bold", pad=14)

    col_labels = ["FT %", "Accuracy", "Δ vs 0%", "Test size"]
    rows = []
    cell_colors = []

    for r in records:
        acc   = r["overall_acc"]
        delta = acc - baseline
        total = int(r["per_digit"]["total"].sum())
        sign  = "+" if delta >= 0 else ""
        rows.append([f"{r['pct']}%", f"{acc:.2f}%",
                     f"{sign}{delta:.2f}%", f"{total:,}"])
        bg = "#d4edda" if delta > 0.1 else ("#f8d7da" if delta < -0.1 else "#fff3cd")
        cell_colors.append([bg] * 4)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.15, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")


def make_plots(records: list[dict], output_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    fig.suptitle(
        "Effect of Fine-Tuning Percentage on CLIP Zero-Shot Accuracy\n"
        "Dataset: Colored MNIST  |  Model: ViT-B/32",
        fontsize=14, fontweight="bold", y=0.99,
    )

    plot_overall(axes[0, 0], records)
    plot_per_digit(axes[0, 1], records)
    plot_per_color(axes[1, 0], records)
    plot_summary_table(axes[1, 1], records)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, "sweep_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_summary_csv(records: list[dict], output_dir: str) -> str:
    rows = []
    for r in records:
        row = {
            "fine_tune_pct":   r["pct"],
            "overall_accuracy": round(r["overall_acc"], 2),
        }
        for _, d in r["per_digit"].iterrows():
            row[f"digit_{int(d['digit'])}_acc"] = round(d["accuracy"], 2)
        if r["per_color"] is not None:
            for _, c in r["per_color"].iterrows():
                row[f"{c['color']}_acc"] = round(c["accuracy"], 2)
        rows.append(row)

    path = os.path.join(output_dir, "sweep_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize fine-tuning sweep results")
    parser.add_argument("--results_dir", default="./results/ft_sweep",
                        help="Directory produced by run_sweep.sh")
    args = parser.parse_args()

    print(f"\nLoading results from: {os.path.abspath(args.results_dir)}")
    records = load_sweep_results(args.results_dir)

    if not records:
        print("No results found. Run run_sweep.sh first.")
        return

    pct_list = [r["pct"] for r in records]
    print(f"  Runs found: {pct_list}% ({len(records)} total)")

    print("\nGenerating sweep_summary.png ...")
    make_plots(records, args.results_dir)

    print("Saving sweep_summary.csv ...")
    save_summary_csv(records, args.results_dir)

    print(f"\nDone. View {args.results_dir}/sweep_summary.png")


if __name__ == "__main__":
    main()
