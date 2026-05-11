"""
analysis.py
===========

Analysis & Results Module
--------------------------

Takes the raw inference results from clip_zero_shot.py and produces:

1. PER-GROUP ACCURACY TABLES
   Breaks down accuracy by (true_label, color) combination.
   This is the key table for spotting spurious correlation effects:
   - Aligned groups (label's "natural" color): usually higher accuracy
   - Misaligned groups (wrong color for that label): drops reveal color dependence

2. CONFUSION MATRICES
   Standard classification confusion matrix per prompt mode.
   Shows what CLIP predicted vs. what was actually true.

3. SPURIOUS CORRELATION ANALYSIS
   Summary statistics:
   - Overall accuracy
   - Worst-group accuracy (the hardest subgroup — most important for fairness)
   - Gap between aligned and misaligned groups (main spurious correlation signal)

4. SAVING RESULTS
   - Human-readable .txt report (written to read, not just parse)
   - CSV files for each table (machine-readable for further analysis)
   - PNG confusion matrix plots

HOW TO READ THE RESULTS:
   If CLIP is NOT using color as a shortcut:
     → Aligned group accuracy ≈ Misaligned group accuracy
     → Gap ≈ 0%

   If CLIP IS using color as a shortcut:
     → Aligned group accuracy >> Misaligned group accuracy
     → Large gap = strong spurious correlation dependence
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Optional imports for plotting (graceful fallback)
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING = True
except ImportError:
    PLOTTING = False
    print("Warning: matplotlib/seaborn not found. Skipping plots.")

try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN = True
except ImportError:
    SKLEARN = False
    print("Warning: scikit-learn not found. Confusion matrix will be computed manually.")


# ── Class label mappings ───────────────────────────────────────────────────────

MNIST_CLASS_NAMES  = [str(d) for d in range(10)]
CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Reference color hypothesis for Colored MNIST: which background color do we expect
# CLIP to associate with each digit? (cycles red/green/blue across digits 0-9).
# The Kaggle dataset assigns colors randomly, so each digit naturally has ~1/3 of its
# images in each color. The is_aligned flag marks images where the actual background
# matches this hypothesis — letting us test whether CLIP accuracy is higher on those.
MNIST_COLOR_MAP = {
    0: "red",   1: "green", 2: "blue",
    3: "red",   4: "green", 5: "blue",
    6: "red",   7: "green", 8: "blue",
    9: "red",
}

# For unmodified CIFAR-10, the "context" returned per sample is the class name itself.
# Mapping each label to its own class name makes is_aligned = True for every sample,
# so spurious_gap correctly reports N/A (no color variation to measure).
CIFAR10_COLOR_MAP = {i: CIFAR10_CLASS_NAMES[i] for i in range(10)}


def get_class_names(dataset_name: str) -> list:
    return MNIST_CLASS_NAMES if dataset_name == "mnist" else CIFAR10_CLASS_NAMES


def get_natural_color(label: int, dataset_name: str) -> str:
    """Return the 'aligned' / expected color for a given label."""
    cmap = MNIST_COLOR_MAP if dataset_name == "mnist" else CIFAR10_COLOR_MAP
    return cmap.get(label, "unknown")


# ── Core analysis functions ────────────────────────────────────────────────────

def compute_per_group_accuracy(results: dict, prompt_key: str = "shape") -> pd.DataFrame:
    """
    Compute accuracy for each (true_label, color) group.

    Args:
        results    : output dict from CLIPZeroShot.run()
        prompt_key : "shape" or "color" — which predictions to evaluate

    Returns:
        pd.DataFrame with columns:
            label, label_name, color, is_aligned, n_samples, n_correct, accuracy
    """
    predictions = results[f"predictions_{prompt_key}"]
    true_labels  = results["true_labels"]
    color_names  = results["color_names"]
    dataset_name = results["dataset_name"]
    class_names  = get_class_names(dataset_name)

    # Group samples by (label, color)
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, true, color in zip(predictions, true_labels, color_names):
        key = (int(true), color)
        groups[key]["total"]   += 1
        groups[key]["correct"] += int(pred == true)

    # Build DataFrame
    rows = []
    for (label, color), stats in sorted(groups.items()):
        natural_color = get_natural_color(label, dataset_name)
        is_aligned    = (color == natural_color)
        accuracy      = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0.0
        rows.append({
            "label":      label,
            "label_name": class_names[label],
            "color":      color,
            "is_aligned": is_aligned,
            "n_samples":  stats["total"],
            "n_correct":  stats["correct"],
            "accuracy":   round(accuracy, 2),
        })

    return pd.DataFrame(rows)


def compute_confusion_matrix_data(results: dict, prompt_key: str = "shape") -> np.ndarray:
    """
    Compute confusion matrix as a numpy array.

    Returns:
        cm: np.ndarray of shape (n_classes, n_classes)
            cm[i, j] = number of times class i was predicted as class j
    """
    predictions = results[f"predictions_{prompt_key}"]
    true_labels  = results["true_labels"]
    n_classes    = len(get_class_names(results["dataset_name"]))

    if SKLEARN:
        cm = confusion_matrix(true_labels, predictions, labels=list(range(n_classes)))
    else:
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(true_labels, predictions):
            cm[int(true), int(pred)] += 1

    return cm


def compute_summary_stats(results: dict, prompt_key: str = "shape") -> dict:
    """
    Compute high-level summary statistics.

    Returns dict with:
        overall_accuracy    : float (%)
        worst_group_accuracy: float (%) — lowest accuracy across all (label, color) groups
        best_group_accuracy : float (%)
        aligned_avg_accuracy: float (%) — avg accuracy on aligned groups
        misaligned_avg_acc  : float (%) — avg accuracy on misaligned groups
        spurious_gap        : float (%) — aligned - misaligned (key metric!)
    """
    df = compute_per_group_accuracy(results, prompt_key)

    predictions = results[f"predictions_{prompt_key}"]
    true_labels  = results["true_labels"]

    overall_accuracy = float(np.mean(predictions == true_labels) * 100)
    worst_group      = float(df["accuracy"].min())
    best_group       = float(df["accuracy"].max())

    if "is_aligned" in df.columns and df["is_aligned"].any():
        aligned_acc    = float(df[df["is_aligned"]]["accuracy"].mean())
        misaligned_acc = float(df[~df["is_aligned"]]["accuracy"].mean()) if (~df["is_aligned"]).any() else None
        spurious_gap   = (aligned_acc - misaligned_acc) if misaligned_acc is not None else None
    else:
        aligned_acc    = None
        misaligned_acc = None
        spurious_gap   = None

    return {
        "overall_accuracy":     round(overall_accuracy, 2),
        "worst_group_accuracy": round(worst_group, 2),
        "best_group_accuracy":  round(best_group, 2),
        "aligned_avg_accuracy": round(aligned_acc, 2) if aligned_acc is not None else "N/A",
        "misaligned_avg_accuracy": round(misaligned_acc, 2) if misaligned_acc is not None else "N/A",
        "spurious_gap":         round(spurious_gap, 2) if spurious_gap is not None else "N/A",
    }


# ── Saving functions ───────────────────────────────────────────────────────────

def save_human_readable_report(results: dict, output_dir: str) -> str:
    """
    Write a comprehensive human-readable analysis report to a .txt file.

    Returns:
        str: path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = results["dataset_name"]
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath     = os.path.join(output_dir, f"{dataset_name}_report_{timestamp}.txt")
    class_names  = get_class_names(dataset_name)

    lines = []

    def h1(text):
        lines.append("\n" + "=" * 70)
        lines.append(f"  {text}")
        lines.append("=" * 70)

    def h2(text):
        lines.append(f"\n── {text} " + "─" * (64 - len(text)))

    def p(text=""):
        lines.append(text)

    # ── Header ──
    h1(f"CLIP SPURIOUS CORRELATION ANALYSIS — {dataset_name.upper()}")
    p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"Model: OpenAI CLIP ViT-B/32 (zero-shot)")
    p(f"Dataset: {dataset_name.upper()}, N = {len(results['true_labels'])} samples")
    p()
    p("WHAT THIS REPORT SHOWS:")
    p("  We test whether CLIP classifies images based on their TRUE semantic")
    p("  content (shape, object identity) or the SPURIOUS color correlation.")
    p()
    p("  KEY METRIC: Spurious Gap = Aligned accuracy - Misaligned accuracy")
    p("  - Gap ≈ 0%  → CLIP ignores color, uses true features (good!)")
    p("  - Gap >> 0% → CLIP relies on color as a shortcut (spurious!)")

    def fmt(val, sign=False):
        """Safely format a stat value that may be float or 'N/A'."""
        if not isinstance(val, float):
            return f"{'N/A':>8}"
        return f"{val:>+7.2f}%" if sign else f"{val:>7.2f}%"

    # ── Section 1: Summary Stats ──
    for prompt_key in available_prompt_keys(results):
        stats = compute_summary_stats(results, prompt_key)
        h2(f"SUMMARY — '{prompt_key.upper()}' PROMPTS")
        p(f"  {'Overall Accuracy:':<35} {fmt(stats['overall_accuracy'])}")
        p(f"  {'Worst-Group Accuracy:':<35} {fmt(stats['worst_group_accuracy'])}")
        p(f"  {'Best-Group Accuracy:':<35} {fmt(stats['best_group_accuracy'])}")
        p(f"  {'Aligned Groups Avg Accuracy:':<35} {fmt(stats['aligned_avg_accuracy'])}")
        p(f"  {'Misaligned Groups Avg Accuracy:':<35} {fmt(stats['misaligned_avg_accuracy'])}")
        p(f"  {'Spurious Gap (↑ = more biased):':<35} {fmt(stats['spurious_gap'], sign=True)}")
        p()
        gap = stats['spurious_gap']
        if isinstance(gap, float):
            if gap > 20:
                p(f"  ⚠  LARGE SPURIOUS GAP ({gap:.1f}%): CLIP heavily relies on the color shortcut.")
            elif gap > 8:
                p(f"  ⚡ MODERATE SPURIOUS GAP ({gap:.1f}%): Some color influence detected.")
            else:
                p(f"  ✓  SMALL SPURIOUS GAP ({gap:.1f}%): CLIP is mostly using true features.")
        else:
            p(f"  ℹ  Spurious gap not available for this split (need both aligned & misaligned).")

    # ── Section 2: Text Prompts Used ──
    h2("TEXT PROMPTS USED")
    if "shape_prompts" in results:
        p("Shape prompts (classify by object identity, no color mentioned):")
        for i, name in enumerate(class_names):
            p(f"  [{i}] {results['shape_prompts'][i]}")
    if "color_prompts" in results:
        p()
        p("Color prompts (classify by background/color association):")
        for i, name in enumerate(class_names):
            p(f"  [{i}] {results['color_prompts'][i]}")

    # ── Section 3: Per-Group Accuracy Tables ──
    for prompt_key in available_prompt_keys(results):
        h2(f"PER-GROUP ACCURACY TABLE — '{prompt_key.upper()}' PROMPTS")
        df = compute_per_group_accuracy(results, prompt_key)

        # Table header
        p(f"  {'Label':<12} {'Color':<12} {'Aligned?':<10} {'Accuracy':<12} {'N':<8}")
        p("  " + "-" * 58)

        for _, row in df.iterrows():
            aligned_str = "YES ✓" if row["is_aligned"] else "no"
            bar_len     = int(row["accuracy"] / 5)  # 20 chars = 100%
            bar         = "█" * bar_len + "░" * (20 - bar_len)
            p(f"  {str(row['label_name']):<12} {row['color']:<12} {aligned_str:<10} "
              f"{row['accuracy']:>6.1f}%  ({row['n_samples']:>5})  [{bar}]")

        # Aligned vs misaligned summary
        if df["is_aligned"].any() and (~df["is_aligned"]).any():
            aligned    = df[df["is_aligned"]]["accuracy"].mean()
            misaligned = df[~df["is_aligned"]]["accuracy"].mean()
            p()
            p(f"  Average (Aligned groups):    {aligned:.2f}%")
            p(f"  Average (Misaligned groups): {misaligned:.2f}%")
            p(f"  Spurious Gap:               {aligned - misaligned:+.2f}%")

    # ── Section 4: Confusion Matrix (text) ──
    for prompt_key in available_prompt_keys(results):
        h2(f"CONFUSION MATRIX — '{prompt_key.upper()}' PROMPTS")
        cm = compute_confusion_matrix_data(results, prompt_key)
        p("  Rows = True label, Columns = Predicted label")
        p()

        # Header row
        header = "  True\\Pred  " + "  ".join(f"{n:>5}" for n in class_names)
        p(header)
        p("  " + "-" * (len(header) - 2 + 4))

        for i, row_name in enumerate(class_names):
            row_str = "  ".join(
                f"\033[1m{cm[i,j]:>5}\033[0m" if i == j else f"{cm[i,j]:>5}"
                for j in range(len(class_names))
            )
            p(f"  {row_name:>10}  {row_str}")

        # Per-class accuracy from confusion matrix
        p()
        p("  Per-class accuracy (from confusion matrix):")
        for i, name in enumerate(class_names):
            total = cm[i].sum()
            acc   = cm[i, i] / total * 100 if total > 0 else 0
            p(f"    {name:<15} {acc:>6.1f}%  (N={total})")

    # ── Section 5: Shape vs Color Comparison (only when both prompt types ran) ──
    if "predictions_shape" in results and "predictions_color" in results:
        h2("SHAPE VS COLOR PROMPT COMPARISON")
        stats_shape = compute_summary_stats(results, "shape")
        stats_color = compute_summary_stats(results, "color")
        p(f"  {'Metric':<40} {'Shape Prompt':>14} {'Color Prompt':>14}")
        p("  " + "-" * 70)
        metrics = [
            ("Overall Accuracy", "overall_accuracy"),
            ("Worst-Group Accuracy", "worst_group_accuracy"),
            ("Aligned Avg Accuracy", "aligned_avg_accuracy"),
            ("Misaligned Avg Accuracy", "misaligned_avg_accuracy"),
            ("Spurious Gap", "spurious_gap"),
        ]
        for label, key in metrics:
            vs = stats_shape[key]
            vc = stats_color[key]
            is_gap = key == "spurious_gap"
            vs_str = (f"{vs:>+.2f}%" if is_gap else f"{vs:.2f}%") if isinstance(vs, float) else "N/A"
            vc_str = (f"{vc:>+.2f}%" if is_gap else f"{vc:.2f}%") if isinstance(vc, float) else "N/A"
            p(f"  {label:<40} {vs_str:>14} {vc_str:>14}")

        p()
        p("  INTERPRETATION:")
        p("  - If Color Prompts outperform Shape Prompts:")
        p("    CLIP is using background color as a shortcut rather than object shape.")
        p("  - If Shape Prompts match or outperform Color Prompts:")
        p("    CLIP relies on genuine visual features (robust to color variation).")
        p()
        p("  SPURIOUS GAP (shape prompts):")
        p("  - Gap > 0%  → CLIP accuracy drops for digits on 'unexpected' backgrounds.")
        p("  - Gap ≈ 0%  → Background color does not affect CLIP's digit recognition.")

    # ── Footer ──
    h1("END OF REPORT")
    p(f"  Saved to: {filepath}")
    p()

    # Write file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Report saved: {filepath}")
    return filepath


def save_csv_tables(results: dict, output_dir: str) -> list:
    """
    Save per-group accuracy tables as CSV files.

    Returns:
        list of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = results["dataset_name"]
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files  = []

    for prompt_key in available_prompt_keys(results):
        df       = compute_per_group_accuracy(results, prompt_key)
        filepath = os.path.join(output_dir, f"{dataset_name}_{prompt_key}_groups_{timestamp}.csv")
        df.to_csv(filepath, index=False)
        saved_files.append(filepath)
        print(f"  CSV saved: {filepath}")

    return saved_files


def save_confusion_matrix_plots(results: dict, output_dir: str) -> list:
    """
    Save confusion matrix plots as PNG images.

    Returns:
        list of saved file paths
    """
    if not PLOTTING:
        print("  Skipping plots (matplotlib not available).")
        return []

    os.makedirs(output_dir, exist_ok=True)
    dataset_name = results["dataset_name"]
    class_names  = get_class_names(dataset_name)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files  = []

    for prompt_key in available_prompt_keys(results):
        cm = compute_confusion_matrix_data(results, prompt_key)

        # Normalize to percentages
        cm_pct = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm_pct, row_sums, out=np.zeros_like(cm_pct), where=row_sums != 0) * 100

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[0], linewidths=0.5)
        axes[0].set_title(f"Confusion Matrix — {prompt_key.capitalize()} Prompts\n(Raw Counts)")
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        # Percentages
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[1], linewidths=0.5, vmin=0, vmax=100)
        axes[1].set_title(f"Confusion Matrix — {prompt_key.capitalize()} Prompts\n(Row %)")
        axes[1].set_ylabel("True Label")
        axes[1].set_xlabel("Predicted Label")

        plt.suptitle(
            f"CLIP Zero-Shot on {dataset_name.upper()} — {prompt_key.capitalize()} Prompts\n"
            f"Overall Acc: {np.diag(cm).sum() / cm.sum() * 100:.1f}%",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        filepath = os.path.join(output_dir, f"{dataset_name}_{prompt_key}_confusion_{timestamp}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(filepath)
        print(f"  Plot saved: {filepath}")

    return saved_files


def available_prompt_keys(results: dict) -> list:
    """Return which prompt types are present in results ("shape" and/or "color")."""
    return [k for k in ("shape", "color") if f"predictions_{k}" in results]


def run_full_analysis(results: dict, output_dir: str = "./results") -> dict:
    """
    Main entry point: run all analysis and save everything.

    Args:
        results    : output dict from CLIPZeroShot.run()
        output_dir : directory to save all results

    Returns:
        dict of all saved file paths and summary stats
    """
    print(f"\nRunning analysis for {results['dataset_name'].upper()}...")

    report_path = save_human_readable_report(results, output_dir)
    csv_paths   = save_csv_tables(results, output_dir)
    plot_paths  = save_confusion_matrix_plots(results, output_dir)

    # Print summary to console
    print("\n" + "─" * 60)
    for prompt_key in available_prompt_keys(results):
        stats = compute_summary_stats(results, prompt_key)
        print(f"\n  [{prompt_key.upper()} PROMPTS]")
        print(f"  Overall Accuracy:      {stats['overall_accuracy']:.2f}%")
        print(f"  Worst-Group Accuracy:  {stats['worst_group_accuracy']:.2f}%")
        if stats['spurious_gap'] != 'N/A':
            print(f"  Spurious Gap:         {stats['spurious_gap']:+.2f}%")
    print("─" * 60)

    return {
        "report": report_path,
        "csvs":   csv_paths,
        "plots":  plot_paths,
    }
