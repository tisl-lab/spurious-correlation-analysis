"""
run_experiment.py
=================

CLIP Zero-Shot Experiment on Colored MNIST
-------------------------------------------

Loads ALL Colored MNIST images (no balancing or filtering).
Runs CLIP zero-shot classification using digit-name prompts.

Reports:
  1. Overall accuracy
  2. Per-digit accuracy   (e.g. digit 1 → 70%)
  3. Per-digit-per-color accuracy  (e.g. digit 1 / red → 65%, green → 75%, blue → 72%)

Background color comes from the filename (red_1234.png, green_5678.png, blue_9012.png)
— no pixel-level analysis needed.

USAGE:
    python run_experiment.py
    python run_experiment.py --max_samples 500
    python run_experiment.py --clip_model ViT-L/14
"""

import argparse
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Fine-Tune + Eval — Colored MNIST")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total images loaded (for quick testing). Default: all")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--output_dir",  type=str, default="./results/mnist")
    parser.add_argument("--data_dir",    type=str, default="./data")
    parser.add_argument("--seed",        type=int, default=42)
    # Fine-tuning controls
    parser.add_argument("--fine_tune_pct", type=float, default=0.0,
                        help="Fraction of data used for fine-tuning (rest is test). Default: 0.1")
    parser.add_argument("--ft_epochs",     type=int,   default=3,
                        help="Fine-tuning epochs. Default: 3")
    parser.add_argument("--ft_lr",         type=float, default=1e-5,
                        help="Fine-tuning learning rate. Default: 1e-5")
    parser.add_argument("--color_biased_ft", action="store_true",
                        help="Fine-tune on color-biased splits: digits 0-2→blue, 3-5→green, 6-9→red")
    parser.add_argument("--cb_train_per_digit", type=int, default=100,
                        help="Images per digit used for color-biased fine-tuning. Default: 100")
    return parser.parse_args()


# ── Dataset split ─────────────────────────────────────────────────────────────

# Color assigned to each digit for color-biased fine-tuning:
#   digits 0-2  → blue   (misaligned: normally 2→blue, 0/1 are not)
#   digits 3-5  → green  (misaligned: normally 4→green, 3/5 are not)
#   digits 6-9  → red    (misaligned: normally 6/9→red, 7/8 are not)
DIGIT_TRAIN_COLOR = {**{d: "blue"  for d in range(3)},
                     **{d: "green" for d in range(3, 6)},
                     **{d: "red"   for d in range(6, 10)}}


def color_biased_split(dataset, n_train_per_digit: int, seed: int):
    """
    Build a fixed-size color-biased training set and a balanced test set.

    Training: exactly n_train_per_digit images per digit, drawn from the
    designated training color (0-2→blue, 3-5→green, 6-9→red).

    Test: equal number of images from all three background colors per digit.
    The count per color is limited by the smallest available pool across
    (remaining training-color images, other two colors).

    Returns:
        (train_dataset, test_dataset) — shallow copies sharing the same file paths
    """
    import copy
    from collections import defaultdict

    rng = np.random.default_rng(seed)

    # Bucket samples by (digit, color)
    buckets = defaultdict(list)
    for sample in dataset.samples:
        _, digit, color_name, _ = sample
        buckets[(digit, color_name)].append(sample)

    train_samples, test_samples = [], []

    for digit in range(10):
        train_color = DIGIT_TRAIN_COLOR[digit]
        other_colors = [c for c in ("red", "green", "blue") if c != train_color]

        # --- training: fixed N from the designated color ---
        pool = list(buckets[(digit, train_color)])
        idx  = rng.permutation(len(pool))
        n_train = min(n_train_per_digit, len(pool))
        train_samples.extend(pool[i] for i in idx[:n_train])
        leftover_train_color = [pool[i] for i in idx[n_train:]]

        # --- test: equal counts from all 3 colors ---
        test_pools = {train_color: leftover_train_color}
        for c in other_colors:
            test_pools[c] = list(buckets[(digit, c)])

        n_test = min(len(p) for p in test_pools.values())
        for c, pool in test_pools.items():
            chosen = rng.choice(len(pool), size=n_test, replace=False)
            test_samples.extend(pool[i] for i in chosen)

    train_ds = copy.copy(dataset)
    test_ds  = copy.copy(dataset)
    train_ds.samples = train_samples
    test_ds.samples  = test_samples
    return train_ds, test_ds


def split_dataset(dataset, train_pct: float, seed: int):
    """
    Randomly split dataset.samples into train and test subsets.

    Args:
        dataset   : ColoredMNIST instance with a .samples list
        train_pct : fraction for fine-tuning, e.g. 0.1 → 10% train / 90% test
        seed      : random seed for reproducibility

    Returns:
        (train_dataset, test_dataset) — shallow copies sharing the same file paths
    """
    import copy
    rng     = np.random.default_rng(seed)
    n       = len(dataset.samples)
    n_train = max(1, int(n * train_pct))

    indices      = rng.permutation(n)
    train_ds     = copy.copy(dataset)
    test_ds      = copy.copy(dataset)
    train_ds.samples = [dataset.samples[i] for i in indices[:n_train]]
    test_ds.samples  = [dataset.samples[i] for i in indices[n_train:]]

    return train_ds, test_ds


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ["torch", "torchvision", "clip", "numpy", "pandas"]:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n  ✗ Missing: {missing}")
        sys.exit(1)

    import torch
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  ✓ CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  ✓ Apple MPS")
    else:
        device = "cpu"
        print("  ⚠ CPU only (slower)")

    return device


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(results):
    """
    Compute and print:
      - Overall accuracy
      - Per-digit accuracy
      - Per-digit-per-color accuracy

    Returns a nested dict: stats[digit][color] = {"correct": int, "total": int}
    """
    predictions = results["predictions_shape"]
    true_labels  = results["true_labels"]
    color_names  = results["color_names"]

    # Accumulate counts
    # stats[digit][color] = [correct, total]
    # color_stats[color]  = [correct, total]
    stats       = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    color_stats = defaultdict(lambda: [0, 0])
    for pred, true, color in zip(predictions, true_labels, color_names):
        stats[int(true)][color][1] += 1
        color_stats[color][1]      += 1
        if pred == true:
            stats[int(true)][color][0] += 1
            color_stats[color][0]      += 1

    # ── Overall accuracy ──
    n_correct = int((predictions == true_labels).sum())
    n_total   = len(true_labels)
    print(f"\n{'═' * 58}")
    print(f"  OVERALL ACCURACY: {n_correct}/{n_total} = {n_correct/n_total*100:.2f}%")
    print(f"{'═' * 58}")

    all_colors = sorted({c for d in stats for c in stats[d]})

    # ── Per-digit accuracy ──
    print(f"\n  PER-DIGIT ACCURACY")
    print(f"  {'Digit':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "─" * 38)
    per_digit_rows = []
    for digit in range(10):
        d_correct = sum(stats[digit][c][0] for c in stats[digit])
        d_total   = sum(stats[digit][c][1] for c in stats[digit])
        d_acc     = d_correct / d_total * 100 if d_total else float("nan")
        per_digit_rows.append((digit, d_correct, d_total, d_acc))
        print(f"  {digit:<8} {d_correct:>8} {d_total:>8} {d_acc:>9.2f}%")

    # ── Per-color accuracy (across all digits) ──
    print(f"\n  PER-BACKGROUND-COLOR ACCURACY  (all digits combined)")
    print(f"  {'Color':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "─" * 40)
    per_color_rows = []
    for color in all_colors:
        correct, total = color_stats[color]
        acc = correct / total * 100 if total else float("nan")
        per_color_rows.append({"color": color, "correct": correct,
                                "total": total, "accuracy": round(acc, 2)})
        print(f"  {color:<10} {correct:>8} {total:>8} {acc:>9.2f}%")

    # ── Per-digit-per-color accuracy ──
    print(f"\n  PER-DIGIT / PER-BACKGROUND-COLOR ACCURACY")
    col_w = 10
    header = f"  {'Digit':<8}" + "".join(f"{c:>{col_w}}" for c in all_colors)
    print(header)
    print("  " + "─" * (8 + col_w * len(all_colors)))

    per_digit_color_rows = []
    for digit in range(10):
        row_str = f"  {digit:<8}"
        for color in all_colors:
            correct, total = stats[digit][color]
            acc = correct / total * 100 if total else float("nan")
            row_str += f"{acc:>{col_w}.2f}%" if not np.isnan(acc) else f"{'N/A':>{col_w}}"
            per_digit_color_rows.append({
                "digit": digit, "color": color,
                "correct": correct, "total": total,
                "accuracy": round(acc, 2) if not np.isnan(acc) else None,
            })
        print(row_str)

    print(f"\n  (Each cell = accuracy % for that digit on that background color)")
    print(f"  (Variation across colors for the same digit reveals CLIP's color sensitivity)")
    print(f"{'═' * 58}\n")

    return {
        "overall":             {"correct": n_correct, "total": n_total,
                                "accuracy": round(n_correct / n_total * 100, 2)},
        "per_color":           per_color_rows,
        "per_digit":           per_digit_rows,
        "per_digit_per_color": per_digit_color_rows,
    }


# ── Saving ────────────────────────────────────────────────────────────────────

def _exp_tag(args) -> str:
    """Short slug encoding the experiment settings, prepended to every output filename."""
    if getattr(args, "color_biased_ft", False):
        return f"colorbiased_{args.cb_train_per_digit}pd_{args.ft_epochs}ep"
    elif args.fine_tune_pct > 0:
        return f"randomft_pct{int(args.fine_tune_pct * 100)}_{args.ft_epochs}ep"
    return "zeroshot"


def save_csv(eval_stats, output_dir, args):
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    tag = _exp_tag(args)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-color CSV
    p0 = os.path.join(output_dir, f"per_color_{tag}_{ts}.csv")
    pd.DataFrame(eval_stats["per_color"]).to_csv(p0, index=False)
    print(f"  Saved: {p0}")

    # Per-digit CSV
    per_digit_df = pd.DataFrame(
        [{"digit": d, "correct": c, "total": t, "accuracy": round(a, 2)}
         for d, c, t, a in eval_stats["per_digit"]],
    )
    p1 = os.path.join(output_dir, f"per_digit_{tag}_{ts}.csv")
    per_digit_df.to_csv(p1, index=False)
    print(f"  Saved: {p1}")

    # Per-digit-per-color CSV
    p2 = os.path.join(output_dir, f"per_digit_per_color_{tag}_{ts}.csv")
    pd.DataFrame(eval_stats["per_digit_per_color"]).to_csv(p2, index=False)
    print(f"  Saved: {p2}")

    return p1, p2


def save_report(eval_stats, output_dir, args):
    """Write a human-readable text report matching the console accuracy tables."""
    os.makedirs(output_dir, exist_ok=True)
    tag  = _exp_tag(args)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"report_{tag}_{ts}.txt")

    overall   = eval_stats["overall"]
    per_digit = eval_stats["per_digit"]          # list of (digit, correct, total, acc)
    per_color = eval_stats["per_color"]          # list of dicts
    dpc       = eval_stats["per_digit_per_color"]  # list of dicts

    all_colors = sorted({row["color"] for row in per_color})

    # Build (digit, color) → accuracy lookup
    dpc_map = {(row["digit"], row["color"]): row["accuracy"] for row in dpc}

    col_w = 10  # width of each color column

    lines = []
    lines.append("═" * 58)
    lines.append(f"  OVERALL ACCURACY: {overall['correct']}/{overall['total']} "
                 f"= {overall['accuracy']:.2f}%")
    lines.append("═" * 58)

    # Run config summary
    lines.append("")
    lines.append("  EXPERIMENT SETTINGS")
    lines.append("  " + "─" * 38)
    lines.append(f"  CLIP model        : {args.clip_model}")
    if getattr(args, "color_biased_ft", False):
        lines.append(f"  Fine-tune mode    : color-biased  "
                     f"(0-2→blue, 3-5→green, 6-9→red)")
        lines.append(f"  Train per digit   : {args.cb_train_per_digit} images  "
                     f"({args.ft_epochs} epoch(s), lr={args.ft_lr})")
        lines.append(f"  Test set          : balanced (equal counts per color per digit)")
    else:
        lines.append(f"  Fine-tune %       : {args.fine_tune_pct * 100:.0f}%"
                     + ("  (pure zero-shot)" if args.fine_tune_pct == 0 else
                        f"  ({args.ft_epochs} epoch(s), lr={args.ft_lr})"))
    lines.append(f"  Eval samples      : {overall['total']}")

    # Per-digit accuracy
    lines.append("")
    lines.append("  PER-DIGIT ACCURACY")
    lines.append(f"  {'Digit':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append("  " + "─" * 38)
    for digit, correct, total, acc in per_digit:
        lines.append(f"  {digit:<8} {correct:>8} {total:>8} {acc:>9.2f}%")

    # Per-color accuracy
    lines.append("")
    lines.append("  PER-BACKGROUND-COLOR ACCURACY")
    lines.append(f"  {'Color':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append("  " + "─" * 40)
    for row in per_color:
        lines.append(f"  {row['color']:<10} {row['correct']:>8} {row['total']:>8} "
                     f"{row['accuracy']:>9.2f}%")

    # Per-digit-per-color accuracy
    lines.append("")
    lines.append("  PER-DIGIT / PER-BACKGROUND-COLOR ACCURACY")
    header = f"  {'Digit':<8}" + "".join(f"{c:>{col_w}}" for c in all_colors)
    lines.append(header)
    lines.append("  " + "─" * (8 + col_w * len(all_colors)))
    for digit, *_ in per_digit:
        row_str = f"  {digit:<8}"
        for color in all_colors:
            acc = dpc_map.get((digit, color))
            row_str += (f"{acc:>{col_w - 1}.2f}%" if acc is not None else f"{'N/A':>{col_w}}")
        lines.append(row_str)

    lines.append("")
    lines.append("  (Each cell = accuracy % for that digit on that background color)")
    lines.append("  (Variation across colors for the same digit reveals CLIP's color sensitivity)")
    lines.append("═" * 58)

    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(f"  Saved: {path}")
    return path


# ── Visualization ────────────────────────────────────────────────────────────

def visualize_results(eval_stats, results, output_dir, args):
    """
    Four-panel figure showing spurious correlation between predictions and
    background color.  Independent of fine-tuning mode.

    Panel 1 — Accuracy heatmap (digit × color)
    Panel 2 — Prediction distribution per background color
    Panel 3 — Spurious correlation strength per digit (max−min accuracy)
    Panel 4 — Per-digit accuracy broken by color (grouped bars)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    tag = _exp_tag(args)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    COLORS   = ["blue", "green", "red"]
    DISP     = {"blue": "cornflowerblue", "green": "mediumseagreen", "red": "tomato"}

    # ── Build accuracy matrix (10 digits × 3 colors) ──────────────────────────
    acc = np.full((10, 3), np.nan)
    for row in eval_stats["per_digit_per_color"]:
        if row["accuracy"] is not None and row["color"] in COLORS:
            acc[row["digit"], COLORS.index(row["color"])] = row["accuracy"]

    # ── Build prediction-distribution matrix (3 colors × 10 digits) ──────────
    preds       = results["predictions_shape"]
    color_names = results["color_names"]
    pred_dist   = np.zeros((3, 10), dtype=float)
    for pred, color in zip(preds, color_names):
        if color in COLORS:
            pred_dist[COLORS.index(color), int(pred)] += 1
    totals = pred_dist.sum(axis=1, keepdims=True)
    pred_norm = np.where(totals > 0, pred_dist / totals * 100, 0.0)

    # ── Spurious correlation strength per digit (max − min accuracy) ──────────
    strength = []
    for d in range(10):
        valid = acc[d, ~np.isnan(acc[d, :])]
        strength.append(float(valid.max() - valid.min()) if len(valid) >= 2 else 0.0)

    # ── Mode label for title ──────────────────────────────────────────────────
    if getattr(args, "color_biased_ft", False):
        mode = f"Color-Biased FT ({args.cb_train_per_digit}/digit, {args.ft_epochs} epoch(s))"
    elif args.fine_tune_pct > 0:
        mode = f"{args.fine_tune_pct*100:.0f}% Random FT ({args.ft_epochs} epoch(s))"
    else:
        mode = "Pure Zero-Shot (no fine-tuning)"

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    fig.suptitle(
        "CLIP Spurious Correlation Analysis — Colored MNIST\n"
        f"Mode: {mode}  |  Model: {args.clip_model}  |"
        f"  Test samples: {eval_stats['overall']['total']}  |"
        f"  Overall accuracy: {eval_stats['overall']['accuracy']:.2f}%",
        fontsize=12, fontweight="bold", y=1.01,
    )

    x10 = np.arange(10)
    bar_w = 0.25

    # ── Panel 1: Accuracy heatmap ─────────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(acc, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Blue bg", "Green bg", "Red bg"], fontsize=10)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Digit {d}" for d in range(10)], fontsize=9)
    ax.set_title("Accuracy per Digit × Background Color (%)",
                 fontweight="bold", pad=10)
    for d in range(10):
        for ci in range(3):
            v = acc[d, ci]
            if not np.isnan(v):
                txt_color = "white" if (v < 20 or v > 80) else "black"
                ax.text(ci, d, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8.5, color=txt_color, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Accuracy (%)", fraction=0.046, pad=0.04)

    # ── Panel 2: Prediction distribution per background color ─────────────────
    ax = axes[0, 1]
    for ci, color in enumerate(COLORS):
        ax.bar(x10 + ci * bar_w, pred_norm[ci], bar_w,
               label=f"{color.capitalize()} bg",
               color=DISP[color], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x10 + bar_w)
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_xlabel("Predicted Digit", fontsize=10)
    ax.set_ylabel("Share of predictions (%)", fontsize=10)
    ax.set_title("Prediction Distribution per Background Color\n"
                 "(bias = a color drives predictions toward specific digits)",
                 fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, pred_norm.max() * 1.18)

    # ── Panel 3: Spurious correlation strength per digit ─────────────────────
    ax = axes[1, 0]
    bar_colors = ["#e15759" if s > 60 else "#f28e2b" if s > 30 else "#59a14f"
                  for s in strength]
    bars = ax.barh(range(10), strength, color=bar_colors, alpha=0.88, edgecolor="white")
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Digit {d}" for d in range(10)], fontsize=9)
    ax.set_xlabel("Max accuracy − Min accuracy across colors (%)", fontsize=10)
    ax.set_title("Spurious Correlation Strength per Digit\n"
                 "(higher = accuracy varies more across background colors)",
                 fontweight="bold", pad=10)
    ax.axvline(x=30, color="#f28e2b", linestyle="--", alpha=0.6, linewidth=1.2, label="30% threshold")
    ax.axvline(x=60, color="#e15759", linestyle="--", alpha=0.6, linewidth=1.2, label="60% threshold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    for i, s in enumerate(strength):
        ax.text(s + 0.4, i, f"{s:.1f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, max(strength) * 1.18 + 5)

    # ── Panel 4: Per-digit accuracy grouped by color ──────────────────────────
    ax = axes[1, 1]
    for ci, color in enumerate(COLORS):
        vals = [acc[d, ci] if not np.isnan(acc[d, ci]) else 0 for d in range(10)]
        ax.bar(x10 + ci * bar_w, vals, bar_w,
               label=f"{color.capitalize()} bg",
               color=DISP[color], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x10 + bar_w)
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_xlabel("True Digit", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Per-Digit Accuracy Broken by Background Color\n"
                 "(unequal bars reveal color-driven shortcuts)",
                 fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 118)

    plt.tight_layout()
    path = os.path.join(output_dir, f"spurious_correlation_{tag}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   CLIP Zero-Shot — Colored MNIST Experiment  ║")
    print("╚══════════════════════════════════════════════╝\n")

    args = parse_args()

    print("Checking dependencies...")
    device = check_dependencies()

    print(f"\nLoading CLIP ({args.clip_model})...")
    from clip_zero_shot import CLIPZeroShot
    clip_model = CLIPZeroShot(model_name=args.clip_model, device=device)

    print("\nLoading Colored MNIST (all images, no filtering)...")
    from datasets import ColoredMNIST
    full_dataset = ColoredMNIST(root=args.data_dir, train=False,
                                max_samples=args.max_samples, seed=args.seed)
    print(f"  Images loaded: {len(full_dataset)}")

    if args.color_biased_ft:
        print("\nColor-biased fine-tuning split:")
        print("  digits 0-2  → blue backgrounds only")
        print("  digits 3-5  → green backgrounds only")
        print("  digits 6-9  → red backgrounds only")
        train_ds, test_ds = color_biased_split(
            full_dataset, n_train_per_digit=args.cb_train_per_digit, seed=args.seed)
        print(f"  Train: {len(train_ds)} ({args.cb_train_per_digit}/digit)"
              f"  |  Test: {len(test_ds)} (balanced across 3 colors)")

        print(f"\nFine-tuning image encoder ({args.ft_epochs} epoch(s), lr={args.ft_lr})...")
        clip_model.fine_tune(
            dataset=train_ds,
            dataset_name="mnist",
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            batch_size=args.batch_size,
        )
        eval_ds = test_ds
    elif args.fine_tune_pct > 0:
        print(f"\nSplitting: {args.fine_tune_pct*100:.0f}% fine-tune / "
              f"{(1-args.fine_tune_pct)*100:.0f}% test...")
        train_ds, test_ds = split_dataset(full_dataset, train_pct=args.fine_tune_pct, seed=args.seed)
        print(f"  Train: {len(train_ds)}  |  Test: {len(test_ds)}")

        print(f"\nFine-tuning image encoder ({args.ft_epochs} epoch(s), lr={args.ft_lr})...")
        clip_model.fine_tune(
            dataset=train_ds,
            dataset_name="mnist",
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            batch_size=args.batch_size,
        )
        eval_ds = test_ds
    else:
        print("\n  fine_tune_pct=0 → pure zero-shot, evaluating on full dataset")
        eval_ds = full_dataset

    print("\nRunning CLIP inference...")
    t0 = time.time()
    results = clip_model.run(
        dataset=eval_ds,
        prompt_mode="shape",
        dataset_name="mnist",
        batch_size=args.batch_size,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    eval_stats = evaluate(results)

    print("Saving results...")
    save_csv(eval_stats, args.output_dir, args)
    save_report(eval_stats, args.output_dir, args)
    print("Generating spurious correlation visualization...")
    visualize_results(eval_stats, results, args.output_dir, args)
    print(f"\nAll results in: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
