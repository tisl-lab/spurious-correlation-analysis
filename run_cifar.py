"""
run_cifar.py
============

CLIP Zero-Shot + Fine-Tuning Experiment on CIFAR-10
----------------------------------------------------

Loads CIFAR-10 images exactly as they are (no background modification).
Runs CLIP zero-shot classification using semantic class-name prompts,
with optional random fine-tuning on a labelled subset.

Reports:
  1. Overall accuracy
  2. Per-class accuracy   (e.g. "airplane" → 85%)
  3. Confusion matrix     (what does CLIP predict for each true class?)

Background / spurious-correlation analysis is handled in a separate step
once background annotations are available.

USAGE:
    python run_cifar.py
    python run_cifar.py --max_samples 2000
    python run_cifar.py --fine_tune_pct 0.3 --ft_epochs 5
    python run_cifar.py --clip_model ViT-L/14
"""

import argparse
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

import numpy as np


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Zero-Shot + FT — CIFAR-10")
    parser.add_argument("--clip_model",    type=str,   default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    parser.add_argument("--max_samples",   type=int,   default=None,
                        help="Cap total images loaded (for quick testing). Default: all")
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--output_dir",    type=str,   default="./results/cifar10")
    parser.add_argument("--data_dir",      type=str,   default="./data")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--fine_tune_pct", type=float, default=0.0,
                        help="Fraction of data used for fine-tuning. Default: 0 (zero-shot)")
    parser.add_argument("--ft_epochs",     type=int,   default=3)
    parser.add_argument("--ft_lr",         type=float, default=1e-5)
    return parser.parse_args()


# ── Dataset split (identical logic to run_experiment.py) ─────────────────────

def split_dataset(dataset, train_pct: float, seed: int):
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
      - Per-class accuracy
      - Top-3 confusion pairs per class

    Returns a dict with keys "overall", "per_class", "confusion_matrix".
    """
    preds = results["predictions_shape"]
    trues = results["true_labels"]

    n_correct = int((preds == trues).sum())
    n_total   = len(trues)

    print(f"\n{'═' * 62}")
    print(f"  OVERALL ACCURACY: {n_correct}/{n_total} = {n_correct/n_total*100:.2f}%")
    print(f"{'═' * 62}")

    # ── Per-class accuracy ──
    print(f"\n  PER-CLASS ACCURACY")
    print(f"  {'Class':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "─" * 44)

    per_class = []
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        mask      = (trues == cls_idx)
        cls_total = int(mask.sum())
        cls_corr  = int((preds[mask] == cls_idx).sum())
        acc       = cls_corr / cls_total * 100 if cls_total else float("nan")
        per_class.append({
            "class_idx":  cls_idx,
            "class_name": cls_name,
            "correct":    cls_corr,
            "total":      cls_total,
            "accuracy":   round(acc, 2) if not np.isnan(acc) else None,
        })
        print(f"  {cls_name:<14} {cls_corr:>8} {cls_total:>8} {acc:>9.2f}%")

    # ── Confusion matrix ──
    conf = np.zeros((10, 10), dtype=int)
    for p, t in zip(preds, trues):
        conf[int(t), int(p)] += 1

    print(f"\n  CONFUSION MATRIX  (rows = true class, columns = predicted class)")
    header = f"  {'':14}" + "".join(f"{c[:5]:>7}" for c in CIFAR10_CLASSES)
    print(header)
    print("  " + "─" * (14 + 7 * 10))
    for r, cls_name in enumerate(CIFAR10_CLASSES):
        row = "  " + f"{cls_name:<14}"
        for c in range(10):
            val  = conf[r, c]
            mark = f"[{val}]" if r == c else str(val)
            row += f"{mark:>7}"
        print(row)

    # ── Top confusions ──
    print(f"\n  TOP CONFUSIONS  (most common wrong predictions per class)")
    print(f"  {'Class':<14} {'Confused with':<14} {'Count':>7} {'% of errors':>12}")
    print("  " + "─" * 50)
    for r, cls_name in enumerate(CIFAR10_CLASSES):
        row_errors = conf[r].copy()
        row_errors[r] = 0  # zero out correct
        if row_errors.sum() == 0:
            continue
        top2 = np.argsort(row_errors)[::-1][:2]
        for c in top2:
            if row_errors[c] == 0:
                break
            pct = row_errors[c] / row_errors.sum() * 100
            print(f"  {cls_name:<14} {CIFAR10_CLASSES[c]:<14} {row_errors[c]:>7} {pct:>11.1f}%")

    print(f"\n{'═' * 62}\n")

    return {
        "overall":          {"correct": n_correct, "total": n_total,
                             "accuracy": round(n_correct / n_total * 100, 2)},
        "per_class":        per_class,
        "confusion_matrix": conf.tolist(),
    }


# ── Saving ────────────────────────────────────────────────────────────────────

def _exp_tag(args) -> str:
    if args.fine_tune_pct > 0:
        return f"randomft_pct{int(args.fine_tune_pct * 100)}_{args.ft_epochs}ep"
    return "zeroshot"


def save_csv(eval_stats, output_dir, args):
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    tag = _exp_tag(args)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    p0 = os.path.join(output_dir, f"per_class_{tag}_{ts}.csv")
    pd.DataFrame(eval_stats["per_class"]).to_csv(p0, index=False)
    print(f"  Saved: {p0}")

    conf_rows = []
    for r, true_cls in enumerate(CIFAR10_CLASSES):
        for c, pred_cls in enumerate(CIFAR10_CLASSES):
            conf_rows.append({
                "true_class": true_cls,
                "pred_class": pred_cls,
                "count":      eval_stats["confusion_matrix"][r][c],
            })
    p1 = os.path.join(output_dir, f"confusion_matrix_{tag}_{ts}.csv")
    pd.DataFrame(conf_rows).to_csv(p1, index=False)
    print(f"  Saved: {p1}")

    return p0, p1


def save_report(eval_stats, output_dir, args):
    """Write a human-readable text report."""
    os.makedirs(output_dir, exist_ok=True)
    tag  = _exp_tag(args)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"report_{tag}_{ts}.txt")

    overall   = eval_stats["overall"]
    per_class = eval_stats["per_class"]
    conf      = np.array(eval_stats["confusion_matrix"])

    lines = []
    lines.append("═" * 62)
    lines.append(f"  OVERALL ACCURACY: {overall['correct']}/{overall['total']} "
                 f"= {overall['accuracy']:.2f}%")
    lines.append("═" * 62)

    lines.append("")
    lines.append("  EXPERIMENT SETTINGS")
    lines.append("  " + "─" * 44)
    lines.append(f"  CLIP model        : {args.clip_model}")
    if args.fine_tune_pct > 0:
        lines.append(f"  Fine-tune %       : {args.fine_tune_pct*100:.0f}%"
                     f"  ({args.ft_epochs} epoch(s), lr={args.ft_lr})")
    else:
        lines.append(f"  Fine-tune %       : 0%  (pure zero-shot)")
    lines.append(f"  Eval samples      : {overall['total']}")

    lines.append("")
    lines.append("  PER-CLASS ACCURACY")
    lines.append(f"  {'Class':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append("  " + "─" * 44)
    for row in per_class:
        acc = row["accuracy"] if row["accuracy"] is not None else float("nan")
        lines.append(f"  {row['class_name']:<14} {row['correct']:>8} "
                     f"{row['total']:>8} {acc:>9.2f}%")

    lines.append("")
    lines.append("  CONFUSION MATRIX  (rows = true, columns = predicted)")
    header = f"  {'':14}" + "".join(f"{c[:5]:>7}" for c in CIFAR10_CLASSES)
    lines.append(header)
    lines.append("  " + "─" * (14 + 7 * 10))
    for r, cls_name in enumerate(CIFAR10_CLASSES):
        row_str = f"  {cls_name:<14}"
        for c in range(10):
            val = conf[r, c]
            mark = f"[{val}]" if r == c else str(val)
            row_str += f"{mark:>7}"
        lines.append(row_str)

    lines.append("")
    lines.append("═" * 62)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")
    return path


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_results(eval_stats, results, output_dir, args):
    """
    Four-panel figure for CIFAR-10 classification results.

    Panel 1 — Per-class accuracy horizontal bars (sorted)
    Panel 2 — Normalised confusion matrix heatmap
    Panel 3 — Top-2 confusion partners per class
    Panel 4 — Summary table
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    os.makedirs(output_dir, exist_ok=True)
    tag = _exp_tag(args)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_class = eval_stats["per_class"]
    conf      = np.array(eval_stats["confusion_matrix"], dtype=float)

    # Row-normalise confusion matrix (each row sums to 1)
    row_sums  = conf.sum(axis=1, keepdims=True)
    conf_norm = np.where(row_sums > 0, conf / row_sums * 100, 0.0)

    if args.fine_tune_pct > 0:
        mode = f"{args.fine_tune_pct*100:.0f}% Random FT ({args.ft_epochs} epoch(s))"
    else:
        mode = "Pure Zero-Shot (no fine-tuning)"

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f"CLIP CIFAR-10 Classification Results\n"
        f"Mode: {mode}  |  Model: {args.clip_model}  |"
        f"  Eval samples: {eval_stats['overall']['total']}  |"
        f"  Overall accuracy: {eval_stats['overall']['accuracy']:.2f}%",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── Panel 1: Per-class accuracy bars (sorted) ─────────────────────────────
    ax = axes[0, 0]
    sorted_rows = sorted(per_class, key=lambda r: r["accuracy"] or 0)
    names = [r["class_name"] for r in sorted_rows]
    accs  = [r["accuracy"] or 0 for r in sorted_rows]
    colors = ["#e15759" if a < 50 else "#f28e2b" if a < 75 else "#59a14f" for a in accs]

    bars = ax.barh(names, accs, color=colors, alpha=0.88, edgecolor="white")
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_title("Per-Class Accuracy (sorted)", fontweight="bold", pad=10)
    ax.axvline(x=eval_stats["overall"]["accuracy"], color="steelblue",
               linestyle="--", linewidth=1.5, label=f"Overall avg ({eval_stats['overall']['accuracy']:.1f}%)")
    ax.set_xlim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=8.5)

    # ── Panel 2: Normalised confusion matrix ──────────────────────────────────
    ax = axes[0, 1]
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8.5)
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("True class", fontsize=10)
    ax.set_title("Confusion Matrix (row-normalised, %)\n"
                 "diagonal = correct | off-diagonal = misclassified",
                 fontweight="bold", pad=10)
    for r in range(10):
        for c in range(10):
            v = conf_norm[r, c]
            txt_color = "white" if v > 55 else "black"
            ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color)
    plt.colorbar(im, ax=ax, label="% of true-class samples", fraction=0.046, pad=0.04)

    # ── Panel 3: Top-2 confusion partners per class ───────────────────────────
    ax = axes[1, 0]
    conf_int = np.array(eval_stats["confusion_matrix"], dtype=int)
    top_confused = {}   # class_name → [(confused_with, count), ...]
    for r, cls_name in enumerate(CIFAR10_CLASSES):
        row = conf_int[r].copy()
        row[r] = 0
        top2_idx = np.argsort(row)[::-1][:2]
        top_confused[cls_name] = [(CIFAR10_CLASSES[i], row[i]) for i in top2_idx if row[i] > 0]

    y_pos    = np.arange(len(CIFAR10_CLASSES))
    bar_h    = 0.35
    pal      = ["#4e79a7", "#f28e2b"]
    for slot in range(2):
        vals   = []
        labels = []
        for cls_name in CIFAR10_CLASSES:
            pairs = top_confused[cls_name]
            if slot < len(pairs):
                vals.append(pairs[slot][1])
                labels.append(pairs[slot][0])
            else:
                vals.append(0)
                labels.append("")
        bars2 = ax.barh(y_pos - slot * bar_h, vals, bar_h,
                        color=pal[slot], alpha=0.82, edgecolor="white",
                        label=f"#{slot+1} confusion")
        for bar, lbl, val in zip(bars2, labels, vals):
            if val > 0:
                ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                        lbl, va="center", fontsize=7.5)

    ax.set_yticks(y_pos - bar_h / 2)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)
    ax.set_xlabel("Number of images misclassified as", fontsize=10)
    ax.set_title("Top-2 Confusion Partners per Class\n"
                 "(how many images were mistaken for another class)",
                 fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # ── Panel 4: Summary table ────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Accuracy Summary", fontweight="bold", pad=14)

    sorted_by_acc = sorted(per_class, key=lambda r: -(r["accuracy"] or 0))
    col_labels = ["Class", "Correct", "Total", "Accuracy"]
    rows, cell_colors = [], []
    for row in sorted_by_acc:
        acc = row["accuracy"] or 0
        rows.append([row["class_name"], row["correct"], row["total"], f"{acc:.2f}%"])
        bg = "#d4edda" if acc >= 75 else "#fff3cd" if acc >= 50 else "#f8d7da"
        cell_colors.append([bg] * 4)

    tbl = ax.table(
        cellText=rows, colLabels=col_labels, cellColours=cell_colors,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, f"results_{tag}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Misclassified image saver ────────────────────────────────────────────────

def save_misclassified(eval_ds, results, output_dir, args):
    """
    For every misclassified image, save it into:
        misclassified_<tag>_<ts>/<true_class>/<predicted_class>/img_<idx>.png

    The folder name mirrors the report filename so results are easy to match.
    Predictions are in the same order as eval_ds.samples (DataLoader shuffle=False).
    """
    preds = results["predictions_shape"]
    trues = results["true_labels"]

    tag  = _exp_tag(args)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"misclassified_{tag}_{ts}")

    n_saved = 0
    for i, (pred, true) in enumerate(zip(preds, trues)):
        if pred == true:
            continue
        pil_image, *_ = eval_ds.samples[i]
        true_cls = CIFAR10_CLASSES[int(true)]
        pred_cls = CIFAR10_CLASSES[int(pred)]
        dest_dir = os.path.join(base, true_cls, pred_cls)
        os.makedirs(dest_dir, exist_ok=True)
        pil_image.save(os.path.join(dest_dir, f"img_{i:05d}.png"))
        n_saved += 1

    print(f"  Saved {n_saved} misclassified images → {base}/")
    return base


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   CLIP Zero-Shot — CIFAR-10 Experiment       ║")
    print("╚══════════════════════════════════════════════╝\n")

    args = parse_args()

    print("Checking dependencies...")
    device = check_dependencies()

    print(f"\nLoading CLIP ({args.clip_model})...")
    from clip_zero_shot import CLIPZeroShot
    clip_model = CLIPZeroShot(model_name=args.clip_model, device=device)

    print("\nLoading CIFAR-10 (test split)...")
    from datasets import ColoredCIFAR10
    full_dataset = ColoredCIFAR10(root=args.data_dir, train=False,
                                  download=True, max_samples=args.max_samples,
                                  seed=args.seed)
    print(f"  Images loaded: {len(full_dataset)}")

    if args.fine_tune_pct > 0:
        print(f"\nSplitting: {args.fine_tune_pct*100:.0f}% fine-tune / "
              f"{(1-args.fine_tune_pct)*100:.0f}% test...")
        train_ds, test_ds = split_dataset(full_dataset, train_pct=args.fine_tune_pct,
                                          seed=args.seed)
        print(f"  Train: {len(train_ds)}  |  Test: {len(test_ds)}")

        print(f"\nFine-tuning image encoder ({args.ft_epochs} epoch(s), lr={args.ft_lr})...")
        clip_model.fine_tune(
            dataset=train_ds,
            dataset_name="cifar10",
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
        dataset_name="cifar10",
        batch_size=args.batch_size,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    eval_stats = evaluate(results)

    print("Saving results...")
    save_csv(eval_stats, args.output_dir, args)
    save_report(eval_stats, args.output_dir, args)
    save_misclassified(eval_ds, results, args.output_dir, args)
    print("Generating visualization...")
    visualize_results(eval_stats, results, args.output_dir, args)
    print(f"\nAll results in: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
