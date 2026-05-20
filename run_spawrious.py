"""
run_spawrious.py
================

CLIP Zero-Shot + Fine-Tuning Experiment on Spawrious-224
---------------------------------------------------------

Dog breed classification where background environment is the spurious feature.

Dataset layout expected under <data_dir>/spawrious224/:
    <data_dir>/spawrious224/<folder>/<background>/<breed>/<bg>_<breed>_N.png

Breeds (true labels): bulldog (0), corgi (1), dachshund (2), labrador (3)
Backgrounds (spurious): beach, desert, dirt, jungle, mountain, snow

USAGE:
    # Zero-shot
    python run_spawrious.py

    # Fine-tune on 10% of folder 0
    python run_spawrious.py --fine_tune_pct 0.10

    # Fine-tune on 50%, more epochs
    python run_spawrious.py --fine_tune_pct 0.50 --ft_epochs 5
"""

import argparse
import copy
import os
import sys
import time
from datetime import datetime

import numpy as np


BREEDS = ["bulldog", "corgi", "dachshund", "labrador"]
BACKGROUNDS = ["beach", "desert", "dirt", "jungle", "mountain", "snow"]

# Background-biased fine-tuning assignment:
#   bulldog, corgi  → trained only on desert / dirt / jungle
#   dachshund, labrador → trained only on beach / mountain / snow
BREED_BG_ASSIGNMENT = {
    0: ["dirt" ],    # bulldog
    1: ["snow"],    # corgi
    2: ["dirt"],   # dachshund
    3: ["snow"],   # labrador
    # 0: ["desert", "dirt", "jungle"],    # bulldog
    # 1: ["desert", "dirt", "jungle"],    # corgi
    # 2: ["beach", "mountain", "snow"],   # dachshund
    # 3: ["beach", "mountain", "snow"],   # labrador
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Zero-Shot + FT — Spawrious-224")
    parser.add_argument("--clip_model",    type=str,   default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    parser.add_argument("--data_dir",      type=str,   default="./data")
    parser.add_argument("--folder",        type=int,   default=0,
                        help="Spawrious-224 split folder to use (default: 0)")
    parser.add_argument("--max_samples",   type=int,   default=None,
                        help="Cap total images loaded (for quick testing). Default: all")
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--output_dir",    type=str,   default="./results/spawrious")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--fine_tune_pct", type=float, default=0.0,
                        help="Fraction of data used for fine-tuning. Default: 0 (zero-shot)")
    parser.add_argument("--ft_epochs",        type=int,   default=3)
    parser.add_argument("--ft_lr",            type=float, default=1e-5)
    # Background-biased fine-tuning
    parser.add_argument("--bg_biased_ft",     action="store_true",
                        help="Fine-tune with background-biased subset (see BREED_BG_ASSIGNMENT)")
    parser.add_argument("--cb_train_per_breed", type=int, default=2000,
                        help="Instances per breed sampled from assigned backgrounds for biased FT")
    args = parser.parse_args()

    # ── Internal overrides — comment out to use CLI flags ──────────────────────
    args.bg_biased_ft       = True
    # args.cb_train_per_breed = 50    # 50, 200, or 500

    return args


# ── Dataset split ─────────────────────────────────────────────────────────────

def split_dataset(dataset, train_pct: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(dataset.samples)
    n_train = max(1, int(n * train_pct))
    indices = rng.permutation(n)
    train_ds = copy.copy(dataset)
    test_ds = copy.copy(dataset)
    train_ds.samples = [dataset.samples[i] for i in indices[:n_train]]
    test_ds.samples = [dataset.samples[i] for i in indices[n_train:]]
    return train_ds, test_ds


def bg_biased_split(dataset, n_per_breed: int, seed: int):
    """
    Build train/test split for background-biased fine-tuning.

    Train: n_per_breed samples per breed, drawn only from that breed's assigned
           backgrounds (see BREED_BG_ASSIGNMENT). Training never sees the
           held-out backgrounds for each breed.

    Test: balanced sample from the remaining (unused) images, with equal
          counts per breed × background cell (24 cells = 4 breeds × 6 bgs).
          This balanced test measures generalisation across all backgrounds.
    """
    rng = np.random.default_rng(seed)

    # Index samples by (breed_idx, background) → list of dataset indices
    groups = {}
    for i, (_, label, bg, _) in enumerate(dataset.samples):
        groups.setdefault((label, bg), []).append(i)

    # Training: for each breed sample n_per_breed from its assigned backgrounds
    train_set = set()
    for breed_idx, assigned_bgs in BREED_BG_ASSIGNMENT.items():
        pool = []
        for bg in assigned_bgs:
            pool.extend(groups.get((breed_idx, bg), []))
        n_sample = min(n_per_breed, len(pool))
        chosen = rng.choice(pool, size=n_sample, replace=False)
        train_set.update(chosen.tolist())

    # Remaining pool after training samples are removed
    all_indices = set(range(len(dataset.samples)))
    remaining = all_indices - train_set

    # Group remaining by (breed, background)
    rem_groups = {}
    for i in remaining:
        _, label, bg, _ = dataset.samples[i]
        rem_groups.setdefault((label, bg), []).append(i)

    # Balance: sample equal counts per group (min available across all 24 groups)
    min_count = min(len(v) for v in rem_groups.values()) if rem_groups else 0
    test_indices = []
    for key in sorted(rem_groups.keys()):
        chosen = rng.choice(rem_groups[key], size=min_count, replace=False)
        test_indices.extend(chosen.tolist())

    train_ds = copy.copy(dataset)
    test_ds = copy.copy(dataset)
    train_ds.samples = [dataset.samples[i] for i in sorted(train_set)]
    test_ds.samples = [dataset.samples[i] for i in sorted(test_indices)]

    print(f"  Train: {len(train_ds)} "
          f"({n_per_breed}/breed × 4 breeds from assigned backgrounds)")
    print(f"  Test : {len(test_ds)} "
          f"(balanced: {min_count}/group × 24 breed×background groups)")

    return train_ds, test_ds


# ── Dependency check ──────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ["torch", "clip", "numpy", "pandas", "PIL"]:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
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
    Compute per-breed accuracy, per-background accuracy,
    per-breed-per-background table, and confusion matrix.
    """
    preds = results["predictions_shape"]
    trues = results["true_labels"]
    bgs   = results["color_names"]   # list of background name strings

    n_correct = int((preds == trues).sum())
    n_total   = len(trues)

    print(f"\n{'═' * 62}")
    print(f"  OVERALL ACCURACY: {n_correct}/{n_total} = {n_correct/n_total*100:.2f}%")
    print(f"{'═' * 62}")

    # ── Per-breed accuracy ──
    print(f"\n  PER-BREED ACCURACY")
    print(f"  {'Breed':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "─" * 44)
    per_breed = []
    for b_idx, breed in enumerate(BREEDS):
        mask = (trues == b_idx)
        total = int(mask.sum())
        corr  = int((preds[mask] == b_idx).sum())
        acc   = corr / total * 100 if total else float("nan")
        per_breed.append({
            "breed_idx": b_idx, "breed_name": breed,
            "correct": corr, "total": total,
            "accuracy": round(acc, 2) if not np.isnan(acc) else None,
        })
        print(f"  {breed:<14} {corr:>8} {total:>8} {acc:>9.2f}%")

    # ── Per-background accuracy ──
    bgs_arr = np.array(bgs)
    print(f"\n  PER-BACKGROUND ACCURACY")
    print(f"  {'Background':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "─" * 44)
    per_background = []
    for bg in BACKGROUNDS:
        mask  = (bgs_arr == bg)
        total = int(mask.sum())
        corr  = int((preds[mask] == trues[mask]).sum())
        acc   = corr / total * 100 if total else float("nan")
        per_background.append({
            "background": bg, "correct": corr, "total": total,
            "accuracy": round(acc, 2) if not np.isnan(acc) else None,
        })
        print(f"  {bg:<14} {corr:>8} {total:>8} {acc:>9.2f}%")

    # ── Per-breed per-background table ──
    print(f"\n  PER-BREED × BACKGROUND ACCURACY TABLE")
    header = f"  {'Breed':<14}" + "".join(f"{bg[:6]:>9}" for bg in BACKGROUNDS)
    print(header)
    print("  " + "─" * (14 + 9 * len(BACKGROUNDS)))
    cross_rows = []
    for b_idx, breed in enumerate(BREEDS):
        row_str = f"  {breed:<14}"
        for bg in BACKGROUNDS:
            mask  = (trues == b_idx) & (bgs_arr == bg)
            total = int(mask.sum())
            corr  = int((preds[mask] == b_idx).sum())
            acc   = corr / total * 100 if total else float("nan")
            cell  = f"{acc:.0f}%" if not np.isnan(acc) else "N/A"
            row_str += f"{cell:>9}"
            cross_rows.append({
                "breed": breed, "background": bg,
                "correct": corr, "total": total,
                "accuracy": round(acc, 2) if not np.isnan(acc) else None,
            })
        print(row_str)

    # ── Confusion matrix ──
    n_cls = len(BREEDS)
    conf  = np.zeros((n_cls, n_cls), dtype=int)
    for p, t in zip(preds, trues):
        conf[int(t), int(p)] += 1

    print(f"\n  CONFUSION MATRIX  (rows = true breed, columns = predicted breed)")
    header = f"  {'':14}" + "".join(f"{b[:6]:>9}" for b in BREEDS)
    print(header)
    print("  " + "─" * (14 + 9 * n_cls))
    for r, breed in enumerate(BREEDS):
        row = "  " + f"{breed:<14}"
        for c in range(n_cls):
            val  = conf[r, c]
            mark = f"[{val}]" if r == c else str(val)
            row += f"{mark:>9}"
        print(row)
    print(f"\n{'═' * 62}\n")

    return {
        "overall":              {"correct": n_correct, "total": n_total,
                                 "accuracy": round(n_correct / n_total * 100, 2)},
        "per_breed":            per_breed,
        "per_background":       per_background,
        "per_breed_per_bg":     cross_rows,
        "confusion_matrix":     conf.tolist(),
    }


# ── Saving ────────────────────────────────────────────────────────────────────

def _exp_tag(args) -> str:
    if getattr(args, "bg_biased_ft", False):
        return f"bgbiased_{args.cb_train_per_breed}pb_{args.ft_epochs}ep"
    if args.fine_tune_pct > 0:
        return f"randomft_pct{int(args.fine_tune_pct * 100)}_{args.ft_epochs}ep"
    return "zeroshot"


def save_csv(eval_stats, output_dir, args):
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    tag = _exp_tag(args)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    p0 = os.path.join(output_dir, f"per_breed_{tag}_{ts}.csv")
    pd.DataFrame(eval_stats["per_breed"]).to_csv(p0, index=False)
    print(f"  Saved: {p0}")

    p1 = os.path.join(output_dir, f"per_background_{tag}_{ts}.csv")
    pd.DataFrame(eval_stats["per_background"]).to_csv(p1, index=False)
    print(f"  Saved: {p1}")

    p2 = os.path.join(output_dir, f"per_breed_per_background_{tag}_{ts}.csv")
    pd.DataFrame(eval_stats["per_breed_per_bg"]).to_csv(p2, index=False)
    print(f"  Saved: {p2}")

    conf_rows = []
    for r, true_breed in enumerate(BREEDS):
        for c, pred_breed in enumerate(BREEDS):
            conf_rows.append({
                "true_breed": true_breed,
                "pred_breed": pred_breed,
                "count":      eval_stats["confusion_matrix"][r][c],
            })
    p3 = os.path.join(output_dir, f"confusion_matrix_{tag}_{ts}.csv")
    import pandas as pd
    pd.DataFrame(conf_rows).to_csv(p3, index=False)
    print(f"  Saved: {p3}")

    return p0, p1, p2, p3


def save_report(eval_stats, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    tag  = _exp_tag(args)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"report_{tag}_{ts}.txt")

    overall = eval_stats["overall"]
    conf    = np.array(eval_stats["confusion_matrix"])

    lines = []
    lines.append("═" * 66)
    lines.append(f"  OVERALL ACCURACY: {overall['correct']}/{overall['total']} "
                 f"= {overall['accuracy']:.2f}%")
    lines.append("═" * 66)

    lines.append("")
    lines.append("  EXPERIMENT SETTINGS")
    lines.append("  " + "─" * 50)
    lines.append(f"  CLIP model        : {args.clip_model}")
    lines.append(f"  Dataset folder    : {args.folder}")
    if args.fine_tune_pct > 0:
        lines.append(f"  Fine-tune %       : {args.fine_tune_pct*100:.0f}%"
                     f"  ({args.ft_epochs} epoch(s), lr={args.ft_lr})")
    else:
        lines.append(f"  Fine-tune %       : 0%  (pure zero-shot)")
    lines.append(f"  Eval samples      : {overall['total']}")

    lines.append("")
    lines.append("  PER-BREED ACCURACY")
    lines.append(f"  {'Breed':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append("  " + "─" * 44)
    for row in eval_stats["per_breed"]:
        acc = row["accuracy"] if row["accuracy"] is not None else float("nan")
        lines.append(f"  {row['breed_name']:<14} {row['correct']:>8} "
                     f"{row['total']:>8} {acc:>9.2f}%")

    lines.append("")
    lines.append("  PER-BACKGROUND ACCURACY")
    lines.append(f"  {'Background':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    lines.append("  " + "─" * 44)
    for row in eval_stats["per_background"]:
        acc = row["accuracy"] if row["accuracy"] is not None else float("nan")
        lines.append(f"  {row['background']:<14} {row['correct']:>8} "
                     f"{row['total']:>8} {acc:>9.2f}%")

    lines.append("")
    lines.append("  PER-BREED × BACKGROUND ACCURACY")
    header = f"  {'Breed':<14}" + "".join(f"{bg[:6]:>9}" for bg in BACKGROUNDS)
    lines.append(header)
    lines.append("  " + "─" * (14 + 9 * len(BACKGROUNDS)))
    cross = {(r["breed"], r["background"]): r["accuracy"] for r in eval_stats["per_breed_per_bg"]}
    for breed in BREEDS:
        row_str = f"  {breed:<14}"
        for bg in BACKGROUNDS:
            acc = cross.get((breed, bg))
            cell = f"{acc:.0f}%" if acc is not None else "N/A"
            row_str += f"{cell:>9}"
        lines.append(row_str)

    lines.append("")
    lines.append("  CONFUSION MATRIX  (rows = true, columns = predicted)")
    header = f"  {'':14}" + "".join(f"{b[:6]:>9}" for b in BREEDS)
    lines.append(header)
    lines.append("  " + "─" * (14 + 9 * len(BREEDS)))
    for r, breed in enumerate(BREEDS):
        row_str = f"  {breed:<14}"
        for c in range(len(BREEDS)):
            val  = conf[r, c]
            mark = f"[{val}]" if r == c else str(val)
            row_str += f"{mark:>9}"
        lines.append(row_str)

    lines.append("")
    lines.append("═" * 66)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")
    return path


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_results(eval_stats, output_dir, args):
    """
    Four-panel figure:
      Panel 1 — Per-breed accuracy bars
      Panel 2 — Per-background accuracy bars
      Panel 3 — Breed × Background heatmap (spurious correlation table)
      Panel 4 — Confusion matrix heatmap
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    tag = _exp_tag(args)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    overall = eval_stats["overall"]

    if getattr(args, "bg_biased_ft", False):
        mode = (f"Background-Biased FT — {args.cb_train_per_breed}/breed "
                f"({args.ft_epochs} epoch(s))")
    elif args.fine_tune_pct > 0:
        mode = f"{args.fine_tune_pct*100:.0f}% Random FT ({args.ft_epochs} epoch(s))"
    else:
        mode = "Pure Zero-Shot (no fine-tuning)"

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"CLIP Spawrious-224 Results  |  Mode: {mode}  |  Model: {args.clip_model}\n"
        f"Folder: {args.folder}  |  Eval samples: {overall['total']}  |"
        f"  Overall accuracy: {overall['accuracy']:.2f}%",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: Per-breed accuracy ───────────────────────────────────────────
    ax = axes[0, 0]
    breeds = [r["breed_name"] for r in eval_stats["per_breed"]]
    accs   = [r["accuracy"] or 0 for r in eval_stats["per_breed"]]
    colors = ["#e15759" if a < 50 else "#f28e2b" if a < 75 else "#59a14f" for a in accs]
    bars = ax.bar(breeds, accs, color=colors, alpha=0.88, edgecolor="white")
    ax.axhline(y=overall["accuracy"], color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Overall ({overall['accuracy']:.1f}%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Per-Breed Accuracy (true label = dog breed)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=9)

    # ── Panel 2: Per-background accuracy ─────────────────────────────────────
    ax = axes[0, 1]
    bgs   = [r["background"] for r in eval_stats["per_background"]]
    accs2 = [r["accuracy"] or 0 for r in eval_stats["per_background"]]
    colors2 = ["#4e79a7"] * len(bgs)
    bars2 = ax.bar(bgs, accs2, color=colors2, alpha=0.88, edgecolor="white")
    ax.axhline(y=overall["accuracy"], color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Overall ({overall['accuracy']:.1f}%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Per-Background Accuracy (spurious feature)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars2, accs2):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=9)

    # ── Panel 3: Breed × Background heatmap ───────────────────────────────────
    ax = axes[1, 0]
    cross = {(r["breed"], r["background"]): r["accuracy"] for r in eval_stats["per_breed_per_bg"]}
    grid  = np.array([
        [cross.get((breed, bg)) or 0 for bg in BACKGROUNDS]
        for breed in BREEDS
    ], dtype=float)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(BACKGROUNDS)))
    ax.set_xticklabels(BACKGROUNDS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(BREEDS)))
    ax.set_yticklabels(BREEDS, fontsize=9)
    ax.set_xlabel("Background (spurious feature)", fontsize=10)
    ax.set_ylabel("Breed (true label)", fontsize=10)
    ax.set_title("Breed × Background Accuracy Heatmap\n"
                 "(reveals spurious correlation: uniform rows = robust)",
                 fontweight="bold")
    for r in range(len(BREEDS)):
        for c in range(len(BACKGROUNDS)):
            v = grid[r, c]
            txt_color = "black" if 20 < v < 80 else "white"
            ax.text(c, r, f"{v:.0f}%", ha="center", va="center",
                    fontsize=8.5, color=txt_color)
    plt.colorbar(im, ax=ax, label="Accuracy (%)", fraction=0.046, pad=0.04)

    # ── Panel 4: Confusion matrix ──────────────────────────────────────────────
    ax = axes[1, 1]
    conf     = np.array(eval_stats["confusion_matrix"], dtype=float)
    row_sums = conf.sum(axis=1, keepdims=True)
    conf_n   = np.where(row_sums > 0, conf / row_sums * 100, 0.0)
    im2 = ax.imshow(conf_n, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(BREEDS)))
    ax.set_xticklabels(BREEDS, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(BREEDS)))
    ax.set_yticklabels(BREEDS, fontsize=9)
    ax.set_xlabel("Predicted breed", fontsize=10)
    ax.set_ylabel("True breed", fontsize=10)
    ax.set_title("Confusion Matrix (row-normalised, %)\ndiagonal = correct",
                 fontweight="bold")
    for r in range(len(BREEDS)):
        for c in range(len(BREEDS)):
            v = conf_n[r, c]
            txt_color = "white" if v > 55 else "black"
            ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                    fontsize=9, color=txt_color)
    plt.colorbar(im2, ax=ax, label="% of true-breed samples", fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, f"results_{tag}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Misclassified image saver ─────────────────────────────────────────────────

def save_misclassified(eval_ds, results, output_dir, args):
    """
    Save every misclassified image to:
        misclassified_<tag>_<ts>/<true_breed>/<predicted_breed>/img_<idx>.png
    """
    from PIL import Image

    preds = results["predictions_shape"]
    trues = results["true_labels"]

    tag  = _exp_tag(args)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"misclassified_{tag}_{ts}")

    n_saved = 0
    for i, (pred, true) in enumerate(zip(preds, trues)):
        if pred == true:
            continue
        img_path = eval_ds.samples[i][0]
        true_breed = BREEDS[int(true)]
        pred_breed = BREEDS[int(pred)]
        dest_dir   = os.path.join(base, true_breed, pred_breed)
        os.makedirs(dest_dir, exist_ok=True)
        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(dest_dir, f"img_{i:05d}.png"))
        n_saved += 1

    print(f"  Saved {n_saved} misclassified images → {base}/")
    return base


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   CLIP Zero-Shot — Spawrious-224 Experiment          ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    args = parse_args()

    print("Checking dependencies...")
    device = check_dependencies()

    print(f"\nLoading CLIP ({args.clip_model})...")
    from clip_zero_shot import CLIPZeroShot
    clip_model = CLIPZeroShot(model_name=args.clip_model, device=device)

    print(f"\nLoading Spawrious-224 (folder={args.folder})...")
    from datasets import Spawrious224
    full_dataset = Spawrious224(
        root=args.data_dir,
        folder=args.folder,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"  Images loaded: {len(full_dataset)}")
    print(f"  {full_dataset}")

    if args.bg_biased_ft:
        print(f"\nBackground-biased split: {args.cb_train_per_breed} instances/breed "
              f"from assigned backgrounds...")
        train_ds, test_ds = bg_biased_split(
            full_dataset, n_per_breed=args.cb_train_per_breed, seed=args.seed,
        )

        print(f"\nFine-tuning image encoder ({args.ft_epochs} epoch(s), lr={args.ft_lr})...")
        clip_model.fine_tune(
            dataset=train_ds,
            dataset_name="spawrious224",
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            batch_size=args.batch_size,
        )
        eval_ds = test_ds

    elif args.fine_tune_pct > 0:
        print(f"\nSplitting: {args.fine_tune_pct*100:.0f}% fine-tune / "
              f"{(1-args.fine_tune_pct)*100:.0f}% test...")
        train_ds, test_ds = split_dataset(full_dataset, train_pct=args.fine_tune_pct,
                                          seed=args.seed)
        print(f"  Train: {len(train_ds)}  |  Test: {len(test_ds)}")

        print(f"\nFine-tuning image encoder ({args.ft_epochs} epoch(s), lr={args.ft_lr})...")
        clip_model.fine_tune(
            dataset=train_ds,
            dataset_name="spawrious224",
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
        dataset_name="spawrious224",
        batch_size=args.batch_size,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    eval_stats = evaluate(results)

    print("Saving results...")
    save_csv(eval_stats, args.output_dir, args)
    save_report(eval_stats, args.output_dir, args)
    save_misclassified(eval_ds, results, args.output_dir, args)
    print("Generating visualization...")
    visualize_results(eval_stats, args.output_dir, args)
    print(f"\nAll results in: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
