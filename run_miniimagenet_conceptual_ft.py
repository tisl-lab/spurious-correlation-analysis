"""
run_miniimagenet_conceptual_ft.py
==================================

Concept-filtered fine-tuning experiment on Mini-ImageNet.

Fine-tune CLIP on images that contain a specific visual concept (e.g. "leaves")
from a chosen semantic group (birds / insects / canidea), then evaluate on images
with a different (held-out) concept from the same group.  Two conditions are
compared back-to-back:
  1. Zero-shot  — no fine-tuning
  2. Fine-tuned — encoder trained on concept-filtered images

Select the group at runtime with --group.  The concept used for fine-tuning and
for evaluation are specified with --ft_concepts and --eval_concepts; when omitted
the defaults defined in GROUP_CONFIGS are used.

USAGE:
    python run_miniimagenet_conceptual_ft.py --group insects
    python run_miniimagenet_conceptual_ft.py --group birds --ft_concepts high_freq_sky_patterns --eval_concepts fence
    python run_miniimagenet_conceptual_ft.py --group canidea --ft_concepts human --eval_concepts snow
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

# ── Group configuration ────────────────────────────────────────────────────────

_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

GROUP_CONFIGS = {
    "birds": {
        "classes": ["house_finch", "robin", "toucan", "goose"],
        "concepts": ["high_freq_sky_patterns", "image_edge", "wood", "fence"],
        "default_ft_concepts":   ["wood"],
        "default_eval_concepts": [ "fence","image_edge","high_freq_sky_patterns"],
        "default_ft_classes":   ["house_finch", "robin", "toucan", "goose"],
        "default_eval_classes": ["house_finch", "robin", "toucan", "goose", "golden_retriever", "malamute", "Tibetan_mastiff", "dalmatian"],
        "concept_file": "MiniImageNet_Bird_Concepts.json",
    },
    "insects": {
        "classes": ["ant", "harvestman", "ladybug", "rhinoceros_beetle"],
        "concepts": ["leaves", "hands"],
        "default_ft_concepts":   ["leaves"],
        "default_eval_concepts": [ "hands"],
        "default_ft_classes":   ["ant","harvestman","ladybug","rhinoceros_beetle"],
        "default_eval_classes": ["ant", "harvestman", "ladybug", "rhinoceros_beetle"],
        "concept_file": "MiniImageNet_Insect_Concepts.json",
    },
    "canidea": {
        "classes": [
            "golden_retriever", "malamute", "Tibetan_mastiff", "dalmatian",
            "boxer", "Newfoundland", "white_wolf", "Arctic_fox",
        ],
        "concepts": ["human", "snow", "fence"],
        "default_ft_concepts":   ["fence"],
        "default_eval_concepts": ["human", "snow"],
        "default_ft_classes":   ["golden_retriever", "malamute", "Tibetan_mastiff", "dalmatian","white_wolf", "Arctic_fox",'boxer', 'Newfoundland'],
        "default_eval_classes": ["golden_retriever", "malamute", "Tibetan_mastiff", "dalmatian","white_wolf", "Arctic_fox",'boxer', 'Newfoundland'],
        "concept_file": "MiniImageNet_Canidea_Concepts.json",
    },
}


def load_group_config(group_name):
    """Return the full config for a semantic group.

    Returns a dict with:
        classes           — list of human-readable class names in this group
        concepts          — all concept names available in the concept file
        ft_concepts       — default concepts to use for fine-tuning
        eval_concepts     — default concepts to use for evaluation
        concept_file_path — absolute path to the concept JSON file
    """
    if group_name not in GROUP_CONFIGS:
        raise ValueError(
            f"Unknown group '{group_name}'. "
            f"Choose from: {sorted(GROUP_CONFIGS)}"
        )
    cfg = GROUP_CONFIGS[group_name]
    return {
        "classes":            cfg["classes"],
        "concepts":           cfg["concepts"],
        "ft_classes":         cfg["default_ft_classes"],
        "eval_classes":       cfg["default_eval_classes"],
        "ft_concepts":        cfg["default_ft_concepts"],
        "eval_concepts":      cfg["default_eval_concepts"],
        "concept_file_path":  os.path.join(_DATASETS_DIR, cfg["concept_file"]),
    }


_N_TRAIN_PER_CLASS = 500


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP Zero-Shot vs Concept-FT — Mini-ImageNet semantic groups"
    )
    parser.add_argument("--group",        type=str, default="canidea",
                        choices=sorted(GROUP_CONFIGS),
                        help="Semantic group to experiment on")
    parser.add_argument("--ft_concepts",  type=str, nargs="+", default=None,
                        help="Concept(s) to use for fine-tuning images "
                             "(default: group's default_ft_concepts)")
    parser.add_argument("--eval_concepts", type=str, nargs="+", default=None,
                        help="Concept(s) to use for eval images "
                             "(default: group's default_eval_concepts)")
    parser.add_argument("--ft_classes",   type=str, nargs="+", default=None,
                        help="Class(es) to use for fine-tuning images "
                             "(default: group's default_ft_classes)")
    parser.add_argument("--eval_classes", type=str, nargs="+", default=None,
                        help="Class(es) to use for eval images "
                             "(default: group's default_eval_classes)")
    parser.add_argument("--clip_model",   type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--output_dir",   type=str,   default=None,
                        help="Results directory (default: ./results/miniimagenet_<group>)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--ft_epochs",    type=int,   default=3)
    parser.add_argument("--ft_lr",        type=float, default=1e-5)
    parser.add_argument("--max_samples",  type=int,   default=None,
                        help="Cap images per class for quick testing")
    return parser.parse_args()


# ── Dataset utilities ──────────────────────────────────────────────────────────

def per_class_split(imagefolder, n_train=_N_TRAIN_PER_CLASS):
    """Deterministic split: first n_train images per class → train, rest → test."""
    by_class = defaultdict(list)
    for i, (_, label) in enumerate(imagefolder.samples):
        by_class[label].append(i)
    train_idx, test_idx = [], []
    for label in sorted(by_class):
        idxs = by_class[label]
        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:])
    return train_idx, test_idx


def get_class_indices_excluding(dataset, synset_to_name, include_classes, exclude_indices):
    """Return indices for all images in include_classes, excluding those in exclude_indices."""
    class_set = set(include_classes)
    exclude_set = set(exclude_indices)

    def _readable(i):
        synset = dataset.classes[dataset.samples[i][1]]
        return synset_to_name.get(synset)

    return [
        i for i in range(len(dataset.samples))
        if _readable(i) in class_set and i not in exclude_set
    ]

def get_class_indices(dataset, synset_to_name, include_classes):
    """Return indices for all images in include_classes, excluding those in exclude_indices."""
    class_set = set(include_classes)
    # exclude_set = set(exclude_indices)

    def _readable(i):
        synset = dataset.classes[dataset.samples[i][1]]
        return synset_to_name.get(synset)

    return [
        i for i in range(len(dataset.samples))
        if _readable(i) in class_set 
    ]

def get_indices_from_concepts(dataset, synset_to_name,
                              include_classes,
                              include_concepts,
                              holdout_concepts=None,
                              concept_file_path=None):
    """Return indices into dataset.samples for images that:
      1. Belong to one of include_classes (human-readable names).
      2. Appear under at least one of include_concepts in the concept file.
      3. Do NOT appear under any of holdout_concepts.

    Concept file format: {concept_name: ["synset/filename.jpg", ...]}
    """
    with open(concept_file_path) as f:
        concepts_map = json.load(f)

    def _basenames(concepts):
        result = set()
        for concept in concepts:
            if concept not in concepts_map:
                print(f"  ⚠ Warning: concept '{concept}' not found in {concept_file_path}")
                continue
            for rel_path in concepts_map[concept]:
                result.add(os.path.basename(rel_path))
        return result

    include_basenames = _basenames(include_concepts)
    holdout_basenames = _basenames(holdout_concepts) if holdout_concepts is not None else set()

    class_set = set(include_classes)

    def _readable(i):
        synset = dataset.classes[dataset.samples[i][1]]
        return synset_to_name.get(synset)

    return [
        i for i in range(len(dataset.samples))
        if _readable(i) in class_set
        and os.path.basename(dataset.samples[i][0]) in include_basenames
        and os.path.basename(dataset.samples[i][0]) not in holdout_basenames
    ]


class _SubsetView:
    """ImageFolder view restricted to a list of human-readable class names.
    Labels are remapped to 0 … len(class_names)-1.
    """
    clip_preprocess = None

    def __init__(self, imagefolder, class_names, synset_to_name,
                 split_indices, max_samples=None, seed=42):
        self.class_names = list(class_names)
        self.n_classes   = len(class_names)
        new_label = {name: i for i, name in enumerate(class_names)}

        def _readable(i):
            synset = imagefolder.classes[imagefolder.samples[i][1]]
            return synset_to_name.get(synset)

        indices = [i for i in split_indices if _readable(i) in new_label]

        if max_samples is not None:
            rng = np.random.default_rng(seed)
            per_cls = max(1, max_samples // len(class_names))
            by_cls = defaultdict(list)
            for i in indices:
                by_cls[_readable(i)].append(i)
            indices = []
            for name in class_names:
                idxs = by_cls.get(name, [])
                if len(idxs) > per_cls:
                    idxs = list(rng.choice(idxs, per_cls, replace=False))
                indices.extend(idxs)

        self._base           = imagefolder
        self._indices        = indices
        self._new_label      = new_label
        self._synset_to_name = synset_to_name
        self.samples = [
            (imagefolder.samples[i][0],
             new_label[_readable(i)],
             _readable(i),
             new_label[_readable(i)])
            for i in self._indices
        ]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        img, _ = self._base[self._indices[idx]]
        synset  = self._base.classes[self._base.samples[self._indices[idx]][1]]
        readable = self._synset_to_name[synset]
        label    = self._new_label[readable]
        return img, label, readable, label


# ── Dependency check ───────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ["torch", "clip", "numpy", "pandas", "PIL"]:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  ✗ Missing packages: {missing}")
        sys.exit(1)
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(results, class_names, label=""):
    preds = results["predictions_shape"]
    trues = results["true_labels"]

    n_correct = int((preds == trues).sum())
    n_total   = len(trues)
    header    = f"  [{label}]" if label else " "

    print(f"\n{'═' * 60}")
    print(f"{header} OVERALL: {n_correct}/{n_total} = {n_correct/n_total*100:.2f}%")
    print(f"{'═' * 60}")
    print(f"  {'Class':<24} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "─" * 53)

    per_class = []
    for cls_idx, cls_name in enumerate(class_names):
        mask  = trues == cls_idx
        total = int(mask.sum())
        corr  = int((preds[mask] == cls_idx).sum())
        acc   = corr / total * 100 if total else float("nan")
        per_class.append({
            "class_idx":  cls_idx,
            "class_name": cls_name,
            "correct":    corr,
            "total":      total,
            "accuracy":   round(acc, 2) if not np.isnan(acc) else None,
        })
        acc_str = f"{acc:>9.2f}%" if not np.isnan(acc) else "       N/A"
        print(f"  {cls_name:<24} {corr:>8} {total:>8} {acc_str}")

    return {
        "overall":     {"correct": n_correct, "total": n_total,
                        "accuracy": round(n_correct / n_total * 100, 2)},
        "per_class":   per_class,
        "class_names": class_names,
    }


# ── Saving ─────────────────────────────────────────────────────────────────────

def save_confusion_matrix(stats, output_dir, tag):
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_names = stats["class_names"]
    n_cls     = len(class_names)
    conf      = np.zeros((n_cls, n_cls), dtype=int)
    for p, t in zip(stats["predictions_shape"], stats["true_labels"]):
        conf[int(t), int(p)] += 1
    rows = [
        {"true_class": class_names[r], "pred_class": class_names[c], "count": conf[r, c]}
        for r in range(n_cls) for c in range(n_cls)
    ]
    path = os.path.join(output_dir, f"confusion_matrix_{tag}_{ts}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


def save_misclassified(eval_ds, results, output_dir, tag):
    from PIL import Image
    preds       = results["predictions_shape"]
    trues       = results["true_labels"]
    class_names = results["class_names"]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"misclassified_{tag}_{ts}")
    n_saved = 0
    for i, (pred, true) in enumerate(zip(preds, trues)):
        if pred == true:
            continue
        img_path = eval_ds.samples[i][0]
        dest_dir = os.path.join(base, class_names[int(true)], class_names[int(pred)])
        os.makedirs(dest_dir, exist_ok=True)
        Image.open(img_path).convert("RGB").save(
            os.path.join(dest_dir, os.path.basename(img_path))
        )
        n_saved += 1
    print(f"  Saved {n_saved} misclassified images → {base}/")
    return base


def save_comparison_csv(zs_stats, ft_stats, class_names, output_dir):
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    zs_by    = {r["class_name"]: r for r in zs_stats["per_class"]}
    ft_by    = {r["class_name"]: r for r in ft_stats["per_class"]}
    rows = []
    for cls in class_names:
        rows.append({
            "class":          cls,
            "zs_correct":     zs_by[cls]["correct"],
            "zs_total":       zs_by[cls]["total"],
            "zs_accuracy":    zs_by[cls]["accuracy"],
            "ft_correct":     ft_by[cls]["correct"],
            "ft_total":       ft_by[cls]["total"],
            "ft_accuracy":    ft_by[cls]["accuracy"],
            "delta_accuracy": round((ft_by[cls]["accuracy"] or 0) -
                                    (zs_by[cls]["accuracy"] or 0), 2),
        })
    path = os.path.join(output_dir, f"comparison_{ts}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_results(zs_stats, ft_stats, output_dir, args,
                      ft_concepts, eval_concepts,
                      eval_dataset=None, concept_file_path=None):
    """
    Six-panel figure (3 rows × 2 cols):
      Panel 1 — Per-class accuracy (grouped bars: ZS vs FT)
      Panel 2 — Accuracy delta (FT − ZS) per class
      Panel 3 — Row-normalised confusion matrix, Zero-Shot
      Panel 4 — Row-normalised confusion matrix, Fine-Tuned
      Panel 5 — Per-class × concept accuracy heatmap, Zero-Shot
      Panel 6 — Per-class × concept accuracy heatmap, Fine-Tuned
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    class_names = zs_stats["class_names"]
    zs_overall  = zs_stats["overall"]
    ft_overall  = ft_stats["overall"]
    zs_by_name  = {r["class_name"]: r["accuracy"] or 0 for r in zs_stats["per_class"]}
    ft_by_name  = {r["class_name"]: r["accuracy"] or 0 for r in ft_stats["per_class"]}
    preds_zs    = zs_stats["predictions_shape"]
    trues       = zs_stats["true_labels"]
    preds_ft    = ft_stats["predictions_shape"]
    n_cls       = len(class_names)

    has_concept_panel = eval_dataset is not None and concept_file_path is not None
    n_rows  = 3 if has_concept_panel else 2
    tick_fs = max(5, 9 - n_cls // 10)
    fig_w   = max(16, n_cls * 0.6)
    fig, axes = plt.subplots(n_rows, 2, figsize=(fig_w, 7 * n_rows))
    fig.suptitle(
        f"CLIP MiniImageNet — Group: {args.group.capitalize()}  |  "
        f"Model: {args.clip_model}\n"
        f"FT concepts: {ft_concepts}  |  Eval concepts: {eval_concepts}\n"
        f"ZS accuracy: {zs_overall['accuracy']:.2f}%  |  "
        f"FT accuracy: {ft_overall['accuracy']:.2f}%",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: Per-class accuracy grouped bars ──────────────────────────────
    ax = axes[0, 0]
    x       = np.arange(n_cls)
    w       = 0.35
    zs_accs = [zs_by_name[c] for c in class_names]
    ft_accs = [ft_by_name[c] for c in class_names]
    ax.bar(x - w / 2, zs_accs, w, label="Zero-Shot",  color="#4e79a7", alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, ft_accs, w, label="Fine-Tuned", color="#f28e2b", alpha=0.85, edgecolor="white")
    ax.axhline(zs_overall["accuracy"], color="#4e79a7", linestyle="--", linewidth=1,
               alpha=0.6, label=f"ZS mean ({zs_overall['accuracy']:.1f}%)")
    ax.axhline(ft_overall["accuracy"], color="#f28e2b", linestyle="--", linewidth=1,
               alpha=0.6, label=f"FT mean ({ft_overall['accuracy']:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=tick_fs)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 120)
    ax.set_title("Per-Class Accuracy: Zero-Shot vs Fine-Tuned", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    if n_cls <= 15:
        for xi, (za, fa) in enumerate(zip(zs_accs, ft_accs)):
            ax.text(xi - w / 2, za + 1, f"{za:.0f}", ha="center", fontsize=7)
            ax.text(xi + w / 2, fa + 1, f"{fa:.0f}", ha="center", fontsize=7)

    # ── Panel 2: Delta accuracy (FT − ZS) ────────────────────────────────────
    ax = axes[0, 1]
    deltas     = [ft_by_name[c] - zs_by_name[c] for c in class_names]
    bar_colors = ["#e15759" if d < 0 else "#59a14f" for d in deltas]
    ax.bar(x, deltas, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=tick_fs)
    ax.set_ylabel("Δ Accuracy (Fine-Tuned − Zero-Shot, %)")
    ax.set_title("Accuracy Change After Fine-Tuning", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for xi, d in enumerate(deltas):
        va  = "bottom" if d >= 0 else "top"
        off = 0.5     if d >= 0 else -0.5
        ax.text(xi, d + off, f"{d:+.1f}", ha="center", va=va, fontsize=8)

    # ── Panels 3 & 4: Confusion matrices ─────────────────────────────────────
    for col, (preds, label, acc_overall) in enumerate([
        (preds_zs, "Zero-Shot",  zs_overall),
        (preds_ft, "Fine-Tuned", ft_overall),
    ]):
        ax = axes[1, col]
        conf = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, trues):
            if 0 <= int(t) < n_cls and 0 <= int(p) < n_cls:
                conf[int(t), int(p)] += 1
        row_sums = conf.sum(axis=1, keepdims=True)
        conf_n   = np.where(row_sums > 0, conf / row_sums * 100, 0.0)
        im = ax.imshow(conf_n, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(n_cls))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(class_names, fontsize=tick_fs)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(
            f"Confusion Matrix — {label} (row-normalised, %)\n"
            f"Overall: {acc_overall['accuracy']:.1f}%",
            fontweight="bold",
        )
        if n_cls <= 15:
            for r in range(n_cls):
                for c in range(n_cls):
                    v = conf_n[r, c]
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="white" if v > 55 else "black")
        plt.colorbar(im, ax=ax, label="% of true-class samples",
                     fraction=0.046, pad=0.04)

    # ── Panels 5 & 6: Per-class × concept accuracy heatmaps ──────────────────
    if has_concept_panel:
        with open(concept_file_path) as f:
            concepts_map = json.load(f)
        all_concepts = list(concepts_map.keys())
        n_concepts   = len(all_concepts)

        # Map each eval image index → set of concepts it belongs to
        basename_to_concepts = defaultdict(set)
        for concept, rel_paths in concepts_map.items():
            for rel_path in rel_paths:
                basename_to_concepts[os.path.basename(rel_path)].add(concept)

        sample_concepts = [
            basename_to_concepts.get(os.path.basename(eval_dataset.samples[i][0]), set())
            for i in range(len(eval_dataset.samples))
        ]

        for col, (preds, label) in enumerate([
            (preds_zs, "Zero-Shot"),
            (preds_ft, "Fine-Tuned"),
        ]):
            ax = axes[2, col]
            correct = np.zeros((n_cls, n_concepts), dtype=int)
            total   = np.zeros((n_cls, n_concepts), dtype=int)
            for i, (pred, true) in enumerate(zip(preds, trues)):
                cls_idx = int(true)
                for ci, concept in enumerate(all_concepts):
                    if concept in sample_concepts[i]:
                        total[cls_idx, ci] += 1
                        if pred == true:
                            correct[cls_idx, ci] += 1

            with np.errstate(invalid="ignore"):
                grid = np.where(total > 0, correct / total * 100, np.nan)

            # Use a masked array so NaN cells show as grey
            masked = np.ma.masked_invalid(grid)
            cmap = plt.cm.RdYlGn.copy()
            cmap.set_bad(color="#dddddd")
            im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(n_concepts))
            ax.set_xticklabels(all_concepts, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(n_cls))
            ax.set_yticklabels(class_names, fontsize=tick_fs)
            ax.set_xlabel("Concept")
            ax.set_ylabel("True class")
            ax.set_title(
                f"Per-Class × Concept Accuracy — {label} (%)\n"
                f"(grey = no eval images with that concept in that class)",
                fontweight="bold",
            )
            for r in range(n_cls):
                for c in range(n_concepts):
                    if total[r, c] > 0:
                        v = grid[r, c]
                        ax.text(c, r, f"{v:.0f}\n(n={total[r,c]})",
                                ha="center", va="center", fontsize=7,
                                color="white" if v > 65 else "black")
            plt.colorbar(im, ax=ax, label="Accuracy (%)", fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, f"results_{args.group}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Split evaluation (seen vs novel) ───────────────────────────────────────────

def _set_prompts(class_names):
    """Register PROMPT_SETS['miniimagenet'] for the given ordered class list."""
    from clip_zero_shot import PROMPT_SETS
    prompts = {i: f"a photo of a {name.replace('_', ' ')}" for i, name in enumerate(class_names)}
    PROMPT_SETS["miniimagenet"] = {"shape": prompts, "color": prompts}


def build_eval_split_datasets(full_ds, ft_indices, ft_class_names, eval_class_names, synset_to_name):
    """Return (ft_eval_ds, novel_ds).

    ft_eval_ds  — the images used during fine-tuning (ft_indices, ft_class_names).
    novel_ds    — all eval_class images *not* in ft_indices.
    """
    ft_eval_ds = _SubsetView(full_ds, ft_class_names, synset_to_name, ft_indices)
    novel_indices = get_class_indices_excluding(
        full_ds, synset_to_name, eval_class_names, exclude_indices=ft_indices,
    )
    novel_ds = _SubsetView(full_ds, eval_class_names, synset_to_name, novel_indices)
    return ft_eval_ds, novel_ds


def run_on_split(model, ft_eval_ds, novel_ds, ft_class_names, eval_class_names,
                 dataset_name, batch_size, label):
    """Run model separately on seen (ft) images and novel images.

    Returns (stats_seen, stats_novel) — each has the same shape as evaluate() output
    plus 'predictions_shape' and 'true_labels'.
    """
    print(f"\n  [{label}] — Seen images (used for fine-tuning, n={len(ft_eval_ds)})")
    _set_prompts(ft_class_names)
    res_seen = model.run(dataset=ft_eval_ds, prompt_mode="shape",
                         dataset_name=dataset_name, batch_size=batch_size)
    stats_seen = evaluate(res_seen, ft_class_names, label=f"{label}/Seen")
    stats_seen["predictions_shape"] = res_seen["predictions_shape"]
    stats_seen["true_labels"]       = res_seen["true_labels"]

    print(f"\n  [{label}] — Novel images (not seen during fine-tuning, n={len(novel_ds)})")
    _set_prompts(eval_class_names)
    res_novel = model.run(dataset=novel_ds, prompt_mode="shape",
                          dataset_name=dataset_name, batch_size=batch_size)
    stats_novel = evaluate(res_novel, eval_class_names, label=f"{label}/Novel")
    stats_novel["predictions_shape"] = res_novel["predictions_shape"]
    stats_novel["true_labels"]       = res_novel["true_labels"]

    return stats_seen, stats_novel


def visualize_split_results(zs_seen, zs_novel, ft_seen, ft_novel,
                            ft_class_names, eval_class_names, ft_concepts ,
                            output_dir, args):
    """3 × 2 figure comparing ZS vs FT across seen and novel image groups.

    Rows: accuracy bars | delta bars | FT confusion matrices
    Cols: seen images   | novel images
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    seen_cls  = ft_class_names
    novel_cls = eval_class_names
    n_seen    = len(seen_cls)
    n_novel   = len(novel_cls)
    tick_fs   = max(5, 9 - max(n_seen, n_novel) // 3)
    fig_w     = max(16, max(n_seen, n_novel) * 0.8)

    fig, axes = plt.subplots(4, 2, figsize=(fig_w, 21))
    fig.suptitle(
        f"CLIP — {args.group.capitalize()}  |  "
        f"Seen (FT) images vs Novel images\n"
        f"Seen: {len(zs_seen['true_labels'])} imgs, ft_classes={ft_class_names}  |  "
        f"Novel: {len(zs_novel['true_labels'])} imgs, eval_classes={eval_class_names}\n"
        f"Fine-tuned over concepts: {ft_concepts} \n ",
        fontsize=10, fontweight="bold",
        
    )

    def _accs(stats):
        by = {r["class_name"]: r["accuracy"] or 0 for r in stats["per_class"]}
        return by

    def _bar_panel(ax, class_names, zs_stats, ft_stats, title):
        x      = np.arange(len(class_names))
        w      = 0.35
        zs_by  = _accs(zs_stats)
        ft_by  = _accs(ft_stats)
        zs_acc = [zs_by[c] for c in class_names]
        ft_acc = [ft_by[c] for c in class_names]
        ax.bar(x - w / 2, zs_acc, w, label="Zero-Shot",  color="#4e79a7", alpha=0.85, edgecolor="white")
        ax.bar(x + w / 2, ft_acc, w, label="Fine-Tuned", color="#f28e2b", alpha=0.85, edgecolor="white")
        ax.axhline(zs_stats["overall"]["accuracy"], color="#4e79a7", linestyle="--", lw=1,
                   alpha=0.5, label=f"ZS mean {zs_stats['overall']['accuracy']:.1f}%")
        ax.axhline(ft_stats["overall"]["accuracy"], color="#f28e2b", linestyle="--", lw=1,
                   alpha=0.5, label=f"FT mean {ft_stats['overall']['accuracy']:.1f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=tick_fs)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 120)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        for xi, (za, fa) in enumerate(zip(zs_acc, ft_acc)):
            ax.text(xi - w / 2, za + 1, f"{za:.0f}", ha="center", fontsize=6)
            ax.text(xi + w / 2, fa + 1, f"{fa:.0f}", ha="center", fontsize=6)

    def _delta_panel(ax, class_names, zs_stats, ft_stats, title):
        x      = np.arange(len(class_names))
        zs_by  = _accs(zs_stats)
        ft_by  = _accs(ft_stats)
        deltas = [ft_by[c] - zs_by[c] for c in class_names]
        colors = ["#59a14f" if d >= 0 else "#e15759" for d in deltas]
        ax.bar(x, deltas, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=tick_fs)
        ax.set_ylabel("Δ Accuracy (FT − ZS, %)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for xi, d in enumerate(deltas):
            va, off = ("bottom", 0.5) if d >= 0 else ("top", -0.5)
            ax.text(xi, d + off, f"{d:+.1f}", ha="center", va=va, fontsize=7)

    def _conf_panel(ax, stats, class_names, title):
        n   = len(class_names)
        preds = stats["predictions_shape"]
        trues = stats["true_labels"]
        conf  = np.zeros((n, n), dtype=int)
        for p, t in zip(preds, trues):
            if 0 <= int(t) < n and 0 <= int(p) < n:
                conf[int(t), int(p)] += 1
        row_s  = conf.sum(axis=1, keepdims=True)
        conf_n = np.where(row_s > 0, conf / row_s * 100, 0.0)
        im = ax.imshow(conf_n, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_names, fontsize=tick_fs)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontweight="bold")
        if n <= 12:
            for r in range(n):
                for c in range(n):
                    v = conf_n[r, c]
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 55 else "black")
        plt.colorbar(im, ax=ax, label="% of true class", fraction=0.046, pad=0.04)

    # Row 0: accuracy bars
    _bar_panel(axes[0, 0], seen_cls,  zs_seen,  ft_seen,
               f"ZS vs FT — Seen images (n={len(zs_seen['true_labels'])})")
    _bar_panel(axes[0, 1], novel_cls, zs_novel, ft_novel,
               f"ZS vs FT — Novel images (n={len(zs_novel['true_labels'])})")

    # Row 1: delta accuracy
    _delta_panel(axes[1, 0], seen_cls,  zs_seen,  ft_seen,  "Δ Accuracy — Seen images")
    _delta_panel(axes[1, 1], novel_cls, zs_novel, ft_novel, "Δ Accuracy — Novel images")

    # Row 2: ZS confusion matrices
    _conf_panel(axes[2, 0], zs_seen,  seen_cls,
                f"ZS Confusion Matrix — Seen  ({zs_seen['overall']['accuracy']:.1f}%)")
    _conf_panel(axes[2, 1], zs_novel, novel_cls,
                f"ZS Confusion Matrix — Novel ({zs_novel['overall']['accuracy']:.1f}%)")

    # Row 3: FT confusion matrices
    _conf_panel(axes[3, 0], ft_seen,  seen_cls,
                f"FT Confusion Matrix — Seen  ({ft_seen['overall']['accuracy']:.1f}%)")
    _conf_panel(axes[3, 1], ft_novel, novel_cls,
                f"FT Confusion Matrix — Novel ({ft_novel['overall']['accuracy']:.1f}%)")

    plt.tight_layout()
    path = os.path.join(output_dir, f"split_results_{args.group}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = f"./results/miniimagenet_{args.group}"

    print(f"\n╔══════════════════════════════════════════════════════╗")
    print(f"║  CLIP Conceptual FT — group: {args.group:<24}║")
    print(f"╚══════════════════════════════════════════════════════╝\n")

    # ── Load group config ─────────────────────────────────────────────────────
    group_cfg     = load_group_config(args.group)
    class_names   = group_cfg["classes"]
    ft_class_names   = group_cfg["ft_classes"]
    ft_concepts   = args.ft_concepts   or group_cfg["ft_concepts"]
    eval_concepts = args.eval_concepts or group_cfg["eval_concepts"]
    eval_class_names = args.eval_classes or group_cfg["eval_classes"]
    concept_path  = group_cfg["concept_file_path"]

    print(f"Group            : {args.group}")
    print(f"Classes          : {class_names}")
    print(f"FT concepts      : {ft_concepts}")
    print(f"Eval concepts    : {eval_concepts}")
    print(f"FT classes       : {ft_class_names}")
    print(f"Eval classes     : {eval_class_names}")
    print(f"All concepts     : {group_cfg['concepts']}")
    print(f"Concept file     : {os.path.basename(concept_path)}")

    # validate concepts
    for c in ft_concepts + eval_concepts:
        if c not in group_cfg["concepts"]:
            print(f"\n  ✗ '{c}' is not a valid concept for group '{args.group}'.")
            print(f"     Available: {group_cfg['concepts']}")
            sys.exit(1)

    device = check_dependencies()
    print(f"\nDevice: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\nLoading Mini-ImageNet...")
    from datasets.mini_imagenet import get_miniimagenet_dataset
    full_ds = get_miniimagenet_dataset()

    _labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "datasets", "mini_imagenet_labels.json")
    with open(_labels_path) as f:
        synset_to_name = json.load(f)

    # ── Build fine-tune and eval subsets ──────────────────────────────────────
    ft_indices = get_indices_from_concepts(
        full_ds, synset_to_name, ft_class_names,
        include_concepts=ft_concepts,
        holdout_concepts=eval_concepts,
        concept_file_path=concept_path,
    )
    ## split to train and val (80/20) for fine-tuning
    # ft_indices = split_train_val(classconcept_indices, val_ratio=0.2, seed=args.seed)
    
    ft_dataset = _SubsetView(
        full_ds, ft_class_names, synset_to_name, ft_indices,
        max_samples=args.max_samples, seed=args.seed,
    )


    # eval_indices = get_indices_from_concepts(
    #     full_ds, synset_to_name, eval_class_names,
    #     include_concepts=eval_concepts,
    #     concept_file_path=concept_path,
    # )
    eval_indices = get_class_indices(
        full_ds, synset_to_name, eval_class_names,
    )
    # eval_indices = get_class_indices_excluding(
    #     full_ds, synset_to_name, eval_class_names, exclude_indices=ft_indices,
    # )
    eval_dataset = _SubsetView(
        full_ds, eval_class_names, synset_to_name, eval_indices,
        max_samples=args.max_samples, seed=args.seed,
    )

    print(f"\n  Fine-tune images : {len(ft_dataset)}  (concepts: {ft_concepts})")
    print(f"  Eval images      : {len(eval_dataset)}  (concepts: {eval_concepts})")

    if len(ft_dataset) == 0:
        print("\n  ✗ Fine-tune dataset is empty — check concept names.")
        sys.exit(1)
    if len(eval_dataset) == 0:
        print("\n  ✗ Eval dataset is empty — check concept names.")
        sys.exit(1)

    # ── Build split datasets (seen = ft images, novel = rest of eval classes) ──
    ft_eval_ds, novel_ds = build_eval_split_datasets(
        full_ds, ft_indices, ft_class_names, eval_class_names, synset_to_name,
    )
    print(f"  Seen (FT) images : {len(ft_eval_ds)}  (ft_class_names: {ft_class_names})")
    print(f"  Novel images     : {len(novel_ds)}  (eval_class_names: {eval_class_names})")

    # ── Import model class ────────────────────────────────────────────────────
    from clip_zero_shot import CLIPZeroShot

    # ── Condition 1: Zero-Shot ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  CONDITION 1 — Zero-Shot (no fine-tuning)")
    print("─" * 60)
    clip_zs = CLIPZeroShot(model_name=args.clip_model, device=device)
    _set_prompts(eval_class_names)
    t0 = time.time()
    zs_results = clip_zs.run(
        dataset=eval_dataset,
        prompt_mode="shape",
        dataset_name="miniimagenet",
        batch_size=args.batch_size,
    )
    zs_results["class_names"] = eval_class_names
    print(f"  Done in {time.time() - t0:.1f}s")
    zs_stats = evaluate(zs_results, eval_class_names, label="Zero-Shot")
    zs_stats["predictions_shape"] = zs_results["predictions_shape"]
    zs_stats["true_labels"]       = zs_results["true_labels"]

    print("\n  --- Zero-Shot split evaluation ---")
    zs_seen, zs_novel = run_on_split(
        clip_zs, ft_eval_ds, novel_ds, ft_class_names, eval_class_names,
        dataset_name="miniimagenet", batch_size=args.batch_size, label="Zero-Shot",
    )

    # ── Condition 2: Fine-Tuned ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  CONDITION 2 — Fine-Tuned ({args.ft_epochs} epoch(s), lr={args.ft_lr})")
    print("─" * 60)
    clip_ft = CLIPZeroShot(model_name=args.clip_model, device=device)
    print(f"  Fine-tuning on {len(ft_dataset)} images (concepts: {ft_concepts})...")
    clip_ft.fine_tune(
        dataset=ft_dataset,
        dataset_name="miniimagenet",
        epochs=args.ft_epochs,
        lr=args.ft_lr,
        batch_size=args.batch_size,
    )
    _set_prompts(eval_class_names)
    t0 = time.time()
    ft_results = clip_ft.run(
        dataset=eval_dataset,
        prompt_mode="shape",
        dataset_name="miniimagenet",
        batch_size=args.batch_size,
    )
    ft_results["class_names"] = eval_class_names
    print(f"  Done in {time.time() - t0:.1f}s")
    ft_stats = evaluate(ft_results, eval_class_names, label="Fine-Tuned")
    ft_stats["predictions_shape"] = ft_results["predictions_shape"]
    ft_stats["true_labels"]       = ft_results["true_labels"]

    print("\n  --- Fine-Tuned split evaluation ---")
    ft_seen, ft_novel = run_on_split(
        clip_ft, ft_eval_ds, novel_ds, ft_class_names, eval_class_names,
        dataset_name="miniimagenet", batch_size=args.batch_size, label="Fine-Tuned",
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)

    save_confusion_matrix(zs_stats, args.output_dir, tag="zeroshot")
    save_confusion_matrix(ft_stats, args.output_dir, tag="finetuned")
    save_misclassified(eval_dataset, zs_results, args.output_dir, tag="zeroshot")
    save_misclassified(eval_dataset, ft_results, args.output_dir, tag="finetuned")
    save_comparison_csv(zs_stats, ft_stats, eval_class_names, args.output_dir)

    print("Generating visualizations...")
    _set_prompts(eval_class_names)
    visualize_results(zs_stats, ft_stats, args.output_dir, args,
                      ft_concepts, eval_concepts,
                      eval_dataset=eval_dataset,
                      concept_file_path=concept_path)

    visualize_split_results(
        zs_seen, zs_novel, ft_seen, ft_novel,
        ft_class_names, eval_class_names, ft_concepts,
        args.output_dir, args,
    )

    print(f"\nAll results in: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
