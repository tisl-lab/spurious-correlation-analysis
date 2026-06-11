"""
run_miniimagenet_perclass_concpt.py
=====================================

Per-class concept fine-tuning experiment on Mini-ImageNet.

Each class in a semantic group is assigned its dominant visual concept.
Fine-tuning uses images from each class that contain that class's assigned concept.
Evaluation compares Zero-Shot vs Fine-Tuned on:
  (a) the seen images used for fine-tuning
  (b) novel images not shown during fine-tuning

USAGE:
    python run_miniimagenet_perclass_concpt.py --group insects
    python run_miniimagenet_perclass_concpt.py --group birds
    python run_miniimagenet_perclass_concpt.py --group canidea
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
        "class_concepts": {
            "house_finch": "high_freq_sky_patterns",
            # "robin":       "image_edge",
            "toucan":      "fence",
            # "goose":       "high_freq_sky_patterns",
        },
        "concept_file": "MiniImageNet_Bird_Concepts.json",
    },
    "insects": {
        "class_concepts": {
            "ant":               "leaves",
            # "harvestman":        "leaves",
            # "ladybug":           "leaves",
            "rhinoceros_beetle": "hands",
        },
        "concept_file": "MiniImageNet_Insect_Concepts.json",
    },
    "canidea": {
        "class_concepts": {
            # "golden_retriever": "human",
            # "malamute":         "human",
            # "Tibetan_mastiff":  "human",
            "dalmatian":        "human",
            # "boxer":            "human",
            # "Newfoundland":     "human",
            "white_wolf":       "snow",
            # "Arctic_fox":       "snow",
        },
        "concept_file": "MiniImageNet_Canidea_Concepts.json",
    },
}



def load_group_config(group_name):
    """Return the full config for a semantic group.

    Returns a dict with:
        classes           — ordered list of class names in this group
        class_concepts    — {class_name: assigned_concept} per-class mapping
        all_concepts      — sorted unique set of concepts used in this group
        concept_file_path — absolute path to the concept JSON file
    """
    if group_name not in GROUP_CONFIGS:
        raise ValueError(
            f"Unknown group '{group_name}'. "
            f"Choose from: {sorted(GROUP_CONFIGS)}"
        )
    cfg = GROUP_CONFIGS[group_name]
    return {
        "classes":            list(cfg["class_concepts"].keys()),
        "class_concepts":     cfg["class_concepts"],
        "all_concepts":       sorted(set(cfg["class_concepts"].values())),
        "concept_file_path":  os.path.join(_DATASETS_DIR, cfg["concept_file"]),
    }


_N_TRAIN_PER_CLASS = 500


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP Zero-Shot vs Per-Class Concept-FT — Mini-ImageNet semantic groups"
    )
    parser.add_argument("--group",       type=str, default="canidea",
                        choices=sorted(GROUP_CONFIGS),
                        help="Semantic group to experiment on")
    parser.add_argument("--clip_model",  type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--output_dir",  type=str,   default=None,
                        help="Results directory (default: ./results/miniimagenet_perclass_<group>)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--ft_epochs",   type=int,   default=3)
    parser.add_argument("--ft_lr",       type=float, default=1e-5)
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Cap images per class for quick testing")
    return parser.parse_args()

# def split_train_val(indices, val_ratio=0.2, seed=42):
#     """Split a list of indices into train and val subsets."""
#     rng = np.random.default_rng(seed)
#     indices = np.array(indices)
#     rng.shuffle(indices)
#     n_val = int(len(indices) * val_ratio)
#     return indices[:-n_val].tolist(), indices[-n_val:].tolist()
    
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
                              include_concepts = "any",
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

    if include_concepts == "any":
        include_basenames = set()

        class_set = set(include_classes)
    # exclude_set = set(exclude_indices)

        def _readable(i):
            synset = dataset.classes[dataset.samples[i][1]]
            return synset_to_name.get(synset)

        include_indices = [
            i for i in range(len(dataset.samples))
            if _readable(i) in class_set 
        ]



    else:     
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


def get_per_class_concept_indices(dataset, synset_to_name, class_concepts, concept_file_path):
    """Return indices for images matching each class's individually assigned concept.

    class_concepts: {class_name: concept_name}
    An image is included when its class matches AND its filename appears under that
    class's assigned concept in the concept file.
    """
    with open(concept_file_path) as f:
        concepts_map = json.load(f)

    concept_basenames = {
        concept: {os.path.basename(p) for p in rel_paths}
        for concept, rel_paths in concepts_map.items()
    }

    result = {cls: [] for cls in class_concepts}
    for i in range(len(dataset.samples)):
        synset     = dataset.classes[dataset.samples[i][1]]
        class_name = synset_to_name.get(synset)
        if class_name not in class_concepts:
            continue
        assigned = class_concepts[class_name]
        if os.path.basename(dataset.samples[i][0]) in concept_basenames.get(assigned, set()):
            result[class_name].append(i)
    return result


def split_train_val(per_class_indices, val_ratio=0.2, seed=42):
    """Split {class_name: [indices]} into (train_dict, val_dict) per class.

    Each class's indices are shuffled independently, then split at val_ratio.
    Returns two dicts with the same keys as per_class_indices.
    """
    rng = np.random.default_rng(seed)
    train, val = {}, {}
    for class_name, indices in per_class_indices.items():
        arr = np.array(indices)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * val_ratio))
        val[class_name]   = arr[:n_val].tolist()
        train[class_name] = arr[n_val:].tolist()
    return train, val


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


def save_comparison_csv(zs_stats, ft_stats, class_names, output_dir, tag=""):
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
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(output_dir, f"comparison{suffix}_{ts}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_results(zs_stats, ft_stats, output_dir, args,
                      class_concepts,
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
    concept_summary = ", ".join(f"{cls}→{c}" for cls, c in class_concepts.items())
    fig.suptitle(
        f"CLIP MiniImageNet — Group: {args.group.capitalize()}  |  "
        f"Model: {args.clip_model}\n"
        f"Per-class concepts: {concept_summary}\n"
        f"ZS accuracy: {zs_overall['accuracy']:.2f}%  |  "
        f"FT accuracy: {ft_overall['accuracy']:.2f}%",
        fontsize=10, fontweight="bold",
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
                            ft_class_names, eval_class_names, class_concepts,
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
    concept_summary = ", ".join(f"{cls}→{c}" for cls, c in class_concepts.items())
    fig.suptitle(
        f"CLIP — {args.group.capitalize()}  |  "
        f"Seen (FT) images vs Novel images\n"
        f"Seen: {len(zs_seen['true_labels'])} imgs  |  "
        f"Novel: {len(zs_novel['true_labels'])} imgs\n"
        f"Per-class concepts: {concept_summary}",
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


# ── Per-class concept split helpers ───────────────────────────────────────────

def visualize_concept_split_results(zs_per_class, ft_per_class,
                                    class_names, class_concepts,
                                    output_dir, args):
    """6-panel figure visualising per-class ZS vs FT accuracy split by concept presence.

    Panel layout (3 rows × 2 cols):
      [0,0] 4-bar accuracy overview  (ZS+, FT+, ZS−, FT−) per class
      [0,1] FT−ZS delta for both conditions per class
      [1,0] Concept reliance gap  (acc_with − acc_without) for ZS and FT
      [1,1] Evaluation set sizes (n_with, n_without) per class
      [2,0] Accuracy heatmap  (condition × class)
      [2,1] Textual summary by concept with ZS/FT accuracy pairs

    Classes fine-tuned on the same concept share a background shade.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_cls   = len(class_names)
    tick_fs = max(6, 10 - n_cls // 3)
    fig_w   = max(18, n_cls * 1.8)

    C_ZS_WITH    = "#5291c8"   # blue  (solid)
    C_FT_WITH    = "#5291c8"   # green  (hatched — same hue as ZS, distinguished by pattern)
    C_ZS_WITHOUT = "#cb619f"   # red    (solid)
    C_FT_WITHOUT = "#cb619f"   # red    (hatched — same hue as ZS, distinguished by pattern)
    H_FT         = "//"        # hatch applied to all FT bars

    unique_concepts = sorted(set(class_concepts.values()))
    _bg_palette     = ["#fff2cc", "#dae8fc", "#d5e8d4", "#f8cecc",
                       "#e1d5e7", "#ffe6cc", "#d0e0e3", "#fff2ae"]
    concept_bg      = {c: _bg_palette[i % len(_bg_palette)]
                       for i, c in enumerate(unique_concepts)}

    x = np.arange(n_cls)
    w = 0.18

    zs_with    = [zs_per_class[cls]["with_acc"]    for cls in class_names]
    zs_without = [zs_per_class[cls]["without_acc"] for cls in class_names]
    ft_with    = [ft_per_class[cls]["with_acc"]    for cls in class_names]
    ft_without = [ft_per_class[cls]["without_acc"] for cls in class_names]

    fig, axes = plt.subplots(4, 2, figsize=(fig_w, 26))

    def _shade_concepts(ax, ylim_bottom=-8):
        for i, cls in enumerate(class_names):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.15,
                       color=concept_bg[class_concepts[cls]], zorder=0)
            ax.text(i, ylim_bottom,
                    class_concepts[cls].replace("_", "\n"),
                    ha="center", va="top",
                    fontsize=max(4, tick_fs - 3), color="#555555", style="italic")

    # ── [0,0] Accuracy overview: 4 bars per class ─────────────────────────────
    ax = axes[0, 0]
    ax.bar(x - 1.5*w, zs_with,    w, label="ZS + concept",  color=C_ZS_WITH,    alpha=0.9, edgecolor="white")
    ax.bar(x - 0.5*w, ft_with,    w, label="FT + concept",  color=C_FT_WITH,    alpha=0.9, edgecolor="white", hatch=H_FT)
    ax.bar(x + 0.5*w, zs_without, w, label="ZS − concept",  color=C_ZS_WITHOUT, alpha=0.9, edgecolor="white")
    ax.bar(x + 1.5*w, ft_without, w, label="FT − concept",  color=C_FT_WITHOUT, alpha=0.9, edgecolor="white", hatch=H_FT)
    _shade_concepts(ax, ylim_bottom=-18)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=tick_fs)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-22, 110)
    ax.set_title("Accuracy Overview: ZS vs FT × With / Without Concept", fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    if n_cls <= 12:
        for xi, vals in enumerate(zip(zs_with, ft_with, zs_without, ft_without)):
            for bi, (offset, v) in enumerate(zip([-1.5, -0.5, 0.5, 1.5], vals)):
                ax.text(xi + offset*w, v + 1, f"{v:.0f}",
                        ha="center", fontsize=6, color="black")

    # ── [0,1] FT−ZS delta per class, both conditions ──────────────────────────
    ax = axes[0, 1]
    delta_with    = [f - z for f, z in zip(ft_with,    zs_with)]
    delta_without = [f - z for f, z in zip(ft_without, zs_without)]
    bw = w * 1.6
    ax.bar(x - bw/2, delta_with,
           bw, label="Δ (+ concept)",
           color=[("#59a14f" if d >= 0 else "#e15759") for d in delta_with],
           alpha=0.9, edgecolor="white")
    ax.bar(x + bw/2, delta_without,
           bw, label="Δ (− concept)",
           color=[("#59a14f" if d >= 0 else "#e15759") for d in delta_without],
           alpha=0.55, edgecolor="white", hatch="//")
    _shade_concepts(ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=tick_fs)
    ax.set_ylabel("Δ Accuracy (FT − ZS, %)")
    ax.set_title("Fine-Tuning Effect: With vs Without Concept", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for xi, (dw, dwo) in enumerate(zip(delta_with, delta_without)):
        for off, d in [(-bw/2, dw), (bw/2, dwo)]:
            va, pad = ("bottom", 0.4) if d >= 0 else ("top", -0.4)
            ax.text(xi + off, d + pad, f"{d:+.1f}", ha="center", va=va, fontsize=7)

    # ── [1,0] Concept reliance gap: acc_with − acc_without ────────────────────
    ax = axes[1, 0]
    reliance_zs = [zs_per_class[cls]["with_acc"] - zs_per_class[cls]["without_acc"]
                   for cls in class_names]
    reliance_ft = [ft_per_class[cls]["with_acc"] - ft_per_class[cls]["without_acc"]
                   for cls in class_names]
    ax.bar(x - bw/2, reliance_zs, bw, label="ZS reliance",
           color=["#59a14f" if v >= 0 else "#e15759" for v in reliance_zs],
           alpha=0.9, edgecolor="white")
    ax.bar(x + bw/2, reliance_ft, bw, label="FT reliance",
           color=["#59a14f" if v >= 0 else "#e15759" for v in reliance_ft],
           alpha=0.9, edgecolor="white", hatch=H_FT)
    _shade_concepts(ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=tick_fs)
    ax.set_ylabel("Acc(with concept) − Acc(without concept)  %")
    ax.set_title("Concept Reliance Gap (positive = relies on concept)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for xi, (rz, rf) in enumerate(zip(reliance_zs, reliance_ft)):
        for off, r in [(-bw/2, rz), (bw/2, rf)]:
            va, pad = ("bottom", 0.4) if r >= 0 else ("top", -0.4)
            ax.text(xi + off, r + pad, f"{r:+.1f}", ha="center", va=va, fontsize=7)

    # ── [1,1] Evaluation set sizes ────────────────────────────────────────────
    ax = axes[1, 1]
    n_with    = [zs_per_class[cls]["with_n"]    for cls in class_names]
    n_without = [zs_per_class[cls]["without_n"] for cls in class_names]
    ax.bar(x - bw/2, n_with,    bw, label="With concept (val)", color=C_ZS_WITH,    alpha=0.85, edgecolor="white")
    ax.bar(x + bw/2, n_without, bw, label="Without concept",    color=C_ZS_WITHOUT, alpha=0.85, edgecolor="white")
    _shade_concepts(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=tick_fs)
    ax.set_ylabel("Number of images")
    ax.set_title("Evaluation Set Sizes per Class", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for xi, (nw, nwo) in enumerate(zip(n_with, n_without)):
        ax.text(xi - bw/2, nw  + max(n_with)  * 0.01, str(nw),  ha="center", fontsize=7)
        ax.text(xi + bw/2, nwo + max(n_without)* 0.01, str(nwo), ha="center", fontsize=7)

    # ── [2,0] Accuracy heatmap: condition × class ─────────────────────────────
    ax = axes[2, 0]
    cond_labels = ["ZS + concept", "FT + concept", "ZS − concept", "FT − concept"]
    matrix = np.array([zs_with, ft_with, zs_without, ft_without])  # (4, n_cls)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticks(range(4))
    ax.set_yticklabels(cond_labels, fontsize=9)
    ax.set_title("Accuracy Heatmap — Condition × Class (%)", fontweight="bold")
    for r in range(4):
        for c in range(n_cls):
            v = matrix[r, c]
            ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if (v > 65 or v < 20) else "black")
    # Highlight FT concept assignment (rows 0-1) with a border per class
    for ci, cls in enumerate(class_names):
        bg = concept_bg[class_concepts[cls]]
        ax.add_patch(plt.Rectangle(
            (ci - 0.5, -0.5), 1, 4, fill=False,
            edgecolor=bg, lw=4, zorder=5,
        ))
    plt.colorbar(im, ax=ax, label="Accuracy (%)", fraction=0.046, pad=0.04)

    # ── [2,1] Textual summary by concept ─────────────────────────────────────
    ax = axes[2, 1]
    ax.axis("off")
    summary_lines = ["Concept assignments & per-class results\n"
                     "Format: ZS acc_with / acc_without   FT acc_with / acc_without\n"]
    for concept in unique_concepts:
        summary_lines.append(f"[{concept}]")
        for cls in class_names:
            if class_concepts[cls] != concept:
                continue
            zr = zs_per_class[cls]
            fr = ft_per_class[cls]
            summary_lines.append(
                f"  {cls:<26}"
                f"  ZS: {zr['with_acc']:5.1f}% / {zr['without_acc']:5.1f}%"
                f"   FT: {fr['with_acc']:5.1f}% / {fr['without_acc']:5.1f}%"
                f"   Reliance Δ: {(fr['with_acc']-fr['without_acc'])-(zr['with_acc']-zr['without_acc']):+.1f}%"
            )
        summary_lines.append("")
    ax.text(0.02, 0.98, "\n".join(summary_lines),
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    # ── [3,0] Spurious correlation strength — horizontal bar, sorted ──────────
    ax = axes[3, 0]
    reliance_zs_all = [zs_per_class[cls]["with_acc"] - zs_per_class[cls]["without_acc"]
                       for cls in class_names]
    reliance_ft_all = [ft_per_class[cls]["with_acc"] - ft_per_class[cls]["without_acc"]
                       for cls in class_names]
    order    = sorted(range(n_cls), key=lambda i: abs(reliance_ft_all[i]), reverse=True)
    s_names  = [class_names[i]       for i in order]
    s_rel_zs = [reliance_zs_all[i]   for i in order]
    s_rel_ft = [reliance_ft_all[i]   for i in order]

    y_h = np.arange(n_cls)
    bh  = 0.35
    ax.barh(y_h + bh/2, s_rel_zs, bh, label="ZS",
            color=["#59a14f" if v >= 0 else "#e15759" for v in s_rel_zs],
            alpha=0.88, edgecolor="white")
    ax.barh(y_h - bh/2, s_rel_ft, bh, label="FT",
            color=["#59a14f" if v >= 0 else "#e15759" for v in s_rel_ft],
            alpha=0.88, edgecolor="white", hatch=H_FT)
    for yi, (vz, vf) in enumerate(zip(s_rel_zs, s_rel_ft)):
        for v, off in [(vz, bh/2), (vf, -bh/2)]:
            ax.text(v + (0.5 if v >= 0 else -0.5), yi + off, f"{v:+.1f}",
                    va="center", ha=("left" if v >= 0 else "right"), fontsize=7)
    ax.axvline(x=0, color="black", lw=0.8)
    ax.set_yticks(y_h)
    ax.set_yticklabels(s_names, fontsize=tick_fs)
    ax.set_xlabel("Acc(with concept) − Acc(without concept)  %")
    ax.set_title("Spurious Correlation Strength per Class  (sorted by |FT reliance|)\n"
                 "solid = ZS · hatched = FT · green = positive · red = negative",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # ── [3,1] FT prediction distribution: true class → predicted class ────────
    axes[3, 1].set_title("FT Prediction Distribution  (True Class → Predicted Class  %)",
                          fontweight="bold", pad=14)
    axes[3, 1].axis("off")
    ax_top = axes[3, 1].inset_axes([0.0, 0.52, 1.0, 0.44])
    ax_bot = axes[3, 1].inset_axes([0.0, 0.0,  1.0, 0.44])

    pred_with_mat    = np.zeros((n_cls, n_cls))
    pred_without_mat = np.zeros((n_cls, n_cls))
    for ci, cls in enumerate(class_names):
        for p in ft_per_class[cls].get("with_preds", []):
            if 0 <= int(p) < n_cls:
                pred_with_mat[ci, int(p)] += 1
        for p in ft_per_class[cls].get("without_preds", []):
            if 0 <= int(p) < n_cls:
                pred_without_mat[ci, int(p)] += 1
    for mat in (pred_with_mat, pred_without_mat):
        totals = mat.sum(axis=1, keepdims=True)
        mat[:] = np.where(totals > 0, mat / totals * 100, 0.0)

    short = [c[:11] for c in class_names]
    for ax_sub, mat, subtitle in [(ax_top, pred_with_mat,    "With Concept"),
                                   (ax_bot, pred_without_mat, "Without Concept")]:
        im_sub = ax_sub.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax_sub.set_xticks(range(n_cls))
        ax_sub.set_xticklabels(short, rotation=40, ha="right", fontsize=max(5, tick_fs - 1))
        ax_sub.set_yticks(range(n_cls))
        ax_sub.set_yticklabels(short, fontsize=max(5, tick_fs - 1))
        ax_sub.set_title(subtitle, fontsize=8, fontweight="bold")
        for r in range(n_cls):
            for c in range(n_cls):
                v = mat[r, c]
                if v > 1:
                    ax_sub.text(c, r, f"{v:.0f}", ha="center", va="center",
                                fontsize=6, color="white" if v > 60 else "black")
        # Highlight diagonal (correct prediction)
        for k in range(n_cls):
            ax_sub.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                             fill=False, edgecolor="gold", lw=1.5, zorder=5))
        plt.colorbar(im_sub, ax=ax_sub, label="% predicted", fraction=0.046, pad=0.04)

    # ── Figure title & concept legend ─────────────────────────────────────────
    concept_summary = ", ".join(f"{cls}→{c}" for cls, c in class_concepts.items())
    fig.suptitle(
        f"CLIP Mini-ImageNet — Group: {args.group.capitalize()}  |  Model: {args.clip_model}\n"
        f"FT: {args.ft_epochs} epoch(s), lr={args.ft_lr}   |   {concept_summary}",
        fontsize=10, fontweight="bold",
    )
    legend_patches = [
        mpatches.Patch(color=concept_bg[c], label=c, alpha=0.7)
        for c in unique_concepts
    ]
    n_concepts = len(unique_concepts)
    fig.legend(handles=legend_patches, loc="lower center", fontsize=8,
               title="Concept (background shade)", title_fontsize=8, framealpha=0.9,
               ncol=n_concepts, bbox_to_anchor=(0.5, 0.0))

    # Reserve bottom margin so the figure legend doesn't overlap any axes
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(output_dir, f"concept_split_{args.group}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def _run_per_class_concept_eval(model, class_names, class_concepts,
                                val_datasets, without_concept_datasets,
                                dataset_name, batch_size):
    """Run model on val_datasets (with-concept) and without_concept_datasets per class.

    val_datasets and without_concept_datasets use single-class label spaces ([cls]),
    so accuracy is computed by comparing predictions to class_names.index(cls) directly.
    Returns {class_name: {concept, with_n, with_correct, with_incorrect, with_acc,
                          without_n, without_correct, without_incorrect, without_acc}}.
    """
    _set_prompts(class_names)
    results = {}
    for cls in class_names:
        cls_idx = class_names.index(cls)
        concept = class_concepts[cls]

        res_with     = model.run(dataset=val_datasets[cls], prompt_mode="shape",
                                 dataset_name=dataset_name, batch_size=batch_size)
        preds_with   = res_with["predictions_shape"]
        n_with       = len(preds_with)
        correct_with = int((preds_with == cls_idx).sum())

        res_without     = model.run(dataset=without_concept_datasets[cls], prompt_mode="shape",
                                    dataset_name=dataset_name, batch_size=batch_size)
        preds_without   = res_without["predictions_shape"]
        n_without       = len(preds_without)
        correct_without = int((preds_without == cls_idx).sum())

        results[cls] = {
            "concept":           concept,
            "with_n":            n_with,
            "with_correct":      correct_with,
            "with_incorrect":    n_with - correct_with,
            "with_acc":          correct_with / n_with * 100 if n_with else 0.0,
            "without_n":         n_without,
            "without_correct":   correct_without,
            "without_incorrect": n_without - correct_without,
            "without_acc":       correct_without / n_without * 100 if n_without else 0.0,
            "with_preds":        preds_with,
            "without_preds":     preds_without,
        }
    return results


def _print_concept_split_table(label, per_class_data, class_names, class_concepts):
    """Print per-class with-concept vs without-concept accuracy table."""
    w = 116
    print(f"\n{'═' * w}")
    print(f"  {label} — Per-Class: With Concept (val) vs Without Concept")
    print(f"{'═' * w}")
    print(f"  {'Class':<24}  {'Concept':<22}  "
          f"{'── WITH CONCEPT ──':^38}  {'── WITHOUT CONCEPT ──':^40}")
    print(f"  {'':24}  {'':22}  "
          f"{'n':>6}  {'Correct':>8}  {'Wrong':>8}  {'Acc%':>7}    "
          f"{'n':>6}  {'Correct':>8}  {'Wrong':>8}  {'Acc%':>7}")
    print("  " + "─" * 112)
    for cls in class_names:
        r = per_class_data[cls]
        print(f"  {cls:<24}  {r['concept']:<22}  "
              f"{r['with_n']:>6}  {r['with_correct']:>8}  {r['with_incorrect']:>8}  {r['with_acc']:>6.1f}%"
              f"    {r['without_n']:>6}  {r['without_correct']:>8}  {r['without_incorrect']:>8}  {r['without_acc']:>6.1f}%")
    print(f"{'═' * w}")


def save_concept_split_csv(per_class_data, output_dir, tag=""):
    """Save per-class with/without-concept results to CSV."""
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows   = [{"class": cls, **vals} for cls, vals in per_class_data.items()]
    suffix = f"_{tag}" if tag else ""
    path   = os.path.join(output_dir, f"concept_split{suffix}_{ts}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


def save_report(args, device, class_names, class_concepts,
                ft_indices_per_class, val_indices_per_class,
                without_concept_datasets,
                zs_per_class, ft_per_class,
                run_start, output_dir):
    """Write a human-readable run report to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"report_{ts}.txt")

    W = 116

    def _line(char="─"):
        return char * W

    lines = []
    lines.append("=" * W)
    lines.append("  CLIP Mini-ImageNet — Per-Class Concept Fine-Tuning Report")
    lines.append("=" * W)
    lines.append("")

    # ── Run configuration ─────────────────────────────────────────────────────
    lines.append("RUN CONFIGURATION")
    lines.append(_line())
    lines.append(f"  Timestamp        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Group            : {args.group}")
    lines.append(f"  CLIP model       : {args.clip_model}")
    lines.append(f"  Device           : {device}")
    lines.append(f"  FT epochs        : {args.ft_epochs}")
    lines.append(f"  FT learning rate : {args.ft_lr}")
    lines.append(f"  Batch size       : {args.batch_size}")
    lines.append(f"  Seed             : {args.seed}")
    lines.append(f"  Max samples      : {args.max_samples if args.max_samples else 'None (all)'}")
    lines.append(f"  Output dir       : {os.path.abspath(output_dir)}")
    lines.append(f"  Total runtime    : {time.time() - run_start:.1f}s")
    lines.append("")

    # ── Class-concept assignments ─────────────────────────────────────────────
    lines.append("CLASS → CONCEPT ASSIGNMENTS")
    lines.append(_line())
    for cls, concept in class_concepts.items():
        lines.append(f"  {cls:<28} → {concept}")
    lines.append("")

    # ── Dataset summary ───────────────────────────────────────────────────────
    lines.append("DATASET SUMMARY")
    lines.append(_line())
    hdr = (f"  {'Class':<28}  {'Concept':<26}  "
           f"{'FT train':>8}  {'FT val (w/ concept)':>20}  {'W/O concept':>12}")
    lines.append(hdr)
    lines.append("  " + "─" * 100)
    for cls in class_names:
        concept   = class_concepts[cls]
        n_ft      = len(ft_indices_per_class[cls])
        n_val     = len(val_indices_per_class[cls])
        n_without = len(without_concept_datasets[cls])
        lines.append(f"  {cls:<28}  {concept:<26}  "
                     f"{n_ft:>8}  {n_val:>20}  {n_without:>12}")
    total_ft  = sum(len(ft_indices_per_class[c]) for c in class_names)
    total_val = sum(len(val_indices_per_class[c]) for c in class_names)
    total_wc  = sum(len(without_concept_datasets[c]) for c in class_names)
    lines.append("  " + "─" * 100)
    lines.append(f"  {'TOTAL':<56}  {total_ft:>8}  {total_val:>20}  {total_wc:>12}")
    lines.append("")

    # ── Per-condition results ─────────────────────────────────────────────────
    for cond_label, per_class in [("ZERO-SHOT", zs_per_class), ("FINE-TUNED", ft_per_class)]:
        lines.append(f"{cond_label} RESULTS")
        lines.append(_line())
        lines.append(f"  {'Class':<24}  {'Concept':<22}  "
                     f"{'── WITH CONCEPT ──':^38}  {'── WITHOUT CONCEPT ──':^40}")
        lines.append(f"  {'':24}  {'':22}  "
                     f"{'n':>6}  {'Correct':>8}  {'Wrong':>8}  {'Acc%':>7}    "
                     f"{'n':>6}  {'Correct':>8}  {'Wrong':>8}  {'Acc%':>7}")
        lines.append("  " + "─" * 112)
        for cls in class_names:
            r = per_class[cls]
            lines.append(
                f"  {cls:<24}  {r['concept']:<22}  "
                f"{r['with_n']:>6}  {r['with_correct']:>8}  {r['with_incorrect']:>8}  {r['with_acc']:>6.1f}%"
                f"    {r['without_n']:>6}  {r['without_correct']:>8}  {r['without_incorrect']:>8}  {r['without_acc']:>6.1f}%"
            )
        lines.append("")

    # ── ZS vs FT comparison ───────────────────────────────────────────────────
    lines.append("ZS vs FT COMPARISON")
    lines.append(_line())
    lines.append(f"  {'Class':<24}  {'Concept':<22}  "
                 f"{'ZS w/concept':>13}  {'FT w/concept':>13}  {'Δ w/concept':>12}  "
                 f"{'ZS w/o concept':>15}  {'FT w/o concept':>15}  {'Δ w/o concept':>14}")
    lines.append("  " + "─" * 112)
    for cls in class_names:
        zs = zs_per_class[cls]
        ft = ft_per_class[cls]
        d_with    = ft["with_acc"]    - zs["with_acc"]
        d_without = ft["without_acc"] - zs["without_acc"]
        lines.append(
            f"  {cls:<24}  {zs['concept']:<22}  "
            f"{zs['with_acc']:>12.1f}%  {ft['with_acc']:>12.1f}%  {d_with:>+11.1f}%  "
            f"{zs['without_acc']:>14.1f}%  {ft['without_acc']:>14.1f}%  {d_without:>+13.1f}%"
        )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    run_start = time.time()

    if args.output_dir is None:
        args.output_dir = f"./results/miniimagenet_perclass_{args.group}"

    print(f"\n╔══════════════════════════════════════════════════════╗")
    print(f"║  CLIP Conceptual FT — group: {args.group:<24}║")
    print(f"╚══════════════════════════════════════════════════════╝\n")

    # ── Load group config ─────────────────────────────────────────────────────
    group_cfg     = load_group_config(args.group)
    class_names   = group_cfg["classes"]
    class_concepts = group_cfg["class_concepts"]
    concept_path  = group_cfg["concept_file_path"]

    print(f"Group            : {args.group}")
    print(f"Classes          : {class_names}")
    print(f"Per-class concepts:")
    for cls, concept in class_concepts.items():
        print(f"  {cls:<28} → {concept}")
    print(f"Concept file     : {os.path.basename(concept_path)}")

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
    # Per-class concept-filtered indices: {class_name: [indices]}
    classconcept_indices = get_per_class_concept_indices(
        full_ds, synset_to_name, class_concepts, concept_path,
    )

    # Split each class 80/20 into ft and val
    ft_indices_per_class, val_indices_per_class = split_train_val(
        classconcept_indices, val_ratio=0.2, seed=args.seed,
    )

    # Per-class _SubsetView datasets
    ft_datasets = {
        cls: _SubsetView(full_ds, [cls], synset_to_name, indices,
                         max_samples=args.max_samples, seed=args.seed)
        for cls, indices in ft_indices_per_class.items()
    }
    val_datasets = {
        cls: _SubsetView(full_ds, [cls], synset_to_name, indices,
                         max_samples=args.max_samples, seed=args.seed)
        for cls, indices in val_indices_per_class.items()
    }

    # Without-concept eval: per class, all class images excluding the ft training indices
    without_concept_indices = {
        cls: get_class_indices_excluding(
            full_ds, synset_to_name,
            include_classes=[cls],
            exclude_indices=classconcept_indices[cls],
        )
        for cls in class_names
    }
    without_concept_datasets = {
        cls: _SubsetView(full_ds, [cls], synset_to_name, indices,
                         max_samples=args.max_samples, seed=args.seed)
        for cls, indices in without_concept_indices.items()
    }

    # Combined ft dataset (all classes together) used for fine-tuning
    ft_indices = [i for indices in ft_indices_per_class.values() for i in indices]
    ft_dataset = _SubsetView(
        full_ds, class_names, synset_to_name, ft_indices,
        max_samples=args.max_samples, seed=args.seed,
    )

    # Combined without-concept eval dataset (all classes, images without their assigned concept)
    wc_all_indices = [i for indices in without_concept_indices.values() for i in indices]
    without_concept_eval_ds = _SubsetView(
        full_ds, class_names, synset_to_name, wc_all_indices,
        max_samples=args.max_samples, seed=args.seed,
    )

    # Combined with-concept eval dataset (all concept-labeled images, all classes)
    with_concept_all_indices = [i for indices in classconcept_indices.values() for i in indices]
    with_concept_eval_ds = _SubsetView(
        full_ds, class_names, synset_to_name, with_concept_all_indices,
        max_samples=args.max_samples, seed=args.seed,
    )

   
    print(f"\n  {'Class':<28}  {'Concept':<26}  {'FT train':>8}  {'FT val':>8}  {'W/ concept':>11}  {'W/O concept':>12} ")
    print("  " + "─" * 108)
    for cls, concept in class_concepts.items():
        n_with    = len(classconcept_indices[cls])
        n_without = len(without_concept_datasets[cls])
        print(f"  {cls:<28}  {concept:<26}  "
              f"{len(ft_datasets[cls]):>8}  {len(val_datasets[cls]):>8}  "
              f"{n_with:>11}  {n_without:>12}  ")
    print(f"  {'TOTAL':<56}  {sum(len(ft_datasets[cls]) for cls in class_names):>8}  "
          f"{sum(len(v) for v in val_indices_per_class.values()):>8}  "
          f"{len(with_concept_eval_ds):>11}  {len(without_concept_eval_ds):>12}  ")

    if len(ft_dataset) == 0:
        print("\n  ✗ Fine-tune dataset is empty — check concept names.")
        sys.exit(1)

    # ── Import model class ────────────────────────────────────────────────────
    from clip_zero_shot import CLIPZeroShot

    # ── Zero-Shot: per-class with-concept vs without-concept ─────────────────
    print("\n" + "─" * 60)
    print("  ZERO-SHOT EVALUATION — per class, with vs without concept")
    print("─" * 60)
    clip_zs = CLIPZeroShot(model_name=args.clip_model, device=device)
    t0 = time.time()
    zs_per_class = _run_per_class_concept_eval(
        clip_zs, class_names, class_concepts,
        val_datasets, without_concept_datasets,
        dataset_name="miniimagenet", batch_size=args.batch_size,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    _print_concept_split_table("ZERO-SHOT", zs_per_class, class_names, class_concepts)

    # ── Fine-Tuning ───────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  FINE-TUNING ({args.ft_epochs} epoch(s), lr={args.ft_lr})")
    print("─" * 60)
    clip_ft = CLIPZeroShot(model_name=args.clip_model, device=device)
    print(f"  Fine-tuning on {len(ft_dataset)} images (per-class concept assignment)...")
    clip_ft.fine_tune(
        dataset=ft_dataset,
        dataset_name="miniimagenet",
        epochs=args.ft_epochs,
        lr=args.ft_lr,
        batch_size=args.batch_size,
    )

    # ── Fine-Tuned: per-class with-concept vs without-concept ────────────────
    print("\n" + "─" * 60)
    print("  FINE-TUNED EVALUATION — per class, with vs without concept")
    print("─" * 60)
    t0 = time.time()
    ft_per_class = _run_per_class_concept_eval(
        clip_ft, class_names, class_concepts,
        val_datasets, without_concept_datasets,
        dataset_name="miniimagenet", batch_size=args.batch_size,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    _print_concept_split_table("FINE-TUNED", ft_per_class, class_names, class_concepts)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    save_concept_split_csv(zs_per_class, args.output_dir, tag="zeroshot")
    save_concept_split_csv(ft_per_class, args.output_dir, tag="finetuned")
    save_report(
        args, device, class_names, class_concepts,
        ft_indices_per_class, val_indices_per_class,
        without_concept_datasets,
        zs_per_class, ft_per_class,
        run_start, args.output_dir,
    )
    visualize_concept_split_results(
        zs_per_class, ft_per_class,
        class_names, class_concepts,
        args.output_dir, args,
    )

    print(f"\nAll results in: {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
