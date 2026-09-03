"""
run_waterbirds.py
=================

CLIP Zero-Shot + Fine-Tuning Experiment on Waterbirds.

Waterbirds is a standard benchmark for spurious correlations:
  - True label   : landbird (0) / waterbird (1)
  - Spurious cue : background — land (0) or water (1)
  - 4 groups     : 2 classes × 2 backgrounds
  - Training bias: ~95% of waterbirds appear on water, ~95% of landbirds on land

This script measures whether CLIP relies on background as a shortcut,
and whether fine-tuning on a biased subset amplifies or reduces that reliance.

Dataset layout expected:
    <data_dir>/
        metadata.csv          — columns: img_filename, y, split, place
        <img_filename>        — JPEG/PNG images

Download:
    https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
    Extract and pass the extracted directory as --data_dir.

USAGE:
    python run_waterbirds.py
    python run_waterbirds.py --biased_ft          # FT on spurious-aligned examples only
    python run_waterbirds.py --ft_epochs 5 --ft_lr 5e-6
    python run_waterbirds.py --data_dir ./data/waterbird_complete95_forest2water2
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper


# ── Constants ──────────────────────────────────────────────────────────────────

CLASS_NAMES = ["landbird", "waterbird"]   # index = y value in metadata
BG_NAMES    = ["land", "water"]           # index = place value in metadata

# group_id = label * 2 + place
GROUP_NAMES = [
    "landbird / land bg",    # group 0 — spurious-aligned
    "landbird / water bg",   # group 1 — counter-spurious
    "waterbird / land bg",   # group 2 — counter-spurious
    "waterbird / water bg",  # group 3 — spurious-aligned
]
ALIGNED_GROUPS = {0, 3}  # class and background match the training bias direction


def _gid(label: int, place: int) -> int:
    return label * 2 + place


# ── CLIP prompts ───────────────────────────────────────────────────────────────

_WB_SHAPE_PROMPTS = {
    0: "a photo of a Landbird",
    1: "a photo of a Waterbird",
}
_WB_BG_PROMPTS = {
    0: "a photo of a bird in a land environment",
    1: "a photo of a bird in a water environment",
}


def _register_prompts():
    """Inject Waterbirds prompt sets into CLIPZeroShot's PROMPT_SETS registry."""
    from clip_zero_shot import PROMPT_SETS
    PROMPT_SETS["waterbirds"] = {
        "shape": _WB_SHAPE_PROMPTS,
        "color": _WB_BG_PROMPTS,
    }


# ── Dataset ────────────────────────────────────────────────────────────────────

class WaterbirdsDataset(WILDSDataset):
    """
    Waterbirds dataset inheriting from WILDSDataset — matches PRISM/data/Waterbird.py.

    Extensions over PRISM:
      - split / group_filter / max_samples filtering at construction time
      - train_group_sizes computed dynamically from metadata.csv (not hardcoded)
      - clip_preprocess injection for CLIPZeroShot compatibility
      - __getitem__ returns (img, label, bg_name, gid) for prism_classify

    Split codes: 0=train, 1=val, 2=test.
    """

    # Required by WILDSDataset (matches PRISM exactly)
    _dataset_name  = "waterbirds"
    _versions_dict = {
        "1.0": {
            "download_url":    "https://worksheets.codalab.org/rest/bundles/"
                               "0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/",
            "compressed_size": None,
        }
    }
    _original_resolution = (224, 224)

    clip_preprocess = None

    def __init__(self, root: str, split=None, group_filter=None,
                 max_samples=None, seed=42, balanced=False):
        """
        Args:
            root         : directory containing metadata.csv and image files
            split        : None (all), int, or list[int] — 0=train, 1=val, 2=test
            group_filter : None (all groups) or set/list of group_ids to keep
            max_samples  : cap images per class (for quick testing / biased subsets)
            seed         : RNG seed used when max_samples subsamples
            balanced     : whether to balance the subset across groups
        """
        import pandas as pd

        meta_path = os.path.join(root, "metadata.csv")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Waterbirds metadata not found: {meta_path}\n"
                f"Download from:\n"
                f"  https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
            )

        full_df = pd.read_csv(meta_path)

        # Training group sizes from the FULL unfiltered train split.
        # adj_acc_avg must be weighted by the real training distribution,
        # regardless of any subsampling applied to this dataset object.
        _tr  = full_df[full_df["split"] == 0]
        _cnt = (_tr["y"] * 2 + _tr["place"]).value_counts().to_dict()
        self.train_group_sizes = {gid: _cnt.get(gid, 0) for gid in range(4)}

        # ── Apply filters ─────────────────────────────────────────────────────
        df = full_df.copy()
        if split is not None:
            splits = [split] if isinstance(split, int) else list(split)
            df = df[df["split"].isin(splits)].reset_index(drop=True)

        rng = np.random.default_rng(seed)
        if max_samples is not None:
            parts = []
            for _, grp in df.groupby("y"):
                if len(grp) > max_samples:
                    chosen = rng.choice(len(grp), size=max_samples, replace=False)
                    parts.append(grp.iloc[sorted(chosen)])
                else:
                    parts.append(grp)
            df = pd.concat(parts).sort_index().reset_index(drop=True)

        if group_filter is not None:
            gids = df["y"] * 2 + df["place"]
            df = df[gids.isin(set(group_filter))].reset_index(drop=True)

        # ── WILDSDataset required attributes (mirrors PRISM/data/Waterbird.py) ─
        # We set these manually and skip super().__init__() because WILDS expects
        # data at root_dir/waterbirds_v1.0/ but ours is directly at root/.
        self._data_dir   = root
        self._version    = "1.0"
        self._split_scheme = "official"
        self._split_dict   = {"train": 0, "val": 1, "test": 2}
        self._split_names  = {"train": "Train", "val": "Validation", "test": "Test"}

        self._y_array   = torch.LongTensor(df["y"].values.copy())
        self._y_size    = 1
        self._n_classes = 2

        # metadata columns: [place/background, y] — matches PRISM exactly
        self._metadata_array = torch.stack(
            (torch.LongTensor(df["place"].values.copy()), self._y_array), dim=1
        )
        self._metadata_fields = ["background", "y"]
        self._metadata_map    = {
            "background": [" land", "water"],       # leading space matches PRISM
            "y":          [" landbird", "waterbird"],
        }

        self._input_array = df["img_filename"].values   # relative paths (as in PRISM)
        self._split_array = torch.LongTensor(df["split"].values.copy())

        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=["background", "y"]
        )

        # ── Our interface ─────────────────────────────────────────────────────
        self.root    = root
        self.samples = [
            (os.path.join(root, row["img_filename"]),
             int(row["y"]),
             BG_NAMES[int(row["place"])],
             _gid(int(row["y"]), int(row["place"])))
            for _, row in df.iterrows()
        ]

        # Numpy arrays for convenient indexing (y_array is already a WILDSDataset property)
        self.place_array    = df["place"].values.astype(np.int64)
        self.group_id_array = (df["y"].values * 2 + df["place"].values).astype(np.int64)

    # ── WILDSDataset interface ────────────────────────────────────────────────

    def get_input(self, idx):
        """Return PIL Image for sample idx (WILDS-compatible)."""
        from PIL import Image
        return Image.open(
            os.path.join(self._data_dir, self._input_array[idx])
        ).convert("RGB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (img, label, bg_name, gid) — compatible with prism_classify."""
        from PIL import Image
        path, label, bg_name, gid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.clip_preprocess is not None:
            img = self.clip_preprocess(img)
        return img, label, bg_name, gid

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """WILDS-compatible eval with PRISM's training-weighted adj_acc_avg.

        Computes per-group accuracy directly from predictions and metadata
        (avoids torch_scatter dependency from standard_group_eval).
        adj_acc_avg uses actual training group sizes from metadata.csv.
        """
        if prediction_fn is not None:
            y_pred = prediction_fn(y_pred)

        places = metadata[:, 0]   # background index (0=land, 1=water)
        ys     = metadata[:, 1]   # class index     (0=landbird, 1=waterbird)

        results     = {}
        results_str = ""
        total_w     = sum(self.train_group_sizes.values())
        weighted    = 0.0

        # Per-group accuracy: gid = y * 2 + place
        for gid in range(4):
            y_val, pl_val = gid // 2, gid % 2
            mask = (ys == y_val) & (places == pl_val)
            n    = int(mask.sum())
            if n > 0:
                acc = (y_pred[mask] == y_true[mask]).float().mean().item()
                weighted += acc * self.train_group_sizes[gid]
            else:
                acc = float("nan")
            gname = GROUP_NAMES[gid]
            key   = f"acc_y:{CLASS_NAMES[y_val]}_background:{BG_NAMES[pl_val]}"
            results[key] = acc
            results_str += f"  {gname:<34} acc: {acc * 100:.1f}%  (n={n})\n"

        overall = (y_pred == y_true).float().mean().item()
        results["acc_avg"]     = overall
        results["adj_acc_avg"] = weighted / total_w

        results_str = (
            f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n"
            + f"Overall acc: {overall:.3f}\n"
            + results_str
        )
        return results, results_str

    def adj_acc_avg(self, per_group_acc: dict) -> float:
        """Training-weighted accuracy for use with evaluate_waterbirds().

        Args:
            per_group_acc: {group_id: accuracy_in_percent} for gid in {0,1,2,3}
        Returns float in [0, 100].
        """
        total_w = sum(self.train_group_sizes.values())
        return sum(
            per_group_acc[gid] * w
            for gid, w in self.train_group_sizes.items()
            if gid in per_group_acc
        ) / total_w


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP Zero-Shot + Fine-Tuning on Waterbirds spurious correlation benchmark"
    )
    parser.add_argument("--clip_model",  type=str, default="ViT-L/14",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    parser.add_argument("--data_dir",    type=str, default="./data/waterbirds")
    parser.add_argument("--output_dir",  type=str, default="./results/waterbirds")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--ft_epochs",   type=int, default=3)
    parser.add_argument("--ft_lr",       type=float, default=1e-5)
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Cap images loaded per split for quick testing")
    parser.add_argument("--biased_ft",   type=bool,  default=True,
                        help="Fine-tune only on spurious-saligned training examples "
                             "(groups 0 and 3); exacerbates the spurious correlation")
    parser.add_argument("--clip_weights_dir", type=str, default=None,
                        help="Path to offline CLIP weights saved by "
                             "CLIPZeroShot.download_and_save() / download_hf_models.py. "
                             "Use on servers without internet access.")
    parser.add_argument("--balanced_ft",   type=bool,  default=False,
                        help="Fine-tune on a group-balanced subset of training examples; "
                             "mutually exclusive with --biased_ft")

    args = parser.parse_args()

    if args.balanced_ft and args.biased_ft:
        parser.error("--balanced_ft and --biased_ft are mutually exclusive; enable only one")

    # On Compute Canada, SLURM_JOB_ID is always set — redirect paths to $HOME
    # only when the user has not explicitly overridden them.
    if "SLURM_JOB_ID" in os.environ:
        _home = os.path.expanduser("~")
        if args.data_dir == "./data/waterbirds":
            args.data_dir = os.path.join(_home, "data", "waterbirds")
        if args.output_dir == "./results/waterbirds":
            args.output_dir = os.path.join(_home, "results", "waterbirds")
        if args.clip_weights_dir is None:
            args.clip_weights_dir = os.path.join(_home, "hf_models")

    return args


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

def evaluate_waterbirds(results, label="", dataset=None):
    """Compute per-class, per-group, and overall accuracy from run() output."""
    preds  = results["predictions_shape"]
    trues  = results["true_labels"]
    groups = results["group_ids"]

    n_total   = len(trues)
    n_correct = int((preds == trues).sum())
    overall   = n_correct / n_total * 100 if n_total else 0.0

    header = f"  [{label}]" if label else "  "
    print(f"\n{'═' * 65}")
    print(f"{header} OVERALL: {n_correct}/{n_total} = {overall:.2f}%")
    print(f"{'═' * 65}")

    # Per-class
    print(f"\n  {'Class':<22} {'Correct':>8} {'Total':>8} {'Acc%':>8}")
    print("  " + "─" * 50)
    per_class = []
    for ci, cls in enumerate(CLASS_NAMES):
        mask = trues == ci
        tot  = int(mask.sum())
        corr = int((preds[mask] == ci).sum())
        acc  = corr / tot * 100 if tot else float("nan")
        per_class.append({"class": cls, "correct": corr, "total": tot, "acc": acc})
        print(f"  {cls:<22} {corr:>8} {tot:>8} {acc:>7.1f}%")

    # Per-group
    print(f"\n  {'Group':<30} {'Correct':>8} {'Total':>8} {'Acc%':>8}  Note")
    print("  " + "─" * 72)
    per_group = []
    for gid, gname in enumerate(GROUP_NAMES):
        mask = groups == gid
        tot  = int(mask.sum())
        corr = int((preds[mask] == trues[mask]).sum())
        acc  = corr / tot * 100 if tot else float("nan")
        per_group.append({"group": gname, "group_id": gid, "correct": corr, "total": tot, "acc": acc})
        note  = "← aligned" if gid in ALIGNED_GROUPS else "← counter-spurious"
        acc_s = f"{acc:>7.1f}%" if not np.isnan(acc) else "      N/A"
        print(f"  {gname:<30} {corr:>8} {tot:>8} {acc_s}  {note}")

    valid_accs = [g["acc"] for g in per_group if g["total"] > 0 and not np.isnan(g["acc"])]
    worst_acc  = min(valid_accs) if valid_accs else float("nan")

    # PRISM-style adj_acc_avg: weighted by actual training-set group frequencies.
    # Sizes are read from the dataset object (computed from metadata.csv, not
    # hardcoded) so they are correct regardless of subsampling.
    # Falls back to equal weight when dataset is not provided.
    train_sizes = dataset.train_group_sizes if dataset is not None \
                  else {gid: 1 for gid in range(4)}
    total_w = sum(train_sizes.values())
    adj_acc = sum(
        per_group[gid]["acc"] * train_sizes[gid]
        for gid in range(4) if per_group[gid]["total"] > 0
    ) / total_w

    print(f"\n  Worst-group accuracy       : {worst_acc:.2f}%")
    print(f"  Adj avg acc (train-weighted): {adj_acc:.2f}%")

    return {
        "overall":           {"acc": overall, "correct": n_correct, "total": n_total},
        "per_class":         per_class,
        "per_group":         per_group,
        "worst_group_acc":   worst_acc,
        "adj_acc_avg":       adj_acc,
        "predictions_shape": preds,
        "true_labels":       trues,
        "group_ids":         groups,
        "class_names":       CLASS_NAMES,
    }


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_waterbirds(zs_stats, ft_stats, output_dir, args):
    """
    3 × 2 panel figure analysing spurious correlation.

    [0,0] Per-group accuracy:  ZS vs FT for all 4 groups
    [0,1] Spurious gap:        aligned − counter-spurious accuracy (ZS & FT)
    [1,0] Per-class accuracy:  ZS vs FT
    [1,1] Test set distribution: images per group
    [2,0] Background × Class heatmap — Zero-Shot
    [2,1] Background × Class heatmap — Fine-Tuned
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    C_ZS = "#4e79a7"   # blue
    C_FT = "#f28e2b"   # orange
    # aligned = green, counter-spurious = red
    GROUP_COLORS = ["#59a14f", "#e15759", "#e15759", "#59a14f"]

    if args.biased_ft:
        ft_label = "Biased FT on " + str(args.max_samples) + " images per aligned group"
    elif args.balanced_ft:
        ft_label = "Balanced FT on " + str(args.max_samples) + " images per group for all 4 groups"
    else:
        ft_label = "Fine-Tuned on " + str(args.max_samples) + " images per group"

    zs_grp = {g["group_id"]: g for g in zs_stats["per_group"]}
    ft_grp = {g["group_id"]: g for g in ft_stats["per_group"]}
    zs_cls = {g["class"]: g for g in zs_stats["per_class"]}
    ft_cls = {g["class"]: g for g in ft_stats["per_class"]}

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle(
        f"CLIP × Waterbirds — Spurious Correlation Analysis  |  Model: {args.clip_model}\n"
        f"ZS overall: {zs_stats['overall']['acc']:.1f}%   {ft_label} overall: {ft_stats['overall']['acc']:.1f}%\n"
        f"ZS worst-group: {zs_stats['worst_group_acc']:.1f}%   "
        f"{ft_label} worst-group: {ft_stats['worst_group_acc']:.1f}%",
        fontsize=11, fontweight="bold",
    )

    short_gnames = [g.replace(" / ", "\n") for g in GROUP_NAMES]
    x4 = np.arange(4)
    w  = 0.35

    # ── [0,0] Per-group accuracy ──────────────────────────────────────────────
    ax = axes[0, 0]
    zs_accs = [zs_grp[g]["acc"] for g in range(4)]
    ft_accs = [ft_grp[g]["acc"] for g in range(4)]
    ax.bar(x4 - w/2, zs_accs, w, color=GROUP_COLORS, alpha=0.9, edgecolor="white",
           label="Zero-Shot")
    ax.bar(x4 + w/2, ft_accs, w, color=GROUP_COLORS, alpha=0.6, edgecolor="white",
           hatch="//", label=ft_label)
    ax.axhline(zs_stats["overall"]["acc"], color=C_ZS, ls="--", lw=1.2, alpha=0.7,
               label=f"ZS mean ({zs_stats['overall']['acc']:.1f}%)")
    ax.axhline(ft_stats["overall"]["acc"], color=C_FT, ls="--", lw=1.2, alpha=0.7,
               label=f"FT mean ({ft_stats['overall']['acc']:.1f}%)")
    ax.set_xticks(x4)
    ax.set_xticklabels(short_gnames, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Per-Group Accuracy: Zero-Shot vs Fine-Tuned\n"
                 "green = spurious-aligned  ·  red = counter-spurious", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    for xi, (za, fa) in enumerate(zip(zs_accs, ft_accs)):
        ax.text(xi - w/2, za + 1, f"{za:.1f}", ha="center", fontsize=8)
        ax.text(xi + w/2, fa + 1, f"{fa:.1f}", ha="center", fontsize=8)

    # ── [0,1] Spurious correlation gap ───────────────────────────────────────
    # gap = aligned_acc − counter_spurious_acc per class, for both ZS and FT
    ax = axes[0, 1]
    gaps = [
        zs_grp[0]["acc"] - zs_grp[1]["acc"],   # ZS landbird: land_bg − water_bg
        zs_grp[3]["acc"] - zs_grp[2]["acc"],   # ZS waterbird: water_bg − land_bg
        ft_grp[0]["acc"] - ft_grp[1]["acc"],   # FT landbird
        ft_grp[3]["acc"] - ft_grp[2]["acc"],   # FT waterbird
    ]
    gap_labels = ["ZS\nlandbird", "ZS\nwaterbird", f"{ft_label}\nlandbird", f"{ft_label}\nwaterbird"]
    gap_colors = ["#59a14f" if g >= 0 else "#e15759" for g in gaps]
    x_gap = np.arange(4)
    ax.bar(x_gap, gaps, color=gap_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_gap)
    ax.set_xticklabels(gap_labels, fontsize=9)
    ax.set_ylabel("Accuracy: aligned − counter-spurious (%)")
    ax.set_title("Spurious Correlation Strength\n"
                 "positive = model favours aligned (biased)  ·  green ≥ 0  red < 0",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for xi, g in enumerate(gaps):
        va, off = ("bottom", 0.4) if g >= 0 else ("top", -0.4)
        ax.text(xi, g + off, f"{g:+.1f}%", ha="center", va=va, fontsize=10, fontweight="bold")

    # ── [1,0] Per-class accuracy ──────────────────────────────────────────────
    ax = axes[1, 0]
    x2 = np.arange(2)
    zs_cls_accs = [zs_cls[c]["acc"] for c in CLASS_NAMES]
    ft_cls_accs = [ft_cls[c]["acc"] for c in CLASS_NAMES]
    ax.bar(x2 - w/2, zs_cls_accs, w, label="Zero-Shot",  color=C_ZS, alpha=0.85, edgecolor="white")
    ax.bar(x2 + w/2, ft_cls_accs, w, label=ft_label,     color=C_FT, alpha=0.85, edgecolor="white")
    ax.set_xticks(x2)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Per-Class Accuracy: Zero-Shot vs Fine-Tuned", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for xi, (za, fa) in enumerate(zip(zs_cls_accs, ft_cls_accs)):
        ax.text(xi - w/2, za + 1, f"{za:.1f}", ha="center", fontsize=9)
        ax.text(xi + w/2, fa + 1, f"{fa:.1f}", ha="center", fontsize=9)
        delta = fa - za
        ax.text(xi, max(za, fa) + 5, f"Δ{delta:+.1f}", ha="center", fontsize=8,
                color="#59a14f" if delta >= 0 else "#e15759")

    # ── [1,1] Test set group distribution ────────────────────────────────────
    ax = axes[1, 1]
    grp_counts = [zs_grp[g]["total"] for g in range(4)]
    ax.bar(x4, grp_counts, color=GROUP_COLORS, alpha=0.85, edgecolor="white")
    ax.set_xticks(x4)
    ax.set_xticklabels(short_gnames, fontsize=9)
    ax.set_ylabel("Number of images")
    ax.set_title("Test Set — Group Distribution\n"
                 "green = spurious-aligned  ·  red = counter-spurious", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    mx = max(grp_counts) if grp_counts else 1
    for xi, n in enumerate(grp_counts):
        ax.text(xi, n + mx * 0.01, str(n), ha="center", fontsize=9, fontweight="bold")

    # ── [2,0] and [2,1]: Background × Class accuracy heatmap ─────────────────
    # Rows = background (land=0, water=1), Cols = class (landbird=0, waterbird=1)
    # Diagonal cells are the spurious-aligned groups.
    # gid = label * 2 + place  →  mat[place, label]
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#dddddd")

    def _draw_heatmap(ax, grp_dict, title, cbar_label):
        mat = np.full((2, 2), np.nan)
        cnt = np.zeros((2, 2), dtype=int)
        for gid in range(4):
            label = gid // 2
            place = gid %  2
            mat[place, label] = grp_dict[gid]["acc"]
            cnt[place, label] = grp_dict[gid]["total"]

        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES, fontsize=11)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(BG_NAMES, fontsize=11)
        ax.set_xlabel("True class")
        ax.set_ylabel("Background")
        ax.set_title(title, fontweight="bold")
        for r in range(2):
            for c in range(2):
                v = mat[r, c]
                n = cnt[r, c]
                if not np.isnan(v):
                    col = "white" if (v > 65 or v < 20) else "black"
                    ax.text(c, r - 0.12, f"{v:.1f}%", ha="center", va="center",
                            fontsize=13, fontweight="bold", color=col)
                    ax.text(c, r + 0.28, f"n={n}", ha="center", va="center",
                            fontsize=9, color=col, alpha=0.85)
        # Red border marks spurious-aligned diagonal cells
        for diag in [0, 1]:
            ax.add_patch(plt.Rectangle(
                (diag - 0.5, diag - 0.5), 1, 1,
                fill=False, edgecolor="red", lw=3, zorder=6,
            ))
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    _draw_heatmap(
        axes[2, 0], zs_grp,
        "Background × Class Accuracy  [Zero-Shot]\n(red border = spurious-aligned cell)",
        "ZS Accuracy (%)",
    )
    _draw_heatmap(
        axes[2, 1], ft_grp,
        f"Background × Class Accuracy  [{ft_label}]\n(red border = spurious-aligned cell)",
        f"{ft_label} Accuracy (%)",
    )

    plt.tight_layout()
    path = os.path.join(output_dir, f"waterbirds_{args.max_samples}_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Dataset manifest ──────────────────────────────────────────────────────────

def save_dataset_manifest(dataset, path, split_name):
    import pandas as pd
    rows = []
    for img_path, label, bg_name, gid in dataset.samples:
        rows.append({
            "img_path":   img_path,
            "label":      label,
            "class":      ["landbird", "waterbird"][label],
            "bg":         bg_name,
            "group_id":   gid,
            "group":      GROUP_NAMES[gid],
            "aligned":    gid in ALIGNED_GROUPS,
            "split":      split_name,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(rows)} images)")


# ── CSV output ─────────────────────────────────────────────────────────────────

def save_results_csv(zs_stats, ft_stats, output_dir, sample_size):
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zs_grp = {g["group_id"]: g for g in zs_stats["per_group"]}
    ft_grp = {g["group_id"]: g for g in ft_stats["per_group"]}
    rows = []
    for gid, gname in enumerate(GROUP_NAMES):
        rows.append({
            "group_id":  gid,
            "group":     gname,
            "aligned":   gid in ALIGNED_GROUPS,
            "zs_acc":    round(zs_grp[gid]["acc"], 2),
            "zs_n":      zs_grp[gid]["total"],
            "ft_acc":    round(ft_grp[gid]["acc"], 2),
            "ft_n":      ft_grp[gid]["total"],
            "delta_acc": round(ft_grp[gid]["acc"] - zs_grp[gid]["acc"], 2),
        })

    path = os.path.join(output_dir, f"waterbirds_results_{sample_size}_{ts}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


# ── Report output ──────────────────────────────────────────────────────────────

def save_report(zs_stats, ft_stats, output_dir, args, runtime_s):
    import json
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    zs_grp = {g["group_id"]: g for g in zs_stats["per_group"]}
    ft_grp = {g["group_id"]: g for g in ft_stats["per_group"]}

    def _spurious_gaps(grp):
        return {
            "landbird_gap":  round(grp[0]["acc"] - grp[1]["acc"], 2),
            "waterbird_gap": round(grp[3]["acc"] - grp[2]["acc"], 2),
        }

    report = {
        "timestamp": ts,
        "config": {
            "clip_model":   args.clip_model,
            "ft_epochs":    args.ft_epochs,
            "ft_lr":        args.ft_lr,
            "max_samples":  args.max_samples,
            "biased_ft":    args.biased_ft,
            "seed":         args.seed,
            "batch_size":   args.batch_size,
            "data_dir":     args.data_dir,
            "output_dir":   args.output_dir,
        },
        "runtime_s": round(runtime_s, 1),
        "zero_shot": {
            "overall":        zs_stats["overall"],
            "per_class":      zs_stats["per_class"],
            "per_group":      zs_stats["per_group"],
            "worst_group_acc": round(zs_stats["worst_group_acc"], 2),
            "spurious_gaps":  _spurious_gaps(zs_grp),
        },
        "fine_tuned": {
            "overall":        ft_stats["overall"],
            "per_class":      ft_stats["per_class"],
            "per_group":      ft_stats["per_group"],
            "worst_group_acc": round(ft_stats["worst_group_acc"], 2),
            "spurious_gaps":  _spurious_gaps(ft_grp),
        },
        "delta": {
            "overall_acc":    round(ft_stats["overall"]["acc"] - zs_stats["overall"]["acc"], 2),
            "worst_group_acc": round(ft_stats["worst_group_acc"] - zs_stats["worst_group_acc"], 2),
            "per_group": [
                {
                    "group_id": gid,
                    "group":    GROUP_NAMES[gid],
                    "delta_acc": round(ft_grp[gid]["acc"] - zs_grp[gid]["acc"], 2),
                }
                for gid in range(4)
            ],
        },
    }

    path = os.path.join(output_dir, f"waterbirds_report_{args.max_samples}_{ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    run_start = time.time()
    args      = parse_args()
    device    = check_dependencies()

    print("\n" + "═" * 65)
    print("  CLIP × Waterbirds — Spurious Correlation Experiment")
    print("═" * 65)
    print(f"  Model      : {args.clip_model}")
    print(f"  Device     : {device}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  FT epochs  : {args.ft_epochs}   lr: {args.ft_lr}")
    print(f"  Biased FT  : {args.biased_ft}")
    print(f"  Balanced FT: {args.balanced_ft}")
    print(f"  Output     : {args.output_dir}")
    
    print("═" * 65 + "\n")

    _register_prompts()

    # ── Load splits ───────────────────────────────────────────────────────────
    print("Loading Waterbirds dataset...")
    train_ds = WaterbirdsDataset(args.data_dir, split=0,
                                 max_samples=args.max_samples, seed=args.seed)
    test_ds  = WaterbirdsDataset(args.data_dir, split=2,
                                 max_samples=None, seed=args.seed)

    for sname, ds in [("train", train_ds), ("test", test_ds)]:
        gid_counts = defaultdict(int)
        for _, _, _, gid in ds.samples:
            gid_counts[gid] += 1
        print(f"  {sname:5s} ({len(ds):5d} images) : " +
              "  |  ".join(f"{GROUP_NAMES[g]}: {gid_counts[g]}" for g in range(4)))

    # ── Fine-tuning subset ────────────────────────────────────────────────────
    if args.biased_ft:
        ft_ds = WaterbirdsDataset(args.data_dir, split=0,
                                  group_filter=ALIGNED_GROUPS,
                                  max_samples=args.max_samples, seed=args.seed)
        print(f"\n  Biased FT: {len(ft_ds)} images — spurious-aligned groups only")
    elif args.balanced_ft:
        ft_ds = WaterbirdsDataset(args.data_dir, split=0,
                                  group_filter=[0, 1, 2, 3],
                                  max_samples=args.max_samples, seed=args.seed,
                                  balanced=True)
        print(f"\n  Balanced FT: {len(ft_ds)} images — group-balanced subset")
    else:
        ft_ds = train_ds
        print(f"\n  Full FT  : {len(ft_ds)} training images")

    if len(ft_ds) == 0:
        print("  ✗ Fine-tune dataset is empty. Check --data_dir or --biased_ft.")
        sys.exit(1)

    # ── Zero-Shot ─────────────────────────────────────────────────────────────
    from clip_zero_shot import CLIPZeroShot

    print("\n" + "─" * 65)
    print("  ZERO-SHOT EVALUATION")
    print("─" * 65)
    clip_zs = CLIPZeroShot(model_name=args.clip_model, device=device,
                           offline_weights_dir=args.clip_weights_dir)
    t0 = time.time()
    res_zs   = clip_zs.run(dataset=test_ds, prompt_mode="shape",
                           dataset_name="waterbirds", batch_size=args.batch_size)
    zs_stats = evaluate_waterbirds(res_zs, label="Zero-Shot / Test", dataset=test_ds)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Fine-Tuning ───────────────────────────────────────────────────────────
    if args.biased_ft:
        ft_tag = "BIASED"
    elif args.balanced_ft:
        ft_tag = "BALANCED"
    else:
        ft_tag = "FULL"
    print("\n" + "─" * 65)
    print(f"  FINE-TUNING ({ft_tag}, {args.ft_epochs} epoch(s), lr={args.ft_lr})")
    print("─" * 65)
    clip_ft = CLIPZeroShot(model_name=args.clip_model, device=device,
                           offline_weights_dir=args.clip_weights_dir)
    clip_ft.fine_tune(
        dataset=ft_ds,
        dataset_name="waterbirds",
        epochs=args.ft_epochs,
        lr=args.ft_lr,
        batch_size=args.batch_size,
    )

    import json
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"clip_ft_{ft_tag}_{args.max_samples}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # ── model weights ─────────────────────────────────────────────────────────
    model_path = os.path.join(run_dir, "model.pt")
    metadata = {
        "clip_model":  args.clip_model,
        "ft_tag":      ft_tag,
        "max_samples": args.max_samples,
        "ft_epochs":   args.ft_epochs,
        "ft_lr":       args.ft_lr,
        "seed":        args.seed,
        "biased_ft":   args.biased_ft,
        "data_dir":    os.path.abspath(args.data_dir),
        "run_dir":     os.path.abspath(run_dir),
    }
    clip_ft.save_model(model_path, metadata=metadata)

    # ── run config ────────────────────────────────────────────────────────────
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {os.path.join(run_dir, 'config.json')}")

    # ── dataset manifests ─────────────────────────────────────────────────────
    print("\nSaving dataset manifests...")
    save_dataset_manifest(train_ds, os.path.join(run_dir, "train_all_manifest.csv"),  "train")
    save_dataset_manifest(ft_ds,    os.path.join(run_dir, "ft_train_manifest.csv"),   "ft_train")
    save_dataset_manifest(test_ds,  os.path.join(run_dir, "test_manifest.csv"),       "test")

    print(f"\n  Run folder: {os.path.abspath(run_dir)}")

    print("\n" + "─" * 65)
    print(f"  FINE-TUNED EVALUATION ({ft_tag})")
    print("─" * 65)
    t0 = time.time()
    res_ft   = clip_ft.run(dataset=test_ds, prompt_mode="shape",
                           dataset_name="waterbirds", batch_size=args.batch_size)
    ft_stats = evaluate_waterbirds(res_ft, label=f"{ft_tag} Fine-Tuned / Test", dataset=test_ds)
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Spurious correlation summary ──────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  SPURIOUS CORRELATION SUMMARY")
    print("═" * 65)
    for mname, grp_list in [("ZS", zs_stats["per_group"]), ("FT", ft_stats["per_group"])]:
        grp = {g["group_id"]: g for g in grp_list}
        gap_land  = grp[0]["acc"] - grp[1]["acc"]
        gap_water = grp[3]["acc"] - grp[2]["acc"]
        print(f"  [{mname}] landbird  aligned−counter: {gap_land:+.1f}%  "
              f"| waterbird aligned−counter: {gap_water:+.1f}%")
    print(f"\n  ZS worst-group : {zs_stats['worst_group_acc']:.2f}%")
    print(f"  FT worst-group : {ft_stats['worst_group_acc']:.2f}%")
    print("═" * 65)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_csv(zs_stats, ft_stats, args.output_dir,args.max_samples)
    visualize_waterbirds(zs_stats, ft_stats, args.output_dir, args)

    print(f"\nAll results in : {os.path.abspath(args.output_dir)}")
    print(f"Total runtime  : {time.time() - run_start:.1f}s\n")





if __name__ == "__main__":
    main()
