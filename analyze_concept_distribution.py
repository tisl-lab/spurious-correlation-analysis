#!/usr/bin/env python3
"""
Analyzes concept distribution across MiniImageNet semantic groups.

For each group (birds, insects, canidea):
  - Counts images per class per concept
  - Computes distribution over concept combinations (single, pairs, triples, ...)
  - Generates a visualized report and saves CSV statistics

Usage:
    python analyze_concept_distribution.py [--output_dir OUTPUT_DIR]
"""

import argparse
import itertools
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "concept_analysis")

GROUP_CONFIGS = {
    "birds": {
        "target_classes": ["house_finch", "robin", "toucan", "goose"],
        "concepts": ["high_freq_sky_patterns", "image_edge", "wood", "fence"],
        "concept_file": "MiniImageNet_Bird_Concepts.json",
    },
    "insects": {
        "target_classes": ["ant", "harvestman", "ladybug", "rhinoceros_beetle"],
        "concepts": ["leaves", "hands"],
        "concept_file": "MiniImageNet_Insect_Concepts.json",
    },
    "canidea": {
        "target_classes": [
            "golden_retriever", "malamute", "Tibetan_mastiff", "dalmatian",
            "boxer", "Newfoundland", "white_wolf", "Arctic_fox",
        ],
        "concepts": ["human", "snow", "fence"],
        "concept_file": "MiniImageNet_Canidea_Concepts.json",
    },
}


def load_labels():
    with open(os.path.join(DATASETS_DIR, "mini_imagenet_labels.json")) as f:
        return json.load(f)  # {synset: readable_name}


def build_image_concept_map(concept_file_path, all_concepts):
    """
    Builds a mapping from (synset, basename) -> frozenset_of_concepts.
    Only includes concepts listed in all_concepts.
    """
    with open(concept_file_path) as f:
        concepts_map = json.load(f)

    img_concepts = defaultdict(set)
    for concept, rel_paths in concepts_map.items():
        if concept not in all_concepts:
            continue
        for rel_path in rel_paths:
            parts = rel_path.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            synset, basename = parts[0], parts[-1]
            img_concepts[(synset, basename)].add(concept)

    return {k: frozenset(v) for k, v in img_concepts.items()}


def analyze_group(group_name, config, synset_to_name):
    """
    Returns:
      class_data: {class_name: [frozenset_of_concepts, ...]}  — one entry per image
      all_concepts: list of concept names for this group
      target_classes: set of intended class names
      spurious_classes: set of non-target class names that appear in the concept file
    """
    all_concepts = config["concepts"]
    target_set = set(config["target_classes"])
    concept_file = os.path.join(DATASETS_DIR, config["concept_file"])

    img_concepts = build_image_concept_map(concept_file, set(all_concepts))

    # Collect all class names that appear in the concept file
    class_data = defaultdict(list)
    for (synset, _basename), concepts in img_concepts.items():
        class_name = synset_to_name.get(synset, synset)
        class_data[class_name].append(concepts)

    class_data = dict(class_data)
    spurious_classes = set(class_data.keys()) - target_set

    return class_data, all_concepts, target_set, spurious_classes


def compute_stats(class_data, all_concepts):
    """
    Returns:
      per_class_concept_count: {class: {concept: count}}
      per_class_combo_count:   {class: {frozenset: count}}
      global_combo_dist:       {k: {frozenset: count}} — exact k concepts
      at_least_combo:          {k: {frozenset: count}} — at-least k concept subset
    """
    per_class_concept_count = {}
    per_class_combo_count = {}

    for cls, images in class_data.items():
        concept_count = defaultdict(int)
        combo_count = defaultdict(int)
        for concepts in images:
            for c in concepts:
                concept_count[c] += 1
            combo_count[concepts] += 1
        per_class_concept_count[cls] = dict(concept_count)
        per_class_combo_count[cls] = dict(combo_count)

    global_combo_dist = defaultdict(lambda: defaultdict(int))
    for images in class_data.values():
        for concepts in images:
            global_combo_dist[len(concepts)][concepts] += 1

    at_least_combo = {}
    for k in range(1, len(all_concepts) + 1):
        at_least_combo[k] = {}
        for combo in itertools.combinations(all_concepts, k):
            combo_fs = frozenset(combo)
            count = sum(
                1
                for images in class_data.values()
                for img_concepts in images
                if combo_fs.issubset(img_concepts)
            )
            at_least_combo[k][combo_fs] = count

    return per_class_concept_count, per_class_combo_count, dict(global_combo_dist), at_least_combo


def save_stats(group_name, class_data, all_concepts, target_classes,
               per_class_concept_count, per_class_combo_count,
               global_combo_dist, at_least_combo, output_dir):
    lines = [
        f"# Concept Distribution Analysis: {group_name.upper()}\n",
        f"# Concepts: {', '.join(all_concepts)}\n",
        f"# Target classes: {', '.join(sorted(target_classes))}\n\n",
    ]

    # Sort: target classes first, then spurious
    ordered_classes = sorted(target_classes & set(class_data.keys())) + \
                      sorted(set(class_data.keys()) - target_classes)

    lines.append("## Per-Class Concept Counts\n")
    lines.append("class,is_target," + ",".join(all_concepts) + ",total_with_concepts\n")
    for cls in ordered_classes:
        counts = per_class_concept_count.get(cls, {})
        is_target = "yes" if cls in target_classes else "no"
        row = f"{cls},{is_target}," + ",".join(str(counts.get(c, 0)) for c in all_concepts)
        row += f",{len(class_data.get(cls, []))}\n"
        lines.append(row)

    all_combos = set()
    for d in per_class_combo_count.values():
        all_combos.update(d.keys())
    all_combos = sorted(all_combos, key=lambda s: (len(s), sorted(s)))

    lines.append("\n## Exact Concept Combination Counts (per class)\n")
    combo_labels = ["+".join(sorted(c)) for c in all_combos]
    lines.append("class," + ",".join(combo_labels) + "\n")
    for cls in ordered_classes:
        row = cls + "," + ",".join(
            str(per_class_combo_count.get(cls, {}).get(c, 0)) for c in all_combos
        ) + "\n"
        lines.append(row)

    lines.append("\n## Images with Exactly k Concepts (global, all classes)\n")
    lines.append("k,count\n")
    for k in sorted(global_combo_dist.keys()):
        total = sum(global_combo_dist[k].values())
        lines.append(f"{k},{total}\n")

    lines.append("\n## At-Least Combo Counts (global)\n")
    for k in sorted(at_least_combo.keys()):
        lines.append(f"\n### Size-{k} combinations\n")
        lines.append("combo,count\n")
        for combo_fs, count in sorted(at_least_combo[k].items(), key=lambda x: -x[1]):
            lines.append(f"{'+'.join(sorted(combo_fs))},{count}\n")

    out_path = os.path.join(output_dir, f"{group_name}_concept_stats.csv")
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"  Saved stats : {out_path}")


def _short(name, maxlen=14):
    return name[:maxlen].replace("_", " ")


def visualize_group(group_name, class_data, all_concepts, target_classes,
                    per_class_concept_count, per_class_combo_count,
                    global_combo_dist, at_least_combo, output_dir):
    ordered_classes = sorted(target_classes & set(class_data.keys())) + \
                      sorted(set(class_data.keys()) - target_classes)
    n_cls = len(ordered_classes)
    n_con = len(all_concepts)
    is_target = [c in target_classes for c in ordered_classes]

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        f"Concept Distribution — {group_name.capitalize()} "
        f"({n_cls} classes, {n_con} concepts)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    # ── Panel 1: Heatmap class × concept ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    heat = np.array([
        [per_class_concept_count.get(cls, {}).get(c, 0) for c in all_concepts]
        for cls in ordered_classes
    ], dtype=float)
    im1 = ax1.imshow(heat, aspect="auto", cmap="YlOrRd")
    ax1.set_xticks(range(n_con))
    ax1.set_xticklabels([_short(c) for c in all_concepts], fontsize=7, rotation=30, ha="right")
    ax1.set_yticks(range(n_cls))
    ylabels = [("★ " if t else "  ") + _short(c) for c, t in zip(ordered_classes, is_target)]
    ax1.set_yticklabels(ylabels, fontsize=7)
    ax1.set_title("Images per Class × Concept\n(★ = target class)", fontsize=8)
    plt.colorbar(im1, ax=ax1, shrink=0.75)
    vmax = heat.max() if heat.max() > 0 else 1
    for ci in range(n_cls):
        for cj in range(n_con):
            ax1.text(cj, ci, str(int(heat[ci, cj])), ha="center", va="center",
                     fontsize=6, color="white" if heat[ci, cj] > 0.6 * vmax else "black")

    # ── Panel 2: Grouped bar chart per class ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(n_cls)
    w = 0.75 / n_con
    colors = plt.cm.tab10(np.linspace(0, 0.7, n_con))
    for ci, concept in enumerate(all_concepts):
        vals = [per_class_concept_count.get(cls, {}).get(concept, 0) for cls in ordered_classes]
        ax2.bar(x + ci * w - 0.375 + w / 2, vals, width=w,
                label=_short(concept, 18), color=colors[ci])
    ax2.set_xticks(x)
    ax2.set_xticklabels([_short(c) for c in ordered_classes], fontsize=6, rotation=35, ha="right")
    ax2.set_ylabel("Count")
    ax2.set_title("Per-Class Concept Image Counts", fontsize=8)
    ax2.legend(fontsize=6, loc="upper right")
    # shade non-target classes
    for xi, t in enumerate(is_target):
        if not t:
            ax2.axvspan(xi - 0.4, xi + 0.4, color="grey", alpha=0.12, zorder=0)

    # ── Panel 3: Distribution by exact concept count ────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ks = sorted(global_combo_dist.keys())
    totals = [sum(global_combo_dist[k].values()) for k in ks]
    bars = ax3.bar([str(k) for k in ks], totals, color="steelblue", edgecolor="white")
    ax3.set_xlabel("# concepts on image")
    ax3.set_ylabel("Image count")
    ax3.set_title("Distribution by Concept Count\nper Image (all classes)", fontsize=8)
    for bar, t in zip(bars, totals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(t), ha="center", va="bottom", fontsize=9)

    # ── Panel 4: Stacked bar — exact combos per class ──────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    all_combos = set()
    for d in per_class_combo_count.values():
        all_combos.update(d.keys())
    all_combos = sorted(all_combos, key=lambda s: (len(s), sorted(s)))
    combo_colors = plt.cm.tab20(np.linspace(0, 1, max(len(all_combos), 1)))

    bottoms = np.zeros(n_cls)
    for ci, combo in enumerate(all_combos):
        vals = np.array(
            [per_class_combo_count.get(cls, {}).get(combo, 0) for cls in ordered_classes],
            dtype=float,
        )
        label = "+".join(sorted(combo)).replace("_", " ")
        ax4.bar(range(n_cls), vals, bottom=bottoms, label=label, color=combo_colors[ci])
        bottoms += vals

    ax4.set_xticks(range(n_cls))
    ax4.set_xticklabels(
        [("★" if t else "") + _short(c) for c, t in zip(ordered_classes, is_target)],
        fontsize=6, rotation=35, ha="right",
    )
    ax4.set_ylabel("Image count")
    ax4.set_title("Exact Concept Combo Distribution per Class  (★ = target)", fontsize=8)
    ax4.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    for xi, t in enumerate(is_target):
        if not t:
            ax4.axvspan(xi - 0.4, xi + 0.4, color="grey", alpha=0.12, zorder=0)

    # ── Panel 5: Concept co-occurrence heatmap (at-least) ──────────────
    ax5 = fig.add_subplot(gs[1, 2])
    co = np.zeros((n_con, n_con))
    for ci, c1 in enumerate(all_concepts):
        co[ci, ci] = at_least_combo[1].get(frozenset([c1]), 0)
        for cj, c2 in enumerate(all_concepts):
            if ci < cj:
                val = at_least_combo[2].get(frozenset([c1, c2]), 0)
                co[ci, cj] = val
                co[cj, ci] = val

    im5 = ax5.imshow(co, cmap="Blues")
    short_con = [_short(c) for c in all_concepts]
    ax5.set_xticks(range(n_con))
    ax5.set_xticklabels(short_con, fontsize=7, rotation=30, ha="right")
    ax5.set_yticks(range(n_con))
    ax5.set_yticklabels(short_con, fontsize=7)
    ax5.set_title("Concept Co-occurrence\n(diag=single, off-diag=pair, at-least)", fontsize=8)
    plt.colorbar(im5, ax=ax5, shrink=0.75)
    vmax5 = co.max() if co.max() > 0 else 1
    for ci in range(n_con):
        for cj in range(n_con):
            ax5.text(cj, ci, str(int(co[ci, cj])), ha="center", va="center",
                     fontsize=7, color="white" if co[ci, cj] > 0.6 * vmax5 else "black")

    out_path = os.path.join(output_dir, f"{group_name}_concept_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def print_summary(group_name, class_data, all_concepts, target_classes,
                  per_class_concept_count, global_combo_dist):
    target = sorted(target_classes & set(class_data.keys()))
    spurious = sorted(set(class_data.keys()) - target_classes)

    print(f"\n{'='*60}")
    print(f"  Group: {group_name.upper()}  |  Concepts: {', '.join(all_concepts)}")
    print(f"{'='*60}")
    print(f"  {'Class':<30} {'Total':>6}  " + "  ".join(f"{c[:10]:>10}" for c in all_concepts))
    print(f"  {'-'*30} {'-'*6}  " + "  ".join("-"*10 for _ in all_concepts))

    for cls in target + spurious:
        marker = "★" if cls in target_classes else " "
        total = len(class_data.get(cls, []))
        counts = per_class_concept_count.get(cls, {})
        con_str = "  ".join(f"{counts.get(c, 0):>10}" for c in all_concepts)
        print(f"  {marker} {cls:<29} {total:>6}  {con_str}")

    print(f"\n  Combo-size distribution:")
    for k in sorted(global_combo_dist.keys()):
        n = sum(global_combo_dist[k].values())
        print(f"    exactly {k} concept(s): {n} images")


def main():
    parser = argparse.ArgumentParser(description="Analyze concept distribution across MiniImageNet groups.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for output figures and CSV files")
    parser.add_argument("--groups", nargs="+", default=list(GROUP_CONFIGS.keys()),
                        choices=list(GROUP_CONFIGS.keys()),
                        help="Which groups to analyze (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    synset_to_name = load_labels()

    for group_name in args.groups:
        config = GROUP_CONFIGS[group_name]
        class_data, all_concepts, target_classes, spurious_classes = \
            analyze_group(group_name, config, synset_to_name)

        per_class_concept_count, per_class_combo_count, global_combo_dist, at_least_combo = \
            compute_stats(class_data, all_concepts)

        print_summary(group_name, class_data, all_concepts, target_classes,
                      per_class_concept_count, global_combo_dist)

        if spurious_classes:
            print(f"\n  Non-target classes with concepts: {', '.join(sorted(spurious_classes))}")

        save_stats(group_name, class_data, all_concepts, target_classes,
                   per_class_concept_count, per_class_combo_count,
                   global_combo_dist, at_least_combo, args.output_dir)

        visualize_group(group_name, class_data, all_concepts, target_classes,
                        per_class_concept_count, per_class_combo_count,
                        global_combo_dist, at_least_combo, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
