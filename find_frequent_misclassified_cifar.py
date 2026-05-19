"""
find_frequent_misclassified_cifar.py
=====================================
Scans all existing CIFAR-10 misclassified folders under results/all_experiments_cifar/
and identifies images that are misclassified in at least --min_runs experiment runs.

For each such image the most common predicted class across those runs is used as the
prediction label, following the same folder structure as the original misclassified
folders:
    frequently_misclassified/<true_class>/<most_common_pred_class>/img_<idx>.png

Usage:
    python find_frequent_misclassified_cifar.py
    python find_frequent_misclassified_cifar.py --min_runs 6
    python find_frequent_misclassified_cifar.py --results_dir results/all_experiments_cifar --min_runs 3
"""

import argparse
import os
import shutil
from collections import Counter, defaultdict


def collect_misclassified(results_dir):
    """
    Walk every misclassified_* folder and build a mapping:
        sample_idx -> list of (true_class, pred_class, src_path)

    Returns (mapping, total_runs)
    """
    # Each run contributes one misclassified folder
    run_dirs = []
    for root, dirs, _ in os.walk(results_dir):
        for d in dirs:
            if d.startswith("misclassified_"):
                run_dirs.append(os.path.join(root, d))

    run_dirs.sort()
    print(f"Found {len(run_dirs)} misclassified folder(s):")
    for d in run_dirs:
        print(f"  {os.path.relpath(d, results_dir)}")

    # idx -> [(true_class, pred_class, abs_path), ...]
    idx_records = defaultdict(list)

    for run_dir in run_dirs:
        for true_class in os.listdir(run_dir):
            true_path = os.path.join(run_dir, true_class)
            if not os.path.isdir(true_path):
                continue
            for pred_class in os.listdir(true_path):
                pred_path = os.path.join(true_path, pred_class)
                if not os.path.isdir(pred_path):
                    continue
                for fname in os.listdir(pred_path):
                    if not fname.endswith(".png"):
                        continue
                    # Extract numeric index from img_<idx>.png
                    stem = os.path.splitext(fname)[0]  # "img_00271"
                    if not stem.startswith("img_"):
                        continue
                    try:
                        idx = int(stem[4:])
                    except ValueError:
                        continue
                    src = os.path.join(pred_path, fname)
                    idx_records[(true_class, idx)].append((pred_class, src))

    return idx_records, len(run_dirs)


def find_frequent(idx_records, min_runs):
    """
    Return list of (true_class, idx, most_common_pred, miss_count, best_src)
    for every image misclassified in >= min_runs runs.
    """
    frequent = []
    for (true_class, idx), records in idx_records.items():
        miss_count = len(records)
        if miss_count < min_runs:
            continue
        pred_counter = Counter(pred for pred, _ in records)
        most_common_pred = pred_counter.most_common(1)[0][0]
        # Prefer a source from a run where pred == most_common_pred
        best_src = next((src for pred, src in records if pred == most_common_pred), records[0][1])
        frequent.append((true_class, idx, most_common_pred, miss_count, best_src))

    # Sort: true_class, then by miss_count descending, then by idx
    frequent.sort(key=lambda r: (r[0], -r[3], r[1]))
    return frequent


def copy_frequent(frequent, output_dir):
    """Copy images to output_dir/<true_class>/<pred_class>/img_<idx>.png."""
    os.makedirs(output_dir, exist_ok=True)
    copied = 0

    for true_class, idx, pred_class, miss_count, src in frequent:
        dest_dir = os.path.join(output_dir, true_class, pred_class)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"img_{idx:05d}.png")
        shutil.copy2(src, dest)
        copied += 1

    return copied


def print_summary(frequent, min_runs, total_runs, output_dir):
    print(f"\n{'='*65}")
    print(f"  Frequently Misclassified Images  (threshold: ≥ {min_runs}/{total_runs} runs)")
    print(f"{'='*65}")
    print(f"  {'True Class':<14} {'Pred Class':<14} {'idx':>6}  {'Runs':>5}")
    print(f"  {'-'*58}")

    for true_class, idx, pred_class, miss_count, _ in frequent:
        print(f"  {true_class:<14} {pred_class:<14} {idx:>6}  {miss_count:>5}")

    print(f"  {'-'*58}")

    # Per-class summary
    from collections import Counter as C
    by_class = C(true_class for true_class, *_ in frequent)
    print(f"\n  Per-class count:")
    for cls, count in sorted(by_class.items()):
        print(f"    {cls:<14}  {count}")

    print(f"\n  Total: {len(frequent)} image(s)")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*65}\n")


def main(args):
    idx_records, total_runs = collect_misclassified(args.results_dir)
    print(f"\nTotal unique (class, sample) pairs seen as misclassified: {len(idx_records)}")

    frequent = find_frequent(idx_records, args.min_runs)
    print(f"Images misclassified in ≥ {args.min_runs}/{total_runs} runs: {len(frequent)}")

    if not frequent:
        print("No images meet the threshold. Try lowering --min_runs.")
        return

    output_dir = os.path.join(args.results_dir, "frequently_misclassified")
    copied = copy_frequent(frequent, output_dir)
    print(f"Copied {copied} image(s).")

    print_summary(frequent, args.min_runs, total_runs, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results/all_experiments_cifar",
        help="Root directory containing the per-run misclassified folders",
    )
    parser.add_argument(
        "--min_runs",
        type=int,
        default=2,
        help="Minimum number of runs in which an image must be misclassified (default: 2)",
    )
    args = parser.parse_args()
    main(args)
