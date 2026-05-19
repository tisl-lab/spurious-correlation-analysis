"""
run_experiment.py
=================

Single Entry Point for CLIP Spurious Correlation Analysis
----------------------------------------------------------

This is the ONLY file you need to run. It orchestrates:
    1. Dataset loading (Colored MNIST and/or Colored CIFAR-10)
    2. CLIP zero-shot inference (shape & color prompts)
    3. Analysis + saving results to ./results/

USAGE:
    # Run everything (both datasets)
    python run_experiment.py

    # Run only MNIST
    python run_experiment.py --dataset mnist

    # Run only CIFAR-10
    python run_experiment.py --dataset cifar10

    # Quick test with fewer samples
    python run_experiment.py --max_samples 500

    # Try both aligned and misaligned splits
    python run_experiment.py --split both

    # Use a different CLIP model
    python run_experiment.py --clip_model RN50
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Allow imports from the project root regardless of where this script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP Spurious Correlation Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "both"],
        help="Which dataset to run. Default: both",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["aligned", "misaligned", "both"],
        help=(
            "Which dataset split to evaluate.\n"
            "  aligned    : color perfectly predicts label (shortcut available)\n"
            "  misaligned : each sample appears in all colors (no shortcut)\n"
            "  both       : run both splits and compare (default)"
        ),
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"],
        help="CLIP model variant to use. Default: ViT-B/32",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of samples per dataset (useful for quick testing). Default: all",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Inference batch size. Default: 64",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results. Default: ./results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to download/cache datasets. Default: ./data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--tint_alpha",
        type=float,
        default=0.6,
        help=(
            "CIFAR-10 tint strength. 0.0 = full color overlay, 1.0 = original image. "
            "Default: 0.6"
        ),
    )
    return parser.parse_args()


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║         CLIP SPURIOUS CORRELATION ANALYSIS PIPELINE             ║
║         OpenAI CLIP (ViT-B/32) — Zero-Shot Evaluation           ║
╠══════════════════════════════════════════════════════════════════╣
║  Datasets  : Colored MNIST, Colored CIFAR-10                    ║
║  Goal      : Does CLIP use color shortcuts or true features?    ║
║  Outputs   : Per-group accuracy tables, confusion matrices      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_dependencies():
    """Check all required packages are installed before starting."""
    missing = []

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        missing.append("torch")

    try:
        import torchvision
        print(f"  ✓ torchvision {torchvision.__version__}")
    except ImportError:
        missing.append("torchvision")

    try:
        import clip
        print(f"  ✓ OpenAI CLIP")
    except ImportError:
        missing.append("clip (install: pip install git+https://github.com/openai/CLIP.git)")

    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        import pandas as pd
        print(f"  ✓ pandas {pd.__version__}")
    except ImportError:
        missing.append("pandas")

    try:
        import matplotlib
        print(f"  ✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        print(f"  ⚠ matplotlib not found — plots will be skipped")

    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print(f"  ⚠ scikit-learn not found — will use fallback confusion matrix")

    if missing:
        print(f"\n  ✗ Missing required packages:")
        for pkg in missing:
            print(f"    - {pkg}")
        print("\n  Install with: pip install -r requirements.txt")
        print("  And CLIP:     pip install git+https://github.com/openai/CLIP.git\n")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  ✓ CUDA available — GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  ⚠ CUDA not available — running on CPU (slower)")

    return device


def run_mnist_experiment(args, clip_model, device):
    """Run the full CLIP analysis on Colored MNIST."""
    from datasets import ColoredMNIST
    from clip_zero_shot import CLIPZeroShot
    from analysis import run_full_analysis

    print("\n" + "═" * 60)
    print("  DATASET: COLORED MNIST")
    print("═" * 60)

    splits_to_run = (
        ["aligned", "misaligned"] if args.split == "both" else [args.split]
    )

    all_results = {}

    for split in splits_to_run:
        print(f"\n  ── Split: {split.upper()} ──")

        # Build dataset (without clip_preprocess — will be injected in CLIPZeroShot.run)
        print(f"  Loading ColoredMNIST ({split})...")
        dataset = ColoredMNIST(
            root=args.data_dir,
            split=split,
            train=False,
            download=True,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(f"  {dataset}")

        # Show group distribution
        group_info = dataset.get_group_info()
        print(f"  Groups: {len(group_info)} (label, color) combinations")

        # Run CLIP inference
        results = clip_model.run(
            dataset=dataset,
            prompt_mode="both",
            dataset_name="mnist",
            batch_size=args.batch_size,
        )

        # Add split info to results for reporting
        results["split"] = split

        # Analyze and save
        split_output_dir = os.path.join(args.output_dir, "mnist", split)
        saved = run_full_analysis(results, output_dir=split_output_dir)
        all_results[split] = {"results": results, "saved": saved}

    # If both splits were run, print a cross-split comparison
    if args.split == "both" and "aligned" in all_results and "misaligned" in all_results:
        _print_cross_split_comparison(
            all_results["aligned"]["results"],
            all_results["misaligned"]["results"],
            dataset_name="mnist",
        )

    return all_results


def run_cifar10_experiment(args, clip_model, device):
    """Run the full CLIP analysis on Colored CIFAR-10."""
    from datasets import ColoredCIFAR10
    from clip_zero_shot import CLIPZeroShot
    from analysis import run_full_analysis

    print("\n" + "═" * 60)
    print("  DATASET: COLORED CIFAR-10")
    print("═" * 60)

    splits_to_run = (
        ["aligned", "misaligned"] if args.split == "both" else [args.split]
    )

    all_results = {}

    for split in splits_to_run:
        print(f"\n  ── Split: {split.upper()} ──")

        print(f"  Loading ColoredCIFAR10 ({split}, tint_alpha={args.tint_alpha})...")
        dataset = ColoredCIFAR10(
            root=args.data_dir,
            split=split,
            train=False,
            download=True,
            tint_alpha=args.tint_alpha,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(f"  {dataset}")

        results = clip_model.run(
            dataset=dataset,
            prompt_mode="both",
            dataset_name="cifar10",
            batch_size=args.batch_size,
        )

        results["split"] = split

        split_output_dir = os.path.join(args.output_dir, "cifar10", split)
        saved = run_full_analysis(results, output_dir=split_output_dir)
        all_results[split] = {"results": results, "saved": saved}

    if args.split == "both" and "aligned" in all_results and "misaligned" in all_results:
        _print_cross_split_comparison(
            all_results["aligned"]["results"],
            all_results["misaligned"]["results"],
            dataset_name="cifar10",
        )

    return all_results


def _print_cross_split_comparison(aligned_results, misaligned_results, dataset_name):
    """
    Print a side-by-side comparison of aligned vs misaligned split performance.
    This is the clearest view of how much CLIP relies on color shortcuts.
    """
    from analysis import compute_summary_stats

    print(f"\n{'═' * 60}")
    print(f"  CROSS-SPLIT COMPARISON — {dataset_name.upper()}")
    print(f"  (This is the key test for spurious correlation!)")
    print(f"{'═' * 60}")

    header = f"  {'Metric':<38} {'Aligned':>10} {'Misaligned':>12}"
    print(header)
    print("  " + "-" * 62)

    for prompt_key in ("shape", "color"):
        print(f"\n  [{prompt_key.upper()} PROMPTS]")
        sa = compute_summary_stats(aligned_results,    prompt_key)
        sm = compute_summary_stats(misaligned_results, prompt_key)

        metrics = [
            ("Overall Accuracy",     "overall_accuracy"),
            ("Worst-Group Accuracy", "worst_group_accuracy"),
            ("Best-Group Accuracy",  "best_group_accuracy"),
        ]
        for label, key in metrics:
            va = sa[key]
            vm = sm[key]
            va_s = f"{va:.2f}%" if isinstance(va, float) else str(va)
            vm_s = f"{vm:.2f}%" if isinstance(vm, float) else str(vm)
            diff = (va - vm) if isinstance(va, float) and isinstance(vm, float) else None
            diff_s = f"({diff:+.1f}%)" if diff is not None else ""
            print(f"  {label:<38} {va_s:>10} {vm_s:>12}  {diff_s}")

    print()
    print("  HOW TO READ THIS:")
    print("  - ALIGNED split: color matches the class label (easy shortcut available)")
    print("  - MISALIGNED split: color does NOT match (shortcut is useless)")
    print("  - Large drop from Aligned→Misaligned = CLIP is using the color shortcut")
    print("  - Small drop = CLIP is using genuine semantic features (robust!)")
    print(f"{'═' * 60}\n")


def save_experiment_config(args, output_dir: str, device: str):
    """Save experiment configuration to a human-readable text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "experiment_config.txt")

    with open(filepath, "w") as f:
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CLIP Model:  {args.clip_model}\n")
        f.write(f"Device:      {device}\n")
        f.write(f"Dataset:     {args.dataset}\n")
        f.write(f"Split:       {args.split}\n")
        f.write(f"Max Samples: {args.max_samples if args.max_samples else 'all'}\n")
        f.write(f"Batch Size:  {args.batch_size}\n")
        f.write(f"Tint Alpha:  {args.tint_alpha} (CIFAR-10 only)\n")
        f.write(f"Seed:        {args.seed}\n")
        f.write(f"Output Dir:  {args.output_dir}\n")
        f.write(f"Data Dir:    {args.data_dir}\n")

    print(f"  Config saved: {filepath}")
    return filepath


def main():
    print_banner()

    # Parse arguments
    args = parse_args()

    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── 1. Check dependencies ──
    print("Checking dependencies...")
    device = check_dependencies()

    # ── 2. Save config ──
    save_experiment_config(args, args.output_dir, device)

    # ── 3. Load CLIP once (shared across all datasets) ──
    print(f"\nLoading CLIP model: {args.clip_model}")
    from clip_zero_shot import CLIPZeroShot
    clip_model = CLIPZeroShot(model_name=args.clip_model, device=device)

    # ── 4. Run experiments ──
    t_start = time.time()
    all_experiment_results = {}

    if args.dataset in ("mnist", "both"):
        all_experiment_results["mnist"] = run_mnist_experiment(args, clip_model, device)

    if args.dataset in ("cifar10", "both"):
        all_experiment_results["cifar10"] = run_cifar10_experiment(args, clip_model, device)

    # ── 5. Final summary ──
    elapsed = time.time() - t_start
    print(f"\n{'═' * 60}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Results saved in: {os.path.abspath(args.output_dir)}")
    print(f"{'═' * 60}")
    print()
    print("  Files written:")
    for dataset_name, exp in all_experiment_results.items():
        for split, data in exp.items():
            saved = data["saved"]
            print(f"\n  [{dataset_name.upper()} — {split}]")
            print(f"    Report : {saved['report']}")
            for csv in saved["csvs"]:
                print(f"    CSV    : {csv}")
            for plot in saved["plots"]:
                print(f"    Plot   : {plot}")

    print()
    print("  To read your results:")
    print(f"    cat {args.output_dir}/mnist/aligned/*_report_*.txt")
    print()


if __name__ == "__main__":
    main()
