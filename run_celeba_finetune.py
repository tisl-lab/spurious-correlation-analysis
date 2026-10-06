#!/usr/bin/env python
"""
Biased fine-tuning of CLIP on CelebA -- the run-dir producer for the 4-stage
pipeline, mirroring run_waterbirds_msae.py.

The fine-tuning set is the two ALIGNED groups only -- blond women and
not-blond men, --max_samples of each -- so the model learns the two-way
shortcut female ⇒ blond / male ⇒ not blond. Evaluated zero-shot and after
fine-tuning on the full test split (19,962 images, 4 groups).

Writes, exactly as the waterbirds producer does:

    <output_dir>/clip_ft_BIASED_<max_samples>_<timestamp>/
        model.pt                  fine-tuned CLIP + metadata (ft_tag, max_samples,
                                  ft_epochs, ft_lr, seed, clip_model, dataset)
        config.json               the same metadata
        ft_train_manifest.csv     the biased fine-tuning set
        test_manifest.csv         the test split
        train_all_manifest.csv    the full training split (reference)
        results.json              zero-shot vs fine-tuned per-group accuracy

which is what pipeline_1_setup.py consumes via --run_dir.

Usage:
    python run_celeba_finetune.py --max_samples 400 --ft_epochs 3
    python run_celeba_finetune.py --max_samples 400 --ft_epochs 3 --skip_zeroshot_eval
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from clip_zero_shot import CLIPZeroShot
from datasets import CelebA
from datasets.celeba import (ALIGNED_GROUPS, MISALIGNED_GROUPS, GROUP_NAMES, CLASS_NAMES,
                             save_dataset_manifest)
from run_celeba_zeroshot import report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip_model", default="ViT-B/32",
                   choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    p.add_argument("--clip_weights_dir", default="hf_models", help="offline weights dir (None = download)")
    p.add_argument("--data_dir", default="data", help="parent of celeba/")
    p.add_argument("--output_dir", default="results/celeba")
    p.add_argument("--max_samples", type=int, default=400,
                   help="fine-tuning images PER CLASS, each drawn from its aligned group only")
    p.add_argument("--ft_epochs", type=int, default=3)
    p.add_argument("--ft_lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_zeroshot_eval", action="store_true",
                   help="skip the zero-shot pass over the test split (results/celeba already has it)")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    print("═" * 65 + "\n  CelebA biased fine-tuning\n" + "═" * 65)
    print(f"  model {args.clip_model} | device {device} | {args.max_samples}/class | "
          f"{args.ft_epochs} epoch(s) | lr {args.ft_lr} | seed {args.seed}")

    # ── datasets ─────────────────────────────────────────────────────────────
    train_ds = CelebA(root=args.data_dir, split=0)                                    # reference only
    ft_ds    = CelebA(root=args.data_dir, split=0, group_filter=ALIGNED_GROUPS,
                      max_samples=args.max_samples, seed=args.seed)
    test_ds  = CelebA(root=args.data_dir, split=2)
    print(f"\n  Fine-tuning set (aligned groups only): {ft_ds.group_counts()}  "
          f"= {', '.join(GROUP_NAMES[g] for g in sorted(ft_ds.group_counts()))}")
    print(f"  Test set: {test_ds.group_counts()}")

    def load_clip():
        if args.clip_weights_dir:
            return CLIPZeroShot.load_offline(args.clip_model, weights_dir=args.clip_weights_dir, device=device)
        return CLIPZeroShot(model_name=args.clip_model, device=device)

    clip_ft = load_clip()

    # ── zero-shot baseline ───────────────────────────────────────────────────
    zs_stats = None
    if not args.skip_zeroshot_eval:
        print("\n" + "─" * 65 + "\n  ZERO-SHOT EVALUATION (test split)\n" + "─" * 65)
        res = clip_ft.run(dataset=test_ds, prompt_mode="shape", dataset_name="celeba",
                          batch_size=args.batch_size)
        zs_stats = report(res["predictions_shape"], res["true_labels"], res["group_ids"], "zero-shot / test")

    # ── fine-tune ────────────────────────────────────────────────────────────
    ft_tag = "BIASED"
    print("\n" + "─" * 65 + f"\n  FINE-TUNING ({ft_tag}: blond women + not-blond men)\n" + "─" * 65)
    t0 = time.time()
    clip_ft.fine_tune(dataset=ft_ds, dataset_name="celeba", epochs=args.ft_epochs,
                      lr=args.ft_lr, batch_size=args.batch_size)
    print(f"  fine-tuned in {time.time() - t0:.0f}s")

    # ── run dir: model + config + manifests ─────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"clip_ft_{ft_tag}_{args.max_samples}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    metadata = {
        "dataset":     "celeba",
        "clip_model":  args.clip_model,
        "ft_tag":      ft_tag,
        "max_samples": args.max_samples,
        "ft_epochs":   args.ft_epochs,
        "ft_lr":       args.ft_lr,
        "seed":        args.seed,
        "biased_ft":   True,
        "bias_balanced_ft": None,
        "ft_groups":   sorted(ALIGNED_GROUPS),
        "target_attr": ft_ds.target_attr, "spurious_attr": ft_ds.spurious_attr,
        "data_dir":    os.path.abspath(args.data_dir),
        "run_dir":     os.path.abspath(run_dir),
    }
    clip_ft.save_model(os.path.join(run_dir, "model.pt"), metadata=metadata)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    save_dataset_manifest(train_ds, os.path.join(run_dir, "train_all_manifest.csv"), "train")
    save_dataset_manifest(ft_ds,    os.path.join(run_dir, "ft_train_manifest.csv"),  "ft_train")
    save_dataset_manifest(test_ds,  os.path.join(run_dir, "test_manifest.csv"),      "test")
    print(f"\n  Run folder: {os.path.abspath(run_dir)}")

    # ── fine-tuned evaluation ────────────────────────────────────────────────
    print("\n" + "─" * 65 + f"\n  FINE-TUNED EVALUATION ({ft_tag}, test split)\n" + "─" * 65)
    res = clip_ft.run(dataset=test_ds, prompt_mode="shape", dataset_name="celeba",
                      batch_size=args.batch_size)
    ft_stats = report(res["predictions_shape"], res["true_labels"], res["group_ids"], f"{ft_tag} fine-tuned / test")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 65 + "\n  SPURIOUS CORRELATION SUMMARY\n" + "═" * 65)
    for name, st in [("ZS", zs_stats), ("FT", ft_stats)]:
        if st is None:
            continue
        pg = st["per_group"]
        print(f"  [{name}] not blond: male(aligned) − female(misaligned) = {pg[1] - pg[0]:+.1f}  |  "
              f"blond: female(aligned) − male(misaligned) = {pg[2] - pg[3]:+.1f}  |  "
              f"worst {st['worst_group']:.1f}%  avg {st['overall']:.1f}%")
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"zero_shot": zs_stats, "fine_tuned": ft_stats, "metadata": metadata}, f, indent=2)
    print(f"\n  results.json -> {run_dir}\n  total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
