#!/usr/bin/env python
"""
Fine-tune CLIP on any registered dataset -- the single run-dir producer for
the 4-stage pipeline. Replaces the per-dataset scripts
(run_waterbirds_msae.py, run_celeba_finetune.py, run_spawrious.py) with one
driver plus dataset_settings.py, which holds everything that differs.

    python run_finetune.py --dataset waterbirds --ft_mode biased --max_samples 400
    python run_finetune.py --dataset celeba     --ft_mode biased --max_samples 400
    python run_finetune.py --dataset spawrious224 --ft_mode bg_biased --max_samples 2000

FINE-TUNING MODES (a dataset supports the subset listed in its spec):

  biased     aligned groups only -- the model sees only class/attribute
             combinations that follow the shortcut, which exaggerates it.
             Waterbirds: landbird/land + waterbird/water. CelebA: blond
             women + not-blond men.
  balanced   max_samples per GROUP rather than per class, so every
             class x attribute cell is equally represented.
  full       the training split as loaded, no group filtering.
  bg_biased  spawrious only: the dataset's own bg_biased_split() builds the
             training set (each breed's assigned backgrounds only) AND a
             background-balanced test set from what is left.

  --from_checkpoint <model.pt> continues fine-tuning an existing model
  instead of starting from base CLIP -- the two-stage BALANCED->BIASED run
  (run_waterbirds_msae's --bias_balanced_ft); the baseline evaluation then
  reports that checkpoint rather than zero-shot, and ft_tag becomes
  BIASEDfromBALANCED.

OUTPUT -- the run-dir contract pipeline_1_setup.py consumes:

    <output_dir>/clip_ft_<TAG>_<max_samples>_<timestamp>/
        model.pt                fine-tuned weights + metadata
        config.json             the same metadata
        ft_train_manifest.csv   the fine-tuning subset
        test_manifest.csv       the evaluation split
        train_all_manifest.csv  the full training split, for reference
        results.json            baseline vs fine-tuned, per class / group
        results.csv             per-group baseline / fine-tuned / delta
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

import dataset_settings as DS


def build_parser():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dataset", default="celeba", choices=sorted(DS.REGISTRY))
    known, _ = pre.parse_known_args()
    spec = DS.get(known.dataset)

    p = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip_model", default=spec.default_clip_model,
                   choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    p.add_argument("--clip_weights_dir", default=None,
                   help="offline CLIP weights (download_hf_models.py); required on servers "
                        "without internet. Under SLURM defaults to ~/hf_models.")
    p.add_argument("--data_dir", default=None, help=f"default: {spec.default_data_dir}")
    p.add_argument("--output_dir", default=None, help=f"default: {spec.default_output_dir}")
    p.add_argument("--ft_mode", default=spec.ft_modes[0], choices=spec.ft_modes,
                   help=f"{spec.name} supports: {', '.join(spec.ft_modes)} (default {spec.ft_modes[0]})")
    p.add_argument("--max_samples", type=int, default=spec.default_max_samples,
                   help="images per class (per GROUP when --ft_mode balanced)")
    p.add_argument("--ft_epochs", type=int, default=3)
    p.add_argument("--ft_lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from_checkpoint", default=None,
                   help="continue from a saved model.pt (two-stage BALANCED->BIASED)")
    p.add_argument("--skip_baseline_eval", action="store_true",
                   help="skip the pre-fine-tuning pass over the test split")
    p.add_argument("--no_plot", action="store_true", help="skip the summary figure")
    for flag, kw in spec.extra_args.items():
        p.add_argument(flag, **kw)
    return p, spec


def device_check():
    missing = [m for m in ("torch", "clip", "numpy", "pandas", "PIL.Image")
               if not _importable(m)]
    if missing:
        sys.exit(f"  Missing packages: {missing}")
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _importable(mod):
    try:
        __import__(mod); return True
    except ImportError:
        return False


def group_counts(ds):
    c = defaultdict(int)
    for *_, gid in ds.samples:
        c[gid] += 1
    return dict(sorted(c.items()))


def save_manifest(spec, dataset, path, split_name):
    """The manifest columns every pipeline stage reads -- identical to
    run_waterbirds_msae.save_dataset_manifest's."""
    import pandas as pd
    rows = [{
        "img_path": img_path, "label": label, "class": spec.class_names[label],
        "bg": attr_name, "group_id": gid, "group": spec.group_names[gid],
        "aligned": spec.is_aligned(gid), "split": split_name,
    } for img_path, label, attr_name, gid in dataset.samples]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(rows)} images)")


def evaluate(spec, results, label=""):
    """Per-class, per-group, overall and worst-group accuracy -- the report
    run_waterbirds_msae.evaluate_waterbirds prints, driven by the spec."""
    preds, trues, groups = (np.asarray(results["predictions_shape"]),
                            np.asarray(results["true_labels"]), np.asarray(results["group_ids"]))
    n, corr = len(trues), int((preds == trues).sum())
    overall = corr / n * 100 if n else 0.0
    print(f"\n{'═' * 70}\n  [{label}] OVERALL: {corr}/{n} = {overall:.2f}%\n{'═' * 70}")

    w = max(len(c) for c in spec.class_names) + 2
    print(f"\n  {'Class':<{w}} {'Correct':>8} {'Total':>8} {'Acc%':>8}\n  " + "─" * (w + 27))
    per_class = []
    for ci, cls in enumerate(spec.class_names):
        m = trues == ci; tot = int(m.sum()); c = int((preds[m] == ci).sum())
        acc = c / tot * 100 if tot else float("nan")
        per_class.append({"class": cls, "correct": c, "total": tot, "acc": acc})
        print(f"  {cls:<{w}} {c:>8} {tot:>8} {acc:>7.1f}%")

    gw = max(len(g) for g in spec.group_names) + 2
    print(f"\n  {'Group':<{gw}} {'Correct':>8} {'Total':>8} {'Acc%':>8}  Note\n  " + "─" * (gw + 40))
    per_group = []
    for gid, gname in enumerate(spec.group_names):
        m = groups == gid; tot = int(m.sum()); c = int((preds[m] == trues[m]).sum())
        acc = c / tot * 100 if tot else float("nan")
        per_group.append({"group_id": gid, "group": gname, "correct": c, "total": tot,
                          "acc": acc, "aligned": spec.is_aligned(gid)})
        note = "← aligned" if spec.is_aligned(gid) else "← misaligned"
        acc_s = f"{acc:>7.1f}%" if tot else "      N/A"
        print(f"  {gname:<{gw}} {c:>8} {tot:>8} {acc_s}  {note}")

    valid = [g for g in per_group if g["total"] > 0 and not np.isnan(g["acc"])]
    worst = min(valid, key=lambda g: g["acc"]) if valid else None
    if worst:
        print(f"\n  Worst-group accuracy: {worst['acc']:.2f}%   ({worst['group']}, n={worst['total']})")
    return {"overall": {"acc": overall, "correct": corr, "total": n},
            "per_class": per_class, "per_group": per_group,
            "worst_group_acc": worst["acc"] if worst else float("nan"),
            "worst_group": worst["group"] if worst else None,
            "class_names": spec.class_names}


def spurious_gaps(spec, per_group):
    """aligned − misaligned accuracy per class: how much the model leans on
    the shortcut. Generalises run_waterbirds_msae's landbird/waterbird gaps."""
    by_gid = {g["group_id"]: g for g in per_group}
    gaps = {}
    for gid in spec.misaligned_groups():
        c = spec.contrast_group(gid)
        if c is None or by_gid[gid]["total"] == 0 or by_gid[c]["total"] == 0:
            continue
        gaps[spec.group_names[gid]] = round(by_gid[c]["acc"] - by_gid[gid]["acc"], 2)
    return gaps


def save_results(spec, base_stats, ft_stats, run_dir, metadata, base_label):
    import pandas as pd
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"metadata": metadata, "baseline_label": base_label,
                   "baseline": base_stats, "fine_tuned": ft_stats,
                   "spurious_gaps": {
                       "baseline": spurious_gaps(spec, base_stats["per_group"]) if base_stats else None,
                       "fine_tuned": spurious_gaps(spec, ft_stats["per_group"])}}, f, indent=2)
    ftg = {g["group_id"]: g for g in ft_stats["per_group"]}
    bsg = {g["group_id"]: g for g in base_stats["per_group"]} if base_stats else {}
    rows = [{"group_id": gid, "group": name, "aligned": spec.is_aligned(gid),
             "n": ftg[gid]["total"],
             "baseline_acc": round(bsg[gid]["acc"], 2) if bsg else "",
             "ft_acc": round(ftg[gid]["acc"], 2),
             "delta_acc": round(ftg[gid]["acc"] - bsg[gid]["acc"], 2) if bsg else ""}
            for gid, name in enumerate(spec.group_names)]
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "results.csv"), index=False)
    print(f"  Saved: {os.path.join(run_dir, 'results.csv')}")


def plot_summary(spec, base_stats, ft_stats, run_dir, base_label):
    """Per-group accuracy before/after + the per-class spurious gap."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gids = list(range(len(spec.group_names)))
    ft = [g["acc"] for g in ft_stats["per_group"]]
    bs = [g["acc"] for g in base_stats["per_group"]] if base_stats else None
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(gids) * 0.8), 4.2))
    x = np.arange(len(gids)); wdt = 0.38
    if bs: axes[0].bar(x - wdt/2, bs, wdt, label=base_label, color="#8aa0b8")
    axes[0].bar(x + (wdt/2 if bs else 0), ft, wdt, label="fine-tuned", color="#b8553f")
    axes[0].set_xticks(x); axes[0].set_xticklabels(spec.group_names, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("accuracy %"); axes[0].set_ylim(0, 100); axes[0].legend(fontsize=8)
    axes[0].set_title(f"{spec.name}: per-group accuracy", fontsize=10)
    for gid in gids:
        axes[0].get_xticklabels()[gid].set_color("#2f7d4f" if spec.is_aligned(gid) else "#b4433a")
    gaps = spurious_gaps(spec, ft_stats["per_group"])
    if gaps:
        axes[1].barh(list(gaps), list(gaps.values()), color="#b8553f")
        axes[1].set_xlabel("aligned − misaligned (pp)"); axes[1].axvline(0, color="k", lw=0.8)
        axes[1].set_title("spurious gap after fine-tuning", fontsize=10)
        axes[1].tick_params(labelsize=7)
    plt.tight_layout()
    path = os.path.join(run_dir, "summary.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser, spec = build_parser()
    args = parser.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = device_check()
    data_dir, output_dir, weights_dir = DS.resolve_paths(
        spec, args.data_dir, args.output_dir, args.clip_weights_dir)

    print("\n" + "═" * 70)
    print(f"  CLIP × {spec.name} — spurious-correlation fine-tuning")
    print("═" * 70)
    print(f"  model {args.clip_model} | device {device} | ft_mode {args.ft_mode} | "
          f"{args.max_samples}/class | {args.ft_epochs} epoch(s) | lr {args.ft_lr} | seed {args.seed}")
    print(f"  data {data_dir}\n  out  {output_dir}")
    if spec.notes: print(f"  note: {spec.notes}")
    print("═" * 70)

    spec.register_prompts()
    from clip_zero_shot import CLIPZeroShot

    extra = {k: getattr(args, k) for k in ("folder",) if hasattr(args, k)}

    # ── splits + fine-tuning subset ──────────────────────────────────────────
    if args.ft_mode == "bg_biased":
        # The dataset builds both splits itself (spawrious).
        full = spec.load_split(data_dir, split=None, seed=args.seed, **extra)
        ft_ds, test_ds = spec.custom_split(full, n_per_class=args.max_samples, seed=args.seed)
        train_ds = full
    else:
        train_ds = spec.load_split(data_dir, split=0, seed=args.seed, **extra)
        test_ds = spec.load_split(data_dir, split=2, seed=args.seed, **extra)
        if args.ft_mode == "biased":
            ft_ds = spec.load_split(data_dir, split=0, group_filter=spec.aligned_groups,
                                    max_samples=args.max_samples, seed=args.seed, **extra)
        elif args.ft_mode == "balanced":
            ft_ds = spec.load_split(data_dir, split=0, group_filter=list(range(len(spec.group_names))),
                                    max_samples=args.max_samples, seed=args.seed, balanced=True, **extra)
        else:
            ft_ds = spec.load_split(data_dir, split=0, max_samples=args.max_samples,
                                    seed=args.seed, **extra)

    for nm, ds in (("train", train_ds), ("ft", ft_ds), ("test", test_ds)):
        cc = group_counts(ds)
        print(f"  {nm:<5} {len(ds):>6} images : " +
              "  |  ".join(f"{spec.group_names[g]}: {n}" for g, n in cc.items()))
    if len(ft_ds) == 0:
        sys.exit("  Fine-tuning set is empty — check --data_dir / --ft_mode.")

    # ── baseline ─────────────────────────────────────────────────────────────
    if args.from_checkpoint:
        print(f"\n  Continuing from checkpoint: {args.from_checkpoint}")
        clip_ft = CLIPZeroShot.load_model(args.from_checkpoint, device=device)
        base_label = "checkpoint (pre-fine-tuning)"
        ft_tag = {"biased": "BIASEDfromBALANCED"}.get(args.ft_mode, args.ft_mode.upper())
    else:
        clip_ft = (CLIPZeroShot.load_offline(args.clip_model, weights_dir=weights_dir, device=device)
                   if weights_dir else CLIPZeroShot(model_name=args.clip_model, device=device))
        base_label = "zero-shot"
        ft_tag = {"biased": "BIASED", "bg_biased": "BGBIASED",
                  "balanced": "BALANCED", "full": "FULL"}[args.ft_mode]

    base_stats = None
    if not args.skip_baseline_eval:
        print("\n" + "─" * 70 + f"\n  BASELINE EVALUATION ({base_label}, test split)\n" + "─" * 70)
        res = clip_ft.run(dataset=test_ds, prompt_mode="shape", dataset_name=spec.name,
                          batch_size=args.batch_size)
        base_stats = evaluate(spec, res, f"{base_label} / test")

    # ── fine-tune ────────────────────────────────────────────────────────────
    print("\n" + "─" * 70 + f"\n  FINE-TUNING ({ft_tag}, {len(ft_ds)} images)\n" + "─" * 70)
    t0 = time.time()
    clip_ft.fine_tune(dataset=ft_ds, dataset_name=spec.name, epochs=args.ft_epochs,
                      lr=args.ft_lr, batch_size=args.batch_size)
    print(f"  fine-tuned in {time.time() - t0:.0f}s")

    # ── run dir ──────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"clip_ft_{ft_tag}_{args.max_samples}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    metadata = {
        "dataset": spec.name, "clip_model": args.clip_model, "ft_tag": ft_tag,
        "ft_mode": args.ft_mode, "max_samples": args.max_samples,
        "ft_epochs": args.ft_epochs, "ft_lr": args.ft_lr, "seed": args.seed,
        "biased_ft": args.ft_mode in ("biased", "bg_biased"),
        "bias_balanced_ft": args.from_checkpoint,
        "ft_groups": sorted(spec.aligned_groups) if args.ft_mode in ("biased", "bg_biased") else None,
        "n_ft_images": len(ft_ds), "n_test_images": len(test_ds),
        "data_dir": os.path.abspath(data_dir), "run_dir": os.path.abspath(run_dir),
    }
    clip_ft.save_model(os.path.join(run_dir, "model.pt"), metadata=metadata)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("\nSaving dataset manifests...")
    save_manifest(spec, train_ds, os.path.join(run_dir, "train_all_manifest.csv"), "train")
    save_manifest(spec, ft_ds, os.path.join(run_dir, "ft_train_manifest.csv"), "ft_train")
    save_manifest(spec, test_ds, os.path.join(run_dir, "test_manifest.csv"), "test")

    # ── fine-tuned evaluation + outputs ──────────────────────────────────────
    print("\n" + "─" * 70 + f"\n  FINE-TUNED EVALUATION ({ft_tag}, test split)\n" + "─" * 70)
    res = clip_ft.run(dataset=test_ds, prompt_mode="shape", dataset_name=spec.name,
                      batch_size=args.batch_size)
    ft_stats = evaluate(spec, res, f"{ft_tag} fine-tuned / test")

    print("\n" + "═" * 70 + "\n  SPURIOUS CORRELATION SUMMARY\n" + "═" * 70)
    for nm, st in (("baseline", base_stats), ("fine-tuned", ft_stats)):
        if st is None: continue
        gaps = spurious_gaps(spec, st["per_group"])
        print(f"  [{nm:<10}] overall {st['overall']['acc']:5.1f}%  worst {st['worst_group_acc']:5.1f}%  "
              f"({st['worst_group']})")
        for g, v in gaps.items():
            print(f"               aligned − {g}: {v:+.1f} pp")
    save_results(spec, base_stats, ft_stats, run_dir, metadata, base_label)
    if not args.no_plot and base_stats:
        plot_summary(spec, base_stats, ft_stats, run_dir, base_label)

    print(f"\n  Run folder: {os.path.abspath(run_dir)}")
    print(f"  Next: python pipeline_1_setup.py --dataset {spec.name} --run_dir {run_dir} ...")
    print(f"  Total {time.time() - t_start:.0f}s\n")


if __name__ == "__main__":
    main()
