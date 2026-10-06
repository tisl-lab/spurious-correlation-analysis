#!/usr/bin/env python
"""
Zero-shot CLIP on CelebA (Blond_Hair x Male): per-class, per-group and
worst-group accuracy, plus how well the spurious attribute (gender) is
recognised and whether the hair prediction leans on it.

Mirrors run_waterbirds_msae.evaluate_waterbirds' report, using the group
definitions in datasets/celeba.py and the prompts in
clip_zero_shot.PROMPT_SETS["celeba"]. Image features are cached per
(model, split) under results/celeba/, so a re-run with different prompts
scores in seconds.

Usage:
    python run_celeba_zeroshot.py                          # test split
    python run_celeba_zeroshot.py --split 1 --max_samples 500
    python run_celeba_zeroshot.py --clip_model ViT-L/14
Outputs: results/celeba/zeroshot_<model>_<split>[_n<max>].{json,csv}
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from clip_zero_shot import CLIPZeroShot, PROMPT_SETS
from datasets import CelebA
from datasets.celeba import CLASS_NAMES, ATTR_NAMES, GROUP_NAMES, ALIGNED_GROUPS

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data", help="parent of celeba/")
    p.add_argument("--split", type=int, default=2, help="0 train, 1 val, 2 test")
    p.add_argument("--max_samples", type=int, default=None, help="cap per class (quick runs)")
    p.add_argument("--clip_model", default="ViT-B/32",
                   choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    p.add_argument("--hf_model_dir", default=None, help="offline weights dir, if any")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out_dir", default="results/celeba")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def encode_images(clip_ft, dataset, batch_size, device):
    """L2-normalised CLIP image features for every sample, in dataset order."""
    from torch.utils.data import DataLoader
    dataset.clip_preprocess = clip_ft.preprocess
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    for images, _, _, _ in tqdm(loader, desc="  Encoding images"):
        f = clip_ft.model.encode_image(images.to(device)).float()
        feats.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(feats)


def encode_prompts(clip_ft, prompts):
    """Same rule CLIPZeroShot.run uses: lists -> multi-prompt averaging."""
    if isinstance(next(iter(prompts.values())), list):
        return clip_ft._encode_text_prompts_multi(prompts).float().cpu()
    return clip_ft.encode_text_prompts(prompts).float().cpu()


def predict(img_feats, txt_feats):
    return (100.0 * img_feats @ txt_feats.T).argmax(dim=-1).numpy()


def report(preds, trues, groups, label):
    n = len(trues); overall = (preds == trues).mean() * 100
    print(f"\n{'═' * 66}\n  [{label}] OVERALL: {int((preds == trues).sum())}/{n} = {overall:.2f}%\n{'═' * 66}")
    print(f"\n  {'Class':<22} {'Correct':>8} {'Total':>8} {'Acc%':>8}\n  " + "─" * 50)
    per_class = {}
    for ci, cls in enumerate(CLASS_NAMES):
        m = trues == ci; tot = int(m.sum()); corr = int((preds[m] == ci).sum())
        per_class[cls] = corr / tot * 100 if tot else float("nan")
        print(f"  {cls:<22} {corr:>8} {tot:>8} {per_class[cls]:>7.1f}%")
    print(f"\n  {'Group':<24} {'Correct':>8} {'Total':>8} {'Acc%':>8}  Note\n  " + "─" * 66)
    per_group = {}
    for gid, gname in enumerate(GROUP_NAMES):
        m = groups == gid; tot = int(m.sum()); corr = int((preds[m] == trues[m]).sum())
        acc = corr / tot * 100 if tot else float("nan")
        per_group[gid] = acc
        note = "← aligned" if gid in ALIGNED_GROUPS else "← misaligned (rare)"
        print(f"  {gname:<24} {corr:>8} {tot:>8} {acc:>7.1f}%  {note}")
    valid = [a for a in per_group.values() if not np.isnan(a)]
    worst = min(valid) if valid else float("nan")
    print(f"\n  Worst-group accuracy: {worst:.2f}%   (group {min(per_group, key=lambda g: per_group[g])}: "
          f"{GROUP_NAMES[min(per_group, key=lambda g: per_group[g])]})")
    return dict(overall=overall, per_class=per_class, per_group=per_group, worst_group=worst)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if args.hf_model_dir:
        clip_ft = CLIPZeroShot.load_offline(args.clip_model, weights_dir=args.hf_model_dir, device=device)
    else:
        clip_ft = CLIPZeroShot(model_name=args.clip_model, device=device)

    ds = CelebA(root=args.data_dir, split=args.split, max_samples=args.max_samples, seed=args.seed)
    print(f"\n{ds}")
    trues  = np.array([s[1] for s in ds.samples])
    groups = np.array([s[3] for s in ds.samples])
    attrs  = groups % 2                                     # the spurious attribute (male)

    # Image features are cached per (model, split, subsample): encoding 20k
    # images is the only real cost here, and prompt wording is what gets
    # revisited -- with the cache, a new wording is scored in seconds.
    os.makedirs(args.out_dir, exist_ok=True)
    feat_tag = f"{args.clip_model.replace('/', '~')}_split{args.split}" + (f"_n{args.max_samples}_s{args.seed}" if args.max_samples else "")
    feat_path = os.path.join(args.out_dir, f"image_features_{feat_tag}.pt")
    if os.path.isfile(feat_path):
        cached = torch.load(feat_path)
        if cached["paths"] == [s[0] for s in ds.samples]:
            img = cached["feats"]; print(f"  Reusing cached image features: {feat_path}")
        else:
            img = None
    else:
        img = None
    if img is None:
        img = encode_images(clip_ft, ds, args.batch_size, device)
        torch.save({"feats": img, "paths": [s[0] for s in ds.samples]}, feat_path)
        print(f"  Image features cached -> {feat_path}")

    results = {"clip_model": args.clip_model, "split": args.split, "n": len(ds),
               "group_counts": ds.group_counts()}

    # ── the spurious attribute: how recognisable is gender zero-shot? ──────
    gpred = predict(img, encode_prompts(clip_ft, PROMPT_SETS["celeba"]["color"]))
    gacc = (gpred == attrs).mean() * 100
    print(f"\n  Spurious attribute (gender) zero-shot accuracy: {gacc:.2f}%  "
          f"(prompts: {PROMPT_SETS['celeba']['color']})")
    results["gender_accuracy"] = gacc

    # ── hair colour ───────────────────────────────────────────────────────
    prompts = PROMPT_SETS["celeba"]["shape"]
    preds = predict(img, encode_prompts(clip_ft, prompts))
    r = report(preds, trues, groups, "hair")
    # does the hair prediction lean on gender?  P(pred blond | female) vs P(pred blond | male),
    # within each TRUE class -- a gap here is the zero-shot model's own shortcut
    lean = {}
    for ci, cls in enumerate(CLASS_NAMES):
        m = trues == ci
        lean[cls] = {ATTR_NAMES[a]: float((preds[m & (attrs == a)] == 1).mean() * 100)
                     if (m & (attrs == a)).any() else float("nan") for a in (0, 1)}
    print("\n  P(predicted blond) by true class x gender:")
    for cls, d in lean.items():
        print(f"    {cls:<10} female {d['female']:6.1f}%   male {d['male']:6.1f}%   "
              f"gap {d['female'] - d['male']:+.1f}")
    r["p_pred_blond"] = lean; r["prompts"] = prompts
    results["hair"] = r
    print(f"\n  prompts: {prompts}")

    tag = f"{args.clip_model.replace('/', '~')}_split{args.split}" + (f"_n{args.max_samples}" if args.max_samples else "")
    with open(os.path.join(args.out_dir, f"zeroshot_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    import csv
    with open(os.path.join(args.out_dir, f"zeroshot_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["overall", "worst_group"] + [f"acc_group_{g}" for g in range(4)] + ["gender_accuracy"])
        w.writerow([round(r["overall"], 3), round(r["worst_group"], 3)]
                   + [round(r["per_group"][g], 3) for g in range(4)] + [round(gacc, 3)])
    print(f"\n  Saved -> {args.out_dir}/zeroshot_{tag}.json / .csv")


if __name__ == "__main__":
    main()
