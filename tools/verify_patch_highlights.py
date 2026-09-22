#!/usr/bin/env python
"""Occlusion test for pipeline_4_maco.save_top_images_with_patch_highlights.

The highlight is only meaningful if (a) the g x g grid is drawn where the ViT
actually looked -- raster token order, mapped onto CLIP's center crop -- and
(b) the highlighted patches are what produces the concept's activation, not
just where a number happens to be large. Both are testable by graying out
patches IN THE ORIGINAL IMAGE, using the very same crop-box mapping the
drawing uses, and re-running preprocess -> CLIP -> RouteSAE:

  alignment   occlude only the top cell: that cell's activation should fall
              more than any other cell's. A transposed or mis-cropped grid
              would make some OTHER cell drop most.
  causality   occlude the top-N highlighted cells vs N random non-highlighted
              cells: the image-level (max-pooled) activation should collapse
              in the first case and barely move in the second.

RouteSAE reads the residual stream at middle layers, where attention has
already mixed patches, so a cell's value is not purely its own pixels --
expect other cells to shift a little under occlusion; the test is about which
change is LARGEST and by how much.

Usage:
    python tools/verify_patch_highlights.py --sae_path <ckpt> --concept_finding_method highmag \
        --n_concepts 3 --top_k 3 --top_patches 5
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline_common as pc      # noqa: E402
import msae_ftclip as core        # noqa: E402
from routesae_adapter import patch_concept_activations   # noqa: E402

# CLIP's normalization mean, in 0-255 -- occluding with it is "no signal"
# after Normalize, rather than a strong black/white edge the model reacts to.
GRAY = (int(0.48145466 * 255), int(0.4578275 * 255), int(0.40821073 * 255))


def crop_box(w, h):
    side = min(w, h)
    return (w - side) // 2, (h - side) // 2, side


def occlude(img, cells, g):
    """Gray out the given (row, col) grid cells of img, in ORIGINAL pixels via
    the same center-crop mapping the drawing uses."""
    w, h = img.size
    x0, y0, side = crop_box(w, h)
    p = side / g
    out = img.copy()
    px = out.load()
    for r, c in cells:
        xs, ys = int(round(x0 + c * p)), int(round(y0 + r * p))
        xe, ye = int(round(x0 + (c + 1) * p)), int(round(y0 + (r + 1) * p))
        for y in range(ys, ye):
            for x in range(xs, xe):
                px[x, y] = GRAY
    return out


def grid_for(images, clip_ft, sae, cid, device):
    pix = torch.stack([clip_ft.preprocess(im) for im in images]).to(device)
    grid, cls = patch_concept_activations(sae, clip_ft.model, pix, [cid])
    return grid[:, 0].float().cpu().numpy(), cls[:, 0].float().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_core_args(p)
    p.add_argument("--concept_finding_method", default="highmag")
    p.add_argument("--n_concepts", type=int, default=3)
    p.add_argument("--concept_ids", type=str, default=None,
                   help="Comma-separated concept ids to test instead of the first --n_concepts.")
    p.add_argument("--top_k", type=int, default=3, help="images per concept")
    p.add_argument("--top_patches", type=int, default=5)
    p.add_argument("--n_random", type=int, default=5, help="random-occlusion repeats")
    p.add_argument("--split", default="ft_train", choices=["ft_train", "test"])
    p.set_defaults(clip_mode="zeroshot", sae_path="routesae_weights/routesae_K32_ViT-B~32_16384.pt",
                   hf_model_dir="hf_models")
    next(a for a in p._actions if a.dest == "sae_path").required = False
    args = p.parse_args()

    run_dir, _ = pc.resolve_run_dir_for_pipeline(args)
    manifest = pc.apply_analysis_params(pc.read_manifest(pc.manifest_path_for(args, run_dir)), args)
    concepts = pc.read_concepts(pc.concepts_path_for(manifest, args.concept_finding_method))
    ctx = pc.load_stage1_context(manifest)
    clip_ft, sae, device = ctx["clip_ft"], ctx["sae_model"], ctx["device"]
    results = ctx["ft_results"] if args.split == "ft_train" else ctx["te_results"]
    rng = np.random.default_rng(0)

    n_align_ok = n_align = 0
    drops_top, drops_rand = [], []
    print(f"\n{'concept':>8} {'image':<34} {'act':>7} {'top cell':>9} "
          f"{'align':>6} {'occl top-N':>11} {'occl rand-N':>12}")
    print("-" * 96)
    cids = ([int(c) for c in args.concept_ids.split(",")] if args.concept_ids
            else concepts["candidate_concepts"][: args.n_concepts])
    for cid in cids:
        hits = core.find_top_concept_images(results, top_k=args.top_k, concept_index=cid)
        images = [Image.open(h["image_path"]).convert("RGB") for h in hits]
        grids, cls = grid_for(images, clip_ft, sae, cid, device)
        g = grids.shape[-1]

        for h, im, grid in zip(hits, images, grids):
            base = max(grid.max(), 0.0)
            flat = grid.flatten()
            order = np.argsort(-flat)
            top_cells = [divmod(int(i), g) for i in order[: args.top_patches] if flat[i] > 0]
            top1 = top_cells[0]

            # alignment: occlude ONLY the top cell; which cell drops most?
            g1, _ = grid_for([occlude(im, [top1], g)], clip_ft, sae, cid, device)
            drop = grid - g1[0]
            biggest = divmod(int(np.argmax(drop)), g)
            ok = biggest == top1
            n_align += 1; n_align_ok += ok
            # On a miss, say how far off: an adjacent cell is attention
            # bleed; a far one would be a real mapping error.
            miss_note = ""
            if not ok:
                dist = max(abs(biggest[0] - top1[0]), abs(biggest[1] - top1[1]))
                miss_note = (f"  <- biggest drop at {biggest} ({'adjacent' if dist == 1 else f'{dist} cells away'}: "
                             f"{drop[biggest]:.2f} vs {drop[top1]:.2f} at the top cell)")

            # causality: top-N vs random non-top N
            gt, _ = grid_for([occlude(im, top_cells, g)], clip_ft, sae, cid, device)
            act_top = gt[0].max()
            others = [divmod(int(i), g) for i in range(g * g) if divmod(int(i), g) not in top_cells]
            acts_rand = []
            for _ in range(args.n_random):
                pick = [others[i] for i in rng.choice(len(others), len(top_cells), replace=False)]
                gr, _ = grid_for([occlude(im, pick, g)], clip_ft, sae, cid, device)
                acts_rand.append(gr[0].max())
            act_rand = float(np.mean(acts_rand))
            drops_top.append(1 - act_top / base if base > 0 else 0)
            drops_rand.append(1 - act_rand / base if base > 0 else 0)

            name = os.path.basename(h["image_path"])[:32]
            print(f"{cid:>8} {name:<34} {base:>7.2f} {str(top1):>9} "
                  f"{'ok' if ok else 'MISS':>6} {act_top:>7.2f} ({-100*(1-act_top/base):+4.0f}%) "
                  f"{act_rand:>7.2f} ({-100*(1-act_rand/base):+4.0f}%){miss_note}")

    print("-" * 96)
    print(f"alignment : top cell was the one that dropped most in {n_align_ok}/{n_align} images")
    print(f"causality : occluding the top-{args.top_patches} highlighted cells removed "
          f"{100*np.mean(drops_top):.0f}% of the activation on average; "
          f"occluding {args.top_patches} random other cells removed {100*np.mean(drops_rand):.0f}%")
    return 0 if n_align_ok == n_align else 1


if __name__ == "__main__":
    sys.exit(main())
