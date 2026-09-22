#!/usr/bin/env python
"""
Stage 4 of 4 — visualization: load a concepts_<method>.json + the stage-1
manifest, and visualize every one of its concepts (up to --maco_max_concepts,
ranked by peak test-split activation -- same cap/ranking msae_ftclip.py has
always used; dialguided in particular can return 12,000+ concepts, so this
is "all concepts the finder selected, up to a bounded cost" rather than
literally uncapped).

Three visualizations -- the first two existing features relocated here
verbatim, the third added for RouteSAE:
  - MACO synthetic renders (run_maco_parallel) -- one PNG per concept, the
    "what maximally activates this concept" image.
  - Real-image montages (save_top_ft_images_per_concept, capped at 50
    regardless of --maco_max_concepts -- a quick browse, not the main
    artifact) -- top-5 real ft-train images per concept. Skip with
    --skip_montages.
  - Patch highlights (save_top_images_with_patch_highlights, this file) --
    for every MACO concept, its top real images with the patches that
    activate the concept most kept bright and outlined, read directly from
    RouteSAE's per-patch codes (no gradient attribution). One montage per
    concept plus a standalone highlighted copy of each image. RouteSAE only;
    skip with --skip_patch_highlights.

All are individually resumable: each already renders/skips per concept (or
per image), so a rerun only fills in what's missing -- nothing extra needed
at this stage's level to make reruns cheap.

Usage:
    python pipeline_4_maco.py --clip_mode finetuned --run_dir <dir> \
        --model RouteSAE --sae_path <checkpoint.pt> --routesae_k 32 \
        --concept_finding_method highmag --maco_max_concepts 200
"""

import argparse
import os
import re

import pipeline_common as pc
import msae_ftclip as core


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_core_args(p)
    p.add_argument("--concept_finding_method", type=str, default="conceptscope",
                   choices=["labelfree", "labelguided", "dialguided", "highmag", "conceptscope", "none"],
                   help="Selects which concepts_<method>.json to load (must already "
                        "exist -- run pipeline_2_find_concepts.py first).")
    pc.add_maco_args(p)
    p.add_argument("--skip_patch_highlights", action="store_true",
                   help="Skip the per-patch highlight montages "
                        "(save_top_images_with_patch_highlights). RouteSAE only; "
                        "silently a no-op for MSAE.")
    p.add_argument("--patch_highlight_top_k", type=int, default=5,
                   help="How many top-activation images per concept to highlight "
                        "(same selection as the montage).")
    p.add_argument("--patch_highlight_top_patches", type=int, default=5,
                   help="How many of the strongest patches to keep bright and outline "
                        "in each highlighted image; the rest are dimmed.")
    p.add_argument("--patch_highlight_split", type=str, default="ft_train",
                   choices=["ft_train", "test"],
                   help="Which split's top images to use. 'ft_train' matches the "
                        "existing montage image for image; 'test' uses the split MACO "
                        "ranks concepts on.")

    # Same debug convenience as pipeline_2/3's build_parser: default the one
    # required field (--sae_path) plus clip_mode/hf_model_dir to the RouteSAE
    # K=32 zeroshot combo, so a no-arg run works. Fully overridable.
    p.set_defaults(
        clip_mode="zeroshot",
        sae_path="routesae_weights/routesae_K32_ViT-B~32_16384.pt",
        hf_model_dir="hf_models",
    )
    next(a for a in p._actions if a.dest == "sae_path").required = False
    return p


def save_top_images_with_patch_highlights(
    candidate_concepts, results, sae_dir, concept_extractor_name,
    clip_ft, sae_model, device,
    vocab_names=None, concept_match_scores=None,
    top_k=5, top_patches=5, dim=0.35, split_label="ft-train",
):
    """For each concept, the top-k highest-activation images of `results`
    (the same selection save_top_ft_images_per_concept makes) -- and, for
    each of them, a second version with the patches that activate the
    concept most highlighted.

    Unlike the montage's second row, which overlays a GRADIENT attribution
    (concept_spatial_heatmap: which pixels the concept's activation is
    sensitive to), this draws RouteSAE's own per-patch codes: the SAE assigns
    every patch a sparse activation, and image_concepts() max-pools those
    into the image-level number the concept selectors rank on. This shows
    the un-pooled values, so the brightest patch IS the one that produced the
    image's activation -- the direct answer to "where in this image is the
    concept?", with no attribution method in between.

    Two artifacts per concept, under
        <sae_dir>/<concept_extractor_name>/top_images_patches/
      concept_<id>_<name>.png                one 3-row montage:
          row 1  image (title: image-level activation)
          row 2  the g x g patch map as a heat overlay -- nearest-neighbour
                 upsampled on purpose, so the blocks ARE the patches. Drawn
                 over the central square CLIP's preprocess crops to (Resize
                 shorter side -> 224, CenterCrop 224): that is all the ViT
                 sees, so on a non-square photo the grid does not span the
                 whole image and the cropped-away margins carry no patches
          row 3  the highlighted version: the top `top_patches` patches at
                 full brightness and outlined, everything else dimmed to
                 `dim` of its brightness
      concept_<id>_<name>/<rank>_<image>.png  row 3 alone, one file per
          image, so the highlighted images can be used on their own.

    RouteSAE only: MSAE has one code per image and no patch structure, so
    there is nothing to draw -- the function says so and returns [].
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image as _PIL
    from routesae import RouteSAE
    from routesae_adapter import patch_concept_activations

    if not isinstance(sae_model, RouteSAE):
        print("  Patch highlights need per-patch codes -- RouteSAE only; skipping for "
              f"{type(sae_model).__name__}.")
        return []

    out_dir = os.path.join(sae_dir, concept_extractor_name, "top_images_patches")
    os.makedirs(out_dir, exist_ok=True)

    def _name(cid):
        if vocab_names is not None and concept_match_scores is not None:
            return vocab_names[concept_match_scores[:, cid].argmax()]
        return str(cid)

    def _safe(x):
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(x)).strip("_") or "x"

    def _crop_box(w, h):
        """The region of the ORIGINAL image the patch grid actually covers.

        CLIP's preprocess is Resize(shorter side -> 224) then CenterCrop(224),
        so the ViT -- and therefore the g x g grid -- only ever sees the
        central square of side min(w, h). Drawing the grid over the full
        image would shift every patch on any non-square photo. Returns
        (x0, y0, side) in original pixels."""
        side = min(w, h)
        return (w - side) // 2, (h - side) // 2, side

    def _highlighted(img_np, grid, n):
        """img with the n strongest patches kept and the rest dimmed --
        including everything outside the center crop, which the model never
        saw at all."""
        h, w = img_np.shape[:2]
        g = grid.shape[0]
        x0, y0, side = _crop_box(w, h)
        flat = grid.flatten()
        n = min(n, int((flat > 0).sum())) if (flat > 0).any() else 0
        keep = np.zeros(g * g, dtype=bool)
        if n > 0:
            keep[np.argsort(-flat)[:n]] = True
        keep = keep.reshape(g, g)
        # Nearest-neighbour resize of the g x g mask to the crop square, so
        # block edges land exactly on patch edges; paste it into a full-size
        # mask at the crop offset.
        crop_mask = np.array(_PIL.fromarray(keep.astype(np.uint8) * 255)
                             .resize((side, side), _PIL.NEAREST)) > 0
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y0 + side, x0:x0 + side] = crop_mask
        out = img_np.astype(np.float32)
        out[~mask] *= dim
        return out.astype(np.uint8), keep

    saved = []
    for cid in candidate_concepts:
        hits = core.find_top_concept_images(results, top_k=top_k, concept_index=cid)
        if not hits:
            continue
        cname = _name(cid)
        imgs, pix = [], []
        for h in hits:
            im = _PIL.open(h["image_path"]).convert("RGB")
            imgs.append(im)
            pix.append(clip_ft.preprocess(im))
        pixel_values = torch.stack(pix).to(device)
        grids, cls_act = patch_concept_activations(
            sae_model, clip_ft.model, pixel_values, [int(cid)])
        grids = grids[:, 0].float().cpu().numpy()          # (k, g, g)
        cls_act = cls_act[:, 0].float().cpu().numpy()      # (k,)

        per_concept_dir = os.path.join(out_dir, f"concept_{cid}_{_safe(cname)}")
        os.makedirs(per_concept_dir, exist_ok=True)

        n = len(hits)
        fig, axes = plt.subplots(3, n, figsize=(3 * n, 9.4), squeeze=False)
        for col, (h, im, grid) in enumerate(zip(hits, imgs, grids)):
            img_np = np.array(im)
            hgt, w = img_np.shape[:2]
            g = grid.shape[0]
            x0, y0, side = _crop_box(w, hgt)
            crop_extent = (x0, x0 + side, y0 + side, y0)   # matplotlib: l, r, b, t
            pw = ph = side / g

            def _outline(ax, lw):
                for r in range(g):
                    for c in range(g):
                        if keep[r, c]:
                            ax.add_patch(Rectangle((x0 + c * pw, y0 + r * ph), pw, ph,
                                                   fill=False, edgecolor="red", linewidth=lw))

            ax = axes[0][col]
            ax.imshow(img_np); ax.axis("off")
            ax.set_title(f"act {h['activation']:.2f}", fontsize=8)

            ax = axes[1][col]
            ax.imshow(img_np, extent=(0, w, hgt, 0))
            ax.imshow(grid, cmap="jet", alpha=0.55, extent=crop_extent,
                      interpolation="nearest", vmin=0, vmax=max(grid.max(), 1e-6))
            ax.set_xlim(0, w); ax.set_ylim(hgt, 0)
            ax.axis("off")
            ax.set_title(f"patch max {grid.max():.2f}  cls {cls_act[col]:.2f}", fontsize=8)

            hl, keep = _highlighted(img_np, grid, top_patches)
            ax = axes[2][col]
            ax.imshow(hl, extent=(0, w, hgt, 0))
            _outline(ax, 1.5)
            ax.axis("off")
            ax.set_title(f"top {int(keep.sum())} patches", fontsize=8)

            # The standalone highlighted image, at native resolution.
            # fig1, ax1 = plt.subplots(figsize=(w / 100, hgt / 100), dpi=100)
            # ax1.imshow(hl, extent=(0, w, hgt, 0)); ax1.axis("off")
            # _outline(ax1, 2)
            # fig1.subplots_adjust(0, 0, 1, 1)
            # stem = _safe(os.path.splitext(os.path.basename(h["image_path"]))[0])
            # fig1.savefig(os.path.join(per_concept_dir, f"{col + 1}_{stem}.png"), dpi=100)
            # plt.close(fig1)

        fig.suptitle(
            f"Concept {cid}: {cname} — top {n} {split_label} images\n"
            "row 1 = image  ·  row 2 = per-patch SAE activation  ·  "
            f"row 3 = top {top_patches} patches highlighted",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"concept_{cid}_{_safe(cname)}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    print(f"  Saved patch-highlight montages (+ per-image highlighted copies) for "
          f"{len(saved)} concept(s) → {out_dir}/")
    return saved


def main():
    args = build_parser().parse_args()

    run_dir, _ = pc.resolve_run_dir_for_pipeline(args)
    manifest_path = pc.manifest_path_for(args, run_dir)
    manifest = pc.apply_analysis_params(pc.read_manifest(manifest_path), args)

    concepts_path = pc.concepts_path_for(manifest, args.concept_finding_method)
    concepts = pc.read_concepts(concepts_path)
    candidate_concepts = concepts["candidate_concepts"]
    concept_extractor_name = concepts["concept_extractor_name"]
    sae_dir = concepts["sae_dir"]

    if len(candidate_concepts) == 0:
        print(f"\nNo candidate concepts in {concepts_path} -- nothing to visualize. Skipping.")
        return

    print(f"\nLoading stage-1 context from {manifest_path} ...")
    ctx = pc.load_stage1_context(manifest)
    clip_ft, sae_model, device = ctx["clip_ft"], ctx["sae_model"], ctx["device"]
    test_ds, ft_results, te_results = ctx["test_ds"], ctx["ft_results"], ctx["te_results"]
    vocab_names, concept_match_scores = ctx["vocab_names"], ctx["concept_match_scores"]
    sae_path = ctx["sae_path"]

    if not args.skip_montages:
        save_concepts = core.top_concepts_by_activation(
            candidate_concepts, ft_results["sae_representations"], max_n=50, label="montage",
        )
        if args.model != "RouteSAE":
            print(f"  Saving top-5 ft-train images for {len(save_concepts)} candidate concepts ...")
            core.save_top_ft_images_per_concept(
                candidate_concepts=save_concepts, ft_results=ft_results, sae_dir=sae_dir,
                concept_extractor_name=concept_extractor_name, clip_ft=clip_ft, sae_model=sae_model,
                device=device, vocab_names=vocab_names, concept_match_scores=concept_match_scores,
                top_k=5,
            )

    maco_concepts = core.top_concepts_by_activation(
        candidate_concepts, te_results["sae_representations"],
        max_n=args.maco_max_concepts, label="MACO",
    )
    if not args.skip_patch_highlights:
        hl_results = ft_results if args.patch_highlight_split == "ft_train" else te_results
        print(f"  Patch-highlighting the top {args.patch_highlight_top_k} "
              f"{args.patch_highlight_split} images for {len(maco_concepts)} concepts ...")
        save_top_images_with_patch_highlights(
            candidate_concepts=maco_concepts, results=hl_results, sae_dir=sae_dir,
            concept_extractor_name=concept_extractor_name, clip_ft=clip_ft,
            sae_model=sae_model, device=device, vocab_names=vocab_names,
            concept_match_scores=concept_match_scores,
            top_k=args.patch_highlight_top_k, top_patches=args.patch_highlight_top_patches,
            split_label=args.patch_highlight_split.replace("_", "-"),
        )

    print(f"  MACO visualizing {len(maco_concepts)} candidate concepts ...")
    core.run_maco_parallel(
        candidate_concepts=maco_concepts, sae_model=sae_model, clip_ft=clip_ft, test_ds=test_ds,
        sae_dir=sae_dir, concept_match_scores=concept_match_scores, vocab_names=vocab_names,
        device=device, args=args, run_dir=run_dir, sae_path=sae_path,
        subfolder=concept_extractor_name,
    )
    print("\nStage 4 complete.")


if __name__ == "__main__":
    main()
