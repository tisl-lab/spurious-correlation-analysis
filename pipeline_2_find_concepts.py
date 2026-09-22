#!/usr/bin/env python
"""
Stage 2 of 4 — concept-finding: load stage 1's manifest, run ONE
--concept_finding_method, save the resulting candidate concept list keyed by
that method name (concepts_<method>.json) for stages 3/4 to reuse.

'prism_baseline' is intentionally not offered here -- in msae_ftclip.py it is
a fully self-contained baseline (runs its own PRISM eval, writes its own
report, and returns immediately) that doesn't produce a concept list to feed
into ablation/MACO at all. Run it via msae_ftclip.py directly with
--concept_finding_method prism_baseline.

No-op on rerun: if concepts_<method>.json already exists, this does nothing.
Delete that file to force the method to re-run.

Usage:
    python pipeline_2_find_concepts.py --clip_mode finetuned --run_dir <dir> \
        --model RouteSAE --sae_path <checkpoint.pt> --routesae_k 32 \
        --concept_finding_method highmag
"""

import argparse
import json
import os

import numpy as np
import torch

import pipeline_common as pc
import msae_ftclip as core


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_core_args(p)
    pc.add_concept_finding_args(p)

    # Debug convenience: every add_core_args() field already has a usable
    # default except --sae_path (required=True there on purpose -- guessing
    # wrong silently analyzes the wrong SAE for a real run). Default it here,
    # plus clip_mode/hf_model_dir to match it, to the RouteSAE K=32 zeroshot
    # combo this pipeline was verified against this session, so hitting
    # Run/Debug in the IDE with no args (empty sys.argv) works out of the
    # box. Still fully overridable from the command line -- this only changes
    # what happens when a flag is omitted.
    p.set_defaults(
        clip_mode="zeroshot",
        sae_path="routesae_weights/routesae_K32_ViT-B~32_16384.pt",
        hf_model_dir="hf_models",
    )
    next(a for a in p._actions if a.dest == "sae_path").required = False

    return p


def main():
    args = build_parser().parse_args()
    method = args.concept_finding_method

    run_dir, _ = pc.resolve_run_dir_for_pipeline(args)
    manifest_path = pc.manifest_path_for(args, run_dir)
    manifest = pc.apply_analysis_params(pc.read_manifest(manifest_path), args)

    concepts_path = pc.concepts_path_for(manifest, method)
    candidate_images_path = pc.candidate_images_path_for(manifest)
    concepts_done = os.path.isfile(concepts_path)
    candidates_done = os.path.isfile(candidate_images_path)

    if concepts_done and candidates_done:
        print(f"\nStage 2 already complete for method '{method}':")
        print(f"  {concepts_path}")
        print("Nothing to do. (Delete that file to force a redo.)")
        return

    print(f"\nLoading stage-1 context from {manifest_path} ...")
    ctx = pc.load_stage1_context(manifest)
    clip_ft, sae_model, device = ctx["clip_ft"], ctx["sae_model"], ctx["device"]
    test_ds, ft_ds = ctx["test_ds"], ctx["ft_ds"]
    te_results, ft_results = ctx["te_results"], ctx["ft_results"]
    concept_match_scores, vocab_names = ctx["concept_match_scores"], ctx["vocab_names"]
    base_sae_dir, sae_dir = ctx["base_sae_dir"], ctx["sae_dir"]

    # ── Candidate image selection -- runs once per (run, SAE, knn_k), shared
    # across every --concept_finding_method (not just labelfree, which used
    # to be the only caller): flags images whose fine-tuned CLIP pseudo-label
    # looks wrong or uncertain (centroid/k-NN disagreement). Saved separately
    # from concepts_<method>.json since it doesn't depend on the method, and
    # reused by pipeline_3_ablate.py for the "*_candidates_only" ablation
    # variants (same technique, applied only to these images).
    if candidates_done:
        print(f"  Candidate images already computed -- reusing {candidate_images_path}")
        candidate_payload = pc.read_candidate_images(candidate_images_path)
        M          = torch.tensor(candidate_payload["is_candidate"], dtype=torch.bool)
        Y_hat      = torch.tensor(candidate_payload["y_hat"], dtype=torch.long)
        m_centroid = torch.tensor(candidate_payload["in_centroid"], dtype=torch.bool)
        m_knn      = torch.tensor(candidate_payload["in_knn"], dtype=torch.bool)
    else:
        print(f"\nRunning candidate selection (knn_k={args.knn_k})...")
        M, Y_hat, m_centroid, m_knn = core.candidate_selection(te_results, clip_ft, k=args.knn_k)
        core.Save_candidate_report(
            M=M, Y_hat=Y_hat, M_centroid=m_centroid, M_knn=m_knn,
            te_results=te_results, test_ds=test_ds,
            save_path=os.path.join(sae_dir, "candidate_report.csv"),
        )
        pc.write_candidate_images(candidate_images_path, dict(
            created_at=pc.now_iso(), knn_k=args.knn_k,
            n_total=len(test_ds), n_candidates=int(M.sum().item()),
            is_candidate=M.cpu().tolist(), y_hat=Y_hat.cpu().tolist(),
            in_centroid=m_centroid.cpu().tolist(), in_knn=m_knn.cpu().tolist(),
        ))

    if concepts_done:
        print(f"\nStage 2 concept-finding already complete for method '{method}':")
        print(f"  {concepts_path}")
        print("Nothing more to do. (Delete that file to force a redo.)")
        return

    # denois_candidate_concepts filters by cosine similarity to the mean SAE
    # *decoder* direction -- RouteSAE (per-layer/per-patch, no single decoder)
    # doesn't have one, same force-off main() applies before concept-finding.
    denoise_concepts = args.denoise_concepts
    if manifest["model"] == "RouteSAE" and denoise_concepts:
        print("  Note: --denoise_concepts has no effect for --model RouteSAE "
              "(no single decoder matrix to compute directions from) -- disabling it.")
        denoise_concepts = False

    print(f"\nRunning concept-finding method: {method}")

    concept_weights = None
    # Method-specific provenance for the JSON (only conceptscope fills it so
    # far): which split/subset it ran on and its per-class tables.
    method_extra = None
    # cid -> score. Every method now has a scoring criterion except labelfree
    # under --labelfree_scoring prevalence, which leaves this None so the JSON
    # field stays explicit about which lists carry a score and which are just
    # in the method's own order.
    concept_scores = None

    if method == "labelfree":
        causal = args.labelfree_scoring == "causal"
        # Both selectors below discard non-candidate images, so the causal
        # search only needs to cover candidates -- ~1/5 of the test set here,
        # for identical output. --labelfree_search_all_images opts back out.
        #
        # --labelfree_pool_knn_ks widens the search to the UNION of the
        # candidate sets for several knn_k values. Each image's flipping
        # concepts are computed independently, so a pool over a superset of
        # images is exactly as good as one over this run's candidates -- and
        # because the cache is keyed by the image set, every run in a knn_k
        # sweep that passes the same list hits the same cache instead of
        # each redoing the whole causal search for its own candidate set.
        if args.labelfree_search_all_images:
            image_subset = None
        elif args.labelfree_pool_knn_ks:
            pool_ks = sorted({int(k) for k in args.labelfree_pool_knn_ks.split(",")}
                             | {args.knn_k})
            union = set(M.nonzero(as_tuple=True)[0].tolist())
            for k in pool_ks:
                if k == args.knn_k:
                    continue
                M_k = core.candidate_selection(te_results, clip_ft, k=k)[0]
                union |= set(M_k.nonzero(as_tuple=True)[0].tolist())
            image_subset = sorted(union)
            print(f"  Concept pool over the union of candidate sets for knn_k in "
                  f"{pool_ks}: {len(image_subset):,} images "
                  f"(this run's knn_k={args.knn_k} alone: {int(M.sum()):,})")
        else:
            image_subset = M.nonzero(as_tuple=True)[0].tolist()
        concept_pool, per_image_pool = core.Generate_Concept_Pool(
            te_results=te_results, test_ds=test_ds, sae_model=sae_model,
            concept_match_scores=concept_match_scores, vocab_names=vocab_names,
            sae_dir=base_sae_dir, clip_ft=clip_ft,
            batch_size=args.concept_pool_batch_size,
            record_deltas=causal,
            image_subset=image_subset,
            cache_images=not args.no_cache_images,
        )

        n_candidates = int(M.sum().item())
        if causal:
            # Ranks by causal influence -- how often a concept is among an
            # image's strongest flippers, and how much probability those
            # flips move. Needs the deltas Generate_Concept_Pool just wrote.
            delta_path = core.concept_pool_delta_path(
                base_sae_dir, clip_ft, te_results["sae_representations"].shape[1],
                core.concept_pool_subset_tag(image_subset),
            )
            with open(delta_path) as f:
                per_image_delta = {
                    int(i): {int(c): float(d) for c, d in cd.items()}
                    for i, cd in json.load(f).items()
                }
            candidate_concepts, concept_scores = core.score_labelfree_concepts(
                M=M, per_image_pool=per_image_pool, per_image_delta=per_image_delta,
                concept_pool=concept_pool,
                top_k_per_image=args.labelfree_top_k_per_image,
                sae_dir=sae_dir, vocab_names=vocab_names,
                concept_match_scores=concept_match_scores,
            )
            suffix = (f"over {n_candidates} candidate images "
                      f"(label-free causal influence score)")
        else:
            candidate_concepts, concept_counts = core.select_prevalent_concepts(
                M=M, per_image_pool=per_image_pool, concept_pool=concept_pool,
                prevalence_threshold=args.prevalence_threshold,
                top_k=args.top_k_concepts, n_cpu_workers=args.n_cpu_workers,
            )
            suffix = f"over {n_candidates} candidate images (label-free causal search)"

    elif method == "dialguided":
        candidate_concepts, concept_scores = core.select_dialguided_concepts(
            te_results=te_results, ft_results=ft_results,
            test_ds=test_ds, ft_ds=ft_ds,
            concept_match_scores=concept_match_scores, vocab_names=vocab_names,
            sae_dir=sae_dir, clip_ft=clip_ft, sae_model=sae_model, device=device,
            sp_arribute_dir="results/" + args.dataset + "/attribute_words",
            top_n_concepts=args.top_k_concepts,
        )
        suffix = "from dial-guided spurious concept detection"

    elif method == "highmag":
        candidate_concepts, concept_source_group, concept_mis_prevalence, concept_scores = \
            core.find_spurious_concepts_highmag(
                te_results=te_results, ft_results=ft_results, test_ds=test_ds, ft_ds=ft_ds,
                concept_match_scores=concept_match_scores, vocab_names=vocab_names,
                sae_dir=sae_dir, clip_ft=clip_ft, top_n_concepts=args.top_k_concepts,
            )
        suffix = "from high-magnitude spurious concept detection"

    elif method == "labelguided":
        candidate_concepts, concept_source_group, concept_mis_prevalence, concept_scores = \
            core.find_and_show_spurious_concepts_binary(
                te_results=te_results, ft_results=ft_results, test_ds=test_ds, ft_ds=ft_ds,
                concept_match_scores=concept_match_scores, vocab_names=vocab_names,
                sae_dir=sae_dir, clip_ft=clip_ft, sae_model=sae_model, device=device,
                top_n_concepts=args.top_k_concepts,
            )
        suffix = "from label-guided spurious concept detection"

    elif method == "conceptscope":
        # ConceptScope-style categorisation, prediction-driven
        # (core.find_concept_scope_concepts): classes are the zero-shot
        # PREDICTIONS, run on --conceptscope_split (default: the fine-tuning
        # training images) over an equal number of images per predicted
        # class. It picks its own top-k latents per class, so it gets the
        # whole dictionary. Returns the "bias" concepts -- firing hard for a
        # predicted class without the prediction depending on them -- with
        # their excess over the bias threshold as the score, plus the full
        # per-class tables (also written to conceptscope_categorization.json).
        n_latents = te_results["sae_representations"].shape[1]
        np.random.seed(args.seed)
        candidate_concepts, concept_scores, categorization = core.find_concept_scope_concepts(
            te_results=te_results, ft_results=ft_results, test_ds=test_ds, ft_ds=ft_ds,
            concept_match_scores=concept_match_scores, vocab_names=vocab_names,
            sae_dir=sae_dir, clip_ft=clip_ft, sae_model=sae_model, device=device,
            candidate_concepts=list(range(n_latents)),
            save_path=os.path.join(sae_dir, "conceptscope_patch_activations.npz"),
            batch_size=args.batch_size,
            split=args.conceptscope_split,
            image_subset_size=(args.conceptscope_image_subset or None),
            num_samples_for_alignment=args.conceptscope_num_samples,
            top_k_for_alignment=args.conceptscope_top_k,
            alignment_mode=args.conceptscope_alignment,
            bias_threshold_sigma=args.conceptscope_bias_sigma,
            target_threshold=args.conceptscope_target_threshold,
            top_n_concepts=args.top_k_concepts,
            seed=args.seed,
        )
        method_extra = dict(
            split=args.conceptscope_split, classes_from="predictions",
            image_subset_per_class=(args.conceptscope_image_subset or None),
            num_samples_per_class=args.conceptscope_num_samples,
            top_k_per_class=args.conceptscope_top_k,
            alignment_mode=args.conceptscope_alignment,
            bias_threshold_sigma=args.conceptscope_bias_sigma,
            target_threshold=args.conceptscope_target_threshold,
            categorization_path=os.path.join(sae_dir, "conceptscope_categorization.json"),
            report_path=os.path.join(sae_dir, "conceptscope_report.txt"),
            patch_activations_path=os.path.join(sae_dir, "conceptscope_patch_activations.npz"),
            # A concept flagged bias for every predicted class is excluded from
            # the list above (it does not separate the classes); kept here so
            # the JSON shows what was set aside.
            excluded_bias_in_multiple_classes=sorted(
                set.intersection(*[{e["latent_idx"] for e in cat["context"] if e["bias"]}
                                   for cat in categorization.values()])
                if len(categorization) > 1 else set()),
            per_predicted_class={
                cls: dict(n_target=cat["n_target"], n_context=cat["n_context"],
                          n_bias=cat["n_bias"], bias_threshold=cat["bias_threshold"],
                          bias_concepts=[e["latent_idx"] for e in cat["context"] if e["bias"]],
                          top_target_concepts=[e["latent_idx"] for e in cat["target"][:5]])
                for cls, cat in categorization.items()
            },
        )
        n_excl = len(method_extra["excluded_bias_in_multiple_classes"])
        suffix = (f"flagged as bias for exactly one of {len(categorization)} predicted classes on the "
                  f"{args.conceptscope_split} split (prediction-driven ConceptScope; "
                  f"{n_excl} flagged for both and set aside)")

    elif method == "none":
        # No concept-finding -- ablation runs with an empty concept list,
        # which for both hooks means "encode then decode, zero nothing" i.e.
        # plain SAE reconstruction with nothing deliberately removed.
        candidate_concepts = []
        suffix = None

    
    else:
        raise ValueError(f"Unknown concept_finding_method: {method}")

    if denoise_concepts:
        candidate_concepts, concept_weights = core.denois_candidate_concepts(
            candidate_concepts=candidate_concepts, sae_model=sae_model,
            percentile=args.denoise_percentile, beta=args.denoise_beta,
        )

    # Count is read fresh here, AFTER any denoise reassignment above -- printing
    # a count captured before denoising (this file's earlier revision did) would
    # mislabel it "after denoise" while actually showing the pre-filter number.
    if method == "none":
        print("  No concept-finding — reconstruction-only baseline (0 concepts).")
    elif method == "conceptscope" and len(candidate_concepts) == 0:
        print("  ConceptScope flagged no bias concepts: no context latent's mean activation "
              f"cleared mean + {args.conceptscope_bias_sigma} sigma in any predicted class. "
              "Lower --conceptscope_bias_sigma or raise --conceptscope_top_k and rerun.")
    else:
        label = "after denoise" if denoise_concepts else "found"
        print(f"  Candidate concepts ({label}): {len(candidate_concepts)} concepts {suffix}")

    # RouteSAE-only: which CLIP layer each concept's activations mostly come
    # from. RouteSAE shares one dictionary across layers and routes each
    # PATCH to a layer per forward pass, so this isn't a static lookup --
    # concept_layer_origins measures it empirically over the test split (see
    # its docstring in routesae_adapter.py for exactly what's counted).
    concept_layers = None
    if manifest["model"] == "RouteSAE" and method != "none" and len(candidate_concepts) > 0:
        from routesae_adapter import concept_layer_origins

        print(f"  Computing per-concept layer origins (RouteSAE, hard routing, "
              f"over {len(test_ds)} test images)...")
        concept_layers = concept_layer_origins(
            sae=sae_model, clip_model=clip_ft.model, dataset=test_ds,
            candidate_concepts=candidate_concepts, device=device,
            preprocess=clip_ft.preprocess, batch_size=args.batch_size,
        )
        for cid in candidate_concepts:
            info = concept_layers[cid]
            if info["mode_layer"] is None:
                print(f"    concept {cid:>6}: never active in the test split")
            else:
                print(f"    concept {cid:>6}: layer {info['mode_layer']:>2} "
                      f"(purity {info['purity']*100:4.1f}%, "
                      f"{info['n_active_images']}/{len(test_ds)} images active)")
        # JSON object keys must be strings -- json.dump stringifies the int
        # concept ids automatically; read them back with int(cid) if you need
        # to index by concept id again.

    # Same naming convention main() uses -- always "ft-<method>" regardless
    # of clip_mode (a pre-existing quirk of the report/output naming, kept
    # as-is rather than "fixed" here).
    concept_extractor_name = f"ft-{method}" if method != "none" else "ft-reconstruction_only"
    report_suffix = "__denoised" if denoise_concepts else ""

    # NOTE: when denoising is on, concept_weights is the FULL weight vector
    # aligned to the pre-filter candidate list (denois_candidate_concepts
    # returns (filtered_concepts, w) where len(w) == len(candidate list
    # passed in), while candidate_concepts here has already been reassigned
    # to the shorter filtered list) -- same mismatch main() has always had.
    # Kept as-is; concept_weights is informational, not indexed against
    # candidate_concepts by any downstream stage.
    if isinstance(concept_weights, (np.ndarray, torch.Tensor)):
        concept_weights = concept_weights.tolist()

    # Scores keyed by concept id, in the same order candidate_concepts is
    # sorted by. JSON object keys are strings -- int(cid) to index by concept
    # id again, same as concept_layers. None for methods without a scoring
    # criterion yet, where candidate_concepts is in the method's own order.
    if concept_scores is not None:
        concept_scores = {int(cid): float(s) for cid, s in concept_scores.items()}

    payload = dict(
        created_at=pc.now_iso(),
        method=method,
        concept_extractor_name=concept_extractor_name,
        candidate_concepts=[int(c) for c in candidate_concepts],
        concept_scores=concept_scores,
        concept_weights=concept_weights,
        concept_layers=concept_layers,
        report_suffix=report_suffix,
        denoise_concepts=denoise_concepts,
        base_sae_dir=base_sae_dir, sae_dir=sae_dir,
        method_extra=method_extra,
    )
    pc.write_concepts(concepts_path, payload)
    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
