#!/usr/bin/env python
"""
Stage 1 of 4 — setup: resolve run/CLIP/SAE identity, validate the SAE
checkpoint loads, extract-or-reuse cached representations, resolve the
concept_match vocab. Writes stage1_manifest.json for stages 2-4 to build on.

This does NOT train an SAE -- training stays in msae/train.py /
routesae_train.py, run separately (see the batch_files/run_*_prep_*.sh
scripts). This stage only validates the checkpoint given via --sae_path
loads correctly, and points at the right prep step if not.

No-op on rerun: if the manifest and both cached representations dirs already
exist, this just re-validates and reprints the summary rather than redoing
any work -- same resume convention as msae_ftclip.py's own representations
cache.

Usage (same core args you'd pass to msae_ftclip.py directly):
    python pipeline_1_setup.py --clip_mode finetuned --run_dir <dir> \
        --model RouteSAE --sae_path <checkpoint.pt> --routesae_k 32
"""

import argparse
import os

import pipeline_common as pc
import msae_ftclip as core


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_core_args(p)
    return p


def main():
    args = build_parser().parse_args()
    if args.hf_model_dir:
        core._hf_model_dir = args.hf_model_dir

    run_dir, metadata = pc.resolve_run_dir_for_pipeline(args)
    print(f"\nRun folder : {os.path.abspath(run_dir)}")
    print(f"Metadata   : {metadata}")

    base_sae_dir, sae_dir = pc.sae_dirs(args, run_dir)
    manifest_path = pc.manifest_path_for(args, run_dir)

    already_done = (
        os.path.isfile(manifest_path)
        and pc.representations_cached(base_sae_dir, "ft_train")
        and pc.representations_cached(base_sae_dir, "test")
    )
    if already_done:
        print(f"\nStage 1 already complete -- manifest + cached representations found:")
        print(f"  {manifest_path}")
        print(f"  {base_sae_dir}/representations_{{ft_train,test}}")
        print("Nothing to do. (Delete the manifest or a representations_* dir to force a redo.)")
        return

    device = pc.resolve_device()
    train_ds = core.ManifestDataset(os.path.join(run_dir, "train_all_manifest.csv"))
    ft_ds    = core.ManifestDataset(os.path.join(run_dir, "ft_train_manifest.csv"))
    test_ds  = core.ManifestDataset(os.path.join(run_dir, "test_manifest.csv"))
    print(f"\nDatasets (from saved manifests):")
    print(f"  train_ds : {len(train_ds):5d} images")
    print(f"  ft_ds    : {len(ft_ds):5d} images  <- used for fine-tuning")
    print(f"  test_ds  : {len(test_ds):5d} images")

    if args.clip_mode == "zeroshot":
        clip_ft = core.CLIPZeroShot(model_name=args.clip_model, device=device)
        print(f"\nZero-shot CLIP loaded  ({args.clip_model}, device={device})")
    else:
        clip_ft = core.CLIPZeroShot.load_model(os.path.join(run_dir, "model.pt"), device=device)
        print(f"\nFine-tuned CLIP loaded  (device={device})")
    clip_ft.model.eval()

    embedding_path = pc.embedding_path_for(run_dir)
    os.makedirs(base_sae_dir, exist_ok=True)
    os.makedirs(sae_dir, exist_ok=True)
    print(f"  Representations folder: {base_sae_dir}")
    print(f"  Analysis output folder: {sae_dir}")

    # ── Validate + load the SAE checkpoint ────────────────────────────────
    if args.model == "RouteSAE":
        from routesae_adapter import load_routesae_for_clip, extract_routesae_representations

        if not args.sae_path or not os.path.isfile(args.sae_path):
            raise FileNotFoundError(
                f"--model RouteSAE requires --sae_path pointing directly at a checkpoint "
                f"file (no auto-discovery). Got: {args.sae_path!r}\n"
                f"Train one first with routesae_train.py (see batch_files/run_routesae_prep_*.sh)."
            )
        # See msae_ftclip.py's main() for why: RouteSAE's hook-based
        # extraction upcasts to fp32, and a fp16 CLIP (possible on non-CPU
        # devices via load_model) crashes at the first matmul otherwise.
        clip_ft.model = clip_ft.model.float()
        print(f"\nLoading RouteSAE from: {args.sae_path}  (k={args.routesae_k})")
        sae_model = load_routesae_for_clip(args.sae_path, device=device, k=args.routesae_k)
        sae_path = args.sae_path
        extract_fn = lambda ds: extract_routesae_representations(
            clip_ft.model, sae_model, ds, device, clip_ft.preprocess)
    else:
        from msae.sae import SAE

        if not args.sae_path or not (os.path.isfile(args.sae_path) or os.path.isdir(args.sae_path)):
            raise FileNotFoundError(
                f"--sae_path not found: {args.sae_path!r}\n"
                f"Train one first with msae/train.py (see batch_files/run_*_prep_*.sh)."
            )
        if os.path.isdir(args.sae_path):
            import glob
            candidates = glob.glob(os.path.join(args.sae_path, "*.pth"))
            if not candidates:
                raise FileNotFoundError(f"No *.pth checkpoints found under {args.sae_path!r}")
            sae_path = max(candidates, key=os.path.getmtime)
        else:
            sae_path = args.sae_path
        print(f"\nLoading SAE from: {sae_path}")
        sae_model = SAE(sae_path).to(device).eval()
        print(f"  n_inputs={sae_model.input_dim}  n_latents={sae_model.latent_dim}")
        extract_fn = lambda ds: core.extract_sae_representations(clip_ft, sae_model, ds, device)

    # ── Representations: reuse cache if present, else extract + save ──────
    ft_results = None
    for split_name, ds in (("ft_train", ft_ds), ("test", test_ds)):
        if pc.representations_cached(base_sae_dir, split_name):
            print(f"  Found existing representations at "
                  f"{base_sae_dir}/representations_{split_name} -- reusing.")
        else:
            print(f"\nExtracting SAE representations -- {split_name}...")
            results = extract_fn(ds)
            core.save_representations(results, split_name, base_sae_dir)

    # ── Resolve concept_match vocab (validates it exists; not loaded here — ─
    # stages 2-4 reload it themselves via load_stage1_context)
    vocab_path_to_load, names_txt = pc.resolve_vocab_paths(args, run_dir, sae_path)
    print(f"Resolved concept_match : {os.path.basename(vocab_path_to_load)}")
    print(f"Resolved vocab names   : "
          f"{os.path.basename(names_txt) if names_txt else 'clip_disect_20k.txt (fallback)'}")

    manifest = dict(
        created_at=pc.now_iso(),
        dataset=args.dataset, clip_model=args.clip_model, clip_mode=args.clip_mode,
        hf_model_dir=args.hf_model_dir, run_dir=run_dir,
        model=args.model, sae_path=sae_path, routesae_k=args.routesae_k,
        activation=args.activation, knn_k=args.knn_k,
        prevalence_threshold=args.prevalence_threshold,
        concept_match_vocab=args.concept_match_vocab,
        base_sae_dir=base_sae_dir, sae_dir=sae_dir,
        vocab_path_to_load=vocab_path_to_load, vocab_names_txt=names_txt,
    )
    pc.write_manifest(manifest_path, manifest)
    print("\nStage 1 complete.")


if __name__ == "__main__":
    main()
