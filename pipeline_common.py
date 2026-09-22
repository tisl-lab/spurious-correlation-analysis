"""
Shared plumbing for the 4-stage concept-removal pipeline
(pipeline_1_setup.py -> pipeline_2_find_concepts.py -> pipeline_3_ablate.py
-> pipeline_4_maco.py).

msae_ftclip.py itself is UNCHANGED and untouched by this split -- it stays a
working single-shot CLI (every existing batch_files/*.sh script keeps running
exactly as before) and, more importantly here, stays the function library
every stage below imports from. Splitting a 7000-line file with this many
cross-referenced helpers (report writers, plotting, concept-finding methods
that call into each other) into 4 physically separate files would mean
re-deriving which helper belongs where and risking regressions in code that
already works. The actual ask -- 4 independently runnable stages, each
picking up the previous one's saved output -- doesn't need that: a thin
driver per stage, sharing one manifest format, gives the same result with
none of that risk.

Two JSON artifacts hand state between stages:

  stage1_manifest.json   (written by pipeline_1_setup.py, read by 2/3/4)
      The resolved run identity + where its cached artifacts live: run_dir,
      clip_mode/model, sae_path, base_sae_dir/sae_dir, the concept_match
      vocab/scores paths, and the analysis params (knn_k, prevalence,
      ablation_coefficient, ...). Stages 2-4 reload CLIP+SAE+representations
      from this rather than re-deriving any of it.

  concepts_<method>.json  (written by pipeline_2_find_concepts.py, read by
      pipeline_3_ablate.py and pipeline_4_maco.py)
      One saved candidate-concept list per concept-finding method, named by
      that method so re-running a different method never overwrites another's
      result -- "reusable by next step" per the request this split
      implements.

NOTE: --sae_path must be given explicitly for every --model here (including
MSAE), not auto-discovered. msae_ftclip.py's MSAE branch will glob for
"most recently modified matching checkpoint" if --sae_path is omitted --
fine for a single-shot run, but wrong for a multi-stage pipeline where stage
3 could run hours or days after stage 1 and silently pick a DIFFERENT
checkpoint if a new one landed in between. RouteSAE already requires this
explicitly (see msae_ftclip.py's --model help text); this just applies the
same rule uniformly.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

import msae_ftclip as core

MANIFEST_NAME = "stage1_manifest.json"


# ── CLI arg groups — split out of msae_ftclip.parse_args()'s ~250 lines so ────
# each stage only exposes the flags it actually reads. All defaults/choices
# match parse_args() exactly.

def add_core_args(p: argparse.ArgumentParser) -> None:
    """Identity of the run + SAE: needed by every stage, since each stage
    re-derives base_sae_dir/sae_dir/manifest_path from these rather than
    requiring a separate '--point at stage 1's output' flag -- run the same
    core args across all 4 stages, same as you'd pass to the old single-shot
    CLI."""
    p.add_argument("--dataset", type=str, default="waterbirds")
    p.add_argument("--clip_model", type=str, default="ViT-B/32",
                   choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    p.add_argument("--clip_mode", type=str, default="finetuned",
                   choices=["zeroshot", "finetuned"])
    p.add_argument("--hf_model_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=400)
    p.add_argument("--biased_ft", type=bool, default=False)
    p.add_argument("--run_dir", type=str,
                   default="results/waterbirds/clip_ft_BIASED_400_20260709_111654")
    p.add_argument("--sae_path", type=str, required=True,
                   help="Trained SAE checkpoint. Required (no auto-discovery) -- "
                        "see this module's docstring for why.")
    p.add_argument("-m", "--model", type=str, default="RouteSAE",
                   choices=["ReLUSAE", "TopKSAE", "BatchTopKSAE", "MSAE_UW", "MSAE_RW", "RouteSAE"])
    p.add_argument("--routesae_k", type=int, default=32)
    p.add_argument("-a", "--activation", type=str, default="TopKReLU_256")
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--prevalence_threshold", type=float, default=0.15)
    p.add_argument("--concept_match_vocab", type=str, default="waterbirds_domain")
    p.add_argument("--stage1_manifest", type=str, default=None,
                   help="Explicit path to a stage1_manifest.json, overriding the "
                        "<base_sae_dir>/stage1_manifest.json this would otherwise "
                        "derive from the args above.")


def add_concept_finding_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--concept_finding_method", type=str, default="conceptscope",
                   choices=["labelfree", "labelguided", "dialguided", "highmag", "conceptscope", "none"],
                   help="'prism_baseline' is intentionally not offered here -- it's a "
                        "fully self-contained baseline that bypasses concept-finding/"
                        "ablation/MACO entirely (writes its own report and returns "
                        "early in msae_ftclip.py). Run it via msae_ftclip.py directly.")
    p.add_argument("--top_k_concepts", type=int, default=20)
    p.add_argument("--denoise_concepts", action="store_true", default=True)
    p.add_argument("--denoise_percentile", type=float, default=0.8)
    p.add_argument("--denoise_beta", type=float, default=0.8)
    p.add_argument("--n_cpu_workers", type=int, default=10)
    p.add_argument("--concept_pool_batch_size", type=int, default=64)
    p.add_argument("--labelfree_scoring", type=str, default="causal",
                   choices=["prevalence", "causal"],
                   help="Which labelfree selector to run (they are alternatives). "
                        "'prevalence' (default, unchanged): select_prevalent_concepts, "
                        "ranking a concept by how many candidate images it flips. "
                        "'causal': score_labelfree_concepts, ranking by how often a "
                        "concept is among an image's strongest flippers and by how "
                        "much probability mass those flips move. 'causal' needs the "
                        "per-flip probability deltas, so it runs Generate_Concept_Pool "
                        "with record_deltas=True -- a pool cached without them forces "
                        "a full recompute of the causal search.")
    p.add_argument("--labelfree_top_k_per_image", type=int, default=10,
                   help="--labelfree_scoring causal only: how many of each candidate "
                        "image's strongest flipping concepts count toward the score.")
    p.add_argument("--conceptscope_split", type=str, default="ft_train",
                   choices=["ft_train", "test"],
                   help="conceptscope only: which split to find and categorise concepts "
                        "on. 'ft_train' (default) is the fine-tuning training set -- the "
                        "images the biased model learned its shortcut from; 'test' is "
                        "the held-out split.")
    p.add_argument("--conceptscope_image_subset", type=int, default=512,
                   help="conceptscope only: images PER PREDICTED CLASS that get per-patch "
                        "codes -- drawn in equal numbers per class regardless of "
                        "background, capped by the smallest class (the biased ft_train "
                        "split has ~380 per class, so 512 there means all of them). "
                        "0 = the whole split.")
    p.add_argument("--conceptscope_num_samples", type=int, default=512,
                   help="conceptscope only: images sampled per PREDICTED class for the "
                        "masking test (ConceptScope's num_samples_for_alignment); capped "
                        "by what --conceptscope_image_subset provides for that class.")
    p.add_argument("--conceptscope_top_k", type=int, default=50,
                   help="conceptscope only: latents scored per predicted class, the top-k "
                        "by class-mean activation (ConceptScope's top_k_for_alignment).")
    p.add_argument("--conceptscope_alignment", type=str, default="pred",
                   choices=["pred", "prob"],
                   help="conceptscope only: categorise on hard prediction fractions ('pred': "
                        "does the prediction survive / flip under the mask) or on the "
                        "probability-ratio form ('prob') -- the continuous fallback when "
                        "hard fractions tie.")
    p.add_argument("--conceptscope_bias_sigma", type=float, default=1.0,
                   help="conceptscope only: a context latent is flagged as bias (spurious) "
                        "when its mean activation for a predicted class is >= mean + "
                        "this many std of the class's context latents. The knob that sets "
                        "how many concepts come out; 1.0 is ConceptScope's default.")
    p.add_argument("--conceptscope_target_threshold", type=float, default=0.0,
                   help="conceptscope only: a latent is target (not context) when its "
                        "z-scored alignment within the predicted class is >= this. "
                        "0.0 (ConceptScope's default) = above the class mean.")
    p.add_argument("--labelfree_pool_knn_ks", type=str, default=None,
                   help="labelfree only: build the concept pool over the UNION of the "
                        "candidate-image sets for these comma-separated knn_k values "
                        "(e.g. 5,10,15,20,25) instead of just this run's --knn_k. The "
                        "pool cache is keyed by the exact image set, and the candidate "
                        "set changes with knn_k, so a sweep over knn_k would otherwise "
                        "redo the full causal search once per value. Every run in the "
                        "sweep must pass the SAME list (this run's --knn_k is always "
                        "included) so they all hash to one cache and only the first "
                        "pays for it. The selectors still use this run's own "
                        "candidate mask; a pool over a superset of images gives "
                        "identical results, since each image's flipping concepts are "
                        "computed independently.")
    p.add_argument("--labelfree_search_all_images", action="store_true",
                   help="labelfree only: run the causal search over the whole test "
                        "split instead of just the candidate images. Both selectors "
                        "discard non-candidate images anyway, so this only costs time "
                        "(~5x on waterbirds: 5,794 images vs 1,105 candidates) without "
                        "changing their output -- it exists for reproducing an older "
                        "full-split concept pool.")
    p.add_argument("--no_cache_images", action="store_true",
                   help="labelfree only: re-decode each image from disk for every "
                        "(concept, image) pair instead of preprocessing it once and "
                        "holding it in RAM. The cache costs ~0.7 GB for the candidate "
                        "subset (~3.5 GB for a full split); without it each image is "
                        "decoded ~1,000 times. Use only if RAM is tight.")


def add_ablation_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ablation_coefficient", type=float, default=.9)
    p.add_argument("--projection_method", type=str, default="qr", choices=["qr", "pinv"])
    p.add_argument("--concept_score_threshold", type=str, default="median",
                   help="Minimum score for a concept to count as ACTIVE, i.e. to be "
                        "eligible for ablation at all (kept when score >= this). "
                        "--ablation_top_percent then applies to the active set, not "
                        "to the whole file. Stage 2 keeps every concept scoring above "
                        "zero, which is nearly the entire dictionary, so without a "
                        "threshold even a small percentage pulls in a long tail of "
                        "concepts scoring barely above nothing. Takes a quantile of "
                        "the loaded file's OWN scores -- 'median' (default) or 'pNN' "
                        "such as p90 -- or a plain number. The quantile forms are "
                        "self-calibrating: scores sit on a method-specific scale "
                        "(highmag reaches ~8.8, labelfree ~1e-3), so one absolute "
                        "number cannot mean the same thing across two files. Pass 0 "
                        "for no filtering, i.e. every concept in the file is active. "
                        "A number > 0 requires a scored --concept_finding_method; "
                        "the quantile forms fall back to no filtering on an unscored "
                        "one.")
    p.add_argument("--ablation_top_percent", type=float, default=100.0,
                   help="Ablate the top P%% of the ACTIVE concepts by score, e.g. 10 "
                        "for the top tenth: N = ceil(P/100 x n_active), at least 1. "
                        "Active means clearing --concept_score_threshold, which "
                        "defaults to no filtering -- so with both left at their "
                        "defaults this is the top P%% of everything stage 2 wrote. "
                        "Must be in (0, 100]; the default 100 ablates the whole "
                        "active set. P and the resolved N both go into the report "
                        "and prediction filenames so different values don't "
                        "overwrite each other. (Replaces --ablation_max_concepts, "
                        "which took an absolute count.)")


def add_maco_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--maco_max_concepts", type=int, default=200)
    p.add_argument("--maco_num_steps", type=int, default=128)
    p.add_argument("--maco_num_crops", type=int, default=16)
    p.add_argument("--maco_reference_samples", type=int, default=64)
    p.add_argument("--maco_concept_batch", type=int, default=4)
    p.add_argument("--maco_early_stop_patience", type=int, default=0)
    p.add_argument("--maco_early_stop_delta", type=float, default=1e-3)
    p.add_argument("--maco_shard", type=str, default="0/1")
    p.add_argument("--maco_workers", type=int, default=10)
    p.add_argument("--skip_montages", action="store_true",
                   help="Skip the real-image top-activation montages alongside MACO's "
                        "synthetic renders (msae_ftclip.save_top_ft_images_per_concept). "
                        "Montages are capped at 50 concepts regardless.")


# ── Manifest path + I/O ──────────────────────────────────────────────────────

def activation_str_for(args) -> str:
    """Exact string main() builds from --activation/--model, verbatim from
    msae_ftclip.py's MSAE branch (lines ~6228-6239) -- only meaningful for
    the MSAE family; RouteSAE's tag comes from its checkpoint filename
    instead (see sae_tag_for)."""
    act = args.activation
    if args.model == "ReLUSAE" and "_" in act:
        act_base, sparse_w = act.split("_", 1)
        return f"{act_base}_{str(float(f'0.{sparse_w}')).split('.')[1]}"
    elif args.model in ["MSAE_UW", "MSAE_RW"]:
        if act is None:
            return "TopK_64_" + args.model.replace("MSAE_", "")
        return act
    return act


def sae_tag_for(args) -> str:
    """Same per-branch sae_tag formula main() uses to name base_sae_dir --
    RouteSAE checkpoints self-describe (K/latent_size/source already in the
    filename), MSAE's is model+activation since msae_ftclip.py's --sae_path
    for that family may point at an arbitrary checkpoint file/dir rather
    than one whose name encodes those."""
    if args.model == "RouteSAE":
        return os.path.splitext(os.path.basename(args.sae_path))[0]
    return f"{args.model}_{activation_str_for(args)}"


def sae_dirs(args, run_dir: str) -> tuple[str, str]:
    """(base_sae_dir, sae_dir), same formula main() uses in msae_ftclip.py --
    base_sae_dir names the representations cache (depends on clip_mode + SAE
    identity only), sae_dir the analysis output (also depends on knn_k/
    prevalence_threshold). Takes `run_dir` explicitly (the value
    core.resolve_run_dir(args) returned) rather than reading args.run_dir --
    resolve_run_dir can auto-discover a directory different from the literal
    --run_dir string (matching ft_tag/max_samples/seed against a
    clip_ft_*/model.pt), and every stage must agree on the same resolved
    path for their manifest/output locations to line up."""
    clip_tag = "zs" if args.clip_mode == "zeroshot" else "ft"
    sae_tag = sae_tag_for(args)
    base_sae_dir = os.path.join(run_dir, f"{clip_tag}_{sae_tag}")
    sae_dir = os.path.join(base_sae_dir, f"knn{args.knn_k}_prev{args.prevalence_threshold}")
    return base_sae_dir, sae_dir


def embedding_path_for(run_dir: str) -> str:
    return os.path.join(os.path.dirname(run_dir), "embeddings")


def resolve_vocab_paths(args, run_dir: str, sae_path: str) -> tuple[str, str | None]:
    """(vocab_path_to_load, names_txt_or_None) -- verbatim port of main()'s
    concept_match/vocab resolution (msae_ftclip.py lines ~6441-6498): picks
    the concept_match .npy matching this SAE's checkpoint basename (+
    --concept_match_vocab substring, newest if still ambiguous), then the
    row-aligned names .txt next to it, falling back to clip_disect_20k.txt
    with a warning if no aligned names file exists. Raises the same
    FileNotFoundError/ValueError main() would on a missing/misaligned file."""
    import glob
    import re

    embedding_path = embedding_path_for(run_dir)
    if args.model in ["ReLUSAE", "TopKSAE", "BatchTopKSAE"]:
        vocab_path = os.path.join(embedding_path, f"concept_match/{args.model}/{args.activation}")
    else:
        vocab_path = os.path.join(embedding_path, f"concept_match/{args.model}")

    sae_basename = os.path.splitext(os.path.basename(sae_path))[0]
    if os.path.isdir(vocab_path):
        candidates = sorted(glob.glob(os.path.join(vocab_path, "*.npy")))
        matches = [p for p in candidates if sae_basename in os.path.basename(p)]
        if args.concept_match_vocab:
            matches = [p for p in matches if args.concept_match_vocab in os.path.basename(p)]
        if not matches:
            raise FileNotFoundError(
                f"No concept_match .npy for SAE '{sae_basename}'"
                + (f" + vocab '{args.concept_match_vocab}'" if args.concept_match_vocab else "")
                + f" in {vocab_path}.\nRun msae/sae_naming.py for this SAE/vocab first."
            )
        vocab_path_to_load = max(matches, key=os.path.getmtime)
    elif os.path.isfile(vocab_path):
        vocab_path_to_load = vocab_path
    else:
        raise FileNotFoundError(
            f"Concept_match path not found: {vocab_path}\n"
            f"(neither a directory of Concept_Interpreter_*.npy files, nor a single "
            f".npy file). This usually means concept naming (msae/sae_naming.py or "
            f"routesae_naming.py) was never run for this SAE against --run_dir "
            f"{run_dir!r} -- check --run_dir points at the run this checkpoint's "
            f"naming/concept_match artifacts actually belong to."
        )

    names_txt = None
    prefix = f"Concept_Interpreter_{sae_basename}_"
    base = os.path.splitext(os.path.basename(vocab_path_to_load))[0]
    if base.startswith(prefix):
        vocab_tag = re.sub(r"_\d+$", "", base[len(prefix):])
        candidate_names_txt = os.path.join(embedding_path, vocab_tag + ".txt")
        if os.path.isfile(candidate_names_txt):
            names_txt = candidate_names_txt

    concept_match_scores = np.load(vocab_path_to_load)
    vocab_names = _read_vocab_names(names_txt)
    if len(vocab_names) != concept_match_scores.shape[0]:
        raise ValueError(
            f"vocab_names ({len(vocab_names)}) does not match concept_match rows "
            f"({concept_match_scores.shape[0]}). The names file is not aligned to "
            f"the selected scores: {vocab_path_to_load}"
        )
    return vocab_path_to_load, names_txt


def _read_vocab_names(names_txt: str | None) -> list[str]:
    if names_txt:
        with open(names_txt) as f:
            return [line.strip() for line in f if line.strip()]
    with open("msae/vocab/clip_disect_20k.txt") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_run_dir_for_pipeline(args) -> tuple[str, dict]:
    """core.resolve_run_dir(args), after applying the same --output_dir
    default main() applies first. Shared by all 4 stages so they agree on
    the same resolved run_dir from the same core args."""
    if args.output_dir is None:
        args.output_dir = os.path.join(".", "results", args.dataset)
    return core.resolve_run_dir(args)


def manifest_path_for(args, run_dir: str) -> str:
    if getattr(args, "stage1_manifest", None):
        return args.stage1_manifest
    base_sae_dir, _ = sae_dirs(args, run_dir)
    return os.path.join(base_sae_dir, MANIFEST_NAME)


def write_manifest(path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved to {path}")


def apply_analysis_params(manifest: dict, args) -> dict:
    """Point manifest['sae_dir'] at the analysis folder for THIS invocation's
    --knn_k / --prevalence_threshold, and make sure it exists.

    Stage 1 records sae_dir once, with whatever knn_k/prevalence it was run
    with, and is a no-op from then on -- so without this every later stage
    would read the frozen value and write its outputs into that one folder
    regardless of the analysis params it was actually given. That made a
    sweep over knn_k/prevalence impossible: every point overwrote the same
    concepts_<method>.json. base_sae_dir (representations, manifest, pool
    cache) is genuinely shared across analysis params and is kept as-is.

    Same formula sae_dirs() uses, so with the default knn_k=10 /
    prevalence=0.15 this resolves to exactly the folder stage 1 recorded and
    nothing already on disk moves. Call it right after read_manifest() in
    every stage that reads one.
    """
    manifest["sae_dir"] = os.path.join(
        manifest["base_sae_dir"], f"knn{args.knn_k}_prev{args.prevalence_threshold}")
    os.makedirs(manifest["sae_dir"], exist_ok=True)
    return manifest


def read_manifest(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No stage-1 manifest at {path}.\n"
            f"Run pipeline_1_setup.py with the same core args first "
            f"(--run_dir/--clip_mode/--model/--sae_path/...)."
        )
    with open(path) as f:
        return json.load(f)


def concepts_path_for(manifest: dict, method: str) -> str:
    return os.path.join(manifest["sae_dir"], f"concepts_{method}.json")


def write_concepts(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Concepts saved to {path}  ({len(payload['candidate_concepts'])} concepts)")


def read_concepts(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No concept list at {path}.\n"
            f"Run pipeline_2_find_concepts.py with this --concept_finding_method first."
        )
    with open(path) as f:
        return json.load(f)


# ── Candidate images (core.candidate_selection) -- one per sae_dir, shared ──
# across every --concept_finding_method (unlike concepts_<method>.json, this
# doesn't depend on the method: candidate_selection only reads te_results +
# clip_ft + knn_k, all fixed once knn_k is). Computed once by
# pipeline_2_find_concepts.py, consumed there (to build candidate_report.csv)
# and by pipeline_3_ablate.py (to build the "*_candidates_only" ablation
# variants).

def candidate_images_path_for(manifest: dict) -> str:
    return os.path.join(manifest["sae_dir"], "candidate_images.json")


def write_candidate_images(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Candidate images saved to {path}  "
          f"({payload['n_candidates']}/{payload['n_total']} candidates)")


def read_candidate_images(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No candidate_images.json at {path}.\n"
            f"Run pipeline_2_find_concepts.py first (candidate selection now "
            f"runs for every --concept_finding_method, before dispatching to "
            f"the method itself)."
        )
    with open(path) as f:
        return json.load(f)


# ── Representations cache (same on-disk format as msae_ftclip.save_representations) ─

def representations_cached(base_sae_dir: str, split_name: str) -> bool:
    return os.path.exists(os.path.join(base_sae_dir, f"representations_{split_name}"))


def load_cached_representations(base_sae_dir: str, split_name: str) -> dict:
    path = os.path.join(base_sae_dir, f"representations_{split_name}")
    m = pd.read_csv(os.path.join(path, "metrics.csv"), index_col="img_path")
    return {
        "clip_representations": torch.load(os.path.join(path, "clip_representations.pt"),
                                            map_location="cpu", weights_only=False),
        "sae_representations":  torch.load(os.path.join(path, "sae_representations.pt"),
                                            map_location="cpu", weights_only=False),
        "sae_reconstructed":    torch.load(os.path.join(path, "sae_reconstructed.pt"),
                                            map_location="cpu", weights_only=False),
        "metrics":     m.to_dict(orient="list"),
        "image_paths": m.index.tolist(),
    }


def resolve_device() -> str:
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")


# ── Stage-1 context reload (used by stages 2, 3, 4) ─────────────────────────

def load_stage1_context(manifest: dict) -> dict:
    """Reload CLIP + SAE + datasets + cached representations + concept_match
    from a saved stage-1 manifest -- mirrors msae_ftclip.main()'s setup
    exactly, just reading resolved paths from the manifest instead of
    re-deriving them from CLI args."""
    core._hf_model_dir = manifest.get("hf_model_dir")
    device = resolve_device()

    run_dir = manifest["run_dir"]
    train_ds = core.ManifestDataset(os.path.join(run_dir, "train_all_manifest.csv"))
    ft_ds    = core.ManifestDataset(os.path.join(run_dir, "ft_train_manifest.csv"))
    test_ds  = core.ManifestDataset(os.path.join(run_dir, "test_manifest.csv"))

    if manifest["clip_mode"] == "zeroshot":
        clip_ft = core.CLIPZeroShot(model_name=manifest["clip_model"], device=device)
    else:
        clip_ft = core.CLIPZeroShot.load_model(os.path.join(run_dir, "model.pt"), device=device)
    clip_ft.model.eval()

    base_sae_dir, sae_dir = manifest["base_sae_dir"], manifest["sae_dir"]

    if manifest["model"] == "RouteSAE":
        from routesae_adapter import load_routesae_for_clip
        # Same fp32 upcast main() applies -- RouteSAE's hook-based extraction
        # (routesae.py) explicitly upcasts intermediate activations, and a
        # still-fp16 CLIP (possible on non-CPU devices via load_model) crashes
        # at the first matmul against a fp16 weight otherwise.
        clip_ft.model = clip_ft.model.float()
        sae_model = load_routesae_for_clip(
            manifest["sae_path"], device=device, k=manifest["routesae_k"])
    else:
        from msae.sae import SAE
        sae_model = SAE(manifest["sae_path"]).to(device).eval()

    for split, ds in (("ft_train", ft_ds), ("test", test_ds)):
        if not representations_cached(base_sae_dir, split):
            raise FileNotFoundError(
                f"No cached representations at {base_sae_dir}/representations_{split} -- "
                f"run pipeline_1_setup.py first (or rerun it if this manifest predates "
                f"a checkpoint/run_dir change)."
            )
    ft_results = load_cached_representations(base_sae_dir, "ft_train")
    te_results = load_cached_representations(base_sae_dir, "test")

    concept_match_scores = np.load(manifest["vocab_path_to_load"])
    vocab_names = _read_vocab_names(manifest.get("vocab_names_txt"))

    return dict(
        device=device, run_dir=run_dir,
        train_ds=train_ds, ft_ds=ft_ds, test_ds=test_ds,
        clip_ft=clip_ft, sae_model=sae_model,
        base_sae_dir=base_sae_dir, sae_dir=sae_dir,
        ft_results=ft_results, te_results=te_results,
        concept_match_scores=concept_match_scores, vocab_names=vocab_names,
        sae_path=manifest["sae_path"],
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
