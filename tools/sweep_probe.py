#!/usr/bin/env python
"""Print shell-eval-able facts about a stage-3 sweep before it runs.

Exists so sweep_labelfree_ablation.sh never re-derives paths itself: it
builds pipeline_3_ablate's own parser and calls the same resolution helpers,
so the concepts file the sweep checks is by construction the one stage 3
will load. Takes the same core args as stage 3, plus --percents.

Output (KEY=VALUE lines, safe for `eval`):
    SAE_DIR, CONCEPTS_PATH, CONCEPTS_EXIST, N_TOTAL, N_ACTIVE, HAS_SCORES,
    THRESHOLD_VALUE, PERCENTS_DISTINCT

PERCENTS_DISTINCT drops the percentages that resolve to a concept count some
smaller percentage already covers -- N = ceil(P/100 x n_active) repeats
whenever the active set is small, and two runs with the same N ablate the
same concepts and produce identical predictions under different filenames.

Usage:
    python tools/sweep_probe.py --sae_path <ckpt> --concept_finding_method labelfree \
        --concept_score_threshold median --percents 2,4,6,...,100
"""

import math
import os
import sys

# Repo root, so this runs as `python tools/sweep_probe.py` from anywhere
# without the caller having to set PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline_common as pc      # noqa: E402
import pipeline_3_ablate as p3    # noqa: E402


def main():
    parser = p3.build_parser()
    parser.add_argument("--percents", type=str, default="",
                        help="Comma-separated percentages the sweep intends to run.")
    args = parser.parse_args()

    run_dir, _ = pc.resolve_run_dir_for_pipeline(args)
    manifest_path = pc.manifest_path_for(args, run_dir)
    print(f"MANIFEST_PATH={manifest_path}")

    if not os.path.isfile(manifest_path):
        # Not an error here either -- the sweep script turns this into a
        # "run stage 1 first" message. Without this branch read_manifest
        # raises and the caller only sees a traceback, which is exactly what
        # happens the first time a sweep is pointed at a K/clip_mode combo
        # whose stage 1 has not been run.
        print("MANIFEST_EXISTS=0")
        print('SAE_DIR=""'); print('CONCEPTS_PATH=""'); print("CONCEPTS_EXIST=0")
        print("N_TOTAL=0"); print("N_ACTIVE=0"); print("HAS_SCORES=0")
        print("THRESHOLD_VALUE=0"); print('PERCENTS_DISTINCT=""')
        return 0

    print("MANIFEST_EXISTS=1")
    manifest = pc.apply_analysis_params(pc.read_manifest(manifest_path), args)
    concepts_path = pc.concepts_path_for(manifest, args.concept_finding_method)

    print(f"SAE_DIR={manifest['sae_dir']}")
    print(f"CONCEPTS_PATH={concepts_path}")

    try:
        concepts = pc.read_concepts(concepts_path)
    except FileNotFoundError:
        # Not an error here -- the sweep script turns this into a clear
        # "run stage 2 first" message (or runs stage 2 itself).
        print("CONCEPTS_EXIST=0")
        print("N_TOTAL=0"); print("N_ACTIVE=0"); print("HAS_SCORES=0")
        print("THRESHOLD_VALUE=0"); print('PERCENTS_DISTINCT=""')
        return 0

    scores = concepts.get("concept_scores")
    print("CONCEPTS_EXIST=1")
    print(f"HAS_SCORES={1 if scores else 0}")

    threshold = 0.0
    if scores:
        threshold, _label = p3.resolve_score_threshold(
            args.concept_score_threshold, scores)
    print(f"THRESHOLD_VALUE={threshold:.10g}")

    _sel, n_total, n_active, _by_score = p3.select_top_percent(
        concepts["candidate_concepts"], scores, 100.0, threshold)
    print(f"N_TOTAL={n_total}")
    print(f"N_ACTIVE={n_active}")

    percents, seen, kept = [], set(), []
    if args.percents.strip():
        percents = [float(x) for x in args.percents.split(",") if x.strip()]
    for p in percents:
        n = max(1, math.ceil(n_active * p / 100.0)) if n_active else 0
        if n not in seen:
            seen.add(n)
            kept.append(f"{p:g}")
    print(f'PERCENTS_DISTINCT="{" ".join(kept)}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
