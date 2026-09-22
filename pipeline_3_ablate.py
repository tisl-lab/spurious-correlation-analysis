#!/usr/bin/env python
"""
Stage 3 of 4 — ablation: load a concepts_<method>.json + the stage-1
manifest, ablate the top P% of that method's concepts with ONE editing
technique, and save both a report and the resulting predictions.

  --editing_method deactivation   zero the selected latents (default)
  --editing_method projection     project them out of the representation

  --ablation_images all           edit every test image (default)
  --ablation_images candidates    edit only the images candidate_images.json
                                  flags (pseudo-label wrong or uncertain by
                                  centroid / k-NN disagreement); every other
                                  image keeps its original prediction

Only the requested combination runs -- one editing technique over one image
set, one report, one predictions file. (Earlier revisions ran all four
combinations every time and wrote a single side-by-side report.)

Restricting to candidate images is applied by selecting between the ablated
and original predictions afterwards, not by hooking only those images: every
ablation hook acts on one image at a time, with no batch-level statistics,
so the two are equivalent for the images that are included -- and doing it
afterwards keeps the CLIP pass batched.

How many concepts get ablated is decided HERE, not in concept-finding, in
two steps -- stage 2 keeps every concept its method scores above zero, which
is very nearly the whole dictionary, so a percentage on its own would still
be a percentage of "everything":

  --concept_score_threshold S   a concept is ACTIVE, i.e. eligible to be
                                ablated at all, when its score >= S.
                                Default 'median' -- the better-scoring half
                                of whatever concept file was loaded.
  --ablation_top_percent P      ablate the top P% of the ACTIVE concepts by
                                score: N = ceil(P/100 x n_active), min 1.

S is resolved PER CONCEPT FILE: 'median' and 'pNN' (e.g. p90) are quantiles
of that file's own scores, so the threshold self-calibrates instead of
needing a new hand-picked number for every method, SAE and dataset -- the
scores are on a method-specific scale (highmag reaches ~8.8 on the run this
was built against, labelfree ~1e-3). A plain number still works, and stage 3
prints the loaded file's distribution (max/p99/p90/median) on every run so
one can be read off the previous run's output. Pass 0 for no filtering.

S, P, the resolved N and the ablation coefficient C all go into the output
filenames, so sweeping any of them leaves one report and one predictions
file per combination instead of overwriting (and the no-op check stays
per-combination); the filename carries the threshold spec ('__smedian'),
while the number it resolved to goes in the summary row.

Outputs, in the SAE dir. <variant> is the editing method, plus a
"_candidates_only" suffix under --ablation_images candidates; the __s<S>
segment is omitted when S is 0:
  ablation_report_<extractor>__<variant>__s<S>__p<P>_top<N>__c<C>.txt
      per-group accuracy vs the original predictions, and the ablated
      concept ids
  predictions_<extractor>__<variant>__s<S>__p<P>_top<N>__c<C>.csv
      per image: img_path, group_id, true_label, orig_pred, ablated_pred
      (plus is_candidate in candidates mode). ablated_pred is the FINAL
      prediction, so in candidates mode the unedited rows repeat orig_pred.
  ablation_summary_<concept_finding_method>.csv
      ONE cumulative table per concept-finding method, one row appended per
      run of that method: per-group / average / worst-group accuracy, how
      many concepts were ablated and what fraction of the file that is, the
      threshold, and the editing method. Created with the unablated model's
      accuracies as its first row, so every later row has its baseline.
      Per method rather than per SAE dir because concept counts and score
      scales differ between methods, so those columns only compare within
      one -- the method is still a column, so the files concatenate cleanly.
  <base_sae_dir>/original_logits.npz
      the unablated CLIP logits, cached one level up: they depend only on
      CLIP, the prompts and the test split -- not on the SAE's analysis
      params -- so every knn*_prev*/ folder under one (clip_mode, SAE)
      shares it and only the first run anywhere pays for that pass.

No-op on rerun: if the report file already exists, this does nothing.
Delete it to force a redo.

Usage:
    python pipeline_3_ablate.py --clip_mode finetuned --run_dir <dir> \
        --model RouteSAE --sae_path <checkpoint.pt> --routesae_k 32 \
        --concept_finding_method labelguided --editing_method deactivation \
        --ablation_top_percent 10        # threshold defaults to the file's median
"""

import argparse
import contextlib
import csv
import fcntl
import math
import os
import random
import re
import time
import uuid

import numpy as np

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
    p.add_argument("--editing_method", type=str, default="projection",
                   choices=["deactivation", "projection"],
                   help="Which ablation technique to run. Only this one runs -- it "
                        "picks the hook (make_sae_ablation_hook vs "
                        "projection_ablation_hook) and names the output files.")
    p.add_argument("--ablation_images", type=str, default="candidates",
                   choices=["all", "candidates"],
                   help="Which images the edit applies to. 'all' (default) ablates "
                        "every test image. 'candidates' ablates only the images "
                        "candidate_images.json flags -- the ones whose pseudo-label "
                        "core.candidate_selection found wrong or uncertain (centroid "
                        "/ k-NN disagreement) -- and leaves every other image at its "
                        "original, unedited prediction. Written by stage 2 for every "
                        "concept-finding method, so it is already on disk.")
    pc.add_ablation_args(p)

    # Debug convenience (same as pipeline_2_find_concepts.py's build_parser):
    # default the one required, unsafe-to-guess field (--sae_path) plus
    # clip_mode/hf_model_dir to match it, to the RouteSAE K=32 zeroshot combo
    # this stage was verified against -- so continuing straight on from a
    # stage-2 run of that same combo (concepts_<method>.json already on disk)
    # needs no CLI args at all. Still fully overridable.
    p.set_defaults(
        clip_mode="zeroshot",
        sae_path="routesae_weights/routesae_K32_ViT-B~32_16384.pt",
        hf_model_dir="hf_models",
    )
    next(a for a in p._actions if a.dest == "sae_path").required = False

    return p


def resolve_score_threshold(spec, concept_scores):
    """Turn a --concept_score_threshold spec into an absolute score.

    Accepts either a plain number, or a quantile of THIS concept file's own
    score distribution:

      'median'  (the default)  the file's 50th percentile
      'pNN'                    its NNth percentile, e.g. p90
      any number               used as-is

    The quantile forms exist because the scores are on a method-specific
    scale -- highmag's prevalence x activation reaches ~8.8 on the run this
    was built against, labelfree's prevalence x probability shift is ~1e-3 --
    so one absolute number cannot mean the same thing across two files, and
    would have to be re-derived by hand for every method, SAE and dataset.
    A quantile is self-calibrating: 'median' keeps the better-scoring half of
    whatever file it is pointed at.

    Returns (value, label), where label is the filename-safe form of the spec
    ('median', 'p90', or the number) -- the resolved value is not used in the
    filename because it is a long float that differs per file, and it is
    recorded in the run's JSON and printed instead.

    Raises SystemExit on an unparseable spec, or on a quantile request with
    no scores to take a quantile of (main() handles unscored files before
    calling, so that is a guard, not the normal path).
    """
    s = str(spec).strip().lower()

    try:
        return float(s), f"{float(s):g}"
    except ValueError:
        pass

    if not concept_scores:
        raise SystemExit(
            f"--concept_score_threshold {spec!r} asks for a quantile of this concept "
            f"file's scores, but the file has none (concept_scores is null)."
        )
    v = np.array([float(x) for x in concept_scores.values()])

    if s == "median":
        return float(np.percentile(v, 50)), "median"
    m = re.fullmatch(r"p(\d+(?:\.\d+)?)", s)
    if m and 0 <= float(m.group(1)) <= 100:
        return float(np.percentile(v, float(m.group(1)))), f"p{m.group(1)}"

    raise SystemExit(
        f"--concept_score_threshold {spec!r} not understood. Pass a number "
        f"(e.g. 0.5), 'median', or a percentile of this file's scores (e.g. p90)."
    )


def score_quantiles(concept_scores):
    """Score distribution of a concepts_<method>.json, for choosing a threshold.

    Returns a list of (label, value) pairs, or [] for an unscored method.
    Printed by main() on every run so a --concept_score_threshold given as a
    plain number can be read off the previous run's output instead of
    guessed. The 'median'/'pNN' forms don't need it -- they read the same
    distribution themselves -- but it still shows what the resolved threshold
    is being compared against.
    """
    if not concept_scores:
        return []
    v = np.sort(np.array([float(s) for s in concept_scores.values()]))[::-1]
    return [("max", v[0]), ("p99", np.percentile(v, 99)), ("p90", np.percentile(v, 90)),
            ("median", np.percentile(v, 50)), ("min", v[-1])]


def select_top_percent(candidate_concepts, concept_scores, percent, score_threshold=0.0):
    """Keep the concepts scoring at or above `score_threshold`, then take the
    top `percent`% of those by score.

    The threshold comes first and the percentage second, because they answer
    different questions. Stage 2 keeps every concept scoring above zero, which
    for these SAEs is very nearly the whole dictionary (16,309 of 16,384 on the
    highmag run this was built against) -- so a percentage alone is a fraction
    of "all concepts", and even a small one drags in a long tail of concepts
    whose score is barely distinguishable from nothing. score_threshold is
    what decides which concepts count as ACTIVE at all; percent then says how
    much of that active set to actually ablate.

    A concept is kept when score >= score_threshold. The default 0.0 keeps
    every concept in the file (stage 2 wrote only score > 0 ones), i.e. the
    behavior before the threshold existed.

    Stage 2 already writes the list score-sorted, but it is re-sorted here
    when scores are present rather than trusted: the slice below is only "the
    top P% by score" if the order actually reflects the scores, and a JSON
    written by an older stage 2 (or by a method whose scores were added after
    the fact) may not. Sorting is stable, so concepts tied on score keep the
    file's relative order.

    Methods without a scoring criterion (dialguided, and labelfree unless it
    ran with --labelfree_scoring causal) write concept_scores=null; for those
    the slice is the head of the method's own ordering, which the caller says
    out loud rather than implying a ranking that doesn't exist. A threshold
    is meaningless there -- there are no scores to compare against it -- so
    passing one raises rather than silently ignoring it.

    Returns (selected_concepts, n_total, n_active, ordered_by_score), where
    n_total is everything in the file and n_active is what cleared the
    threshold (equal when the threshold is 0).
    """
    n_total = len(candidate_concepts)
    ordered_by_score = bool(concept_scores)

    if not ordered_by_score:
        if score_threshold > 0:
            raise SystemExit(
                f"--concept_score_threshold {score_threshold:g} was given, but this "
                f"concept list has no scores (concept_scores is null in the JSON), so "
                f"nothing can be compared against it. Either drop the threshold or use "
                f"a scored --concept_finding_method (highmag, labelguided, or labelfree "
                f"with --labelfree_scoring causal)."
            )
        # Unscored: no filtering possible, the file's own order is the ranking.
        n_select = max(1, math.ceil(n_total * percent / 100.0)) if n_total else 0
        return candidate_concepts[:n_select], n_total, n_total, False

    # JSON object keys are strings -- back to ints to index by concept id.
    scores = {int(cid): float(s) for cid, s in concept_scores.items()}
    active = [c for c in candidate_concepts if scores.get(int(c), 0.0) >= score_threshold]
    active.sort(key=lambda cid: scores.get(int(cid), 0.0), reverse=True)
    n_active = len(active)

    # ceil so a small percentage of a long list still selects something, and
    # at least 1 so a valid P can never resolve to an empty ablation (P is
    # validated > 0 by the caller). An empty ACTIVE set is a different thing
    # -- the threshold excluded everything -- and is the caller's to report.
    n_select = max(1, math.ceil(n_active * percent / 100.0)) if n_active else 0
    return active[:n_select], n_total, n_active, ordered_by_score


def save_predictions(path_csv, image_paths, group_ids, true_labels,
                     orig_preds, ablated_preds, candidate_mask=None):
    """Write this run's per-image predictions.

    Keeps the original prediction alongside the ablated one so a downstream
    analysis can recover any subset comparison -- per group, per class, or
    over an image mask -- without re-running the ablation. `ablated_pred` is
    the FINAL prediction: under --ablation_images candidates the non-candidate
    rows hold the original prediction, because those images were not edited.
    The is_candidate column (written only in that mode) says which is which.
    """
    header = ["img_path", "group_id", "true_label", "orig_pred", "ablated_pred"]
    if candidate_mask is not None:
        header.append("is_candidate")
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(image_paths)):
            r = [image_paths[i], int(group_ids[i]), int(true_labels[i]),
                 int(orig_preds[i]), int(ablated_preds[i])]
            if candidate_mask is not None:
                r.append(int(bool(candidate_mask[i])))
            w.writerow(r)
    print(f"\nPredictions saved to {path_csv}")


def accuracy_fields(preds, true_labels, group_ids):
    """Per-group, overall and worst-group accuracy, as a dict of columns.

    Accuracies are stored as fractions rather than percentages so the file
    can be read straight into a plot or a table without parsing a '%' off
    the end; the reports alongside it print percentages for reading.
    """
    preds = np.asarray(preds)
    row = {}
    for gid in sorted(set(group_ids.tolist())):
        m = group_ids == gid
        row[f"acc_group_{int(gid)}"] = round(float((preds[m] == true_labels[m]).mean()), 6)
    row["accuracy_avg"] = round(float((preds == true_labels).mean()), 6)
    row["accuracy_worst_group"] = round(
        min(v for k, v in row.items() if k.startswith("acc_group_")), 6)
    return row


@contextlib.contextmanager
def _file_lock(path, timeout=180.0):
    """Hold an exclusive lock on <path>.lock, or yield False on timeout.

    The summary CSV is read-modify-written (and fully rewritten when a new
    column appears), so two stage-3 processes finishing at the same moment --
    routine under a SLURM array -- would interleave into a corrupt file. One
    lock around the whole sequence serializes them; contention is negligible
    since the critical section is a few milliseconds against runs of minutes.

    Yields False rather than raising when the lock can't be taken (a
    filesystem that doesn't honor flock, or a stale holder), so the caller
    can fall back to writing its row somewhere private instead of losing it.
    """
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    handle = open(lock_path, "w")
    deadline = time.time() + timeout
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.time() >= deadline:
                    break
                # Jittered so tasks that collided once don't collide again in
                # lockstep on every retry.
                time.sleep(0.1 + random.random() * 0.3)
        yield acquired
    finally:
        if acquired:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _write_shard_row(path, row):
    """Last resort when the summary lock can't be taken: write this run's row
    to its own file under summary_shards/, header included, so nothing is
    lost. tools/merge_summary_shards.py folds these back into the main CSV.
    """
    shard_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "summary_shards")
    os.makedirs(shard_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    shard = os.path.join(shard_dir, f"{stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.csv")
    with open(shard, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"  Could not lock {path} -- row written to {shard} instead.\n"
          f"  Fold it in with: python tools/merge_summary_shards.py {path}")


def append_summary_row(path, row, baseline_row=None):
    """Append one run to the cumulative summary CSV, creating it if needed.

    One row per stage-3 run, accumulating across every threshold, percentage,
    concept-finding method and editing method tried in this sae_dir, so a
    sweep can be read as a table instead of by opening one report per run.
    On creation the file gets `baseline_row` first -- the unablated model's
    accuracies -- since every later row is only meaningful next to it.

    Rows are matched to the existing header rather than assumed to fit it. A
    row carrying a column the file doesn't have yet -- a file written before
    a new flag existed, or one whose test split had different groups -- grows
    the header instead of losing the value: the new columns are appended on
    the right, every existing row is padded with blanks, and the file is
    rewritten in place. Column order is otherwise preserved, so anything
    already reading these files by name keeps working.

    The whole sequence runs under an exclusive lock, so parallel stage-3
    processes (a SLURM array over the sweep) append one at a time instead of
    interleaving into a corrupt file. If the lock cannot be taken the row is
    written to its own shard file rather than dropped -- see _write_shard_row.
    """
    with _file_lock(path) as locked:
        if not locked:
            _write_shard_row(path, row)
            return

        header, existing = None, []
        if os.path.isfile(path):
            with open(path, newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                existing = [line for line in r if line]

        if not header:
            header = list(row.keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                if baseline_row is not None:
                    w.writerow({k: baseline_row.get(k, "") for k in header})
            print(f"\nCreated summary {path} (with the unablated baseline row)")
        else:
            new_cols = [k for k in row if k not in header]
            if new_cols:
                header = header + new_cols
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    for line in existing:
                        w.writerow(line + [""] * (len(header) - len(line)))
                print(f"  Added column(s) {', '.join(new_cols)} to {path} "
                      f"({len(existing)} earlier row(s) left blank there)")

        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(
                {k: row.get(k, "") for k in header})
        print(f"Summary row appended to {path}")


def main():
    args = build_parser().parse_args()

    percent = args.ablation_top_percent
    if not (0 < percent <= 100):
        raise SystemExit(
            f"--ablation_top_percent must be in (0, 100], got {percent}. "
            f"(For an ablation with no concepts removed at all, run stage 2 "
            f"with --concept_finding_method none instead.)"
        )

    run_dir, _ = pc.resolve_run_dir_for_pipeline(args)
    manifest_path = pc.manifest_path_for(args, run_dir)
    manifest = pc.apply_analysis_params(pc.read_manifest(manifest_path), args)

    concepts_path = pc.concepts_path_for(manifest, args.concept_finding_method)
    concepts = pc.read_concepts(concepts_path)
    concept_extractor_name = concepts["concept_extractor_name"]
    report_suffix = concepts["report_suffix"]
    sae_dir = concepts["sae_dir"]

    concept_scores = concepts.get("concept_scores")

    quantiles = score_quantiles(concept_scores)
    if quantiles:
        print(f"\nScore distribution of {concepts_path} "
              f"({len(concept_scores):,} scored concepts):")
        print("  " + "   ".join(f"{lbl}={val:.6g}" for lbl, val in quantiles))

    # Resolved per concept file, since 'median'/'pNN' are quantiles of that
    # file's own scores. An unscored file has no quantile to take: a numeric
    # threshold there is a real mistake and select_top_percent raises on it,
    # but the default is a quantile, and erroring out of a default the user
    # never typed would make every unscored method unrunnable -- so those
    # degrade to no filtering, out loud.
    if concept_scores:
        threshold, thr_label = resolve_score_threshold(
            args.concept_score_threshold, concept_scores)
        if thr_label != f"{threshold:g}":
            print(f"  --concept_score_threshold {thr_label} resolves to "
                  f"{threshold:.6g} for this file")
    else:
        threshold, thr_label = 0.0, "0"
        try:
            numeric = float(str(args.concept_score_threshold).strip())
        except ValueError:
            numeric = 0.0
            print(f"\n  Note: --concept_score_threshold "
                  f"{args.concept_score_threshold!r} needs scores to take a quantile "
                  f"of, and this concept list has none -- no threshold applied.")
        if numeric > 0:
            # Not silently ignored: an explicit number was typed.
            select_top_percent(concepts["candidate_concepts"], None, percent, numeric)

    candidate_concepts, n_scored, n_active, ordered_by_score = select_top_percent(
        concepts["candidate_concepts"], concept_scores, percent, threshold,
    )
    # The threshold belongs in the name alongside P: two runs with the same P
    # but different thresholds ablate different concepts, and without it the
    # second would be skipped as already-complete by the check below. The spec
    # goes in rather than the resolved value ('__smedian', not '__s0.0484695')
    # -- shorter, and it says what was asked for. Omitted at 0 so filenames
    # from before the threshold existed still match.
    if threshold > 0:
        report_suffix += f"__s{thr_label}"
    # Both P and the resolved N go in the name: N alone would collide across
    # runs whose concept lists differ in length, and P alone hides how many
    # concepts that actually came to.
    report_suffix += f"__p{percent:g}_top{len(candidate_concepts)}"
    # The coefficient too: it changes the edit as much as anything above
    # (deactivation scales the latents by 1 - coef, so 0.1 barely touches
    # them and 1.0 removes them). Left out of the name, two runs differing
    # only in coefficient share a report path, and the second is skipped as
    # already complete -- which is what happened to every coefficient after
    # the first in the labelfree sweep before this line existed.
    report_suffix += f"__c{args.ablation_coefficient:g}"
    if args.editing_method == "projection" and args.projection_method != "qr":
        report_suffix += f"__{args.projection_method}"

    # One tag names the variant everywhere -- the report's column header, the
    # report filename and the predictions filename -- so an all-images run and
    # a candidates-only run of the same concepts never overwrite each other.
    candidates_only = args.ablation_images == "candidates"
    variant_tag = args.editing_method + ("_candidates_only" if candidates_only else "")

    report_fname = (f"ablation_report_{concept_extractor_name}__"
                    f"{variant_tag}{report_suffix}.txt")
    report_path = os.path.join(sae_dir, report_fname)
    if os.path.isfile(report_path):
        print(f"\nStage 3 already complete for method '{args.concept_finding_method}' "
              f"/ {variant_tag} / threshold {thr_label} / top {percent:g}%:")
        print(f"  {report_path}")
        print("Nothing to do. (Delete that file to force a redo.)")
        return

    if n_active == 0 and n_scored > 0:
        best = max(float(s) for s in concept_scores.values()) if concept_scores else 0.0
        print(f"\nNo concept scores >= --concept_score_threshold {thr_label} "
              f"(= {threshold:.6g}; the best score in {concepts_path} is {best:.6g}), "
              f"so nothing is active to ablate. Lower the threshold and rerun.")
        return

    if len(candidate_concepts) == 0 and args.concept_finding_method != "none":
        print(f"\nNo candidate concepts in {concepts_path} -- nothing to ablate. Skipping.")
        return

    print(f"\nLoading stage-1 context from {manifest_path} ...")
    ctx = pc.load_stage1_context(manifest)
    clip_ft, sae_model, device = ctx["clip_ft"], ctx["sae_model"], ctx["device"]
    test_ds, te_results = ctx["test_ds"], ctx["te_results"]
    vocab_names, concept_match_scores = ctx["vocab_names"], ctx["concept_match_scores"]

    group_ids = np.array([s[3] for s in test_ds.samples])
    true_labels = np.array([s[1] for s in test_ds.samples])
    image_paths = [s[0] for s in test_ds.samples]

    # Which images the edit applies to. Loaded before the ablation runs so a
    # missing or stale candidate_images.json fails immediately rather than
    # after a full pass over the test set.
    candidate_mask = None
    if candidates_only:
        candidate_images_path = pc.candidate_images_path_for(manifest)
        candidate_mask = np.asarray(
            pc.read_candidate_images(candidate_images_path)["is_candidate"], dtype=bool)
        if len(candidate_mask) != len(test_ds.samples):
            raise SystemExit(
                f"candidate_images.json has {len(candidate_mask)} images but test_ds "
                f"has {len(test_ds.samples)} -- {candidate_images_path} is stale "
                f"(test_manifest.csv changed since it was written). Delete it and "
                f"rerun pipeline_2_find_concepts.py."
            )
        print(f"\nEditing candidate images only: {int(candidate_mask.sum()):,} of "
              f"{len(candidate_mask):,} images ({100 * candidate_mask.mean():.1f}%); "
              f"the rest keep their original prediction.")

    ordering = ("score, descending" if ordered_by_score
                else f"the {args.concept_finding_method} method's own order (unscored)")
    active_note = (f"{n_active:,} of {n_scored:,} concepts are active "
                   f"(score >= {thr_label} = {threshold:.6g})" if threshold > 0
                   else f"all {n_scored:,} concepts in the file are active "
                        f"(no --concept_score_threshold)")
    print(f"\n{active_note}; ablating the top {percent:g}% of them "
          f"= {len(candidate_concepts):,}, ranked by {ordering}, "
          f"with {args.editing_method}...")
    if len(candidate_concepts) > 500:
        print(f"  WARNING: {len(candidate_concepts):,} concepts is a large fraction of "
              f"the dictionary -- ablating that many zeroes the representation rather "
              f"than removing a targeted set. Lower --ablation_top_percent.")

    # The projection hook needs P built from the selected concepts; the
    # deactivation hook doesn't, and building it is a QR over as many columns
    # as there are concepts, so it is skipped unless it will be used.
    P = None
    if args.editing_method == "projection":
        P = core.build_projection_matrix(
            concepts=candidate_concepts, sae_model=sae_model, method=args.projection_method,
        )

    # The unablated pass depends only on CLIP, the prompts and the test split
    # -- none of which any stage-3 flag changes, and not knn_k/prevalence
    # either -- so it lives in base_sae_dir, shared by every knn*_prev*/
    # analysis folder under it, and is computed once per (clip_mode, SAE).
    # (Earlier revisions kept it in sae_dir; a sweep over knn_k/prevalence
    # would then have recomputed the same baseline once per folder.)
    orig_logits_cache = os.path.join(manifest["base_sae_dir"], "original_logits.npz")

    if manifest["model"] == "RouteSAE":
        orig_preds, ablated_preds = core.ablate_spurious_concepts_routesae(
            clip_ft=clip_ft, sae_model=sae_model, test_ds=test_ds,
            spurious_concept_indices=candidate_concepts, device=device, sae_dir=sae_dir,
            lambda_coefficient=args.ablation_coefficient,
            vocab_names=vocab_names, concept_match_scores=concept_match_scores,
            concept_extractor_name=concept_extractor_name, clip_mode=manifest["clip_mode"],
            report_suffix=report_suffix, editing_method=args.editing_method, P=P,
            write_report=False, orig_logits_cache=orig_logits_cache,
        )[:2]
    else:
        ablation_hook = (
            core.make_sae_ablation_hook(
                sae_model, candidate_concepts, device, lambda_coef=args.ablation_coefficient)
            if args.editing_method == "deactivation"
            else core.projection_ablation_hook(P=P, lambda_coef=args.ablation_coefficient)
        )
        orig_preds, ablated_preds = core.ablate_spurious_concepts(
            clip_ft=clip_ft, sae_model=sae_model, test_ds=test_ds,
            spurious_concept_indices=candidate_concepts, device=device, sae_dir=sae_dir,
            vocab_names=vocab_names, concept_match_scores=concept_match_scores,
            ablation_hook=ablation_hook, lambda_coefficient=args.ablation_coefficient,
            concept_extractor_name=concept_extractor_name, candidate_mask_only=False,
            clip_mode=manifest["clip_mode"], te_sae_reps=te_results["sae_representations"],
            report_suffix=report_suffix, write_report=False,
            orig_logits_cache=orig_logits_cache,
        )[:2]

    orig_preds = np.asarray(orig_preds)
    ablated_preds = np.asarray(ablated_preds)

    # Restricting the edit to the candidate images is done here, by selecting
    # between the two prediction sets, rather than by hooking only those
    # images during the pass above -- the two are equivalent because every
    # ablation hook acts on one image at a time (LayerNorm/attention, no
    # batch-level statistics), so whether an image is excluded before the hook
    # runs or after makes no difference to the images that are included. Doing
    # it after keeps the pass batched, which the per-image alternative is not.
    if candidate_mask is not None:
        ablated_preds = np.where(candidate_mask, ablated_preds, orig_preds)

    core._write_combined_ablation_report(
        group_ids=group_ids, true_labels=true_labels, orig_preds=orig_preds,
        ablation_results={variant_tag: ablated_preds},
        concept_idx=candidate_concepts,
        vocab_names=vocab_names, concept_match_scores=concept_match_scores,
        sae_dir=sae_dir, concept_extractor_name=concept_extractor_name,
        report_suffix=report_suffix, report_tag=variant_tag,
    )

    stem = (f"predictions_{concept_extractor_name}__"
            f"{variant_tag}{report_suffix}")
    save_predictions(
        path_csv=os.path.join(sae_dir, stem + ".csv"),
        image_paths=image_paths, group_ids=group_ids, true_labels=true_labels,
        orig_preds=orig_preds, ablated_preds=ablated_preds,
        candidate_mask=candidate_mask,
    )

    # One cumulative table per concept-finding method, not one per sae_dir:
    # the methods produce concept lists of different lengths and their scores
    # are on different scales, so n_concepts_* and the threshold columns only
    # compare within a method. Keeping them in separate files means a sweep
    # can be read (or plotted) straight off one file without filtering first.
    # The method column stays anyway, so concatenating the files later still
    # gives a well-formed table. The baseline row is written only when a file
    # is created, since the unablated model is the same for every run here --
    # repeating it per run would just be the same numbers again.
    summary_row = dict(
        timestamp=pc.now_iso(),
        run="ablated",
        concept_finding_method=args.concept_finding_method,
        editing_method=args.editing_method,
        ablation_images=args.ablation_images,
        ablation_coefficient=args.ablation_coefficient,
        projection_method=(args.projection_method
                           if args.editing_method == "projection" else ""),
        n_images_edited=(int(candidate_mask.sum()) if candidate_mask is not None
                         else len(image_paths)),
        n_images_total=len(image_paths),
        score_threshold=thr_label,
        score_threshold_value=round(threshold, 6),
        ablation_top_percent=percent,
        n_concepts_total=n_scored,
        n_concepts_active=n_active,
        n_concepts_ablated=len(candidate_concepts),
        pct_of_all_concepts=(round(100.0 * len(candidate_concepts) / n_scored, 4)
                             if n_scored else 0.0),
        **accuracy_fields(ablated_preds, true_labels, group_ids),
    )
    baseline_row = dict(
        timestamp=pc.now_iso(),
        run="original",
        # The unablated model owes nothing to a concept-finding method, but
        # this is that method's file and its baseline row -- naming the method
        # here keeps a group-by working if the per-method files are ever
        # concatenated. run="original" is what marks it as the baseline.
        concept_finding_method=args.concept_finding_method,
        editing_method="none",
        ablation_images="none", ablation_coefficient="", projection_method="",
        n_images_edited=0, n_images_total=len(image_paths),
        score_threshold="", score_threshold_value="",
        ablation_top_percent=0.0,
        n_concepts_total=n_scored, n_concepts_active="", n_concepts_ablated=0,
        pct_of_all_concepts=0.0,
        **accuracy_fields(orig_preds, true_labels, group_ids),
    )
    append_summary_row(
        os.path.join(sae_dir, f"ablation_summary_{args.concept_finding_method}.csv"),
        summary_row, baseline_row,
    )
    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
