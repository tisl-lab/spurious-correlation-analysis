"""(A) Does a RouteSAE concept's origin layer shift by group?

Recomputes routesae_adapter.concept_layer_origins over the test split in one
pass -- same image order and batch size as stage 2, so the numbers match
stage 2's exactly -- tallying each image under its group, for the union of
every method's candidate concepts (see origins_by_group for why the batching
must match). Then:

  - per group: a layer histogram per method (the main figure)
  - aligned vs misaligned: each side is the sum of its groups' counts; per
    concept, whether the mode layer flips and the total-variation distance
    between the two layer distributions

Regression check: the groups partition the test split, so summing a
concept's per-group layer_counts must give exactly the layer_counts stage 2
stored in concepts_<method>.json (same include_cls/aggre/routing defaults).
Stage 2 and the server's concept files are the reference: differences are
reported with their size (the run continues; --strict_check stops instead).
Skipped under --limit.

Outputs, in the analysis dir next to concepts_<method>.json
(suffix _limit<N> on test runs):
  concept_layers_by_group.json              raw per-group origins (cache)
  concept_layers_by_group_layers.csv        method, layer, group, counts, share, per-image
  concept_layers_by_group_concepts.csv      one row per (method, concept)
  concept_layers_by_group_summary.csv       per method: mode-flip fraction, TV distance
  concept_layers_by_group_groups.png        per-group histograms, one panel per method
  concept_layers_by_group_sides.png         aligned vs misaligned, one panel per method

--sweep_dir restricts each method to the concepts its best sweep run ablated
(tables/plots get _sweepbest_<best_by>; the passes cache is shared).

Usage:
    python tools/find_concept_layers.py --dataset celeba \\
        --run_dir results/celeba/clip_ft_BIASED_400_20260921_214244 \\
        --clip_mode zeroshot --hf_model_dir hf_models \\
        --sae_path "routesae_weights/routesae_K32_celeba_ViT-B~32_16384.pt" [--limit 64]
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pipeline_common as pc  # noqa: E402

METHODS = ["labelguided", "dialguided", "highmag", "labelfree", "conceptscope"]
# Dashboard setting order -- decides ties between equally good sweep points.
SWEEP_SETTINGS = [("deactivation", "all"), ("deactivation", "candidates"),
                  ("projection", "all"), ("projection", "candidates")]
SIDES = ["aligned", "misaligned"]
# Marks a cache written by origins_by_group's single stage-2-batched pass, so
# caches from the older one-pass-per-group version are recomputed, not reused.
PASS_TAG = "single_pass_stage2_batches"
# dataviz reference palette, categorical slots in fixed order (light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def int_keys(d):
    return {int(k): v for k, v in d.items()}


def merged_counts(origins, cid, gids):
    """Sum one concept's layer_counts over several groups -> Counter {layer: n}."""
    total = Counter()
    for g in gids:
        total.update(int_keys(origins[str(g)][str(cid)]["layer_counts"]))
    return total


def mode_of(counts):
    """Most common layer, lowest layer on ties (matches concept_layer_origins' argmax)."""
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def tv_distance(p_counts, q_counts, layers):
    """Total-variation distance between two layer distributions: 0 same, 1 disjoint."""
    p_n, q_n = sum(p_counts.values()), sum(q_counts.values())
    return 0.5 * sum(abs(p_counts[l] / p_n - q_counts[l] / q_n) for l in layers)


def origins_by_group(sae, clip_model, samples, candidate_concepts, gids, device,
                     preprocess, batch_size, include_cls=True, aggre="sum", routing="hard"):
    """routesae_adapter.concept_layer_origins, tallied per group in ONE pass.

    The forward pass and the "which layer did this concept come from" rule
    are concept_layer_origins' own lines, using its helpers; only the tally
    differs (per group, and vectorised instead of a Python loop -- same
    integer counts). What matters is that it walks `samples` in the same
    order and the same batch_size as stage 2 did: on a GPU an image's
    activations can change in the last bits with its batch-mates, and RouteSAE
    has three knife-edge decisions (top-K membership, the router's argmax,
    max > 0) that such bits can flip. Running each group as its own pass gave
    every image different batch-mates than stage 2, and ~0.3% of concepts
    then failed the regression check. Same batches -> same numbers.

    Returns {str(gid): {str(cid): {layer_counts, mode_layer, purity,
    n_active_images}}}, i.e. concept_layer_origins' format per group.
    """
    import torch
    from PIL import Image
    from routesae_adapter import clip_layer_stack, pre_process

    if routing != "hard":
        raise ValueError("origin layers need routing='hard' (see concept_layer_origins).")

    g_index = {g: i for i, g in enumerate(gids)}
    cand_t = torch.tensor(list(candidate_concepts), device=device, dtype=torch.long)
    counts = torch.zeros(len(gids), len(cand_t), sae.n_routed_layers, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            pixel_values = torch.stack(
                [preprocess(Image.open(s[0]).convert("RGB")) for s in chunk]).to(device)

            stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
            x, _, _ = pre_process(stack)
            _, _, latents, _, router_weights = sae(x, aggre, routing)
            layer_per_patch = router_weights.argmax(dim=-1)                 # (B, T)

            patches = latents if include_cls else latents[:, 1:, :]
            layers = layer_per_patch if include_cls else layer_per_patch[:, 1:]
            max_vals, max_patch = patches[:, :, cand_t].max(dim=1)          # (B, n_cand)
            origin_layer = layers.gather(1, max_patch).cpu()

            b_idx, c_idx = (max_vals > 0).cpu().nonzero(as_tuple=True)
            g_of_row = torch.tensor([g_index[s[3]] for s in chunk], dtype=torch.long)
            counts.index_put_((g_of_row[b_idx], c_idx, origin_layer[b_idx, c_idx]),
                              torch.ones(len(b_idx), dtype=torch.long), accumulate=True)
            if (start // batch_size) % 20 == 0:
                print(f"    {min(start + batch_size, len(samples))}/{len(samples)} images")

    origins = {}
    for gi, g in enumerate(gids):
        per_concept = {}
        for ci, cid in enumerate(candidate_concepts):
            row = counts[gi, ci]
            total = int(row.sum())
            if total == 0:
                per_concept[str(cid)] = dict(layer_counts={}, mode_layer=None,
                                             purity=0.0, n_active_images=0)
                continue
            mode_idx = int(row.argmax())   # first max on ties, as concept_layer_origins
            per_concept[str(cid)] = dict(
                layer_counts={str(sae.start_layer + j): int(c)
                              for j, c in enumerate(row.tolist()) if c > 0},
                mode_layer=sae.start_layer + mode_idx,
                purity=int(row[mode_idx]) / total,
                n_active_images=total)
        origins[str(g)] = per_concept
    return origins


def best_sweep_subset(sweep_dir, method, concepts, best_by):
    """The concepts ablated by the best row of ablation_summary_<method>.csv.

    Rebuilt with pipeline_3_ablate's own threshold + top-percent selection,
    and checked against the counts that row recorded, so a concepts file
    that differs from the one the sweep ran on fails loudly instead of
    silently giving a different set. None if the CSV is missing.
    """
    from pipeline_3_ablate import resolve_score_threshold, select_top_percent

    path = os.path.join(sweep_dir, f"ablation_summary_{method}.csv")
    if not os.path.isfile(path):
        print(f"  {method}: no {path} -- left out of this run")
        return None
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["run"] == "ablated"]

    # Same selection as sweap_results*/generate_sweap_ablation_report.mjs:
    # average repeated runs at one (editing method, images, coefficient,
    # concepts removed) setting, worst group = min of the averaged groups,
    # then the highest; first point wins ties, in the dashboard's order.
    points = []
    for editing, images in SWEEP_SETTINGS:
        buckets = {}
        for r in rows:
            if r["editing_method"] == editing and r["ablation_images"] == images:
                buckets.setdefault((int(r["n_concepts_ablated"]),
                                    r.get("ablation_coefficient", "")), []).append(r)
        for (_, _), obs in sorted(buckets.items(), key=lambda kv: kv[0][0]):
            groups = [statistics.mean(float(o[f"acc_group_{i}"]) for o in obs) for i in range(4)]
            points.append(dict(rows=obs, worst=min(groups),
                               avg=statistics.mean(float(o["accuracy_avg"]) for o in obs)))
    if not points:
        print(f"  {method}: no ablated rows in {path} -- left out of this run")
        return None
    key = "worst" if best_by == "worst_group" else "avg"
    best_point = None
    for p in points:
        if best_point is None or p[key] > best_point[key]:
            best_point = p
    # For a scored method any (threshold, top %) giving N concepts selects the
    # same top N by score, so one row of the bucket rebuilds the whole set.
    best = best_point["rows"][0]

    threshold = 0.0
    if best["score_threshold"]:
        threshold, _ = resolve_score_threshold(best["score_threshold"],
                                               concepts["concept_scores"])
    selected, n_total, n_active, _ = select_top_percent(
        concepts["candidate_concepts"], concepts["concept_scores"],
        float(best["ablation_top_percent"]), threshold)
    if (len(selected), n_total) != (int(best["n_concepts_ablated"]),
                                    int(best["n_concepts_total"])):
        sys.exit(f"{method}: rebuilt {len(selected)}/{n_total} concepts but the sweep row "
                 f"ablated {best['n_concepts_ablated']}/{best['n_concepts_total']} -- "
                 f"the concepts file is not the one {path} ran on.")
    return dict(concepts=[int(c) for c in selected], acc=best_point[key],
                editing_method=best["editing_method"], threshold=best["score_threshold"],
                top_percent=best["ablation_top_percent"])


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


def plot_histograms(path, layer_rows, series, key, methods, layers, title, ylabel,
                    method_labels):
    """One panel per method, one bar series per entry of `series` [(id, label)]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(r["method"], r["layer"], r["group"]): r[key] for r in layer_rows}
    fig, axes = plt.subplots(1, len(methods), figsize=(3.8 * len(methods), 3.6),
                             squeeze=False, facecolor=SURFACE)
    x = np.arange(len(layers))
    w = 0.8 / len(series)
    for ax, method in zip(axes[0], methods):
        ax.set_facecolor(SURFACE)
        for i, (sid, label) in enumerate(series):
            vals = [lookup.get((method, l, sid), 0) for l in layers]
            # 2px surface gap between adjacent bars.
            ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, color=SERIES[i],
                   edgecolor=SURFACE, linewidth=2, label=label, zorder=2)
        ax.set_title(method_labels[method], color=INK, fontsize=10, fontweight="bold")
        ax.set_xticks(x, [str(l) for l in layers])
        ax.set_xlabel("CLIP layer", color=INK_2, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
    axes[0][0].set_ylabel(ylabel, color=INK_2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=len(series),
               fontsize=8.5, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pc.add_core_args(ap)
    ap.add_argument("--limit", type=int, default=None,
                    help="Test run: use only the first N images of each group.")
    ap.add_argument("--min_support", type=int, default=20,
                    help="Per-concept aligned-vs-misaligned comparison: a concept needs at "
                         "least this many active images on BOTH sides to be compared.")
    ap.add_argument("--check_tolerance", type=int, default=1,
                    help="Regression check: the largest per-concept count difference from "
                         "stage 2's stored layer_counts counted as a match. 1 = one image "
                         "counted active in one run and not the other (a knife-edge "
                         "max > 0 flip from GPU floating-point noise); an image moving "
                         "between layers counts 2. 0 = exact match.")
    ap.add_argument("--strict_check", action="store_true",
                    help="Stop when any concept differs from stage 2 by more than "
                         "--check_tolerance. Default: report the differences and continue.")
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore a saved concept_layers_by_group.json and rerun the passes.")
    ap.add_argument("--sweep_dir", default=None,
                    help="Folder of ablation_summary_<method>.csv (e.g. "
                         "sweap_results_celeba/zs-k32). Restricts each method to the "
                         "concepts its best sweep row ablated, selected exactly as "
                         "pipeline_3_ablate does.")
    ap.add_argument("--best_by", default="worst_group", choices=["worst_group", "mean"],
                    help="--sweep_dir only: pick the row with the highest worst-group "
                         "or mean accuracy.")
    args = ap.parse_args()

    run_dir, _ = pc.resolve_run_dir_for_pipeline(args)
    manifest_path = pc.manifest_path_for(args, run_dir)
    manifest = pc.apply_analysis_params(pc.read_manifest(manifest_path), args)
    if manifest["model"] != "RouteSAE":
        sys.exit("Origin layers only exist for RouteSAE manifests.")
    spec = pc.apply_dataset(manifest)   # spec from the run's dataset
    sae_dir = manifest["sae_dir"]

    # Groups and sides from the dataset spec (not hardcoded)
    gids = list(range(len(spec.group_names)))
    side_gids = {"aligned": sorted(spec.aligned_groups),
                 "misaligned": sorted(spec.misaligned_groups())}

    # Test-run outputs get their own name so a full run never reads them.
    # The passes (out_json) always cover every concept, so a --sweep_dir run
    # reuses the full run's cache; only its tables/plots get their own name.
    suffix = f"_limit{args.limit}" if args.limit else ""
    out_json = os.path.join(sae_dir, f"concept_layers_by_group{suffix}.json")
    subset_tag = f"_sweepbest_{args.best_by}" if args.sweep_dir else ""
    out_stem = os.path.join(sae_dir, f"concept_layers_by_group{suffix}{subset_tag}")

    # ── Step 3: load concepts (before the slow model load) ───────────────────
    concepts_by_method = {}
    for method in METHODS:
        path = pc.concepts_path_for(manifest, method)
        if not os.path.isfile(path):
            print(f"  Skipping {method}: no {path}")
            continue
        d = pc.read_concepts(path)
        if d["candidate_concepts"]:
            concepts_by_method[method] = d
        else:
            print(f"  Skipping {method}: no candidate concepts")
    if not concepts_by_method:
        sys.exit(f"No non-empty concepts_<method>.json in {sae_dir}")
    methods = list(concepts_by_method)
    union = sorted({c for d in concepts_by_method.values() for c in d["candidate_concepts"]})
    print(f"\n{sae_dir}\n  methods: {', '.join(methods)}  ({len(union)} concepts in union)")

    # Optional: keep only the concepts each method's best sweep run ablated.
    # Done after `union` so the passes still cover every concept.
    method_labels = {m: m for m in methods}
    if args.sweep_dir:
        for method in methods:
            best = best_sweep_subset(args.sweep_dir, method, concepts_by_method[method],
                                     args.best_by)
            if best is None:   # no sweep for this method: leave it out of this run
                del concepts_by_method[method]
                continue
            concepts_by_method[method] = {**concepts_by_method[method],
                                          "candidate_concepts": best["concepts"]}
            method_labels[method] = f"{method} · top {len(best['concepts'])}"
            print(f"  {method:>12}: best {args.best_by} {best['acc']:.3f} "
                  f"({best['editing_method']}, {best['threshold']}, "
                  f"top {best['top_percent']}%) -> {len(best['concepts'])} concepts")
        methods = list(concepts_by_method)
        if not methods:
            sys.exit(f"No method in {sae_dir} has a sweep CSV in {args.sweep_dir}")

    # ── Step 4: one pass per group, or reuse the saved result ────────────────
    cache = None
    if os.path.isfile(out_json) and not args.recompute:
        with open(out_json) as f:
            cache = json.load(f)
        if cache.get("pass") != PASS_TAG or cache.get("batch_size") != args.batch_size:
            print("  Saved origins come from an older per-group pass or another batch "
                  "size -- recomputing.")
            cache = None
        elif not set(map(str, union)) <= set(cache["origins"][str(gids[0])]):
            print("  Saved origins miss some concepts -- recomputing.")
            cache = None
        else:
            print(f"  Reusing saved origins: {out_json}")

    if cache is None:
        print(f"\nLoading stage-1 context from {manifest_path} ...")
        ctx = pc.load_stage1_context(manifest)
        test_ds = ctx["test_ds"]

        # Each sample is (img_path, label, bg_name, group_id). Every image must
        # belong to a known group, or the per-group counts can't add up.
        assert {s[3] for s in test_ds.samples} <= set(gids)
        samples = test_ds.samples   # stage 2's order -- do not reorder or split
        # Test run: first N images of each group, still in manifest order
        if args.limit:
            seen, kept = Counter(), []
            for s in samples:
                seen[s[3]] += 1
                if seen[s[3]] <= args.limit:
                    kept.append(s)
            samples = kept
        group_samples = {g: [s for s in samples if s[3] == g] for g in gids}

        print(f"  One pass over {len(samples)} test images, batch size {args.batch_size} "
              f"(stage 2's order), {len(union)} concepts...")
        origins = origins_by_group(
            sae=ctx["sae_model"], clip_model=ctx["clip_ft"].model, samples=samples,
            candidate_concepts=union, gids=gids, device=ctx["device"],
            preprocess=ctx["clip_ft"].preprocess, batch_size=args.batch_size)

        sae = ctx["sae_model"]
        cache = dict(
            created_at=pc.now_iso(), **{"pass": PASS_TAG}, limit=args.limit,
            batch_size=args.batch_size,
            group_names=spec.group_names, side_gids=side_gids,
            n_images={str(g): len(group_samples[g]) for g in gids},
            layers=list(range(sae.start_layer, sae.start_layer + sae.n_routed_layers)),
            origins=origins,
        )
        with open(out_json, "w") as f:
            json.dump(cache, f)
        print(f"  Saved {out_json}")
        # Round-trip so fresh and reloaded results both have string keys.
        cache = json.loads(json.dumps(cache))

    origins, layers = cache["origins"], cache["layers"]
    n_images = {int(g): n for g, n in cache["n_images"].items()}
    n_side = {s: sum(n_images[g] for g in side_gids[s]) for s in SIDES}
    print("  images per group: " + ", ".join(f"g{g}={n_images[g]}" for g in gids)
          + f"  (aligned {n_side['aligned']}, misaligned {n_side['misaligned']})")

    # ── Step 5: regression check -- groups sum to stage 2's stored counts ────
    if args.limit:
        print("  Test run: regression check skipped.")
    else:
        bad, n_checked = [], 0
        for method, d in concepts_by_method.items():
            for cid, stored in (d.get("concept_layers") or {}).items():
                n_checked += 1
                got = merged_counts(origins, cid, gids)
                want = int_keys(stored["layer_counts"])
                if dict(got) != want:
                    diff = sum(abs(got[l] - want.get(l, 0)) for l in set(got) | set(want))
                    bad.append((diff, method, cid, sum(want.values())))
        bad.sort(reverse=True)
        beyond = [b for b in bad if b[0] > args.check_tolerance]
        if beyond:
            print(f"  Regression check: {len(beyond)}/{n_checked} (method, concept) pairs "
                  f"differ from stage 2 by more than {args.check_tolerance} (an image moved "
                  f"between layers counts 2). Largest differences:")
            for diff, method, cid, n in beyond[:5]:
                print(f"    {method} concept {cid}: {diff} over {n} active images")
            if args.strict_check:
                sys.exit("Stopping: this run does not reproduce stage 2's stored counts. "
                         "Check --batch_size matches stage 2's, and that the SAE checkpoint, "
                         "test manifest and code are the ones stage 2 ran with.")
            print("  WARNING: continuing anyway (pass --strict_check to stop here). Large "
                  "differences would mean the SAE checkpoint, test manifest or code differ "
                  "from stage 2's.")
        elif bad:
            print(f"  Regression check passed within tolerance {args.check_tolerance}: "
                  f"{n_checked - len(bad)}/{n_checked} (method, concept) pairs exact, "
                  f"{len(bad)} off by at most {bad[0][0]} "
                  f"(worst: {bad[0][1]} concept {bad[0][2]}, over {bad[0][3]} active images).")
        elif n_checked == 0:
            print("  Regression check not possible: no concepts_<method>.json stores "
                  "concept_layers (written by stage 2 for RouteSAE).")
        else:
            print(f"  Regression check passed for all {n_checked} (method, concept) pairs.")

    # ── Step 6: per-concept comparison, aligned vs misaligned ────────────────
    concept_rows, summary_rows = [], []
    for method, d in concepts_by_method.items():
        tvs, flips, n_never, n_low = [], 0, 0, 0
        for cid in d["candidate_concepts"]:
            side_counts = {s: merged_counts(origins, cid, side_gids[s]) for s in SIDES}
            row = dict(method=method, concept=cid)
            for g in gids:
                info = origins[str(g)][str(cid)]
                row[f"n_active_g{g}"] = info["n_active_images"]
                row[f"mode_g{g}"] = info["mode_layer"]
            for s in SIDES:
                row[f"n_active_{s}"] = sum(side_counts[s].values())
                row[f"mode_{s}"] = mode_of(side_counts[s])

            if row["n_active_aligned"] + row["n_active_misaligned"] == 0:
                n_never += 1
                row["status"] = "never_active"
            elif min(row["n_active_aligned"], row["n_active_misaligned"]) < args.min_support:
                n_low += 1
                row["status"] = "low_support"
            else:
                row["tv_distance"] = round(tv_distance(
                    side_counts["aligned"], side_counts["misaligned"], layers), 4)
                row["mode_flip"] = row["mode_aligned"] != row["mode_misaligned"]
                row["status"] = "compared"
                tvs.append(row["tv_distance"])
                flips += row["mode_flip"]
            concept_rows.append(row)

        summary_rows.append(dict(
            method=method, n_concepts=len(d["candidate_concepts"]),
            n_compared=len(tvs), n_never_active=n_never, n_low_support=n_low,
            frac_mode_flip=round(flips / len(tvs), 4) if tvs else None,
            mean_tv=round(statistics.mean(tvs), 4) if tvs else None,
            median_tv=round(statistics.median(tvs), 4) if tvs else None,
        ))

    # ── Step 7: per-layer histograms, per group and per side ─────────────────
    # count       activations (concept, image) originating at this layer
    # share       count / all of this group's activations   (sums to 1 per group)
    # per_image   count / images in this group              (comparable across sizes)
    layer_rows = []
    for method, d in concepts_by_method.items():
        units = [(f"g{g}", [g], n_images[g]) for g in gids]
        units += [(s, side_gids[s], n_side[s]) for s in SIDES]
        for unit, unit_gids, n_img in units:
            counts = Counter()
            for cid in d["candidate_concepts"]:
                counts.update(merged_counts(origins, cid, unit_gids))
            total = sum(counts.values())
            for l in layers:
                layer_rows.append(dict(
                    method=method, layer=l, group=unit, n_images=n_img, count=counts[l],
                    share=round(counts[l] / total, 5) if total else 0.0,
                    per_image=round(counts[l] / n_img, 5) if n_img else 0.0,
                ))

    # ── Step 8: write everything ─────────────────────────────────────────────
    write_csv(out_stem + "_layers.csv", layer_rows)
    write_csv(out_stem + "_concepts.csv", concept_rows)
    write_csv(out_stem + "_summary.csv", summary_rows)

    title = (f"{manifest.get('dataset', 'waterbirds')} · {manifest['clip_mode']} · "
             f"RouteSAE K{manifest.get('routesae_k')}"
             + (f" · limit {args.limit}/group" if args.limit else "")
             + (f" · concepts ablated in best-{args.best_by} sweep run" if args.sweep_dir else ""))
    ylabel = "share of group's activations"
    if len(gids) <= len(SERIES):
        group_series = [(f"g{g}", f"g{g} {spec.group_names[g]}"
                         f" ({'al' if g in spec.aligned_groups else 'mis'})") for g in gids]
        plot_histograms(out_stem + "_groups.png", layer_rows, group_series, "share",
                        methods, layers, title + " · per group", ylabel, method_labels)
    else:
        print(f"  {len(gids)} groups exceed the {len(SERIES)}-colour palette -- "
              f"per-group plot skipped (numbers are in the _layers.csv).")
    plot_histograms(out_stem + "_sides.png", layer_rows,
                    [("aligned", "aligned"), ("misaligned", "misaligned")], "share",
                    methods, layers, title + " · aligned vs misaligned", ylabel,
                    method_labels)

    print("\n  Aligned vs misaligned, per concept "
          f"(needs ≥{args.min_support} active images on both sides):")
    for r in summary_rows:
        print(f"  {r['method']:>12}: {r['n_compared']}/{r['n_concepts']} compared "
              f"(never-active {r['n_never_active']}, low-support {r['n_low_support']}); "
              f"mode flips {r['frac_mode_flip']}, TV mean {r['mean_tv']} / "
              f"median {r['median_tv']}")


if __name__ == "__main__":
    sys.exit(main())
