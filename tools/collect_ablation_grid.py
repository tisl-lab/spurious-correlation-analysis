#!/usr/bin/env python
"""Collect every ablation row across a knn_k / prevalence_threshold grid.

Walks each <RD>/{zs,ft}_routesae_K*/knn<k>_prev<p>/ablation_summary_<method>.csv
-- for the waterbirds run that is these four SAE folders:

    ft_routesae_K256_clip_ft_BIASED_400_<ts>_16384
    ft_routesae_K32_clip_ft_BIASED_400_<ts>_16384
    zs_routesae_K256_ViT-B~32_16384
    zs_routesae_K32_ViT-B~32_16384

-- and writes ONE csv holding every `ablated` row found, with all of that
row's own columns preserved verbatim, plus the knn_k and prevalence the
folder name encodes, the clip_mode and K the SAE folder encodes, and that
point's own baseline / delta columns for convenience.

How this differs from tools/summarize_grid.py, which covers the same files:
that script selects a fixed handful of columns and keeps one ablated row per
point (the last), because it exists to rank points. This keeps EVERYTHING --
every column, every ablated row -- so the output is the raw table to pivot
or filter, not a ranking. Both are stdlib-only and run on a login node
without the venv.

Missing files are skipped quietly and counted; a point whose csv has no
baseline row still contributes its ablated rows, with empty baseline/delta
columns.

Usage:
    python tools/collect_ablation_grid.py --rd results/waterbirds/clip_ft_BIASED_400_20260709_111654
    python tools/collect_ablation_grid.py --rd ... --method labelguided
    python tools/collect_ablation_grid.py --rd ... --out /tmp/grid.csv --top 10
"""

import argparse
import csv
import glob
import os
import re
import sys

DIR_RE = re.compile(r"^(zs|ft)_routesae_K(\d+)_.*$")
POINT_RE = re.compile(r"^knn(\d+)_prev([0-9.]+)$")

# Columns this script adds in front of whatever the summary csv already holds.
LEAD = ["clip_mode", "K", "knn_k", "prevalence", "sae_dir", "point_dir"]
# ...and behind it.
TRAIL = ["baseline_accuracy_avg", "baseline_accuracy_worst_group",
         "delta_accuracy_avg", "delta_accuracy_worst_group", "source_csv"]


def _f(d, k):
    """float(d[k]) or None for missing/blank -- these csvs leave cells empty."""
    v = (d or {}).get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect(rd, method):
    rows, seen_cols, n_files, n_missing_base = [], [], 0, 0
    pattern = os.path.join(rd, "*_routesae_K*", "knn*_prev*",
                           f"ablation_summary_{method}.csv")
    for path in sorted(glob.glob(pattern)):
        point_dir = os.path.basename(os.path.dirname(path))
        sae_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m_sae, m_pt = DIR_RE.match(sae_dir), POINT_RE.match(point_dir)
        if not (m_sae and m_pt):
            continue
        n_files += 1
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for c in (reader.fieldnames or []):
                if c not in seen_cols:
                    seen_cols.append(c)     # union, first-seen order
            recs = list(reader)

        base = next((r for r in recs if r.get("run") == "original"), None)
        if base is None:
            n_missing_base += 1
        b_avg, b_wg = _f(base, "accuracy_avg"), _f(base, "accuracy_worst_group")

        for r in recs:
            if r.get("run") != "ablated":
                continue
            a_avg, a_wg = _f(r, "accuracy_avg"), _f(r, "accuracy_worst_group")
            out = dict(r)                         # every original column, verbatim
            out.update(
                clip_mode={"zs": "zeroshot", "ft": "finetuned"}[m_sae.group(1)],
                K=int(m_sae.group(2)),
                knn_k=int(m_pt.group(1)),
                prevalence=float(m_pt.group(2)),
                sae_dir=sae_dir,
                point_dir=point_dir,
                baseline_accuracy_avg="" if b_avg is None else b_avg,
                baseline_accuracy_worst_group="" if b_wg is None else b_wg,
                delta_accuracy_avg="" if None in (a_avg, b_avg) else round(a_avg - b_avg, 6),
                delta_accuracy_worst_group="" if None in (a_wg, b_wg) else round(a_wg - b_wg, 6),
                source_csv=path,
            )
            rows.append(out)
    return rows, seen_cols, n_files, n_missing_base


def best_by(rows, key):
    """Highest `key` among rows that have it. Ties -> fewest concepts ablated,
    because a smaller edit achieving the same accuracy is the better finding."""
    scored = [r for r in rows if _f(r, key) is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: (_f(r, key), -(_f(r, "n_concepts_ablated") or 0)))


def describe(r, key):
    n = _f(r, "n_concepts_ablated")
    return (f"knn_k={r['knn_k']:<3} prevalence={r['prevalence']:<5} "
            f"-> {key}={_f(r, key):.4f}  "
            f"(avg {_f(r, 'accuracy_avg'):.4f}, worst {_f(r, 'accuracy_worst_group'):.4f}, "
            f"{int(n) if n is not None else '?'} concepts, "
            f"{r.get('editing_method', '?')}, coef {r.get('ablation_coefficient', '?')})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rd", required=True,
                    help="Run dir holding the *_routesae_K* folders.")
    ap.add_argument("--method", default="labelfree",
                    help="Which ablation_summary_<method>.csv to read (default labelfree).")
    ap.add_argument("--out", default=None,
                    help="Output csv (default <rd>/<method>_grid_all_rows.csv).")
    ap.add_argument("--top", type=int, default=5,
                    help="Rows to list per SAE under each ranking (default 5).")
    args = ap.parse_args()

    rows, cols, n_files, n_missing_base = collect(args.rd, args.method)
    if not rows:
        print(f"No ablated rows found in "
              f"{args.rd}/*_routesae_K*/knn*_prev*/ablation_summary_{args.method}.csv",
              file=sys.stderr)
        return 1

    out = args.out or os.path.join(args.rd, f"{args.method}_grid_all_rows.csv")
    fieldnames = LEAD + [c for c in cols if c not in LEAD and c not in TRAIL] + TRAIL
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["clip_mode"], r["K"],
                                             r["knn_k"], r["prevalence"])):
            w.writerow(r)

    print(f"{len(rows)} ablated row(s) from {n_files} csv file(s) -> {out}")
    if n_missing_base:
        print(f"  ({n_missing_base} file(s) had no 'original' row; "
              f"their baseline/delta cells are empty)")

    # ── Recommendation, per SAE and overall ────────────────────────────────
    groups = sorted({(r["clip_mode"], r["K"]) for r in rows})
    for cm, k in groups:
        sub = [r for r in rows if r["clip_mode"] == cm and r["K"] == k]
        print(f"\n=== {cm}  K={k}   ({len(sub)} ablated rows, "
              f"{len({(r['knn_k'], r['prevalence']) for r in sub})} grid points)")
        for key, label in (("accuracy_worst_group", "highest WORST-GROUP accuracy"),
                           ("accuracy_avg", "highest AVERAGE accuracy")):
            b = best_by(sub, key)
            if b is None:
                continue
            print(f"  {label}:")
            print(f"    {describe(b, key)}")
            ranked = sorted((r for r in sub if _f(r, key) is not None),
                            key=lambda r: -_f(r, key))[:args.top]
            for r in ranked[1:]:
                print(f"      also  knn_k={r['knn_k']:<3} prev={r['prevalence']:<5} "
                      f"{key}={_f(r, key):.4f}")

    print("\n=== Overall ===")
    for key, label in (("accuracy_worst_group", "worst-group"),
                       ("accuracy_avg", "average")):
        b = best_by(rows, key)
        print(f"  best {label:<11} {b['clip_mode']} K={b['K']}: {describe(b, key)}")
    bw, ba = best_by(rows, "accuracy_worst_group"), best_by(rows, "accuracy_avg")
    if (bw["clip_mode"], bw["K"], bw["knn_k"], bw["prevalence"]) != \
       (ba["clip_mode"], ba["K"], ba["knn_k"], ba["prevalence"]):
        print("  NOTE: the two metrics disagree. Worst-group accuracy is the one "
              "spurious-correlation\n        work is judged on; average accuracy "
              "rises whenever a method trades the small\n        misaligned groups "
              "for the large aligned ones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
