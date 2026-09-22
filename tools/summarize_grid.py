#!/usr/bin/env python
"""Rank the knn_k / prevalence_threshold grid (grid_knnprev_server.sh).

Walks every <RD>/{zs,ft}_routesae_K*/knn<k>_prev<p>/ablation_summary_labelfree.csv,
pairs each point's ablated row with its own baseline row, and ranks the
points per (clip_mode, K) by worst-group accuracy after ablation -- the
metric spurious-correlation work is judged on -- with average accuracy and
the number of concepts the pair selected alongside.

Writes <RD>/grid_knnprev_summary.csv (every point, every SAE) and prints the
top rows per SAE. Stdlib only, so it runs on a login node without the venv.

Usage:
    python tools/summarize_grid.py --rd results/waterbirds/clip_ft_BIASED_400_<ts>
    python tools/summarize_grid.py --rd ... --top 10 --by accuracy_avg
"""

import argparse
import csv
import glob
import os
import re
import sys

DIR_RE = re.compile(r"^(zs|ft)_routesae_K(\d+)_.*$")
POINT_RE = re.compile(r"^knn(\d+)_prev([0-9.]+)$")


def load_points(rd):
    rows = []
    pattern = os.path.join(rd, "*_routesae_K*", "knn*_prev*", "ablation_summary_labelfree.csv")
    for path in sorted(glob.glob(pattern)):
        point_dir = os.path.basename(os.path.dirname(path))
        sae_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m_sae, m_pt = DIR_RE.match(sae_dir), POINT_RE.match(point_dir)
        if not (m_sae and m_pt):
            continue
        with open(path, newline="") as f:
            recs = list(csv.DictReader(f))
        base = next((r for r in recs if r.get("run") == "original"), None)
        abl = [r for r in recs if r.get("run") == "ablated"]
        if base is None or not abl:
            print(f"  skipping {path}: baseline or ablated row missing", file=sys.stderr)
            continue
        # One pinned ablation per point; if a folder somehow holds several,
        # the latest row is the one the current settings produced.
        r = abl[-1]
        f_ = lambda d, k: float(d[k]) if d.get(k, "") != "" else float("nan")
        rows.append(dict(
            clip_mode={"zs": "zeroshot", "ft": "finetuned"}[m_sae.group(1)],
            K=int(m_sae.group(2)),
            knn_k=int(m_pt.group(1)),
            prevalence=float(m_pt.group(2)),
            n_concepts=int(float(r.get("n_concepts_ablated", 0) or 0)),
            accuracy_avg=f_(r, "accuracy_avg"),
            accuracy_worst_group=f_(r, "accuracy_worst_group"),
            baseline_avg=f_(base, "accuracy_avg"),
            baseline_worst_group=f_(base, "accuracy_worst_group"),
            delta_avg=round(f_(r, "accuracy_avg") - f_(base, "accuracy_avg"), 6),
            delta_worst_group=round(f_(r, "accuracy_worst_group")
                                    - f_(base, "accuracy_worst_group"), 6),
            editing_method=r.get("editing_method", ""),
            ablation_coefficient=r.get("ablation_coefficient", ""),
            path=path,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rd", required=True, help="The fine-tuned run dir the grid ran under.")
    ap.add_argument("--top", type=int, default=8, help="Rows to print per SAE.")
    ap.add_argument("--by", default="accuracy_worst_group",
                    choices=["accuracy_worst_group", "accuracy_avg",
                             "delta_worst_group", "delta_avg"],
                    help="Ranking column (default: worst-group accuracy after ablation).")
    args = ap.parse_args()

    rows = load_points(args.rd)
    if not rows:
        print(f"No grid points found under {args.rd}/*_routesae_K*/knn*_prev*/")
        return 1

    out = os.path.join(args.rd, "grid_knnprev_summary.csv")
    cols = ["clip_mode", "K", "knn_k", "prevalence", "n_concepts",
            "accuracy_avg", "accuracy_worst_group", "baseline_avg", "baseline_worst_group",
            "delta_avg", "delta_worst_group", "editing_method", "ablation_coefficient", "path"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["clip_mode"], r["K"], -r[args.by])):
            w.writerow(r)

    print(f"{len(rows)} grid points -> {out}\n")
    for cm in ("zeroshot", "finetuned"):
        for k in sorted({r["K"] for r in rows}):
            sub = [r for r in rows if r["clip_mode"] == cm and r["K"] == k]
            if not sub:
                continue
            sub.sort(key=lambda r: -r[args.by])
            b = sub[0]
            print(f"=== {cm}  K={k}   ({len(sub)} points; baseline avg "
                  f"{b['baseline_avg']:.4f}, worst-group {b['baseline_worst_group']:.4f}) "
                  f"ranked by {args.by}")
            print(f"  {'knn_k':>5}  {'prev':>5}  {'#concepts':>9}  {'avg':>7}  {'worst':>7}  "
                  f"{'Δavg':>7}  {'Δworst':>7}")
            for r in sub[:args.top]:
                print(f"  {r['knn_k']:>5}  {r['prevalence']:>5.2f}  {r['n_concepts']:>9,}  "
                      f"{r['accuracy_avg']:>7.4f}  {r['accuracy_worst_group']:>7.4f}  "
                      f"{r['delta_avg']:>+7.4f}  {r['delta_worst_group']:>+7.4f}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
