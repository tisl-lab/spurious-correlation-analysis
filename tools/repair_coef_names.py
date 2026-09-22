#!/usr/bin/env python
"""Retro-label sweep outputs written before the ablation coefficient was part
of the filename, so a resubmitted sweep redoes only the coefficients that
never actually ran.

BACKGROUND. Until pipeline_3_ablate.py put __c<coef> in its output names, two
runs differing only in --ablation_coefficient shared a report path, and the
second was skipped as "already complete". Every sweep loops coefficients
innermost with 0.1 first, so each existing report/prediction file IS the
coef=0.1 run, and the other coefficients were logged as done without running.

WHAT THIS DOES, per <RD>/{zs,ft}_routesae_K<K>_*/knn10_prev0.15/:
  1. ablation_report_ft-<m>__<variant>[__s<S>]__p<P>_top<N>.txt  and the
     matching predictions_*.csv are renamed to end in __c0.1 -- exactly the
     name the fixed stage 3 gives a coef=0.1 run, so it will no-op on them.
     Files not in that current naming (e.g. *__both.txt, *__routesae_*.txt)
     are older artifacts and are left alone.
  2. ablation_summary_<m>.csv gains ablation_coefficient / projection_method
     columns if missing; every "ablated" row with a blank coefficient gets
     0.1 (the baseline row stays blank).
  3. logs/sweep_<m>_<clipmode>_K<K>_array/done.task*.log lose every line
     whose coefficient field is not 0.1, so the resumed sweep re-runs
     exactly those combinations.

Dry run by default -- prints the plan; --apply performs it. Stdlib only.
Run it while no sweep is writing to those folders.

Usage:
    python tools/repair_coef_names.py --rd results/waterbirds/clip_ft_BIASED_400_<ts> --k 32
    python tools/repair_coef_names.py --rd ... --k 32 --apply
"""

import argparse
import csv
import glob
import os
import re
import sys

# Current stage-3 naming WITHOUT the coefficient segment. The (?!.*__c) guard
# skips anything already labelled.
NAME_RE = re.compile(
    r"^(?P<kind>ablation_report|predictions)_ft-(?P<method>[a-z]+)__"
    r"(?P<variant>deactivation|projection)(_candidates_only)?"
    r"(__s[^_]+)?__p[0-9.]+_top\d+(?P<ext>\.txt|\.csv)$"
)
COEF = "0.1"


def repair_dir(sae_dir, apply):
    n_ren = n_skip = 0
    for f in sorted(os.listdir(sae_dir)):
        m = NAME_RE.match(f)
        if not m or "__c" in f:
            continue
        new = f[: -len(m.group("ext"))] + f"__c{COEF}" + m.group("ext")
        src, dst = os.path.join(sae_dir, f), os.path.join(sae_dir, new)
        if os.path.exists(dst):
            n_skip += 1
            print(f"    exists, left alone: {new}")
            continue
        n_ren += 1
        if apply:
            os.rename(src, dst)
    print(f"  {'renamed' if apply else 'would rename'} {n_ren} report/prediction file(s)"
          + (f", {n_skip} skipped" if n_skip else ""))

    for path in sorted(glob.glob(os.path.join(sae_dir, "ablation_summary_*.csv"))):
        with open(path, newline="") as fh:
            r = csv.reader(fh)
            header = next(r, None)
            rows = [line for line in r if line]
        if not header:
            continue
        for col in ("ablation_coefficient", "projection_method"):
            if col not in header:
                header.append(col)
                rows = [line + [""] for line in rows]
        ci, ri = header.index("ablation_coefficient"), header.index("run")
        filled = 0
        for line in rows:
            line += [""] * (len(header) - len(line))
            if line[ri] == "ablated" and line[ci] == "":
                line[ci] = COEF
                filled += 1
        print(f"  {os.path.basename(path)}: {'set' if apply else 'would set'} "
              f"coefficient={COEF} on {filled} ablated row(s)")
        if apply and filled:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerows(rows)


def repair_done_logs(log_root, method, clip_mode, k, apply):
    d = os.path.join(log_root, f"sweep_{method}_{clip_mode}_K{k}_array")
    logs = sorted(glob.glob(os.path.join(d, "done.task*.log")))
    if not logs:
        return
    kept = dropped = 0
    for path in logs:
        with open(path) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        # key: method|editing|images|coef|percent
        keep = [ln for ln in lines if ln.split("|")[3:4] == [COEF]]
        kept += len(keep)
        dropped += len(lines) - len(keep)
        if apply:
            with open(path, "w") as fh:
                fh.write("".join(ln + "\n" for ln in keep))
    print(f"  {os.path.relpath(d)}: {'kept' if apply else 'would keep'} {kept} "
          f"coef={COEF} entries, {'dropped' if apply else 'would drop'} {dropped} "
          f"(never actually ran)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rd", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--methods", default="labelfree highmag labelguided dialguided")
    ap.add_argument("--analysis", default="knn10_prev0.15")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--apply", action="store_true", help="Perform the changes (default: dry run).")
    args = ap.parse_args()

    if not args.apply:
        print("DRY RUN -- nothing is changed. Re-run with --apply to perform it.\n")

    found = False
    for base in sorted(glob.glob(os.path.join(args.rd, f"*_routesae_K{args.k}_*"))):
        clip_mode = "zeroshot" if os.path.basename(base).startswith("zs_") else "finetuned"
        sae_dir = os.path.join(base, args.analysis)
        if not os.path.isdir(sae_dir):
            continue
        found = True
        print(f"== {os.path.relpath(sae_dir)}  ({clip_mode}, K={args.k})")
        repair_dir(sae_dir, args.apply)
        for m in args.methods.split():
            repair_done_logs(args.logs, m, clip_mode, args.k, args.apply)
        print()
    if not found:
        print(f"No *_routesae_K{args.k}_*/{args.analysis} under {args.rd}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
