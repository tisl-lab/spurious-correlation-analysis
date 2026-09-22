#!/usr/bin/env python
"""Fold summary_shards/*.csv back into an ablation_summary_<method>.csv.

pipeline_3_ablate.append_summary_row writes a shard when it cannot lock the
summary file -- rare, but possible on a filesystem that does not honor flock,
which is exactly the situation a SLURM array can run into. Each shard holds
one run's row with its own header. This appends them through the same
locked, header-growing path the normal runs use, then deletes what it merged.

Safe to run repeatedly and while the sweep is still going: merged shards are
removed, unmerged ones are left alone, and the append is lock-protected.

Usage:
    python tools/merge_summary_shards.py <path/to/ablation_summary_labelfree.csv>
    python tools/merge_summary_shards.py <path> --keep    # don't delete shards
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline_3_ablate as p3   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary_csv", help="The ablation_summary_<method>.csv to merge into.")
    ap.add_argument("--keep", action="store_true",
                    help="Leave the shard files in place after merging.")
    args = ap.parse_args()

    path = os.path.abspath(args.summary_csv)
    stem = os.path.splitext(os.path.basename(path))[0]
    shard_dir = os.path.join(os.path.dirname(path), "summary_shards")
    shards = sorted(glob.glob(os.path.join(shard_dir, f"{stem}.*.csv")))

    if not shards:
        print(f"No shards for {os.path.basename(path)} in {shard_dir} -- nothing to merge.")
        return 0

    merged = 0
    for shard in shards:
        with open(shard, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"  {os.path.basename(shard)} is empty -- left in place.")
            continue
        for row in rows:
            # baseline_row=None: a shard only ever exists because the summary
            # already existed and was locked, so its baseline row is there.
            p3.append_summary_row(path, row)
            merged += 1
        if not args.keep:
            os.remove(shard)

    print(f"\nMerged {merged} row(s) from {len(shards)} shard(s) into {path}"
          f"{' (shards kept)' if args.keep else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
