#!/usr/bin/env python3
"""Generate the "SUMMARY TABLE — Two-Stage ..." report (same format as
results/waterbirds/twostage_vitb32_100_400/.../summary_table_twostage_vitb32.txt)
from the ablation_report_*.txt files in a run directory.

Columns:
  G0-G3      per-group ABLATED accuracy (Den=Y variants only)
  Ovr        real overall accuracy, test-weighted (taken directly from each
             report's "Overall" line)
  Adj        adjusted accuracy: G0-G3 reweighted by Waterbirds' OFFICIAL
             train-group proportions (3498, 184, 56, 1057 / 4795) --
             standard in the spurious-correlation literature, independent
             of this run's actual (roughly balanced) eval set
  WG         worst-group accuracy = min(G0..G3)
  Delta cols same, ablated - original

Usage:
    python tools/summary_table_twostage.py <run_dir> --model "ViT-L/14" [--out NAME]
"""
import argparse
import re
from pathlib import Path

FNAME_RE = re.compile(r"^ablation_report_ft-(.+?)__(.+?)(?:__denoised)?\.txt$")
HOOK_LABELS = {
    "make_sae_ablation_hook": "deactivation",
    "qr_projection_ablation_hook": "projection",
}
GROUP_RE = re.compile(
    r"^Group (\d+) \((.*?)\)\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%"
)
OVERALL_RE = re.compile(r"^Overall\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%")

# Official Waterbirds TRAIN group counts (Sagawa et al. 2019), total 4795.
TRAIN_COUNTS = {0: 3498, 1: 184, 2: 56, 3: 1057}
GROUP_LABELS = {
    0: "Landbirds on Land", 1: "Landbirds on Water",
    2: "Waterbirds on Land", 3: "Waterbirds on Water",
}
METHOD_ORDER = ["labelfree", "labelguided", "dialguided", "highmag"]
EDITING_ORDER = ["deactivation", "projection"]


def adjusted(group_vals: dict) -> float:
    total = sum(TRAIN_COUNTS.values())
    return sum(group_vals[g] * TRAIN_COUNTS[g] for g in TRAIN_COUNTS) / total


def parse_report(path: Path) -> dict:
    m = FNAME_RE.match(path.name)
    method, hook_raw = m.group(1), m.group(2)
    editing = HOOK_LABELS.get(hook_raw, hook_raw)

    groups_orig, groups_abl = {}, {}
    overall = None
    for line in path.read_text().splitlines():
        if (gm := GROUP_RE.match(line)):
            idx = int(gm.group(1))
            groups_orig[idx] = float(gm.group(3))
            groups_abl[idx] = float(gm.group(4))
        elif (om := OVERALL_RE.match(line)):
            overall = {"orig": float(om.group(1)), "ablated": float(om.group(2))}

    return {
        "method": method, "editing": editing,
        "groups_orig": groups_orig, "groups_abl": groups_abl,
        "overall": overall,
    }


def fmt(v):
    return f"{v:6.1f}" if v is not None else "     —"


def fmt_delta(v):
    return f"{v:+6.1f}" if v is not None else "     —"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--model", type=str, required=True, help='e.g. "ViT-L/14"')
    ap.add_argument("--out", type=str, default=None,
                     help="output basename (default: summary_table_twostage_<model tag>)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    reports = {}
    for p in sorted(run_dir.glob("ablation_report_ft-*.txt")):
        r = parse_report(p)
        reports[(r["method"], r["editing"])] = r

    if not reports:
        raise SystemExit(f"No ablation_report_ft-*.txt files found under {run_dir}")

    # Baseline: original (unablated) accuracy, identical across all reports.
    any_report = next(iter(reports.values()))
    base_groups = any_report["groups_orig"]
    base_ovr = any_report["overall"]["orig"]
    base_adj = adjusted(base_groups)
    base_wg = min(base_groups.values())

    tag = args.model.replace("/", "").replace(" ", "").lower()
    out_name = args.out or f"summary_table_twostage_{tag}"

    header = f"SUMMARY TABLE — Two-Stage (BALANCED→BIASED) CLIP {args.model}, all combinations (Ablated Acc %)"
    col_hdr = (
        f"{'Concept':<17} {'Editing':<16} {'Den':<4} "
        f"{'G0':>7} {'G1':>7} {'G2':>7} {'G3':>7} {'Ovr':>7} {'Adj':>7} {'WG':>7} "
        f"{'ΔG0':>7} {'ΔG1':>7} {'ΔG2':>7} {'ΔG3':>7} {'ΔOvr':>7} {'ΔAdj':>7} {'ΔWG':>7}"
    )
    rule = "─" * len(col_hdr)

    lines = [header, rule, col_hdr, rule]
    lines.append(
        f"{'TwoStage (original)':<17} {'—':<16} {'—':<4} "
        f"{fmt(base_groups.get(0))} {fmt(base_groups.get(1))} {fmt(base_groups.get(2))} {fmt(base_groups.get(3))} "
        f"{fmt(base_ovr)} {fmt(base_adj)} {fmt(base_wg)} "
        f"{'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7}"
    )
    lines.append(rule)

    footnotes = []
    for method in METHOD_ORDER:
        any_editing_found = any((method, e) in reports for e in EDITING_ORDER)
        if not any_editing_found:
            continue
        for editing in EDITING_ORDER:
            r = reports.get((method, editing))
            if r is None:
                # Method ran but this editing variant's report is missing --
                # e.g. 0 candidate concepts, ablation skipped for the whole method.
                lines.append(
                    f"{method:<17} {editing:<16} {'Y':<4} "
                    f"{'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} "
                    f"{'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—':>7} {'—*':>7}"
                )
                continue
            g_abl = r["groups_abl"]
            adj_abl = adjusted(g_abl)
            wg_abl = min(g_abl.values())
            ovr_abl = r["overall"]["ablated"]

            d_groups = {g: g_abl[g] - base_groups[g] for g in g_abl}
            d_adj = adj_abl - base_adj
            d_wg = wg_abl - base_wg
            d_ovr = ovr_abl - base_ovr

            lines.append(
                f"{method:<17} {editing:<16} {'Y':<4} "
                f"{fmt(g_abl.get(0))} {fmt(g_abl.get(1))} {fmt(g_abl.get(2))} {fmt(g_abl.get(3))} "
                f"{fmt(ovr_abl)} {fmt(adj_abl)} {fmt(wg_abl)} "
                f"{fmt_delta(d_groups.get(0))} {fmt_delta(d_groups.get(1))} {fmt_delta(d_groups.get(2))} {fmt_delta(d_groups.get(3))} "
                f"{fmt_delta(d_ovr)} {fmt_delta(d_adj)} {fmt_delta(d_wg)}"
            )
        lines.append("")

    if not any((m, e) in reports for m in ["labelguided"] for e in EDITING_ORDER):
        footnotes.append("* labelguided found 0 candidate concepts for this model — ablation was skipped.")
        # Insert a labelguided placeholder block if it's simply absent (not attempted at all is
        # indistinguishable from 0-concepts from report files alone; both render identically here).
        idx = next((i for i, l in enumerate(lines) if l.startswith("dialguided") or l.startswith("highmag")
                    or l.startswith("labelfree")), len(lines))
        # Only add the block if not already present as a genuine section.
        if not any(l.startswith("labelguided") for l in lines):
            block = [
                f"{'labelguided':<17} {'deactivation':<16} {'Y':<4} " + " ".join([f"{'—':>7}"] * 13),
                f"{'labelguided':<17} {'projection':<16} {'Y':<4} " + " ".join([f"{'—':>7}"] * 12) + f" {'—*':>7}",
                "",
            ]
            # Place it after labelfree's block if present, else at top of method rows.
            insert_at = len(lines)
            for i, l in enumerate(lines):
                if l == "" and i > 0 and lines[i - 1].startswith("labelfree"):
                    insert_at = i + 1
                    break
            lines[insert_at:insert_at] = block

    lines.append(rule)
    lines.append("G0: Landbirds on Land           Ovr: Real overall Accuracy (test-weighted)")
    lines.append("G1: Landbirds on Water          Adj: Adjusted Accuracy (train-group-weighted)")
    lines.append("G2: Waterbirds on Land          WG:  Worst Group Accuracy")
    lines.append("G3: Waterbirds on Water")
    for f in footnotes:
        lines.append(f)
    lines.append("No PRISM baseline was run for this two-stage model.")

    out_text = "\n".join(lines) + "\n"
    out_path = run_dir / f"{out_name}.txt"
    out_path.write_text(out_text)
    print(f"Wrote {out_path}")
    print()
    print(out_text)


if __name__ == "__main__":
    main()
