#!/usr/bin/env python3
"""Aggregate all ablation_report_*.txt files in a run directory into one
summary table (Markdown + plain text), covering every
(concept_finding_method x editing_method) combo found.

Usage:
    python tools/summarize_ablation_reports.py <run_dir> [--out NAME]

Writes <run_dir>/<NAME>.md and <run_dir>/<NAME>.txt (default NAME:
ablation_summary).
"""
import argparse
import re
from pathlib import Path

FNAME_RE = re.compile(r"^ablation_report_(.+?)__(.+?)(?:__denoised)?\.txt$")
HOOK_LABELS = {
    "make_sae_ablation_hook": "deactivation",
    "qr_projection_ablation_hook": "projection",
}
CONCEPTS_RE = re.compile(r"^Ablated concepts \((\d+)\):")
GROUP_RE = re.compile(
    r"^Group (\d+) \((.*?)\)\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%"
)
OVERALL_RE = re.compile(r"^Overall\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%")


def parse_report(path: Path) -> dict:
    m = FNAME_RE.match(path.name)
    method, hook_raw = (m.group(1), m.group(2)) if m else (path.stem, "")
    method = method[3:] if method.startswith("ft-") else method
    editing = HOOK_LABELS.get(hook_raw, hook_raw)

    n_concepts = None
    groups = {}
    overall = None
    for line in path.read_text().splitlines():
        if (cm := CONCEPTS_RE.match(line)):
            n_concepts = int(cm.group(1))
        elif (gm := GROUP_RE.match(line)):
            idx, name, orig, ablated, delta = gm.groups()
            groups[int(idx)] = {
                "name": name.strip(), "orig": float(orig),
                "ablated": float(ablated), "delta": float(delta),
            }
        elif (om := OVERALL_RE.match(line)):
            orig, ablated, delta = om.groups()
            overall = {"orig": float(orig), "ablated": float(ablated), "delta": float(delta)}

    return {
        "method": method, "editing": editing, "n_concepts": n_concepts,
        "groups": groups, "overall": overall, "file": path.name,
    }


def build_tables(rows: list[dict]) -> tuple[str, str]:
    rows = sorted(rows, key=lambda r: (r["method"], r["editing"]))
    group_ids = sorted({gid for r in rows for gid in r["groups"]})

    header = ["Method", "Editing", "#Concepts", "Orig Acc", "Ablated Acc", "Δ Overall"]
    header += [f"Δ Group {gid}" for gid in group_ids]

    md_lines = ["| " + " | ".join(header) + " |",
                "|" + "|".join(["---"] * len(header)) + "|"]
    txt_rows = [header]

    for r in rows:
        ov = r["overall"] or {}
        row = [
            r["method"], r["editing"], str(r["n_concepts"]),
            f"{ov.get('orig', float('nan')):.1f}%",
            f"{ov.get('ablated', float('nan')):.1f}%",
            f"{ov.get('delta', float('nan')):+.1f}%",
        ]
        for gid in group_ids:
            g = r["groups"].get(gid)
            row.append(f"{g['delta']:+.1f}%" if g else "n/a")
        md_lines.append("| " + " | ".join(row) + " |")
        txt_rows.append(row)

    # Plain-text aligned table
    widths = [max(len(txt_rows[i][c]) for i in range(len(txt_rows))) for c in range(len(header))]
    txt_lines = []
    for i, row in enumerate(txt_rows):
        txt_lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
        if i == 0:
            txt_lines.append("  ".join("-" * w for w in widths))

    group_legend = "\n".join(
        f"  Group {gid}: {next((r['groups'][gid]['name'] for r in rows if gid in r['groups']), '?')}"
        for gid in group_ids
    )
    return "\n".join(md_lines), "\n".join(txt_lines) + "\n\nGroups:\n" + group_legend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--out", type=str, default="ablation_summary")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    report_paths = sorted(run_dir.glob("ablation_report_*.txt"))
    if not report_paths:
        raise SystemExit(f"No ablation_report_*.txt files found under {run_dir}")

    rows = [parse_report(p) for p in report_paths]
    md, txt = build_tables(rows)

    (run_dir / f"{args.out}.md").write_text(md + "\n")
    (run_dir / f"{args.out}.txt").write_text(txt + "\n")
    print(f"Parsed {len(rows)} report(s) from {run_dir}")
    print(f"Wrote {run_dir / (args.out + '.md')}")
    print(f"Wrote {run_dir / (args.out + '.txt')}")
    print()
    print(txt)


if __name__ == "__main__":
    main()
