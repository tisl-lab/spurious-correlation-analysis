#!/usr/bin/env python3
"""Scan a run dir for ablation_report_*.txt files and print clean comparison
tables (reconstruction-only baseline, and concept-finding/removal results),
grouped by SAE architecture and CLIP mode.

Self-contained, stdlib only -- copy this file to the server and run it
directly with plain `python3`, no venv/activation needed.

Usage:
    python3 summarize_ablation_results.py --rd results/waterbirds/clip_ft_BIASED_400_<timestamp>
    python3 summarize_ablation_results.py --rd /path/to/run_dir --markdown out.md
"""
import argparse
import glob
import os
import re
import sys

GROUP_RE = re.compile(
    r"^Group (\d) \((.*?)\)\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%"
)
OVERALL_RE = re.compile(
    r"^Overall\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%"
)
ADJ_RE = re.compile(
    r"^Adj avg acc.*?\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%"
)
WORST_RE = re.compile(
    r"^Worst-group acc\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%"
)
ABLATED_RE = re.compile(r"^Ablated concepts \((\d+)\)")

# Folder-name patterns for each SAE family this project produces.
# zs_routesae_K<K>_...  /  ft_routesae_K<K>_...
ROUTESAE_DIR_RE = re.compile(r"^(zs|ft)_routesae_K(\d+)_")
# zs_MSAE_RW_<activation>  /  ft_MSAE_RW_<activation>
MSAE_DIR_RE = re.compile(r"^(zs|ft)_(MSAE_RW|MSAE_UW)_(.+)$")

# ablation_report_<zs|ft>-<method>__<hook_tag>.txt
REPORT_FNAME_RE = re.compile(r"^ablation_report_(?:zs|ft)-(.+?)__(.+)\.txt$")

CLIP_MODE_NAME = {"zs": "zero-shot", "ft": "fine-tuned"}


def parse_report(path):
    with open(path) as f:
        text = f.read()
    result = {"groups": {}, "overall": None, "adj": None, "worst": None, "n_ablated": None}
    for line in text.splitlines():
        m = ABLATED_RE.match(line)
        if m:
            result["n_ablated"] = int(m.group(1))
            continue
        m = GROUP_RE.match(line)
        if m:
            gid, gname, orig, abl, delta = m.groups()
            result["groups"][int(gid)] = (gname.strip(), float(orig), float(abl), delta)
            continue
        m = OVERALL_RE.match(line)
        if m:
            result["overall"] = (float(m.group(1)), float(m.group(2)), m.group(3))
            continue
        m = ADJ_RE.match(line)
        if m:
            result["adj"] = (float(m.group(1)), float(m.group(2)), m.group(3))
            continue
        m = WORST_RE.match(line)
        if m:
            result["worst"] = (float(m.group(1)), float(m.group(2)), m.group(3))
            continue
    return result


def classify_dir(dirname):
    """Return (sae_label, clip_mode) for a representations-folder basename, or None."""
    m = ROUTESAE_DIR_RE.match(dirname)
    if m:
        clip_mode, k = m.groups()
        return f"RouteSAE (K={k})", clip_mode
    m = MSAE_DIR_RE.match(dirname)
    if m:
        clip_mode, model, activation = m.groups()
        return f"{model} ({activation})", clip_mode
    return None


# ── sweep_archive_*/ conventions ────────────────────────────────────────────
# The three array scripts archive a combo's knn<k>_prev<p>/ folder by copying
# it to sweep_archive_*/<combo>/ -- which drops the parent "zs_routesae_K.../"
# folder name entirely, so the SAE label can't be read off the path the way
# classify_dir() does for the live folders. Instead these map straight to
# whatever each script currently hardcodes for K/activation -- if a script's
# K changes, update the mapping here too.
ARCHIVE_COMBO_RE = re.compile(r"^(zeroshot|finetuned)_(.+)$")
ARCHIVE_COMBO_TYPED_RE = re.compile(r"^(routesae|msae)_(zeroshot|finetuned)_(.+)$")
CLIP_MODE_LONG_TO_SHORT = {"zeroshot": "zs", "finetuned": "ft"}

# run_routesae_array_server.sh -> sweep_archive_routesae/<zeroshot|finetuned>_<method>/
# run_routesae_k32_array_server.sh -> sweep_archive_routesae_k32/<zeroshot|finetuned>_<method>/
ARCHIVE_ROOT_SAE_LABEL = {
    "sweep_archive_routesae": "RouteSAE (K=256)",
    "sweep_archive_routesae_k32": "RouteSAE (K=32)",
}
# run_sae_comparison_array_server.sh -> sweep_archive_sae_comparison/<routesae|msae>_<zeroshot|finetuned>_<method>/
ARCHIVE_SAE_TYPE_LABEL = {
    "routesae": "RouteSAE (K=256)",
    "msae": "MSAE_RW (TopKReLU_64)",
}


def classify_archive(parts):
    """Return (sae_label, clip_mode) for a sweep_archive_*/<combo>/... path, or None."""
    if len(parts) < 2:
        return None
    root, combo = parts[0], parts[1]
    if root in ARCHIVE_ROOT_SAE_LABEL:
        m = ARCHIVE_COMBO_RE.match(combo)
        if m:
            clip_long, _method = m.groups()
            return ARCHIVE_ROOT_SAE_LABEL[root], CLIP_MODE_LONG_TO_SHORT[clip_long]
    if root == "sweep_archive_sae_comparison":
        m = ARCHIVE_COMBO_TYPED_RE.match(combo)
        if m:
            sae_type, clip_long, _method = m.groups()
            return ARCHIVE_SAE_TYPE_LABEL[sae_type], CLIP_MODE_LONG_TO_SHORT[clip_long]
    return None


def find_reports(rd):
    """Yield (sae_label, clip_mode, method, path) for every ablation report
    found under rd, including sweep_archive_*/ copies. Dedupes by
    (sae_label, clip_mode, method), preferring the live knn*/ copy over an
    archived one, and the newest mtime when still tied."""
    candidates = {}
    for path in glob.glob(os.path.join(rd, "**", "ablation_report_*.txt"), recursive=True):
        rel = os.path.relpath(path, rd)
        parts = rel.split(os.sep)
        fname = parts[-1]
        fm = REPORT_FNAME_RE.match(fname)
        if not fm:
            continue
        method, hook_tag = fm.groups()

        # The SAE/clip_mode-bearing folder is the first path component
        # (representations folder), whether reached directly or via
        # sweep_archive_*/<combo>/... which mirrors the same report.
        sae_clip = None
        for part in parts[:-1]:
            sae_clip = classify_dir(part)
            if sae_clip:
                break
        if sae_clip is None:
            sae_clip = classify_archive(parts)
        if sae_clip is None:
            continue
        sae_label, clip_mode = sae_clip
        is_archived = "sweep_archive" in rel
        key = (sae_label, clip_mode, method)
        mtime = os.path.getmtime(path)
        if key not in candidates:
            candidates[key] = (path, is_archived, mtime)
        else:
            _, prev_archived, prev_mtime = candidates[key]
            # Prefer non-archived; among equal archived-ness, prefer newest.
            if (prev_archived and not is_archived) or (
                prev_archived == is_archived and mtime > prev_mtime
            ):
                candidates[key] = (path, is_archived, mtime)

    for (sae_label, clip_mode, method), (path, _, _) in candidates.items():
        yield sae_label, clip_mode, method, path


GROUP_ORDER = [0, 1, 2, 3]
GROUP_SHORT = {0: "G0 (land/land)", 1: "G1 (land/water)", 2: "G2 (water/land)", 3: "G3 (water/water)"}


def fmt_cell(orig, abl, delta):
    return f"{orig:.1f}%→{abl:.1f}% ({delta}%)"


def build_tables(rows):
    """rows: list of (sae_label, clip_mode, method, parsed_report)."""
    recon_rows = [r for r in rows if r[2] == "reconstruction_only"]
    concept_rows = [r for r in rows if r[2] != "reconstruction_only"]

    def sort_key(r):
        return (r[0], 0 if r[1] == "zs" else 1, r[2])

    recon_rows.sort(key=sort_key)
    concept_rows.sort(key=sort_key)

    out = []

    out.append("## Reconstruction-only (0 concepts removed)\n")
    if recon_rows:
        header = ["SAE", "CLIP mode", "Orig Acc", "Recon Acc", "Δ Overall"] + [GROUP_SHORT[g] for g in GROUP_ORDER]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for sae_label, clip_mode, _, rep in recon_rows:
            overall = rep["overall"]
            if overall is None:
                continue
            row = [sae_label, CLIP_MODE_NAME[clip_mode], f"{overall[0]:.1f}%", f"{overall[1]:.1f}%", f"**{overall[2]}%**"]
            for g in GROUP_ORDER:
                if g in rep["groups"]:
                    _, o, a, d = rep["groups"][g]
                    row.append(fmt_cell(o, a, d))
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
    else:
        out.append("*(none found)*")
    out.append("")

    out.append("## Concept-finding + removal\n")
    if concept_rows:
        header = ["SAE", "CLIP mode", "Method", "#Concepts", "Orig Acc", "Ablated Acc", "Δ Overall"] + [GROUP_SHORT[g] for g in GROUP_ORDER]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for sae_label, clip_mode, method, rep in concept_rows:
            overall = rep["overall"]
            if overall is None:
                continue
            n = rep["n_ablated"] if rep["n_ablated"] is not None else "?"
            row = [sae_label, CLIP_MODE_NAME[clip_mode], method, str(n),
                   f"{overall[0]:.1f}%", f"{overall[1]:.1f}%", f"**{overall[2]}%**"]
            for g in GROUP_ORDER:
                if g in rep["groups"]:
                    _, o, a, d = rep["groups"][g]
                    row.append(fmt_cell(o, a, d))
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
    else:
        out.append("*(none found yet)*")
    out.append("")

    # Adjusted / worst-group accuracy, when present (not every report computes these).
    extra_rows = [r for r in rows if r[3]["adj"] or r[3]["worst"]]
    if extra_rows:
        out.append("## Adjusted / worst-group accuracy (where reported)\n")
        header = ["SAE", "CLIP mode", "Method", "Adj avg acc (orig→abl)", "Worst-group acc (orig→abl)"]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        extra_rows.sort(key=sort_key)
        for sae_label, clip_mode, method, rep in extra_rows:
            adj = f"{rep['adj'][0]:.1f}%→{rep['adj'][1]:.1f}% ({rep['adj'][2]}%)" if rep["adj"] else "—"
            worst = f"{rep['worst'][0]:.1f}%→{rep['worst'][1]:.1f}% ({rep['worst'][2]}%)" if rep["worst"] else "—"
            out.append(f"| {sae_label} | {CLIP_MODE_NAME[clip_mode]} | {method} | {adj} | {worst} |")
        out.append("")

    return "\n".join(out)


def build_collapsed_table(rows):
    """Compact form: one baseline row per CLIP mode (raw CLIP, no SAE at all),
    then one row per (SAE, method) showing only the FINAL (post-ablation)
    values -- no orig→final arrows -- keeping Δ only for Overall and
    Worst-group, since those are the two numbers that matter for comparing
    runs at a glance."""

    def sort_key(r):
        method_rank = 0 if r[2] == "reconstruction_only" else 1
        return (0 if r[1] == "zs" else 1, r[0], method_rank, r[2])

    rows = sorted(rows, key=sort_key)

    # Baseline (raw CLIP, no SAE) group/overall values -- same regardless of
    # SAE/method for a given clip_mode, so take them from the first report
    # seen for that mode. Small (~0.1%) cross-report differences in the
    # "orig" numbers reflect floating-point/batching nondeterminism, not a
    # real difference in the underlying CLIP model.
    baseline = {}
    for sae_label, clip_mode, method, rep in rows:
        if clip_mode in baseline or rep["overall"] is None:
            continue
        baseline[clip_mode] = rep

    header = ["SAE", "CLIP mode", "Method", "#Concepts"] + [GROUP_SHORT[g] for g in GROUP_ORDER] + [
        "Overall", "Δ Overall", "Worst-group", "Δ Worst-group",
    ]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    seen_baseline = set()
    for sae_label, clip_mode, method, rep in rows:
        if rep["overall"] is None:
            continue
        if clip_mode not in seen_baseline and clip_mode in baseline:
            seen_baseline.add(clip_mode)
            b = baseline[clip_mode]
            brow = ["**CLIP baseline (no SAE)**", CLIP_MODE_NAME[clip_mode], "—", "—"]
            for g in GROUP_ORDER:
                brow.append(f"{b['groups'][g][1]:.1f}%" if g in b["groups"] else "—")
            brow.append(f"{b['overall'][0]:.1f}%")
            brow.append("—")
            brow.append(f"{b['worst'][0]:.1f}%" if b["worst"] else "—")
            brow.append("—")
            out.append("| " + " | ".join(brow) + " |")

        n = rep["n_ablated"] if rep["n_ablated"] is not None else "?"
        row = [sae_label, CLIP_MODE_NAME[clip_mode], method, str(n)]
        for g in GROUP_ORDER:
            row.append(f"{rep['groups'][g][2]:.1f}%" if g in rep["groups"] else "—")
        row.append(f"{rep['overall'][1]:.1f}%")
        row.append(f"**{rep['overall'][2]}%**")
        if rep["worst"]:
            row.append(f"{rep['worst'][1]:.1f}%")
            row.append(f"**{rep['worst'][2]}%**")
        else:
            row.append("—")
            row.append("—")
        out.append("| " + " | ".join(row) + " |")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rd", required=True, help="Run dir, e.g. results/waterbirds/clip_ft_BIASED_400_<timestamp>")
    ap.add_argument("--full", action="store_true",
                     help="Also print the verbose orig→final tables (build_tables), split into "
                          "reconstruction-only / concept-finding / adj-worst-group sections. "
                          "Default is just the collapsed final-values table.")
    ap.add_argument("--out", default=None,
                     help="Path to write the collapsed table as .txt (default: "
                          "<rd_basename>_summary.txt next to this script)")
    args = ap.parse_args()

    if not os.path.isdir(args.rd):
        sys.exit(f"ERROR: {args.rd} is not a directory")

    rows = []
    for sae_label, clip_mode, method, path in find_reports(args.rd):
        rows.append((sae_label, clip_mode, method, parse_report(path)))

    if not rows:
        sys.exit(f"No ablation_report_*.txt files found under {args.rd}")

    title = f"# SAE ablation comparison — {os.path.basename(args.rd.rstrip('/'))}\n"
    collapsed = build_collapsed_table(rows)

    print(title)
    print(collapsed)

    if args.full:
        print()
        print(build_tables(rows))

    out_path = args.out or f"{os.path.basename(args.rd.rstrip('/'))}_summary.txt"
    with open(out_path, "w") as f:
        f.write(title + "\n")
        f.write(collapsed + "\n")
    print(f"\n[written to {out_path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
