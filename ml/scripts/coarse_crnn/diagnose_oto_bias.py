"""Diagnose systematic OTO parameter bias between a reference and one or more
auto-generated variants.

The listener observation that motivates this tool: "approximate boundary
location is fine, but preutterance is positioned such that the consonant
sounds dragged, and offset is off, in at least 50% of rows." That points at
**systematic bias** in one of the OTO anchors, which is what the LightGBM
residual is meant to remove. Before tuning the residual we want a precise
read on which anchor, in which role, and in which direction.

For every (wav, alias) pair found in both the reference and the compared
OTO, this script computes deltas on the *absolute* anchor positions:

- `offset_abs`        = offset
- `pre_abs`           = offset + preutterance        (vowel onset)
- `ovl_abs`           = offset + min(overlap, pre)
- `cons_abs`          = offset + max(consonant, pre)
- `cutoff_abs`        = offset + |cutoff|             (cutoff negative encodes offset-relative magnitude)

Positive delta means the compared version is *later/longer* than the
reference; negative means *earlier/shorter*. Rows are grouped by role
(same alias→role classifier used by the rest of the pipeline).

Outputs a markdown summary plus an optional machine-readable JSON.

Example::

    python -m ml.scripts.coarse_crnn.diagnose_oto_bias `
        --manifest ml/eval/ab_listening/manifests/2026-05-15_baseline.json `
        --reference-version manual `
        --compared-versions autooto_baseline,autooto_F1 `
        --out ml/eval/ab_listening/reports/2026-05-15_baseline.bias.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

from core.coarse_crnn.alias_role import classify_alias_role, normalize_role
from core.coarse_crnn.boundary_targets import _infer_alias_type as _boundary_alias_type
from core.coarse_crnn.boundary_targets import _infer_transition_type as _boundary_transition_type
from core.coarse_crnn.lang import normalize_language
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


ANCHOR_KEYS: tuple[str, ...] = (
    "offset_abs",
    "pre_abs",
    "ovl_abs",
    "cons_abs",
    "cutoff_abs",
)
ANCHOR_LABELS: dict[str, str] = {
    "offset_abs": "offset",
    "pre_abs": "preutterance(abs)",
    "ovl_abs": "overlap(abs)",
    "cons_abs": "consonant(abs)",
    "cutoff_abs": "cutoff(abs)",
}


def _read_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "run_id" not in data or "voicebanks" not in data:
        raise SystemExit("manifest missing run_id or voicebanks")
    return data


def _read_oto_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    text = read_text_with_fallback(path)
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        parsed = parse_oto_line(line)
        if not parsed:
            continue
        out.append(parsed)
    return out


def compute_abs_anchors(parsed: dict[str, Any]) -> dict[str, float]:
    """Map an OTO row to its absolute anchor positions in ms (project convention)."""
    offset = max(0.0, float(parsed.get("offset", 0.0) or 0.0))
    pre_rel = max(0.0, float(parsed.get("pre", 0.0) or 0.0))
    ovl_rel = max(0.0, float(parsed.get("ovl", 0.0) or 0.0))
    cons_rel = max(0.0, float(parsed.get("cons", 0.0) or 0.0))
    cutoff_raw = float(parsed.get("cutoff", 0.0) or 0.0)
    pre_abs = offset + pre_rel
    ovl_abs = offset + min(pre_rel, ovl_rel)
    cons_abs = offset + max(pre_rel, cons_rel)
    cutoff_abs = offset + (abs(cutoff_raw) if cutoff_raw < 0.0 else cutoff_raw)
    return {
        "offset_abs": float(offset),
        "pre_abs": float(pre_abs),
        "ovl_abs": float(ovl_abs),
        "cons_abs": float(cons_abs),
        "cutoff_abs": float(cutoff_abs),
    }


def role_for_alias(alias: str, language: str) -> str:
    lang = normalize_language(language) or "korean"
    alias_type = _boundary_alias_type(alias, language=lang)
    transition_type = _boundary_transition_type(alias, language=lang)
    return normalize_role(
        classify_alias_role(
            lang, alias,
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["wav"]).strip(), str(row["alias"]).strip())
        table[key].append(row)
    return table


def compute_deltas(
    *,
    reference_rows: list[dict[str, Any]],
    compared_rows: list[dict[str, Any]],
    language: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Match by (wav, alias, occurrence) and emit per-row delta records."""
    ref_index = _index_rows(reference_rows)
    seen: dict[tuple[str, str], int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    matched = 0
    unmatched = 0
    for row in compared_rows:
        wav = str(row["wav"]).strip()
        alias = str(row["alias"]).strip()
        occ = seen[(wav, alias)]
        seen[(wav, alias)] += 1
        ref_bucket = ref_index.get((wav, alias))
        if not ref_bucket or occ >= len(ref_bucket):
            unmatched += 1
            continue
        ref_row = ref_bucket[occ]
        ref_abs = compute_abs_anchors(ref_row)
        comp_abs = compute_abs_anchors(row)
        deltas = {key: float(comp_abs[key] - ref_abs[key]) for key in ANCHOR_KEYS}
        out.append(
            {
                "wav": wav,
                "alias": alias,
                "role": role_for_alias(alias, language),
                **deltas,
                "ref_offset": ref_abs["offset_abs"],
                "comp_offset": comp_abs["offset_abs"],
            }
        )
        matched += 1
    return out, {"matched": matched, "unmatched_in_reference": unmatched, "compared_total": matched + unmatched}


def _compute_stats(deltas: list[float]) -> dict[str, float]:
    if not deltas:
        return {
            "count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "mae": 0.0,
            "pct_late_5ms": 0.0, "pct_early_5ms": 0.0,
            "pct_within_5ms": 0.0, "pct_within_10ms": 0.0, "pct_within_20ms": 0.0,
        }
    arr = np.asarray(deltas, dtype=np.float64)
    abs_arr = np.abs(arr)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "mae": float(abs_arr.mean()),
        "pct_late_5ms": float((arr > 5.0).mean() * 100.0),
        "pct_early_5ms": float((arr < -5.0).mean() * 100.0),
        "pct_within_5ms": float((abs_arr <= 5.0).mean() * 100.0),
        "pct_within_10ms": float((abs_arr <= 10.0).mean() * 100.0),
        "pct_within_20ms": float((abs_arr <= 20.0).mean() * 100.0),
    }


def aggregate_per_role(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """{role: {anchor_key: stats}} plus a magic 'ALL' role for the whole set."""
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_role[rec["role"]].append(rec)
        by_role["ALL"].append(rec)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for role, rows in by_role.items():
        out[role] = {}
        for key in ANCHOR_KEYS:
            out[role][key] = _compute_stats([float(r[key]) for r in rows])
    return out


def worst_rows(records: list[dict[str, Any]], *, anchor_key: str, n: int = 10) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: -abs(float(r[anchor_key])))[:n]


def diagnose(
    *,
    manifest: dict[str, Any],
    run_dir: str,
    reference_version: str,
    compared_versions: list[str],
) -> dict[str, Any]:
    voicebanks = list(manifest.get("voicebanks") or [])
    results: list[dict[str, Any]] = []
    for vb in voicebanks:
        vb_id = str(vb.get("id", "") or "")
        language = str(vb.get("language", "") or "")
        ref_oto = os.path.join(run_dir, reference_version, vb_id, "oto.ini")
        ref_rows = _read_oto_rows(ref_oto)
        if not ref_rows:
            results.append(
                {"voicebank_id": vb_id, "version_id": None, "status": "reference_missing", "reference_oto": ref_oto}
            )
            continue
        for ver in compared_versions:
            comp_oto = os.path.join(run_dir, ver, vb_id, "oto.ini")
            comp_rows = _read_oto_rows(comp_oto)
            if not comp_rows:
                results.append(
                    {"voicebank_id": vb_id, "version_id": ver, "status": "compared_missing", "compared_oto": comp_oto}
                )
                continue
            records, counts = compute_deltas(
                reference_rows=ref_rows,
                compared_rows=comp_rows,
                language=language,
            )
            agg = aggregate_per_role(records)
            results.append(
                {
                    "voicebank_id": vb_id,
                    "version_id": ver,
                    "status": "ok",
                    "language": language,
                    "reference_oto": ref_oto,
                    "compared_oto": comp_oto,
                    "matched": counts["matched"],
                    "unmatched_in_reference": counts["unmatched_in_reference"],
                    "compared_total": counts["compared_total"],
                    "stats_by_role": agg,
                    "worst_offset": worst_rows(records, anchor_key="offset_abs", n=10),
                    "worst_pre": worst_rows(records, anchor_key="pre_abs", n=10),
                }
            )
    return {
        "run_id": str(manifest["run_id"]),
        "reference_version": reference_version,
        "compared_versions": compared_versions,
        "voicebank_results": results,
    }


def render_markdown(diagnosis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# OTO Bias Diagnostic — {diagnosis['run_id']}")
    lines.append("")
    lines.append(f"생성: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M KST')}")
    lines.append(f"Reference version: `{diagnosis['reference_version']}`")
    lines.append(f"Compared versions: {', '.join(f'`{v}`' for v in diagnosis['compared_versions'])}")
    lines.append("")
    lines.append("**Delta 부호 규약**: positive = 비교 버전이 reference보다 *늦음/큼*. negative = *빠름/작음*.")
    lines.append("")

    # -- Summary table -------------------------------------------------------
    lines.append("## 요약")
    lines.append("")
    header = ["voicebank", "version", "matched", "offset MAE", "pre MAE", "ovl MAE", "cons MAE", "cutoff MAE"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in diagnosis["voicebank_results"]:
        if r.get("status") != "ok":
            lines.append(
                f"| {r['voicebank_id']} | {r.get('version_id') or '-'} | "
                f"_(status={r.get('status')})_ | - | - | - | - | - |"
            )
            continue
        all_stats = r["stats_by_role"].get("ALL", {})
        cells = [
            r["voicebank_id"],
            r["version_id"],
            f"{r['matched']}/{r['compared_total']}",
            *(f"{all_stats.get(k, {}).get('mae', 0.0):.1f}" for k in ANCHOR_KEYS),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # -- Per-(voicebank, version) detail ------------------------------------
    for r in diagnosis["voicebank_results"]:
        if r.get("status") != "ok":
            continue
        lines.extend(_render_voicebank_version_block(r))

    return "\n".join(lines).rstrip() + "\n"


def _render_voicebank_version_block(r: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {r['voicebank_id']} / {r['version_id']} (vs reference, lang={r.get('language', '?')})")
    lines.append("")
    lines.append(f"matched={r['matched']}/{r['compared_total']} (unmatched in reference={r['unmatched_in_reference']})")
    lines.append("")

    # Per-role mean / median table for each anchor
    for anchor in ANCHOR_KEYS:
        lines.append(f"### {ANCHOR_LABELS[anchor]} — per-role delta")
        lines.append("")
        header = ["role", "n", "mean (ms)", "median (ms)", "MAE", "% late >5ms", "% early <-5ms", "% within ±5ms", "±10ms", "±20ms"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        roles = sorted([role for role in r["stats_by_role"] if role != "ALL"])
        roles = roles + ["ALL"]
        for role in roles:
            stats = r["stats_by_role"][role].get(anchor, {})
            if stats.get("count", 0) == 0:
                continue
            row = [
                role,
                str(int(stats["count"])),
                f"{stats['mean']:+.1f}",
                f"{stats['median']:+.1f}",
                f"{stats['mae']:.1f}",
                f"{stats['pct_late_5ms']:.0f}%",
                f"{stats['pct_early_5ms']:.0f}%",
                f"{stats['pct_within_5ms']:.0f}%",
                f"{stats['pct_within_10ms']:.0f}%",
                f"{stats['pct_within_20ms']:.0f}%",
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Worst-10 by offset and preutterance for inspection
    if r.get("worst_offset"):
        lines.append("### Worst 10 by |offset Δ|")
        lines.append("")
        lines.append("| wav | alias | role | offset Δ | pre Δ | overlap Δ |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in r["worst_offset"]:
            lines.append(
                f"| `{row['wav']}` | `{row['alias']}` | {row['role']} | "
                f"{row['offset_abs']:+.1f} | {row['pre_abs']:+.1f} | {row['ovl_abs']:+.1f} |"
            )
        lines.append("")
    if r.get("worst_pre"):
        lines.append("### Worst 10 by |preutterance(abs) Δ|")
        lines.append("")
        lines.append("| wav | alias | role | pre Δ | offset Δ | consonant Δ |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in r["worst_pre"]:
            lines.append(
                f"| `{row['wav']}` | `{row['alias']}` | {row['role']} | "
                f"{row['pre_abs']:+.1f} | {row['offset_abs']:+.1f} | {row['cons_abs']:+.1f} |"
            )
        lines.append("")

    # Interpretation hint
    all_stats = r["stats_by_role"].get("ALL", {})
    hints: list[str] = []
    off_mean = all_stats.get("offset_abs", {}).get("mean", 0.0)
    if abs(off_mean) >= 5.0:
        direction = "late (autooto begins after manual)" if off_mean > 0 else "early (autooto begins before manual)"
        hints.append(f"- offset bias **{off_mean:+.1f} ms** → systematic ({direction}). Phase A (zero-crossing snap) or residual offset target should reduce this.")
    pre_mean = all_stats.get("pre_abs", {}).get("mean", 0.0)
    if abs(pre_mean) >= 5.0:
        direction = (
            "vowel onset late → renderer treats more region as consonant → consonant gets stretched/dragged"
            if pre_mean > 0
            else "vowel onset early → consonant gets clipped"
        )
        hints.append(f"- preutterance(abs) bias **{pre_mean:+.1f} ms** → {direction}.")
    if hints:
        lines.append("### 자동 해석")
        lines.append("")
        lines.extend(hints)
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose per-role OTO parameter bias against a reference version.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-dir", default="", help="Run directory (default: ml/eval/ab_listening/runs/<run_id>).")
    ap.add_argument("--reference-version", default="manual", help="Reference version_id (default: 'manual').")
    ap.add_argument(
        "--compared-versions",
        default="",
        help="Comma-separated version_ids to compare. Default: all autooto-kind versions in manifest.",
    )
    ap.add_argument("--out", default="", help="Output markdown path (default: reports/<run_id>.bias.md).")
    ap.add_argument("--json-out", default="", help="Optional JSON output path.")
    args = ap.parse_args()

    manifest = _read_manifest(os.path.abspath(args.manifest))
    run_id = str(manifest["run_id"])
    run_dir = os.path.abspath(args.run_dir or os.path.join("ml", "eval", "ab_listening", "runs", run_id))

    compared = [t.strip() for t in str(args.compared_versions or "").split(",") if t.strip()]
    if not compared:
        compared = [
            str(v.get("id"))
            for v in manifest.get("versions", [])
            if str(v.get("kind", "")).strip().lower() == "autooto"
        ]
    if not compared:
        raise SystemExit("no compared versions resolved (manifest has no autooto-kind versions, and --compared-versions empty)")

    diagnosis = diagnose(
        manifest=manifest,
        run_dir=run_dir,
        reference_version=str(args.reference_version),
        compared_versions=compared,
    )

    out_md = os.path.abspath(args.out or os.path.join("ml", "eval", "ab_listening", "reports", f"{run_id}.bias.md"))
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(diagnosis))
    print(f"[bias] wrote markdown {out_md}")

    if args.json_out:
        json_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(diagnosis, handle, ensure_ascii=False, indent=2)
        print(f"[bias] wrote JSON {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
