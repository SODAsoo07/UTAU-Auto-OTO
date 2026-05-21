"""Aggregate a filled A/B listening label CSV into a markdown report.

Input CSV schema (produced by build_ab_listening_set.py):
    case_id, voicebank_id, language, wav_name, alias, version_id,
    click, consonant_clip, vc_misalign, overlap_unnatural, cutoff_overreach,
    notes

Defect columns are 0 (or blank) for "no audible defect" and 1 for "defect".
Free-text in defect columns is treated as 1 (defect noted). The aggregator
emits a markdown report covering:

- Overall defect rate per version (per-category + "any defect")
- KO/JA breakdown + gap
- Per-voicebank breakdown
- Delta vs a reference version, if specified
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
from collections import defaultdict
from typing import Any, Iterable


DEFECT_COLUMNS: tuple[str, ...] = (
    "click",
    "consonant_clip",
    "vc_misalign",
    "overlap_unnatural",
    "cutoff_overreach",
)

DEFECT_LABEL_KO: dict[str, str] = {
    "click": "클릭",
    "consonant_clip": "자음잘림",
    "vc_misalign": "VC어긋남",
    "overlap_unnatural": "overlap",
    "cutoff_overreach": "cutoff과욕",
}


def _normalize_defect(value: object) -> int:
    text = str(value if value is not None else "").strip().lower()
    if not text or text in {"0", "false", "no", "n"}:
        return 0
    return 1


def read_label_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        raise SystemExit(f"labels CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in ("version_id", "voicebank_id") if col not in (reader.fieldnames or [])]
        missing.extend(col for col in DEFECT_COLUMNS if col not in (reader.fieldnames or []))
        if missing:
            raise SystemExit(f"labels CSV missing required columns: {missing}")
        for raw in reader:
            row = {key: (value or "").strip() if isinstance(value, str) else value for key, value in raw.items()}
            rows.append(row)
    return rows


def aggregate(
    rows: Iterable[dict[str, Any]],
    *,
    reference_version: str | None = None,
) -> dict[str, Any]:
    rows = list(rows)
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ver = str(row.get("version_id", "") or "").strip()
        if not ver:
            continue
        by_version[ver].append(row)

    overall: dict[str, dict[str, float]] = {}
    per_lang: dict[str, dict[str, dict[str, float]]] = {}
    per_vb: dict[str, dict[str, dict[str, float]]] = {}

    for ver, vrows in by_version.items():
        overall[ver] = _compute_stats(vrows)
        # per language
        per_lang.setdefault(ver, {})
        by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in vrows:
            lang = str(row.get("language", "") or "").strip().lower() or "unknown"
            by_lang[lang].append(row)
        for lang, lrows in by_lang.items():
            per_lang[ver][lang] = _compute_stats(lrows)
        # per voicebank
        per_vb.setdefault(ver, {})
        by_vb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in vrows:
            vb_id = str(row.get("voicebank_id", "") or "").strip() or "unknown"
            by_vb[vb_id].append(row)
        for vb_id, vbrows in by_vb.items():
            per_vb[ver][vb_id] = _compute_stats(vbrows)

    deltas: dict[str, dict[str, float]] = {}
    if reference_version and reference_version in overall:
        ref = overall[reference_version]
        for ver, stats in overall.items():
            if ver == reference_version:
                continue
            deltas[ver] = {
                key: round(float(stats[key]) - float(ref[key]), 4)
                for key in stats
                if isinstance(stats.get(key), (int, float))
            }

    return {
        "overall": overall,
        "per_language": per_lang,
        "per_voicebank": per_vb,
        "deltas_vs_reference": deltas,
        "reference_version": reference_version,
        "row_count": len(rows),
        "version_count": len(by_version),
    }


def _compute_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return per-category defect rate, any-defect rate, and counts."""
    n = len(rows)
    if n == 0:
        out = {col: 0.0 for col in DEFECT_COLUMNS}
        out.update({"any_defect": 0.0, "count": 0})
        return out
    counts = {col: 0 for col in DEFECT_COLUMNS}
    any_defect = 0
    for row in rows:
        any_flag = False
        for col in DEFECT_COLUMNS:
            v = _normalize_defect(row.get(col))
            counts[col] += v
            if v:
                any_flag = True
        if any_flag:
            any_defect += 1
    stats: dict[str, float] = {col: round(counts[col] / n, 4) for col in DEFECT_COLUMNS}
    stats["any_defect"] = round(any_defect / n, 4)
    stats["count"] = n
    return stats


def render_markdown(
    *,
    run_id: str,
    aggregated: dict[str, Any],
    labels_path: str,
) -> str:
    overall = aggregated["overall"]
    per_lang = aggregated["per_language"]
    per_vb = aggregated["per_voicebank"]
    deltas = aggregated.get("deltas_vs_reference") or {}
    reference_version = aggregated.get("reference_version")

    lines: list[str] = []
    lines.append(f"# A/B Listening Report — {run_id}")
    lines.append("")
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M KST")
    lines.append(f"생성: {now}")
    lines.append(f"라벨 파일: `{labels_path}`")
    lines.append(f"전체 행 수: {aggregated['row_count']} | 버전 수: {aggregated['version_count']}")
    lines.append("")

    # --- Overall table ---
    lines.append("## 전체 결함 비율")
    lines.append("")
    header = ["버전"] + [DEFECT_LABEL_KO[col] for col in DEFECT_COLUMNS] + ["총합(any)", "케이스 수"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for ver in sorted(overall.keys()):
        stats = overall[ver]
        cells = [ver] + [
            _fmt_pct(float(stats.get(col, 0.0))) for col in DEFECT_COLUMNS
        ] + [_fmt_pct(float(stats.get("any_defect", 0.0))), str(int(stats.get("count", 0)))]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # --- Language breakdown (KO / JA gap focus) ---
    lines.append("## 언어별 결함 비율")
    lines.append("")
    langs_seen: list[str] = []
    for ver_map in per_lang.values():
        for lang in ver_map.keys():
            if lang not in langs_seen:
                langs_seen.append(lang)
    langs_sorted = sorted(langs_seen)
    header = ["버전"] + [f"{lang} any" for lang in langs_sorted]
    if "korean" in langs_sorted and "japanese" in langs_sorted:
        header.append("KO-JA pp 격차")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for ver in sorted(per_lang.keys()):
        cells = [ver]
        ko = float(per_lang[ver].get("korean", {}).get("any_defect", 0.0)) if "korean" in per_lang[ver] else None
        ja = float(per_lang[ver].get("japanese", {}).get("any_defect", 0.0)) if "japanese" in per_lang[ver] else None
        for lang in langs_sorted:
            stats = per_lang[ver].get(lang)
            if stats is None:
                cells.append("-")
            else:
                cells.append(_fmt_pct(float(stats.get("any_defect", 0.0))))
        if "korean" in langs_sorted and "japanese" in langs_sorted:
            if ko is None or ja is None:
                cells.append("-")
            else:
                cells.append(f"{(ko - ja) * 100.0:+.1f}pp")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # --- Per voicebank breakdown ---
    lines.append("## 보이스뱅크별 결함 비율 (any defect)")
    lines.append("")
    vbs_seen: list[str] = []
    for ver_map in per_vb.values():
        for vb_id in ver_map.keys():
            if vb_id not in vbs_seen:
                vbs_seen.append(vb_id)
    vbs_sorted = sorted(vbs_seen)
    header = ["버전"] + vbs_sorted
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for ver in sorted(per_vb.keys()):
        cells = [ver]
        for vb_id in vbs_sorted:
            stats = per_vb[ver].get(vb_id)
            if stats is None:
                cells.append("-")
            else:
                cells.append(_fmt_pct(float(stats.get("any_defect", 0.0))))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # --- Deltas vs reference ---
    if reference_version and deltas:
        lines.append(f"## 기준 버전 대비 변화 (reference = `{reference_version}`)")
        lines.append("")
        header = ["버전"] + [DEFECT_LABEL_KO[col] for col in DEFECT_COLUMNS] + ["총합(any)"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for ver in sorted(deltas.keys()):
            d = deltas[ver]
            cells = [ver] + [
                _fmt_pp(float(d.get(col, 0.0))) for col in DEFECT_COLUMNS
            ] + [_fmt_pp(float(d.get("any_defect", 0.0)))]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "- 음수(파선)일수록 개선. `+1.0pp` 이상의 카테고리별 악화는 회귀로 간주하고 후속 조치 권장."
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _fmt_pct(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _fmt_pp(value: float) -> str:
    pp = float(value) * 100.0
    return f"{pp:+.1f}pp"


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate A/B listening labels into a markdown report.")
    ap.add_argument("--labels", required=True, help="Path to the filled labeling CSV.")
    ap.add_argument("--out", required=True, help="Path to write the markdown report.")
    ap.add_argument("--reference-version", default="", help="Optional reference version_id for delta table.")
    ap.add_argument("--run-id", default="", help="Override run_id label (default: derived from labels filename).")
    ap.add_argument("--json-out", default="", help="Optional path to also write the aggregated JSON.")
    args = ap.parse_args()

    labels_path = os.path.abspath(args.labels)
    rows = read_label_rows(labels_path)
    aggregated = aggregate(rows, reference_version=(args.reference_version or None))

    run_id = args.run_id or os.path.splitext(os.path.basename(labels_path))[0]
    markdown = render_markdown(run_id=run_id, aggregated=aggregated, labels_path=labels_path)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"[report] wrote {out_path}")

    if args.json_out:
        json_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(aggregated, handle, ensure_ascii=False, indent=2)
        print(f"[json] wrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
