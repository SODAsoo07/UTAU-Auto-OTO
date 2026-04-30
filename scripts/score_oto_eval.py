from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.oto_file_utils import build_occurrence_map, read_oto_rows_for_profile
from core.oto_normalization import normalize_wav_key


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text or "").strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "eval"


def _norm_alias(alias: str, strip_suffix: str = "") -> str:
    text = " ".join(str(alias or "").strip().split()).lower()
    suffix = str(strip_suffix or "").strip().lower()
    if suffix:
        suffix = suffix if suffix.startswith("_") else f"_{suffix}"
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return text


def _classify_alias(language: str, alias: str) -> str:
    lang = str(language or "").strip().lower()
    try:
        if lang == "japanese":
            from core.ja_oto_mapping import classify_ja_alias

            return str(classify_ja_alias(alias) or "unknown")
        from core.kr_oto_rules import classify_alias

        return str(classify_alias(alias) or "unknown")
    except Exception:
        return "unknown"


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(statistics.median(vals))


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _p95(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = int(math.ceil(0.95 * len(vals))) - 1
    return float(vals[max(0, min(idx, len(vals) - 1))])


def _abs_fields(row: Dict[str, object]) -> Dict[str, float]:
    offset = float(row.get("offset", 0.0) or 0.0)
    cons = float(row.get("cons", 0.0) or 0.0)
    cutoff = float(row.get("cutoff", 0.0) or 0.0)
    pre = float(row.get("pre", 0.0) or 0.0)
    ovl = float(row.get("ovl", 0.0) or 0.0)
    return {
        "offset": offset,
        "cons": cons,
        "cutoff": cutoff,
        "pre": pre,
        "ovl": ovl,
        "pre_abs": offset + pre,
        "cons_abs": offset + cons,
        "cutoff_abs": offset + abs(cutoff),
        "ovl_abs": offset + ovl,
    }


def _is_zeroish(row: Dict[str, object]) -> bool:
    vals = _abs_fields(row)
    return all(abs(float(vals[k])) <= 1e-6 for k in ("offset", "cons", "cutoff", "pre", "ovl"))


def _load_rows(path: str, *, strip_suffix: str = "") -> List[Dict[str, object]]:
    return read_oto_rows_for_profile(
        path,
        wav_normalizer=lambda name: normalize_wav_key(os.path.basename(str(name or ""))),
        alias_normalizer=lambda alias: _norm_alias(alias, strip_suffix),
    )


def _case_from_manifest_row(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": row.get("id") or row.get("name") or "",
        "language": row.get("language", ""),
        "format": row.get("format", ""),
        "manual_oto": row.get("manual_oto", ""),
        "candidate_oto": row.get("candidate_oto", ""),
    }


def _discover_cases_from_dataset(dataset_root: str, candidate_name: str) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    root = os.path.abspath(dataset_root)
    for base, _dirs, files in os.walk(root):
        if "oto.ini" not in {str(x) for x in files}:
            continue
        candidate = os.path.join(base, candidate_name)
        rel = os.path.relpath(base, root)
        parts = rel.split(os.sep)
        cases.append(
            {
                "id": _safe_name(rel),
                "language": parts[0].lower() if len(parts) >= 1 else "",
                "format": parts[1].lower() if len(parts) >= 2 else "",
                "manual_oto": os.path.join(base, "oto.ini"),
                "candidate_oto": candidate,
            }
        )
    return cases


def _summarize_deltas(rows: List[Dict[str, object]]) -> Dict[str, float]:
    field_names = ("offset", "pre_abs", "cons_abs", "cutoff_abs", "ovl_abs", "pre", "cons", "cutoff", "ovl")
    out: Dict[str, float] = {"matched_rows": float(len(rows))}
    for field in field_names:
        vals = [abs(float(row.get(f"{field}_err_ms", 0.0) or 0.0)) for row in rows]
        out[f"{field}_mae_ms"] = _mean(vals)
        out[f"{field}_median_ms"] = _median(vals)
        out[f"{field}_p95_ms"] = _p95(vals)
    pre_errs = [abs(float(row.get("pre_abs_err_ms", 0.0) or 0.0)) for row in rows]
    cutoff_errs = [abs(float(row.get("cutoff_abs_err_ms", 0.0) or 0.0)) for row in rows]
    zeroish = [1.0 if row.get("candidate_zeroish") else 0.0 for row in rows]
    out["pre_abs_gt_25ms_rate"] = _mean(1.0 if v > 25.0 else 0.0 for v in pre_errs)
    out["pre_abs_gt_50ms_rate"] = _mean(1.0 if v > 50.0 else 0.0 for v in pre_errs)
    out["cutoff_abs_gt_75ms_rate"] = _mean(1.0 if v > 75.0 else 0.0 for v in cutoff_errs)
    out["candidate_zeroish_rate"] = _mean(zeroish)
    return out


def _compare_case(case: Dict[str, object], *, strip_suffix: str = "") -> tuple[Dict[str, object], List[Dict[str, object]]]:
    manual_path = os.path.abspath(str(case.get("manual_oto", "") or ""))
    candidate_path = os.path.abspath(str(case.get("candidate_oto", "") or ""))
    language = str(case.get("language", "") or "")
    fmt = str(case.get("format", "") or "")
    case_id = str(case.get("id", "") or os.path.basename(os.path.dirname(manual_path)))

    manual_rows = _load_rows(manual_path)
    candidate_rows = _load_rows(candidate_path, strip_suffix=strip_suffix) if os.path.isfile(candidate_path) else []
    manual_map = build_occurrence_map(manual_rows)
    candidate_map = build_occurrence_map(candidate_rows)

    detail_rows: List[Dict[str, object]] = []
    matched = 0
    for key, manual in manual_map.items():
        candidate = candidate_map.get(key)
        if not candidate:
            continue
        matched += 1
        m_abs = _abs_fields(manual)
        c_abs = _abs_fields(candidate)
        alias = str(manual.get("alias", ""))
        detail = {
            "case_id": case_id,
            "language": language,
            "format": fmt,
            "alias_type": _classify_alias(language, alias),
            "wav": manual.get("wav", ""),
            "alias": alias,
            "candidate_zeroish": _is_zeroish(candidate),
        }
        for field in ("offset", "pre_abs", "cons_abs", "cutoff_abs", "ovl_abs", "pre", "cons", "cutoff", "ovl"):
            detail[f"{field}_manual"] = float(m_abs[field])
            detail[f"{field}_candidate"] = float(c_abs[field])
            detail[f"{field}_err_ms"] = float(c_abs[field] - m_abs[field])
        detail_rows.append(detail)

    missing = max(0, len(manual_map) - matched)
    extra = max(0, len(candidate_map) - matched)
    case_summary: Dict[str, object] = {
        "case_id": case_id,
        "language": language,
        "format": fmt,
        "manual_oto": manual_path,
        "candidate_oto": candidate_path,
        "candidate_exists": os.path.isfile(candidate_path),
        "manual_rows": len(manual_map),
        "candidate_rows": len(candidate_map),
        "matched_rows": matched,
        "missing_rows": missing,
        "extra_rows": extra,
        "match_rate": float(matched / max(len(manual_map), 1)),
    }
    case_summary.update(_summarize_deltas(detail_rows))
    return case_summary, detail_rows


def _group_summary(rows: List[Dict[str, object]], key_fn: Callable[[Dict[str, object]], str]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return {key: _summarize_deltas(vals) for key, vals in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score generated OTO against manual dataset_staged oto.ini files.")
    parser.add_argument("--manifest", default="", help="Manifest from build_dataset_eval_cases.py")
    parser.add_argument("--dataset-root", default=os.path.join(ROOT_DIR, "dataset_staged"))
    parser.add_argument("--candidate-name", default="", help="Candidate filename inside each manual OTO folder.")
    parser.add_argument("--strip-alias-suffix", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        cases = [_case_from_manifest_row(row) for row in manifest.get("cases", []) if isinstance(row, dict)]
        default_out_dir = os.path.join(os.path.dirname(os.path.abspath(args.manifest)), "scores")
    elif args.candidate_name:
        cases = _discover_cases_from_dataset(args.dataset_root, args.candidate_name)
        default_out_dir = os.path.join(ROOT_DIR, "eval_runs", "dataset_oto_eval", "scores")
    else:
        raise SystemExit("provide --manifest or --candidate-name")

    if args.limit:
        cases = cases[: int(args.limit)]
    if not cases:
        raise SystemExit("no cases to score")

    out_dir = os.path.abspath(args.out_dir or default_out_dir)
    os.makedirs(out_dir, exist_ok=True)

    case_summaries: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    for case in cases:
        summary, details = _compare_case(case, strip_suffix=args.strip_alias_suffix)
        case_summaries.append(summary)
        detail_rows.extend(details)

    global_summary = _summarize_deltas(detail_rows)
    global_summary.update(
        {
            "case_count": len(case_summaries),
            "candidate_exists_count": sum(1 for row in case_summaries if row.get("candidate_exists")),
            "manual_rows": sum(int(row.get("manual_rows", 0) or 0) for row in case_summaries),
            "candidate_rows": sum(int(row.get("candidate_rows", 0) or 0) for row in case_summaries),
            "matched_rows": sum(int(row.get("matched_rows", 0) or 0) for row in case_summaries),
        }
    )
    global_summary["match_rate"] = float(
        global_summary["matched_rows"] / max(float(global_summary["manual_rows"]), 1.0)
    )

    payload = {
        "summary": global_summary,
        "by_language_format": _group_summary(
            detail_rows,
            lambda row: f"{row.get('language', '')}/{row.get('format', '')}",
        ),
        "by_alias_type": _group_summary(detail_rows, lambda row: str(row.get("alias_type", "unknown") or "unknown")),
        "cases": case_summaries,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    cases_path = os.path.join(out_dir, "cases.csv")
    details_path = os.path.join(out_dir, "details.csv")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if case_summaries:
        with open(cases_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(case_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(case_summaries)
    if detail_rows:
        with open(details_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    print(f"[DONE] cases={len(case_summaries)} matched_rows={int(global_summary['matched_rows'])}")
    print(f"[DONE] match_rate={float(global_summary['match_rate']):.4f}")
    print(f"[DONE] pre_abs_median_ms={float(global_summary.get('pre_abs_median_ms', 0.0)):.3f}")
    print(f"[DONE] cutoff_abs_median_ms={float(global_summary.get('cutoff_abs_median_ms', 0.0)):.3f}")
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] cases_csv={cases_path}")
    if detail_rows:
        print(f"[DONE] details_csv={details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
