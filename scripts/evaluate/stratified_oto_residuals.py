"""Build role-aware residual buckets for gold-vs-generated OTO comparisons.

This is an evaluation/training aid.  It intentionally lives outside runtime code:
use it to decide which OTO role families need new boundary targets or residual
models, not to change generated timings directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dev.diff_oto import build_report

DEFAULT_DIMENSIONS = (
    "alias_family",
    "vc_left_token",
    "vc_right_token",
    "vc_right_class",
    "vc_pair",
    "alias_family_x_vc_right_class",
)
DEFAULT_PARAMS = ("offset", "preutterance", "overlap", "consonant", "cutoff", "cutoff_end_abs")
VOWEL_TOKENS = {"a", "i", "u", "e", "o"}
NASAL_TOKENS = {"m", "n", "ng", "ny", "my"}
LIQUID_TOKENS = {"r", "l", "ry"}
GLIDE_TOKENS = {"w", "y", "j"}
YOON_TOKENS = {"ky", "gy", "ny", "hy", "by", "py", "my", "ry", "sh", "ch", "j", "ty", "dy"}
_ZERO_TIMING_EPS_MS = 1e-3
_KOREAN_CATASTROPHIC_RESIDUAL_SKIP_MS = 1200.0


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    vals = [float(value) for value in values]
    if not vals:
        return {}
    abs_vals = sorted(abs(value) for value in vals)
    return {
        "n": len(vals),
        "mean_delta_ms": round(sum(vals) / len(vals), 1),
        "abs_median_ms": round(abs_vals[len(abs_vals) // 2], 1),
        "abs_p90_ms": round(abs_vals[int(0.9 * (len(abs_vals) - 1))], 1),
        "MAE_ms": round(sum(abs_vals) / len(abs_vals), 1),
        "over_50ms": sum(1 for value in abs_vals if value >= 50.0),
        "over_100ms": sum(1 for value in abs_vals if value >= 100.0),
        "over_250ms": sum(1 for value in abs_vals if value >= 250.0),
        "early_count": sum(1 for value in vals if value < -1e-6),
        "late_count": sum(1 for value in vals if value > 1e-6),
    }


def _strip_token_suffix(token: str) -> str:
    value = str(token or "").strip()
    if not value:
        return ""
    if value.startswith("-"):
        return "-"
    stripped = re.sub(r"\d+$", "", value)
    return stripped or value


def _vc_tokens(alias: str) -> tuple[str, str] | None:
    parts = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(parts) < 2:
        return None
    return _strip_token_suffix(parts[0]).lower(), _strip_token_suffix(parts[-1]).lower()


def _token_class(token: str) -> str:
    value = _strip_token_suffix(token).lower()
    if not value:
        return "blank"
    if value == "-":
        return "terminal"
    if value in VOWEL_TOKENS:
        return "vowel"
    if value in NASAL_TOKENS:
        return "nasal"
    if value in LIQUID_TOKENS:
        return "liquid"
    if value in GLIDE_TOKENS:
        return "glide"
    if value in YOON_TOKENS or value.endswith("y"):
        return "yoon"
    return "obstruent"


def _is_ascii_vc_right_token(token: str, token_class: str) -> bool:
    value = _strip_token_suffix(token).lower()
    if not value or value == "-":
        return True
    if token_class == "vowel":
        return False
    return re.fullmatch(r"[a-z][a-z0-9_#b-]*", value) is not None


def _row_tags(row: dict) -> dict[str, str]:
    family = str(row.get("alias_family") or "")
    alias = str(row.get("alias") or "")
    left = ""
    right = ""
    right_class = ""
    pair = ""
    tokens = _vc_tokens(alias)
    if tokens is not None:
        candidate_left, candidate_right = tokens
        candidate_right_class = _token_class(candidate_right)
        if family in {"vc", "terminal_v_dash"} or _is_ascii_vc_right_token(
            candidate_right,
            candidate_right_class,
        ):
            left, right = candidate_left, candidate_right
            right_class = candidate_right_class
            pair = f"{left}->{right}"
    return {
        "alias_family": family,
        "vc_left_token": left,
        "vc_right_token": right,
        "vc_right_class": right_class,
        "vc_pair": pair,
        "alias_family_x_vc_right_class": f"{family}:{right_class}" if right_class else family,
    }


def row_role_tags(row: dict) -> dict[str, str]:
    return _row_tags(row)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _near_zero(value: object, *, eps: float = _ZERO_TIMING_EPS_MS) -> bool:
    return abs(_float(value)) <= float(eps)


def _manual_timing_has_zero_anchor_profile(gold: dict) -> bool:
    return (
        _near_zero(gold.get("offset"))
        and _near_zero(gold.get("consonant"))
        and _near_zero(gold.get("preutterance"))
        and _near_zero(gold.get("overlap"))
    )


def _is_korean(language: str) -> bool:
    lang = str(language or "").strip().lower()
    return bool(lang in {"korean", "ko", "kr"} or lang.startswith("ko") or lang.startswith("kr"))


def _row_quality(row: dict, *, language: str = "") -> dict[str, object]:
    gold = dict(row.get("gold") or {})
    delta = dict(row.get("delta") or {})
    reason = ""
    trusted = True
    if _manual_timing_has_zero_anchor_profile(gold):
        trusted = False
        reason = "manual_zero_anchor_profile"
    elif _is_korean(language) and abs(_float(delta.get("offset"))) >= _KOREAN_CATASTROPHIC_RESIDUAL_SKIP_MS:
        trusted = False
        reason = "korean_catastrophic_offset_residual"
    return {
        "trusted": bool(trusted),
        "reason": reason,
    }


def _aggregate_by_alias_family(rows: list[dict], params: list[str]) -> dict:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        family = str((row.get("tags") or {}).get("alias_family") or "")
        if not family:
            continue
        delta = dict(row.get("delta") or {})
        for param in params:
            if param in delta:
                buckets[family][param].append(float(delta[param]))
    return {
        family: {
            "n": max((len(values) for values in values_by_param.values()), default=0),
            "by_param": {param: _stats(values) for param, values in sorted(values_by_param.items())},
        }
        for family, values_by_param in sorted(buckets.items())
    }


def _quality_summary(rows: list[dict], params: list[str]) -> dict:
    trusted_rows = [row for row in rows if (row.get("quality") or {}).get("trusted")]
    excluded_rows = [row for row in rows if not (row.get("quality") or {}).get("trusted")]
    by_reason: dict[str, int] = defaultdict(int)
    for row in excluded_rows:
        reason = str((row.get("quality") or {}).get("reason") or "unknown")
        by_reason[reason] += 1
    trusted_by_param = {
        param: _stats(float(row["delta"][param]) for row in trusted_rows if param in row.get("delta", {}))
        for param in params
    }
    return {
        "rows_total": len(rows),
        "trusted_rows": len(trusted_rows),
        "excluded_rows": len(excluded_rows),
        "by_exclusion_reason": dict(sorted(by_reason.items())),
        "trusted_by_param": trusted_by_param,
        "trusted_by_alias_family": _aggregate_by_alias_family(trusted_rows, params),
    }


def build_stratified_report(
    *,
    gold_path: str,
    pred_path: str,
    wav_dir: str = "",
    cutoff_end_mode: str = "compatible",
    dimensions: Iterable[str] = DEFAULT_DIMENSIONS,
    min_bucket_n: int = 1,
    top_n: int = 40,
    language: str = "",
    trusted_only: bool = False,
) -> dict:
    base_report, per_row = build_report(
        gold_path=gold_path,
        pred_path=pred_path,
        wav_dir=wav_dir,
        cutoff_end_mode=cutoff_end_mode,
    )
    params = [param for param in DEFAULT_PARAMS if param in base_report["by_param"]]
    bucket_values: dict[str, dict[str, dict[str, list[float]]]] = {
        dimension: defaultdict(lambda: defaultdict(list)) for dimension in dimensions
    }
    tagged_rows = []
    for row in per_row:
        tags = _row_tags(row)
        quality = _row_quality(row, language=language)
        tagged = {
            "wav": row.get("wav", ""),
            "alias": row.get("alias", ""),
            "occurrence": row.get("occurrence", 0),
            "tags": tags,
            "delta": row.get("delta", {}),
            "quality": quality,
        }
        tagged_rows.append(tagged)
        if bool(trusted_only) and not bool(quality.get("trusted")):
            continue
        for dimension in dimensions:
            bucket = tags.get(dimension, "")
            if not bucket:
                continue
            for param in params:
                if param in row["delta"]:
                    bucket_values[dimension][bucket][param].append(float(row["delta"][param]))

    by_dimension = {}
    flat_buckets = []
    for dimension, buckets in bucket_values.items():
        by_dimension[dimension] = {}
        for bucket, param_values in sorted(buckets.items()):
            n = max((len(values) for values in param_values.values()), default=0)
            if n < int(min_bucket_n):
                continue
            by_param = {param: _stats(values) for param, values in sorted(param_values.items())}
            payload = {"n": int(n), "by_param": by_param}
            by_dimension[dimension][bucket] = payload
            offset_stats = by_param.get("offset", {})
            flat_buckets.append(
                {
                    "dimension": dimension,
                    "bucket": bucket,
                    "n": int(n),
                    "offset_MAE_ms": offset_stats.get("MAE_ms", 0.0),
                    "offset_abs_p90_ms": offset_stats.get("abs_p90_ms", 0.0),
                    "offset_mean_delta_ms": offset_stats.get("mean_delta_ms", 0.0),
                }
            )

    worst_rows = {}
    for param in params:
        candidates = [
            {
                "wav": row["wav"],
                "alias": row["alias"],
                "occurrence": row["occurrence"],
                "delta_ms": round(float(row["delta"][param]), 3),
                "abs_delta_ms": round(abs(float(row["delta"][param])), 3),
                "tags": row["tags"],
            }
            for row in tagged_rows
            if param in row["delta"] and (not bool(trusted_only) or bool((row.get("quality") or {}).get("trusted")))
        ]
        worst_rows[param] = sorted(candidates, key=lambda item: item["abs_delta_ms"], reverse=True)[:top_n]

    return {
        "gold": os.path.abspath(gold_path),
        "pred": os.path.abspath(pred_path),
        "wav_dir": os.path.abspath(wav_dir) if wav_dir else "",
        "cutoff_end_mode": cutoff_end_mode,
        "language": str(language or ""),
        "trusted_only": bool(trusted_only),
        "base_summary": base_report,
        "quality_summary": _quality_summary(tagged_rows, params),
        "by_dimension": by_dimension,
        "priority_buckets": sorted(
            flat_buckets,
            key=lambda item: (float(item["offset_MAE_ms"]), float(item["offset_abs_p90_ms"]), int(item["n"])),
            reverse=True,
        )[:top_n],
        "worst_rows": worst_rows,
    }


def write_bucket_csv(report: dict, path: str) -> None:
    rows = []
    for dimension, buckets in report.get("by_dimension", {}).items():
        for bucket, payload in buckets.items():
            for param, stats in payload.get("by_param", {}).items():
                row = {
                    "dimension": dimension,
                    "bucket": bucket,
                    "param": param,
                    **stats,
                }
                rows.append(row)
    fieldnames = [
        "dimension",
        "bucket",
        "param",
        "n",
        "mean_delta_ms",
        "abs_median_ms",
        "abs_p90_ms",
        "MAE_ms",
        "over_50ms",
        "over_100ms",
        "over_250ms",
        "early_count",
        "late_count",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _print_text(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(str(text).encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Manual/gold oto.ini")
    parser.add_argument("--pred", required=True, help="Generated oto.ini")
    parser.add_argument("--wav-dir", default="", help="Directory containing wavs for cutoff end diagnostics")
    parser.add_argument(
        "--cutoff-end-mode",
        choices=("raw", "relative", "utau", "compatible"),
        default="compatible",
    )
    parser.add_argument("--min-bucket-n", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--language", default="", help="Optional language for trusted-row quality gates")
    parser.add_argument(
        "--trusted-only",
        action="store_true",
        help="Build buckets and worst rows from trusted rows only; base_summary remains unfiltered.",
    )
    parser.add_argument("--out", default="", help="JSON report path")
    parser.add_argument("--csv-out", default="", help="Optional CSV bucket table path")
    args = parser.parse_args()

    wav_dir = args.wav_dir or os.path.dirname(os.path.abspath(args.gold))
    report = build_stratified_report(
        gold_path=args.gold,
        pred_path=args.pred,
        wav_dir=wav_dir,
        cutoff_end_mode=args.cutoff_end_mode,
        min_bucket_n=args.min_bucket_n,
        top_n=args.top_n,
        language=args.language,
        trusted_only=bool(args.trusted_only),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    _print_text(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    if args.csv_out:
        write_bucket_csv(report, args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
