from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.format_type_utils import normalize_format_type, normalize_language_name
from core.runtime.runtime_encoding import bootstrap_utf8_runtime


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_blank_attach_rates(eval_summary: Dict[str, object]) -> Dict[str, float]:
    kpi = dict((eval_summary or {}).get("blank_attach_kpi") or {})
    scope_rows = _as_float(kpi.get("scope_rows"), 0.0)
    if scope_rows <= 0.0:
        return {}
    return {
        "overall": _as_float(kpi.get("pred_attach_rate"), 0.0),
        "head": _as_float(kpi.get("head_pred_attach_rate"), 0.0),
        "vv": _as_float(kpi.get("vv_pred_attach_rate"), 0.0),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    qq = min(max(float(q), 0.0), 1.0)
    pos = (len(sorted_values) - 1) * qq
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    mix = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * mix)


def _resolve_threshold(
    values: Sequence[float],
    *,
    quantile: float,
    margin: float,
    fallback: float,
) -> float:
    if not values:
        return float(fallback)
    tuned = _quantile(values, quantile) + float(margin)
    tuned = max(0.0, min(1.0, tuned))
    return float(tuned)


def _append_record(
    per_bucket: Dict[Tuple[str, str], Dict[str, List[float]]],
    global_bucket: Dict[str, List[float]],
    *,
    language: str,
    format_type: str,
    eval_summary: Dict[str, object],
) -> None:
    lang = normalize_language_name(language)
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower()
    if lang not in {"korean", "japanese"} or not fmt:
        return
    rates = _extract_blank_attach_rates(eval_summary)
    if not rates:
        return
    key = (lang, fmt)
    bucket = per_bucket.setdefault(key, {"overall": [], "head": [], "vv": []})
    for metric in ("overall", "head", "vv"):
        val = float(rates.get(metric, 0.0))
        bucket[metric].append(val)
        global_bucket[metric].append(val)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object: {path}")
    return payload


def _collect_from_matrix_summary(
    path: str,
    *,
    per_bucket: Dict[Tuple[str, str], Dict[str, List[float]]],
    global_bucket: Dict[str, List[float]],
) -> int:
    payload = _load_json(path)
    jobs = list(payload.get("jobs") or [])
    seen = 0
    for job in jobs:
        if not isinstance(job, dict):
            continue
        lang = str(job.get("language", "") or "").strip().lower()
        fmt = str(job.get("format_type", "") or "").strip().lower()
        eval_summary = job.get("eval_summary")
        if not isinstance(eval_summary, dict):
            eval_path = str(job.get("eval_report", "") or "").strip()
            if eval_path and os.path.isfile(eval_path):
                try:
                    eval_summary = _load_json(eval_path)
                except Exception:
                    eval_summary = None
        if not isinstance(eval_summary, dict):
            continue
        _append_record(
            per_bucket,
            global_bucket,
            language=lang,
            format_type=fmt,
            eval_summary=eval_summary,
        )
        seen += 1
    return int(seen)


def _collect_from_eval_report_paths(
    paths: Iterable[str],
    *,
    per_bucket: Dict[Tuple[str, str], Dict[str, List[float]]],
    global_bucket: Dict[str, List[float]],
) -> int:
    seen = 0
    for path in paths:
        report_path = os.path.abspath(str(path or "").strip())
        if not report_path or not os.path.isfile(report_path):
            continue
        norm = report_path.replace("\\", "/")
        parts = norm.split("/")
        language = ""
        format_type = ""
        for idx, token in enumerate(parts):
            lang = normalize_language_name(token)
            if lang in {"korean", "japanese"} and idx + 1 < len(parts):
                language = lang
                format_type = str(parts[idx + 1]).strip().lower()
                break
        if not language or not format_type:
            continue
        payload = _load_json(report_path)
        _append_record(
            per_bucket,
            global_bucket,
            language=language,
            format_type=format_type,
            eval_summary=payload,
        )
        seen += 1
    return int(seen)


def _collect_from_model_root(
    model_root: str,
    *,
    per_bucket: Dict[Tuple[str, str], Dict[str, List[float]]],
    global_bucket: Dict[str, List[float]],
) -> int:
    root = os.path.abspath(str(model_root or "").strip())
    if not root or not os.path.isdir(root):
        return 0
    seen = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        if "eval_summary.json" not in filenames:
            continue
        report_path = os.path.join(dirpath, "eval_summary.json")
        rel = os.path.relpath(report_path, root).replace("\\", "/")
        parts = [p for p in rel.split("/") if p]
        if len(parts) < 3:
            continue
        lang = ""
        fmt = ""
        for idx, token in enumerate(parts):
            maybe_lang = normalize_language_name(token)
            if maybe_lang not in {"korean", "japanese"}:
                continue
            if idx + 1 >= len(parts):
                continue
            maybe_fmt = normalize_format_type(maybe_lang, parts[idx + 1]) or str(parts[idx + 1]).strip().lower()
            if not maybe_fmt:
                continue
            lang = maybe_lang
            fmt = maybe_fmt
            break
        if lang not in {"korean", "japanese"} or not fmt:
            continue
        payload = _load_json(report_path)
        _append_record(
            per_bucket,
            global_bucket,
            language=lang,
            format_type=fmt,
            eval_summary=payload,
        )
        seen += 1
    return int(seen)


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Tune per-language/format blank-attach gate thresholds from eval reports.")
    ap.add_argument("--matrix-summary", action="append", default=[], help="train_matrix output summary JSON path (repeatable)")
    ap.add_argument("--eval-report", action="append", default=[], help="Direct eval_summary.json path (repeatable)")
    ap.add_argument("--model-root", default="", help="Model root to scan for */*/**/eval_summary.json")
    ap.add_argument("--out", default="", help="Output JSON path. Empty prints to stdout only.")
    ap.add_argument("--quantile-overall", type=float, default=0.85)
    ap.add_argument("--quantile-head", type=float, default=0.85)
    ap.add_argument("--quantile-vv", type=float, default=0.85)
    ap.add_argument("--margin-overall", type=float, default=0.01)
    ap.add_argument("--margin-head", type=float, default=0.01)
    ap.add_argument("--margin-vv", type=float, default=0.01)
    ap.add_argument("--default-overall", type=float, default=-1.0)
    ap.add_argument("--default-head", type=float, default=-1.0)
    ap.add_argument("--default-vv", type=float, default=-1.0)
    ap.add_argument(
        "--lock-defaults",
        action="store_true",
        help="Keep default thresholds fixed to --default-* values (do not auto-tune global defaults).",
    )
    ap.add_argument("--min-samples-per-key", type=int, default=2, help="Minimum eval reports needed to emit a key override")
    args = ap.parse_args()

    per_bucket: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    global_bucket: Dict[str, List[float]] = {"overall": [], "head": [], "vv": []}
    source_counts = {
        "matrix_summary": 0,
        "eval_report": 0,
        "model_root": 0,
    }

    for path in list(args.matrix_summary or []):
        source_counts["matrix_summary"] += _collect_from_matrix_summary(
            os.path.abspath(path),
            per_bucket=per_bucket,
            global_bucket=global_bucket,
        )
    source_counts["eval_report"] += _collect_from_eval_report_paths(
        list(args.eval_report or []),
        per_bucket=per_bucket,
        global_bucket=global_bucket,
    )
    if str(args.model_root or "").strip():
        source_counts["model_root"] += _collect_from_model_root(
            str(args.model_root),
            per_bucket=per_bucket,
            global_bucket=global_bucket,
        )

    if bool(args.lock_defaults):
        default_gate = {
            "max_blank_attach_rate": float(args.default_overall),
            "max_head_blank_attach_rate": float(args.default_head),
            "max_vv_blank_attach_rate": float(args.default_vv),
        }
    else:
        default_gate = {
            "max_blank_attach_rate": _resolve_threshold(
                global_bucket["overall"],
                quantile=float(args.quantile_overall),
                margin=float(args.margin_overall),
                fallback=float(args.default_overall),
            ),
            "max_head_blank_attach_rate": _resolve_threshold(
                global_bucket["head"],
                quantile=float(args.quantile_head),
                margin=float(args.margin_head),
                fallback=float(args.default_head),
            ),
            "max_vv_blank_attach_rate": _resolve_threshold(
                global_bucket["vv"],
                quantile=float(args.quantile_vv),
                margin=float(args.margin_vv),
                fallback=float(args.default_vv),
            ),
        }

    overrides: Dict[str, Dict[str, float]] = {}
    min_samples = max(1, int(args.min_samples_per_key))
    for (lang, fmt), bucket in sorted(per_bucket.items()):
        sample_count = len(bucket["overall"])
        if sample_count < min_samples:
            continue
        gate = {
            "max_blank_attach_rate": _resolve_threshold(
                bucket["overall"],
                quantile=float(args.quantile_overall),
                margin=float(args.margin_overall),
                fallback=float(default_gate["max_blank_attach_rate"]),
            ),
            "max_head_blank_attach_rate": _resolve_threshold(
                bucket["head"],
                quantile=float(args.quantile_head),
                margin=float(args.margin_head),
                fallback=float(default_gate["max_head_blank_attach_rate"]),
            ),
            "max_vv_blank_attach_rate": _resolve_threshold(
                bucket["vv"],
                quantile=float(args.quantile_vv),
                margin=float(args.margin_vv),
                fallback=float(default_gate["max_vv_blank_attach_rate"]),
            ),
            "samples": int(sample_count),
        }
        overrides[f"{lang}/{fmt}"] = gate

    out_payload: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "default": default_gate,
        "overrides": overrides,
        "stats": {
            "sources": source_counts,
            "global_samples": int(len(global_bucket["overall"])),
            "keys_total": int(len(per_bucket)),
            "keys_emitted": int(len(overrides)),
            "min_samples_per_key": int(min_samples),
        },
    }

    out_text = json.dumps(out_payload, ensure_ascii=False, indent=2)
    print(out_text)
    out_path = os.path.abspath(str(args.out or "").strip())
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text)


if __name__ == "__main__":
    main()
