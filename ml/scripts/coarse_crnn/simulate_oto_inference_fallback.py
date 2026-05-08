"""Simulate the inference-time low-confidence fallback trigger over an
evaluate_oto.py result JSON, with optional slice-AND mode (Plan B).

This is the verification step (B) for the v2 0-order Plan B pipeline. The
script does NOT regenerate OTO files — it only replays the trigger logic
of ``core/coarse_crnn/oto_predictor_generator._apply_low_confidence_fallback``
against the per-row ``confidence`` / ``predicted_error_ms`` columns the
evaluation JSON already contains. That gives a faithful fallback-rate
delta for each slice without running the model.

The trigger logic mirrored here is:

- ``trigger_conf = confidence < min_confidence`` (env: ``UTOA_OTO_CRNN_LOW_CONF_MIN_CONFIDENCE``, default 0.54)
- ``trigger_err  = predicted_error_ms > max_error_ms`` (env: ``UTOA_OTO_CRNN_LOW_CONF_MAX_PREDICTED_ERROR_MS``, default 95)
- OR mode  (legacy):  fire iff ``trigger_conf or trigger_err``
- AND mode (Plan B):  fire iff ``trigger_err or (trigger_conf and trigger_err)``
                      (i.e. confidence-only is suppressed)

The slice list controlling AND mode is read from the same env var that the
runtime uses (``UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES``). Pattern matches
``language/format`` with ``*`` wildcards. Default
``korean/cvc,japanese/cvvc`` is the Plan B recommendation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

from core.coarse_crnn.oto_predictor_generator import _is_multi_signal_required


def _confidence(row: dict) -> float:
    try:
        return float(row.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _predicted_error_ms(row: dict):
    raw = row.get("predicted_error_ms")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _row_is_correct(row: dict, threshold_ms: float) -> bool:
    errors = row.get("param_abs_errors_ms")
    if not isinstance(errors, dict):
        return False
    pre = errors.get("preutterance")
    try:
        pre = float(pre)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(pre):
        return False
    return pre <= float(threshold_ms)


def _heatmap_confidence(row: dict) -> float:
    components = row.get("confidence_components")
    if isinstance(components, dict):
        try:
            return float(components.get("heatmap", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _classify_row(
    row: dict,
    *,
    min_conf: float,
    max_err: float,
    heatmap_threshold: float = 0.20,
) -> dict:
    """Return per-row trigger flags for several modes.

    - ``conf``: confidence < min_conf
    - ``err``: predicted_error_ms > max_err
    - ``shift``: evaluation flag ``syllable_shift_flag``
    - ``voiced_miss``: evaluation flag ``fn_voiced_miss``
    - ``hard_fail``: evaluation flag ``hard_failure``
    - ``heatmap_low``: heatmap confidence component <= ``heatmap_threshold``

    Then composite triggers:

    - ``or``: conf or err  (legacy inference)
    - ``and``: err or (conf and err)  (legacy inference under AND mode)
    - ``multi_v1``: shift OR voiced_miss OR hard_fail OR heatmap_low
    - ``multi_v2``: (conf or err) AND (shift or voiced_miss)
                   — conf/err must be corroborated by an activity signal
    """
    conf = _confidence(row)
    err = _predicted_error_ms(row)
    flag_conf = conf < float(min_conf)
    flag_err = err is not None and err > float(max_err)
    flag_shift = bool(row.get("syllable_shift_flag", False))
    flag_voiced = bool(row.get("fn_voiced_miss", False))
    flag_hard = bool(row.get("hard_failure", False))
    flag_heatmap_low = _heatmap_confidence(row) <= float(heatmap_threshold)
    return {
        "conf": flag_conf,
        "err": flag_err,
        "shift": flag_shift,
        "voiced_miss": flag_voiced,
        "hard_fail": flag_hard,
        "heatmap_low": flag_heatmap_low,
        "or": bool(flag_conf or flag_err),
        "and": bool(flag_err or (flag_conf and flag_err)),
        "multi_v1": bool(flag_shift or flag_voiced or flag_hard or flag_heatmap_low),
        "multi_v2": bool((flag_conf or flag_err) and (flag_shift or flag_voiced)),
    }


def _slice_summary(
    rows: list[dict],
    *,
    min_conf: float,
    max_err: float,
    correct_threshold_ms: float,
) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"rows": 0}

    modes = ("or", "and", "multi_v1", "multi_v2")
    fire_count = {m: 0 for m in modes}
    fire_correct = {m: 0 for m in modes}  # rows that were good AND were triggered
    correct_total = 0
    for row in rows:
        is_correct = _row_is_correct(row, correct_threshold_ms)
        if is_correct:
            correct_total += 1
        flags = _classify_row(row, min_conf=min_conf, max_err=max_err)
        for m in modes:
            if flags[m]:
                fire_count[m] += 1
                if is_correct:
                    fire_correct[m] += 1

    out = {
        "rows": n,
        "correct_rate": correct_total / n,
    }
    for m in modes:
        fc = fire_count[m]
        out[f"fb_{m}"] = fc / n
        out[f"model_kept_{m}"] = (n - fc) / n
        out[f"good_kept_{m}"] = (
            (correct_total - fire_correct[m]) / correct_total if correct_total else 0.0
        )
    out["delta_and_minus_or"] = out["fb_and"] - out["fb_or"]
    out["delta_multi_v1_minus_or"] = out["fb_multi_v1"] - out["fb_or"]
    out["delta_multi_v2_minus_or"] = out["fb_multi_v2"] - out["fb_or"]
    return out


def _slice_rows(
    files: list[dict],
    *,
    language: str,
    format_type: str,
    alias_role: str,
) -> list[dict]:
    out: list[dict] = []
    for row in files:
        if language and str(row.get("language", "") or "") != language:
            continue
        if format_type and str(row.get("format_type", "") or "") != format_type:
            continue
        if alias_role and str(row.get("alias_role", "") or "") != alias_role:
            continue
        out.append(row)
    return out


def _parse_slice(raw: str) -> tuple[str, str, str]:
    parts = str(raw or "").split("/")
    while len(parts) < 3:
        parts.append("*")
    return (
        "" if parts[0] == "*" else parts[0],
        "" if parts[1] == "*" else parts[1],
        "" if parts[2] == "*" else parts[2],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate inference fallback trigger over an evaluate_oto JSON.")
    parser.add_argument("--eval", required=True, help="Path to evaluate_oto.py output JSON.")
    parser.add_argument("--out", required=True, help="Path to write simulation result JSON.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.environ.get("UTOA_OTO_CRNN_LOW_CONF_MIN_CONFIDENCE", "0.54")),
        help="Mirrors UTOA_OTO_CRNN_LOW_CONF_MIN_CONFIDENCE (default 0.54).",
    )
    parser.add_argument(
        "--max-predicted-error-ms",
        type=float,
        default=float(os.environ.get("UTOA_OTO_CRNN_LOW_CONF_MAX_PREDICTED_ERROR_MS", "95")),
        help="Mirrors UTOA_OTO_CRNN_LOW_CONF_MAX_PREDICTED_ERROR_MS (default 95).",
    )
    parser.add_argument(
        "--correct-threshold-ms",
        type=float,
        default=50.0,
        help="A row counts as 'correct' when preutterance abs error <= this (ms). Default 50.",
    )
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        help='Slice in "language/format/role" form, "*" wildcard. Repeatable.',
    )
    args = parser.parse_args(argv)

    eval_path = os.path.abspath(args.eval)
    if not os.path.isfile(eval_path):
        print(f"[fb-sim] eval JSON not found: {eval_path}", file=sys.stderr)
        return 2
    with open(eval_path, encoding="utf-8") as fh:
        eval_data = json.load(fh)
    files = eval_data.get("files") or []
    if not isinstance(files, list) or not files:
        print("[fb-sim] eval JSON has no per-row 'files' array", file=sys.stderr)
        return 2

    raw_slices = list(args.slice) if args.slice else [
        "*/*/*",
        "korean/*/*",
        "japanese/*/*",
        "korean/cvc/*",
        "korean/cvvc/*",
        "japanese/cvvc/*",
        "korean/vcv/*",
        "japanese/vcv/*",
    ]

    out: dict[str, Any] = {
        "source_eval": eval_path,
        "min_confidence": float(args.min_confidence),
        "max_predicted_error_ms": float(args.max_predicted_error_ms),
        "correct_threshold_ms": float(args.correct_threshold_ms),
        "and_mode_slices_env": os.environ.get(
            "UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES",
            "(default) korean/cvc,japanese/cvvc",
        ),
        "slices": [],
    }

    for raw in raw_slices:
        lang, fmt, role = _parse_slice(raw)
        rows = _slice_rows(files, language=lang, format_type=fmt, alias_role=role)
        if not rows:
            continue
        summary = _slice_summary(
            rows,
            min_conf=float(args.min_confidence),
            max_err=float(args.max_predicted_error_ms),
            correct_threshold_ms=float(args.correct_threshold_ms),
        )
        # Whether the inference would actually use AND mode for this slice
        # (i.e. the env-pattern matches). For roles, AND mode dispatch is
        # by language+format only, so role-narrowed slices inherit their
        # parent's mode.
        and_mode_active = _is_multi_signal_required(lang or "*", fmt or "*")
        summary["slice"] = f"{lang or '*'}/{fmt or '*'}/{role or '*'}"
        summary["language"] = lang or "*"
        summary["format_type"] = fmt or "*"
        summary["alias_role"] = role or "*"
        summary["and_mode_active_for_inference"] = bool(and_mode_active)
        # The "effective fallback rate" inference would actually realise:
        # AND mode only when the pattern matches; OR mode otherwise.
        if and_mode_active:
            summary["inference_effective_fallback_rate"] = summary["fb_and"]
            summary["inference_effective_model_kept"] = summary["model_kept_and"]
        else:
            summary["inference_effective_fallback_rate"] = summary["fb_or"]
            summary["inference_effective_model_kept"] = summary["model_kept_or"]
        out["slices"].append(summary)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    # Compact stdout summary.
    print(json.dumps(
        {
            "source_eval": eval_path,
            "out": os.path.abspath(args.out),
            "slices": [
                {
                    "slice": s["slice"],
                    "rows": s["rows"],
                    "and_active": s["and_mode_active_for_inference"],
                    "fb_or": round(s["fb_or"], 4),
                    "fb_and": round(s["fb_and"], 4),
                    "fb_multi_v1": round(s["fb_multi_v1"], 4),
                    "fb_multi_v2": round(s["fb_multi_v2"], 4),
                    "model_kept_or": round(s["model_kept_or"], 4),
                    "model_kept_multi_v1": round(s["model_kept_multi_v1"], 4),
                    "model_kept_multi_v2": round(s["model_kept_multi_v2"], 4),
                    "good_kept_or": round(s["good_kept_or"], 4),
                    "good_kept_multi_v1": round(s["good_kept_multi_v1"], 4),
                    "good_kept_multi_v2": round(s["good_kept_multi_v2"], 4),
                }
                for s in out["slices"]
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
