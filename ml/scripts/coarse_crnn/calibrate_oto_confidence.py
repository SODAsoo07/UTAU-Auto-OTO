"""Fit confidence calibration on an evaluate_oto.py result JSON.

0-order item from [[CRNN-정확도-개선-방안-v2]]: when the trained model's
confidence head ends up collapsed (good_mean ≈ bad_mean), the fallback
trigger fires for ~100% of korean/cvvc rows. This script picks a stronger
calibration (temperature scaling, optionally isotonic lookup) on the val
set and reports before/after fallback rates.

Inputs:
- ``--eval``: path to an existing evaluate_oto.py output JSON. Must
  include a ``files`` array with per-row ``confidence``,
  ``param_abs_errors_ms.preutterance``, optionally ``language``,
  ``format_type``, ``alias_role`` for slicing.

Outputs:
- ``--out``: JSON with calibration parameters and before/after diagnostic
  metrics. The original eval JSON is NOT modified — calibration is meant
  to be applied at inference time after acceptance is met.

Acceptance (v2 0-order):
- ``korean/cvvc`` low-confidence fallback rate: 100% → 85% or lower
- One of ``korean/cvvc/{vc, other}`` pre_acc50 must improve by ≥ 5%p
  (this script does NOT verify pre_acc improvement; that requires a
  re-evaluated runtime — see follow-up script)
- hard_failure_rate must not worsen by more than +0.02
- Overall pre_acc50 must not drop by more than -1%p

This script reports calibration metrics only; the ≥5%p / hard_failure
checks belong to the runtime sanity script that consumes these
calibration parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

from core.coarse_crnn.oto_confidence_calibration import (
    apply_temperature_batch,
    compute_low_confidence_threshold,
    fallback_rate,
    fit_isotonic_lookup,
    fit_temperature_scaling,
    good_bad_mean,
)


def _row_is_correct(row: dict[str, Any], threshold_ms: float) -> bool:
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


def _slice_rows(
    files: list[dict[str, Any]],
    *,
    language: str,
    format_type: str,
    alias_role: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in files:
        if language and str(row.get("language", "") or "") != language:
            continue
        if format_type and str(row.get("format_type", "") or "") != format_type:
            continue
        if alias_role and str(row.get("alias_role", "") or "") != alias_role:
            continue
        out.append(row)
    return out


def _slice_diagnostics(
    rows: list[dict[str, Any]],
    *,
    threshold_ms: float,
    target_low_rate: float,
) -> dict[str, Any]:
    raw_probs = [float(r.get("confidence", 0.0) or 0.0) for r in rows]
    is_correct = [_row_is_correct(r, threshold_ms) for r in rows]
    raw_low = [bool(r.get("low_confidence", False)) for r in rows]

    base_low_rate = sum(1 for v in raw_low if v) / len(raw_low) if raw_low else 0.0
    base_good, base_bad = good_bad_mean(raw_probs, is_correct)

    # Temperature fit on the slice.
    best_t, best_nll = fit_temperature_scaling(raw_probs, is_correct)
    cal_probs = apply_temperature_batch(raw_probs, best_t)
    cal_threshold = compute_low_confidence_threshold(cal_probs, target_low_rate=target_low_rate)
    cal_low_rate = fallback_rate(cal_probs, cal_threshold)
    cal_good, cal_bad = good_bad_mean(cal_probs, is_correct)

    # Isotonic fit (sklearn-free) for comparison.
    iso = fit_isotonic_lookup(raw_probs, is_correct)
    iso_probs = [iso.apply(p) for p in raw_probs]
    iso_threshold = compute_low_confidence_threshold(iso_probs, target_low_rate=target_low_rate)
    iso_low_rate = fallback_rate(iso_probs, iso_threshold)
    iso_good, iso_bad = good_bad_mean(iso_probs, is_correct)

    return {
        "rows": len(rows),
        "correct_rate": (sum(1 for v in is_correct if v) / len(is_correct)) if is_correct else 0.0,
        "baseline": {
            "low_confidence_fallback_rate": base_low_rate,
            "conf_good_mean": base_good,
            "conf_bad_mean": base_bad,
            "good_minus_bad": (
                (base_good - base_bad)
                if (math.isfinite(base_good) and math.isfinite(base_bad))
                else float("nan")
            ),
        },
        "temperature": {
            "T": best_t,
            "nll_per_row": best_nll,
            "threshold": cal_threshold,
            "low_confidence_fallback_rate": cal_low_rate,
            "conf_good_mean": cal_good,
            "conf_bad_mean": cal_bad,
            "good_minus_bad": (
                (cal_good - cal_bad)
                if (math.isfinite(cal_good) and math.isfinite(cal_bad))
                else float("nan")
            ),
        },
        "isotonic": {
            "table": list(iso.table),
            "threshold": iso_threshold,
            "low_confidence_fallback_rate": iso_low_rate,
            "conf_good_mean": iso_good,
            "conf_bad_mean": iso_bad,
            "good_minus_bad": (
                (iso_good - iso_bad)
                if (math.isfinite(iso_good) and math.isfinite(iso_bad))
                else float("nan")
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit confidence calibration on an evaluate_oto JSON.")
    parser.add_argument("--eval", required=True, help="Path to evaluate_oto.py output JSON.")
    parser.add_argument("--out", required=True, help="Path to write calibration result JSON.")
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=50.0,
        help="A row counts as 'correct' when preutterance abs error <= this (ms). Default 50.",
    )
    parser.add_argument(
        "--target-low-rate",
        type=float,
        default=0.85,
        help="Target maximum fraction of rows allowed below the calibrated low_confidence threshold. "
             "v2 0-order acceptance suggests <= 0.85 (down from 1.00 saturation). Default 0.85.",
    )
    parser.add_argument(
        "--slice",
        action="append",
        default=[],
        help='One or more "lang/format/role" filters, "*" wildcards. '
             'Example: --slice "korean/cvvc/*" --slice "korean/cvvc/vc". '
             'Defaults to global + a few common diagnostic slices when empty.',
    )
    args = parser.parse_args(argv)

    eval_path = os.path.abspath(args.eval)
    if not os.path.isfile(eval_path):
        print(f"[calibrate] eval JSON not found: {eval_path}", file=sys.stderr)
        return 2
    with open(eval_path, encoding="utf-8") as fh:
        eval_data = json.load(fh)
    files = eval_data.get("files") or []
    if not isinstance(files, list) or not files:
        print("[calibrate] eval JSON has no per-row 'files' array", file=sys.stderr)
        return 2

    slices: list[tuple[str, str, str]] = []
    raw_slices = list(args.slice) if args.slice else [
        "*/*/*",
        "korean/cvvc/*",
        "japanese/cvvc/*",
        "korean/cvvc/vc",
        "korean/cvvc/other",
        "korean/cvvc/cv",
        "korean/cvc/*",
    ]
    for raw in raw_slices:
        parts = raw.split("/")
        while len(parts) < 3:
            parts.append("*")
        lang, fmt, role = parts[0], parts[1], parts[2]
        slices.append((
            "" if lang == "*" else lang,
            "" if fmt == "*" else fmt,
            "" if role == "*" else role,
        ))

    out: dict[str, Any] = {
        "source_eval": eval_path,
        "threshold_ms": float(args.threshold_ms),
        "target_low_rate": float(args.target_low_rate),
        "total_rows": len(files),
        "slices": [],
    }
    for lang, fmt, role in slices:
        rows = _slice_rows(files, language=lang, format_type=fmt, alias_role=role)
        if not rows:
            continue
        diag = _slice_diagnostics(
            rows,
            threshold_ms=float(args.threshold_ms),
            target_low_rate=float(args.target_low_rate),
        )
        diag["language"] = lang or "*"
        diag["format_type"] = fmt or "*"
        diag["alias_role"] = role or "*"
        out["slices"].append(diag)

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
                    "key": f"{s['language']}/{s['format_type']}/{s['alias_role']}",
                    "rows": s["rows"],
                    "baseline_fallback": s["baseline"]["low_confidence_fallback_rate"],
                    "temperature_T": s["temperature"]["T"],
                    "temperature_fallback": s["temperature"]["low_confidence_fallback_rate"],
                    "temperature_good_minus_bad": s["temperature"]["good_minus_bad"],
                    "isotonic_fallback": s["isotonic"]["low_confidence_fallback_rate"],
                    "isotonic_good_minus_bad": s["isotonic"]["good_minus_bad"],
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
