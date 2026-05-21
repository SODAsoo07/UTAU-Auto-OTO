from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.coarse_crnn.boundary_types import DecodedOtoRow
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params


def evaluate_decoded_rows(
    decoded_rows: list[DecodedOtoRow],
    *,
    reference_rows: dict[tuple[str, str, int], dict[str, float]] | None = None,
) -> dict[str, Any]:
    total = len(decoded_rows)
    if total <= 0:
        return {
            "rows": 0,
            "position_bad_rate_100ms": 0.0,
            "position_bad_rate_250ms": 0.0,
            "alias_boundary_error_rate_100ms": 0.0,
            "alias_boundary_error_rate_50ms": 0.0,
            "param_mae_abs_ms": {},
            "param_bad_rate_50ms": {},
            "param_bad_rate_100ms": {},
            "full_param_bad_rate_100ms": 0.0,
            "stretch_risk_rate": 0.0,
            "monotonic_violation_rate": 0.0,
            "fallback_rate_by_role": {},
            "hard_failure_rate": 0.0,
        }
    bad100 = 0
    bad50 = 0
    bad250 = 0
    stretch_bad = 0
    hard_fail = 0
    fallback_by_role: dict[str, list[int]] = defaultdict(list)
    bad100_by_role: dict[str, list[int]] = defaultdict(list)
    bad250_by_role: dict[str, list[int]] = defaultdict(list)
    monotonic_violations = 0
    prev_pre_abs_by_wav: dict[str, float] = {}
    param_names = ("offset_abs", "overlap_abs", "pre_abs", "consonant_abs", "cutoff_abs")
    param_err_sum: dict[str, float] = {name: 0.0 for name in param_names}
    param_err_count: dict[str, int] = {name: 0 for name in param_names}
    param_bad_50: dict[str, int] = {name: 0 for name in param_names}
    param_bad_100: dict[str, int] = {name: 0 for name in param_names}
    full_param_bad_100 = 0
    full_param_count = 0
    for row in sorted(decoded_rows, key=lambda item: (item.spec.wav_path, int(item.spec.line_index))):
        params = absolute_anchors_to_oto_params(row.anchors, duration_ms=float(row.spec.duration_ms))
        flags = _stretch_risk_from_anchors(row)
        if any(flags.values()):
            stretch_bad += 1
        role = str(row.spec.role or "other").lower()
        fallback_by_role[role].append(1 if row.fallback_used else 0)
        prev = prev_pre_abs_by_wav.get(row.spec.wav_path)
        if prev is not None and float(row.anchors.pre_abs) + 4.0 < float(prev):
            monotonic_violations += 1
        prev_pre_abs_by_wav[row.spec.wav_path] = float(row.anchors.pre_abs)
        if reference_rows:
            key = (row.spec.wav_name, row.spec.alias, int(row.spec.line_index))
            ref = reference_rows.get(key)
            if ref:
                pred_abs = {
                    "offset_abs": float(row.anchors.offset_abs),
                    "overlap_abs": float(row.anchors.overlap_abs),
                    "pre_abs": float(row.anchors.pre_abs),
                    "consonant_abs": float(row.anchors.consonant_abs),
                    "cutoff_abs": float(row.anchors.cutoff_abs),
                }
                any_bad_100 = False
                any_param = False
                for name in param_names:
                    if name not in ref:
                        continue
                    any_param = True
                    err = abs(float(pred_abs[name]) - float(ref[name]))
                    param_err_sum[name] += float(err)
                    param_err_count[name] += 1
                    if err > 50.0:
                        param_bad_50[name] += 1
                    if err > 100.0:
                        param_bad_100[name] += 1
                        any_bad_100 = True
                if any_param:
                    full_param_count += 1
                    if any_bad_100:
                        full_param_bad_100 += 1

                if "pre_abs" in ref:
                    pred_pre = float(pred_abs["pre_abs"])
                    ref_pre = float(ref.get("pre_abs", pred_pre))
                    err = abs(pred_pre - ref_pre)
                    bad50_flag = 1 if err > 50.0 else 0
                    bad100_flag = 1 if err > 100.0 else 0
                    bad250_flag = 1 if err > 250.0 else 0
                    bad100_by_role[role].append(bad100_flag)
                    bad250_by_role[role].append(bad250_flag)
                    if err > 50.0:
                        bad50 += 1
                    if err > 100.0:
                        bad100 += 1
                    if err > 250.0:
                        bad250 += 1
        if float(params.get("preutterance", 0.0)) <= 0.1 or float(params.get("consonant", 0.0)) <= 0.1:
            hard_fail += 1
    fallback_rate_by_role = {
        role: float(sum(items)) / float(len(items)) if items else 0.0
        for role, items in sorted(fallback_by_role.items())
    }
    role_position_bad_rate_100ms = {
        role: float(sum(items)) / float(len(items)) if items else 0.0
        for role, items in sorted(bad100_by_role.items())
    }
    role_position_bad_rate_250ms = {
        role: float(sum(items)) / float(len(items)) if items else 0.0
        for role, items in sorted(bad250_by_role.items())
    }
    param_mae_abs_ms = {
        name: (float(param_err_sum[name]) / float(param_err_count[name]) if param_err_count[name] > 0 else None)
        for name in param_names
    }
    param_bad_rate_50ms = {
        name: (float(param_bad_50[name]) / float(param_err_count[name]) if param_err_count[name] > 0 else None)
        for name in param_names
    }
    param_bad_rate_100ms = {
        name: (float(param_bad_100[name]) / float(param_err_count[name]) if param_err_count[name] > 0 else None)
        for name in param_names
    }
    return {
        "rows": int(total),
        "position_bad_rate_100ms": float(bad100) / float(total) if reference_rows else None,
        "position_bad_rate_250ms": float(bad250) / float(total) if reference_rows else None,
        "alias_boundary_error_rate_50ms": float(bad50) / float(total) if reference_rows else None,
        "alias_boundary_error_rate_100ms": float(bad100) / float(total) if reference_rows else None,
        "param_mae_abs_ms": param_mae_abs_ms if reference_rows else {},
        "param_bad_rate_50ms": param_bad_rate_50ms if reference_rows else {},
        "param_bad_rate_100ms": param_bad_rate_100ms if reference_rows else {},
        "full_param_bad_rate_100ms": (
            float(full_param_bad_100) / float(full_param_count)
            if reference_rows and full_param_count > 0
            else None
        ),
        "stretch_risk_rate": float(stretch_bad) / float(total),
        "monotonic_violation_rate": float(monotonic_violations) / float(total),
        "fallback_rate_by_role": fallback_rate_by_role,
        "role_position_bad_rate_100ms": role_position_bad_rate_100ms if reference_rows else {},
        "role_position_bad_rate_250ms": role_position_bad_rate_250ms if reference_rows else {},
        "hard_failure_rate": float(hard_fail) / float(total),
    }


def _stretch_risk_from_anchors(row: DecodedOtoRow) -> dict[str, bool]:
    a = row.anchors
    cons_gap = max(0.0, float(a.consonant_abs) - float(a.pre_abs))
    tail_gap = max(0.0, float(a.cutoff_abs) - float(a.consonant_abs))
    pre_rel = max(1.0, float(a.pre_abs) - float(a.offset_abs))
    ovl_rel = max(0.0, float(a.overlap_abs) - float(a.offset_abs))
    overlap_ratio = ovl_rel / pre_rel
    return {
        "cons_gap_outlier": bool(cons_gap > 120.0),
        "tail_gap_outlier": bool(tail_gap > 170.0),
        "overlap_outlier": bool(overlap_ratio > 0.92),
    }


__all__ = ["evaluate_decoded_rows"]
