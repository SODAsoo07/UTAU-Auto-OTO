from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.coarse_crnn.boundary_types import DecodedOtoRow
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params
from core.coarse_crnn.oto_param_builder import stretch_risk_flags


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
            "stretch_risk_rate": 0.0,
            "monotonic_violation_rate": 0.0,
            "fallback_rate_by_role": {},
            "hard_failure_rate": 0.0,
        }
    bad100 = 0
    bad250 = 0
    stretch_bad = 0
    hard_fail = 0
    fallback_by_role: dict[str, list[int]] = defaultdict(list)
    monotonic_violations = 0
    prev_pre_abs_by_wav: dict[str, float] = {}
    for row in sorted(decoded_rows, key=lambda item: (item.spec.wav_path, int(item.spec.line_index))):
        params = absolute_anchors_to_oto_params(row.anchors, duration_ms=float(row.spec.duration_ms))
        flags = stretch_risk_flags(params)
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
                pred_pre = float(row.anchors.pre_abs)
                ref_pre = float(ref.get("pre_abs", pred_pre))
                err = abs(pred_pre - ref_pre)
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
    return {
        "rows": int(total),
        "position_bad_rate_100ms": float(bad100) / float(total) if reference_rows else None,
        "position_bad_rate_250ms": float(bad250) / float(total) if reference_rows else None,
        "stretch_risk_rate": float(stretch_bad) / float(total),
        "monotonic_violation_rate": float(monotonic_violations) / float(total),
        "fallback_rate_by_role": fallback_rate_by_role,
        "hard_failure_rate": float(hard_fail) / float(total),
    }


__all__ = ["evaluate_decoded_rows"]

