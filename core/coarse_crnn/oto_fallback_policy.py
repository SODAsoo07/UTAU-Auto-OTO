"""Multi-signal fallback trigger policy exploration (Plan B, 0-order).

Plan B from [[CRNN-정확도-개선-방안-v2]] 0-order: when the model's confidence
head is collapsed (good_mean ≈ bad_mean), monotonic recalibration cannot
recover separation. Instead this module composes multiple per-row signals
(`syllable_shift_flag`, `fn_voiced_miss`, `low_confidence`, `hard_failure`,
heatmap, predicted_error_ms) into combined triggers and evaluates each
candidate against per-row "bad" labels from the eval JSON.

Inputs are dict rows in the format produced by ``oto_evaluate.py``'s
``files`` array. The module is pure: no I/O, no torch.

Companion script: ``ml/scripts/coarse_crnn/explore_oto_fallback_combinations.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

PARAM_NAMES = ("preutterance", "offset", "consonant", "cutoff_abs", "overlap")


def _param_error(row: dict, name: str) -> float:
    errors = row.get("param_abs_errors_ms")
    if not isinstance(errors, dict):
        return 0.0
    try:
        return float(errors.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _confidence(row: dict) -> float:
    try:
        return float(row.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _predicted_error_ms(row: dict) -> float:
    raw = row.get("predicted_error_ms")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _heatmap_confidence(row: dict) -> float:
    components = row.get("confidence_components")
    if isinstance(components, dict):
        try:
            return float(components.get("heatmap", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _is_bad_row(
    row: dict,
    *,
    bad_pre_ms: float = 50.0,
    bad_offset_ms: float = 700.0,
    bad_cutoff_ms: float = 900.0,
) -> bool:
    """Mirror the definition used in analyze_oto_fallback_candidates.py."""
    return bool(
        _param_error(row, "preutterance") > float(bad_pre_ms)
        or _param_error(row, "offset") > float(bad_offset_ms)
        or _param_error(row, "cutoff_abs") > float(bad_cutoff_ms)
        or bool(row.get("hard_failure", False))
        or bool(row.get("fn_voiced_miss", False))
    )


# ---------------------------------------------------------------------------
# Atomic signals
# ---------------------------------------------------------------------------


def signal_low_confidence(row: dict) -> bool:
    return bool(row.get("low_confidence", False))


def signal_hard_failure(row: dict) -> bool:
    return bool(row.get("hard_failure", False))


def signal_fn_voiced_miss(row: dict) -> bool:
    return bool(row.get("fn_voiced_miss", False))


def signal_syllable_shift(row: dict) -> bool:
    return bool(row.get("syllable_shift_flag", False))


def signal_predicted_error(row: dict, *, threshold_ms: float = 80.0) -> bool:
    return _predicted_error_ms(row) >= float(threshold_ms)


def signal_heatmap_low(row: dict, *, threshold: float = 0.25) -> bool:
    return _heatmap_confidence(row) <= float(threshold)


def signal_confidence_low(row: dict, *, threshold: float = 0.55) -> bool:
    return _confidence(row) <= float(threshold)


# ---------------------------------------------------------------------------
# Combined triggers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerSpec:
    name: str
    fn: Callable[[dict], bool]


def default_trigger_grid() -> list[TriggerSpec]:
    """A focused grid covering the most plausible Plan B combinations.

    Logic priority order:
    1. Atomic strong signals (`syllable_shift`, `fn_voiced_miss`).
    2. OR combinations of activity-based signals — broader recall.
    3. AND combinations — higher precision, lower fallback rate.
    4. Layered combinations adding `hard_failure` as a safety net.
    5. A few legacy comparisons (`low_confidence` only, etc.).
    """
    return [
        # Atomic
        TriggerSpec("low_confidence", signal_low_confidence),
        TriggerSpec("hard_failure", signal_hard_failure),
        TriggerSpec("fn_voiced_miss", signal_fn_voiced_miss),
        TriggerSpec("syllable_shift", signal_syllable_shift),
        # Activity-based OR (recall focus)
        TriggerSpec(
            "syllable_shift OR fn_voiced_miss",
            lambda r: signal_syllable_shift(r) or signal_fn_voiced_miss(r),
        ),
        TriggerSpec(
            "syllable_shift OR fn_voiced_miss OR hard_failure",
            lambda r: signal_syllable_shift(r) or signal_fn_voiced_miss(r) or signal_hard_failure(r),
        ),
        # Activity-based AND (precision focus)
        TriggerSpec(
            "syllable_shift AND fn_voiced_miss",
            lambda r: signal_syllable_shift(r) and signal_fn_voiced_miss(r),
        ),
        # Activity OR + heatmap fallback (cover quiet bad rows)
        TriggerSpec(
            "syllable_shift OR fn_voiced_miss OR heatmap<=0.20",
            lambda r: signal_syllable_shift(r)
            or signal_fn_voiced_miss(r)
            or signal_heatmap_low(r, threshold=0.20),
        ),
        TriggerSpec(
            "syllable_shift OR fn_voiced_miss OR heatmap<=0.25",
            lambda r: signal_syllable_shift(r)
            or signal_fn_voiced_miss(r)
            or signal_heatmap_low(r, threshold=0.25),
        ),
        # Activity-based + predicted error gate
        TriggerSpec(
            "syllable_shift OR fn_voiced_miss OR predicted_error>=120",
            lambda r: signal_syllable_shift(r)
            or signal_fn_voiced_miss(r)
            or signal_predicted_error(r, threshold_ms=120.0),
        ),
        # Plan B candidate: full multi-signal OR (no confidence)
        TriggerSpec(
            "shift OR voiced_miss OR hard_failure OR heatmap<=0.20",
            lambda r: signal_syllable_shift(r)
            or signal_fn_voiced_miss(r)
            or signal_hard_failure(r)
            or signal_heatmap_low(r, threshold=0.20),
        ),
        # Plan B candidate: drop low_confidence dependency entirely
        TriggerSpec(
            "shift OR voiced_miss OR hard_failure OR predicted_error>=120",
            lambda r: signal_syllable_shift(r)
            or signal_fn_voiced_miss(r)
            or signal_hard_failure(r)
            or signal_predicted_error(r, threshold_ms=120.0),
        ),
        # Legacy comparison: current runtime fallback path
        TriggerSpec(
            "runtime_legacy (low_conf OR hard_fail OR voiced_miss)",
            lambda r: signal_low_confidence(r)
            or signal_hard_failure(r)
            or signal_fn_voiced_miss(r),
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _ratio(flags: Iterable[bool]) -> float:
    flags_list = list(flags)
    return float(sum(1 for flag in flags_list if flag)) / float(len(flags_list)) if flags_list else 0.0


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return float(sum(values_list)) / float(len(values_list)) if values_list else 0.0


def evaluate_policy(
    rows: list[dict],
    trigger: TriggerSpec,
    *,
    bad_pre_ms: float = 50.0,
    bad_offset_ms: float = 700.0,
    bad_cutoff_ms: float = 900.0,
) -> dict:
    """Apply ``trigger.fn`` to each row, return classifier-style metrics
    plus the "effective" eval metrics that result from substituting the
    ground-truth (zero error) into rows where the trigger fires.

    The "effective" view is the same accounting used by the legacy
    `analyze_oto_fallback_candidates.py`. It assumes a fallback that
    perfectly recovers the ground truth, so it's an UPPER BOUND on what
    each policy could buy us; in production the source-shape fallback only
    partially recovers errors.
    """
    if not rows:
        return {
            "name": trigger.name,
            "rows": 0,
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "bad_recall": 0.0,
            "precision": 0.0,
            "good_keep_rate": 0.0,
            "effective_preutterance_acc_50ms": 0.0,
            "effective_preutterance_mae_ms": 0.0,
            "effective_offset_mae_ms": 0.0,
            "effective_cutoff_abs_mae_ms": 0.0,
        }
    bad_flags = [
        _is_bad_row(row, bad_pre_ms=bad_pre_ms, bad_offset_ms=bad_offset_ms, bad_cutoff_ms=bad_cutoff_ms)
        for row in rows
    ]
    triggered = [bool(trigger.fn(row)) for row in rows]
    total = len(rows)
    bad_total = sum(1 for flag in bad_flags if flag)
    good_total = max(0, total - bad_total)
    fallback_count = sum(1 for flag in triggered if flag)
    true_positive = sum(1 for tr, bad in zip(triggered, bad_flags) if tr and bad)
    false_positive = sum(1 for tr, bad in zip(triggered, bad_flags) if tr and not bad)

    pre = []
    offset = []
    cutoff = []
    for row, fallback in zip(rows, triggered):
        pre.append(0.0 if fallback else _param_error(row, "preutterance"))
        offset.append(0.0 if fallback else _param_error(row, "offset"))
        cutoff.append(0.0 if fallback else _param_error(row, "cutoff_abs"))
    return {
        "name": trigger.name,
        "rows": total,
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / total) if total else 0.0,
        "bad_recall": float(true_positive / bad_total) if bad_total else 0.0,
        "precision": float(true_positive / fallback_count) if fallback_count else 0.0,
        "good_keep_rate": float((good_total - false_positive) / good_total) if good_total else 0.0,
        "effective_preutterance_acc_50ms": _ratio([value <= 50.0 for value in pre]),
        "effective_preutterance_mae_ms": _mean(pre),
        "effective_offset_mae_ms": _mean(offset),
        "effective_cutoff_abs_mae_ms": _mean(cutoff),
    }


def evaluate_policy_grid(
    rows: list[dict],
    *,
    grid: list[TriggerSpec] | None = None,
    bad_pre_ms: float = 50.0,
    bad_offset_ms: float = 700.0,
    bad_cutoff_ms: float = 900.0,
) -> list[dict]:
    """Evaluate every trigger in the grid (default: ``default_trigger_grid()``)
    and return the result list sorted by Plan B priority:
    higher bad_recall, then higher good_keep_rate, then lower fallback_rate.
    """
    triggers = list(grid) if grid is not None else default_trigger_grid()
    results = [
        evaluate_policy(
            rows,
            trigger,
            bad_pre_ms=bad_pre_ms,
            bad_offset_ms=bad_offset_ms,
            bad_cutoff_ms=bad_cutoff_ms,
        )
        for trigger in triggers
    ]
    results.sort(
        key=lambda row: (row["bad_recall"], row["good_keep_rate"], -row["fallback_rate"]),
        reverse=True,
    )
    return results


__all__ = [
    "TriggerSpec",
    "default_trigger_grid",
    "evaluate_policy",
    "evaluate_policy_grid",
    "signal_confidence_low",
    "signal_fn_voiced_miss",
    "signal_hard_failure",
    "signal_heatmap_low",
    "signal_low_confidence",
    "signal_predicted_error",
    "signal_syllable_shift",
]
