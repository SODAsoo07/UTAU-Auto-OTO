"""Unit tests for the multi-signal fallback policy module (Plan B, 0-order)."""
from __future__ import annotations


def _row(
    *,
    pre_err: float = 0.0,
    offset_err: float = 0.0,
    cutoff_err: float = 0.0,
    consonant_err: float = 0.0,
    overlap_err: float = 0.0,
    confidence: float = 1.0,
    predicted_error: float | None = 0.0,
    heatmap: float = 1.0,
    low_conf: bool = False,
    hard_fail: bool = False,
    voiced_miss: bool = False,
    syllable_shift: bool = False,
):
    return {
        "param_abs_errors_ms": {
            "preutterance": pre_err,
            "offset": offset_err,
            "cutoff_abs": cutoff_err,
            "consonant": consonant_err,
            "overlap": overlap_err,
        },
        "confidence": confidence,
        "predicted_error_ms": predicted_error,
        "confidence_components": {"heatmap": heatmap},
        "low_confidence": low_conf,
        "hard_failure": hard_fail,
        "fn_voiced_miss": voiced_miss,
        "syllable_shift_flag": syllable_shift,
    }


def test_atomic_signals_match_row_keys():
    from core.coarse_crnn.oto_fallback_policy import (
        signal_low_confidence,
        signal_hard_failure,
        signal_fn_voiced_miss,
        signal_syllable_shift,
        signal_predicted_error,
        signal_heatmap_low,
        signal_confidence_low,
    )

    on = _row(
        low_conf=True,
        hard_fail=True,
        voiced_miss=True,
        syllable_shift=True,
        confidence=0.30,
        predicted_error=200.0,
        heatmap=0.10,
    )
    off = _row()
    assert signal_low_confidence(on) and not signal_low_confidence(off)
    assert signal_hard_failure(on) and not signal_hard_failure(off)
    assert signal_fn_voiced_miss(on) and not signal_fn_voiced_miss(off)
    assert signal_syllable_shift(on) and not signal_syllable_shift(off)
    assert signal_predicted_error(on, threshold_ms=80.0)
    assert not signal_predicted_error(off, threshold_ms=80.0)
    assert signal_heatmap_low(on, threshold=0.20)
    assert not signal_heatmap_low(off, threshold=0.20)
    assert signal_confidence_low(on, threshold=0.55)
    assert not signal_confidence_low(off, threshold=0.55)


def test_default_grid_includes_expected_combinations():
    from core.coarse_crnn.oto_fallback_policy import default_trigger_grid

    grid = default_trigger_grid()
    names = {t.name for t in grid}
    assert "low_confidence" in names
    assert "syllable_shift" in names
    assert "fn_voiced_miss" in names
    assert "syllable_shift OR fn_voiced_miss" in names
    assert "syllable_shift AND fn_voiced_miss" in names
    assert "shift OR voiced_miss OR hard_failure OR heatmap<=0.20" in names
    assert "runtime_legacy (low_conf OR hard_fail OR voiced_miss)" in names


def test_evaluate_policy_metrics_match_definitions():
    from core.coarse_crnn.oto_fallback_policy import (
        TriggerSpec,
        evaluate_policy,
        signal_syllable_shift,
    )

    rows = [
        _row(pre_err=10.0, syllable_shift=False),
        _row(pre_err=200.0, syllable_shift=True),
        _row(pre_err=300.0, syllable_shift=True),
        _row(pre_err=20.0, syllable_shift=False),
    ]
    trig = TriggerSpec("syllable_shift", signal_syllable_shift)
    out = evaluate_policy(rows, trig)
    assert out["rows"] == 4
    # Two rows triggered; both happen to be bad → recall 2/2, precision 2/2.
    assert abs(out["fallback_rate"] - 0.5) < 1e-9
    assert abs(out["bad_recall"] - 1.0) < 1e-9
    assert abs(out["precision"] - 1.0) < 1e-9
    # Both good rows kept → good_keep_rate 1.0.
    assert abs(out["good_keep_rate"] - 1.0) < 1e-9
    # Effective preutterance after triggered rows fallback to 0:
    # row1=10, row2=0, row3=0, row4=20 → pre_acc50 = 4/4 = 1.0.
    assert abs(out["effective_preutterance_acc_50ms"] - 1.0) < 1e-9


def test_evaluate_policy_grid_orders_by_plan_b_priority():
    from core.coarse_crnn.oto_fallback_policy import (
        TriggerSpec,
        evaluate_policy_grid,
    )

    rows = [
        _row(pre_err=10.0),                 # good, no signals
        _row(pre_err=200.0, syllable_shift=True),  # bad, shift fires
        _row(pre_err=300.0, voiced_miss=True),     # bad, voiced fires
        _row(pre_err=400.0),                # bad, no signals (caught only by always-on)
        _row(pre_err=20.0),                 # good
    ]
    grid = [
        TriggerSpec("never", lambda r: False),
        TriggerSpec("shift only", lambda r: bool(r.get("syllable_shift_flag", False))),
        TriggerSpec(
            "shift OR voiced",
            lambda r: bool(r.get("syllable_shift_flag", False))
            or bool(r.get("fn_voiced_miss", False)),
        ),
        TriggerSpec("always", lambda r: True),
    ]
    sorted_grid = evaluate_policy_grid(rows, grid=grid)
    # Plan B priority sorts highest bad_recall first.
    names = [c["name"] for c in sorted_grid]
    # 'always' has bad_recall=1.0 but good_keep=0 -> tied with shift OR voiced on recall? Let's check:
    # bad rows: 3 (rows 1,2,3). 'shift only' catches 1/3=0.33, 'shift OR voiced' catches 2/3=0.67,
    # 'always' catches 3/3=1.0, 'never' 0.
    assert names[0] == "always"  # recall 1.0
    assert names[1] == "shift OR voiced"  # recall 0.67
    assert names[2] == "shift only"  # recall 0.33
    assert names[-1] == "never"  # recall 0


def test_evaluate_policy_handles_empty_input():
    from core.coarse_crnn.oto_fallback_policy import (
        TriggerSpec,
        evaluate_policy,
    )

    out = evaluate_policy([], TriggerSpec("noop", lambda r: True))
    assert out["rows"] == 0
    assert out["fallback_rate"] == 0.0
    assert out["bad_recall"] == 0.0
    assert out["effective_preutterance_acc_50ms"] == 0.0


def test_is_bad_row_modes_disagree_on_voiced_miss():
    """errors_only must NOT mark a voiced-miss-only row as bad; strict must."""
    from core.coarse_crnn.oto_fallback_policy import _is_bad_row

    row = _row(pre_err=10.0, offset_err=10.0, cutoff_err=10.0, voiced_miss=True)
    assert _is_bad_row(row, mode="errors_only") is False
    assert _is_bad_row(row, mode="strict") is True


def test_is_bad_row_threshold_breach_in_both_modes():
    from core.coarse_crnn.oto_fallback_policy import _is_bad_row

    row = _row(pre_err=200.0)
    assert _is_bad_row(row, mode="errors_only") is True
    assert _is_bad_row(row, mode="strict") is True


def test_evaluate_policy_grid_respects_bad_mode():
    """In errors_only mode a fn_voiced_miss-based trigger should NOT auto-win
    just because every voiced-miss row is auto-marked bad."""
    from core.coarse_crnn.oto_fallback_policy import (
        TriggerSpec,
        evaluate_policy_grid,
        signal_fn_voiced_miss,
    )

    rows = [
        # bad on errors AND voiced_miss
        _row(pre_err=200.0, voiced_miss=True),
        # voiced_miss only — strict marks bad, errors_only does NOT
        _row(pre_err=10.0, voiced_miss=True),
        _row(pre_err=10.0, voiced_miss=True),
        _row(pre_err=20.0, voiced_miss=False),
    ]
    grid = [TriggerSpec("voiced_miss only", signal_fn_voiced_miss)]
    strict = evaluate_policy_grid(rows, grid=grid, bad_mode="strict")[0]
    relaxed = evaluate_policy_grid(rows, grid=grid, bad_mode="errors_only")[0]
    # strict: 3 bad, voiced_miss catches all 3 → recall 1.0, precision 1.0
    assert abs(strict["bad_recall"] - 1.0) < 1e-9
    assert abs(strict["precision"] - 1.0) < 1e-9
    # errors_only: only 1 bad row, but voiced_miss fires on 3 → low precision
    assert abs(relaxed["bad_recall"] - 1.0) < 1e-9
    assert relaxed["precision"] < strict["precision"]


def test_default_grid_evaluates_without_runtime_signals():
    """Rows missing optional fields should not raise; they default to no-trigger."""
    from core.coarse_crnn.oto_fallback_policy import (
        default_trigger_grid,
        evaluate_policy_grid,
    )

    rows = [
        {"param_abs_errors_ms": {"preutterance": 10.0}},
        {"param_abs_errors_ms": {"preutterance": 200.0, "offset": 1000.0}},
    ]
    out = evaluate_policy_grid(rows, grid=default_trigger_grid())
    assert len(out) > 0
    # All triggers must produce finite metrics.
    for cand in out:
        assert 0.0 <= cand["fallback_rate"] <= 1.0
        assert 0.0 <= cand["bad_recall"] <= 1.0
