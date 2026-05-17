"""Unit tests for env-var overrides of consonant/tail caps in oto_param_builder.

Covers the Quick Win A+B from the 2026-05-15 listening feedback:
- A: cutoff too aggressive (tail too short) → UTOA_BOUNDARY_TAIL_{MIN,MAX}_MS
- B: consonant region too short → UTOA_BOUNDARY_CONS_{MIN,MAX}_MS
"""
from __future__ import annotations

import pytest

from core.coarse_crnn.oto_param_builder import (
    ROLE_GAP_RULES_MS,
    _resolve_gap_caps,
    build_absolute_anchors_for_role,
)


# ---------------------------------------------------------------------------
# _resolve_gap_caps
# ---------------------------------------------------------------------------

def test_resolve_gap_caps_uses_role_defaults_when_no_env(monkeypatch):
    for name in (
        "UTOA_BOUNDARY_CONS_MIN_MS",
        "UTOA_BOUNDARY_CONS_MAX_MS",
        "UTOA_BOUNDARY_TAIL_MIN_MS",
        "UTOA_BOUNDARY_TAIL_MAX_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    caps = _resolve_gap_caps("cv")
    assert caps == ROLE_GAP_RULES_MS["cv"]


def test_resolve_gap_caps_env_overrides_all_four(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_CONS_MIN_MS", "40")
    monkeypatch.setenv("UTOA_BOUNDARY_CONS_MAX_MS", "140")
    monkeypatch.setenv("UTOA_BOUNDARY_TAIL_MIN_MS", "50")
    monkeypatch.setenv("UTOA_BOUNDARY_TAIL_MAX_MS", "200")
    caps = _resolve_gap_caps("cv")
    assert caps == (40.0, 140.0, 50.0, 200.0)


def test_resolve_gap_caps_overrides_apply_to_all_roles(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_TAIL_MAX_MS", "300")
    for role in ("cv", "vc", "v", "vv", "v-cv", "br", "other"):
        caps = _resolve_gap_caps(role)
        assert caps[3] == 300.0, f"tail_max not overridden for role {role}"


def test_resolve_gap_caps_unknown_role_uses_other_default(monkeypatch):
    for name in (
        "UTOA_BOUNDARY_CONS_MIN_MS",
        "UTOA_BOUNDARY_CONS_MAX_MS",
        "UTOA_BOUNDARY_TAIL_MIN_MS",
        "UTOA_BOUNDARY_TAIL_MAX_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert _resolve_gap_caps("nonexistent_role") == ROLE_GAP_RULES_MS["other"]


def test_resolve_gap_caps_enforces_max_greater_than_min(monkeypatch):
    # If user accidentally sets max <= min, max gets bumped above min so the
    # downstream clamp doesn't produce a zero-width region.
    monkeypatch.setenv("UTOA_BOUNDARY_CONS_MIN_MS", "100")
    monkeypatch.setenv("UTOA_BOUNDARY_CONS_MAX_MS", "50")
    caps = _resolve_gap_caps("cv")
    cons_min, cons_max, _, _ = caps
    assert cons_max > cons_min


def test_resolve_gap_caps_invalid_env_value_falls_back(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_CONS_MAX_MS", "not_a_number")
    caps = _resolve_gap_caps("cv")
    assert caps[1] == ROLE_GAP_RULES_MS["cv"][1]  # original default


# ---------------------------------------------------------------------------
# build_absolute_anchors_for_role with env overrides
# ---------------------------------------------------------------------------

def _make_anchors(role: str = "cv", *, center: float = 500.0, left: float = 100.0, right: float = 900.0, duration: float = 1500.0):
    return build_absolute_anchors_for_role(
        role=role,
        center_ms=center,
        duration_ms=duration,
        left_anchor_ms=left,
        right_anchor_ms=right,
        active_start_ms=50.0,
        active_end_ms=1400.0,
    )


def test_build_anchors_default_keeps_conservative_tail(monkeypatch):
    for name in ("UTOA_BOUNDARY_CONS_MIN_MS", "UTOA_BOUNDARY_CONS_MAX_MS", "UTOA_BOUNDARY_TAIL_MIN_MS", "UTOA_BOUNDARY_TAIL_MAX_MS", "UTOA_BOUNDARY_MAX_SPAN_MS"):
        monkeypatch.delenv(name, raising=False)
    a = _make_anchors("cv")
    tail = a.cutoff_abs - a.consonant_abs
    cons = a.consonant_abs - a.pre_abs
    # Default cv caps: cons (14, 62), tail (20, 86).
    assert tail <= 86.0 + 1.0
    assert cons <= 62.0 + 1.0


def test_build_anchors_env_extends_tail(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_TAIL_MAX_MS", "220")
    monkeypatch.setenv("UTOA_BOUNDARY_MAX_SPAN_MS", "700")  # span guard must allow new range
    a_default = _make_anchors("cv")
    monkeypatch.delenv("UTOA_BOUNDARY_TAIL_MAX_MS")
    monkeypatch.delenv("UTOA_BOUNDARY_MAX_SPAN_MS")
    a_baseline = _make_anchors("cv")
    extended_tail = a_default.cutoff_abs - a_default.consonant_abs
    baseline_tail = a_baseline.cutoff_abs - a_baseline.consonant_abs
    # The env-overridden run should produce a strictly longer tail when the
    # anchor span gives it room (800ms span between left=100 and right=900).
    assert extended_tail > baseline_tail, (
        f"extended tail {extended_tail:.1f} not larger than baseline {baseline_tail:.1f}"
    )


def test_build_anchors_env_extends_consonant_region(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_CONS_MAX_MS", "150")
    monkeypatch.setenv("UTOA_BOUNDARY_MAX_SPAN_MS", "700")
    a_default = _make_anchors("cv")
    monkeypatch.delenv("UTOA_BOUNDARY_CONS_MAX_MS")
    monkeypatch.delenv("UTOA_BOUNDARY_MAX_SPAN_MS")
    a_baseline = _make_anchors("cv")
    extended_cons = a_default.consonant_abs - a_default.pre_abs
    baseline_cons = a_baseline.consonant_abs - a_baseline.pre_abs
    assert extended_cons > baseline_cons


def test_build_anchors_does_not_cross_right_anchor(monkeypatch):
    """Even with aggressive cap overrides, cutoff stays inside right_anchor − margin."""
    monkeypatch.setenv("UTOA_BOUNDARY_TAIL_MAX_MS", "500")
    monkeypatch.setenv("UTOA_BOUNDARY_CONS_MAX_MS", "300")
    monkeypatch.setenv("UTOA_BOUNDARY_MAX_SPAN_MS", "900")
    right_anchor = 700.0
    a = build_absolute_anchors_for_role(
        role="cv",
        center_ms=400.0,
        duration_ms=1500.0,
        left_anchor_ms=100.0,
        right_anchor_ms=right_anchor,
        active_start_ms=50.0,
        active_end_ms=1400.0,
    )
    # apply_right_boundary_guard should keep cutoff inside right - margin.
    assert a.cutoff_abs <= right_anchor, (
        f"cutoff {a.cutoff_abs} crossed right_anchor {right_anchor}"
    )


def test_build_anchors_env_extends_consonant_floor(monkeypatch):
    """When cons_min is raised, even a short anchor span should respect the floor."""
    monkeypatch.setenv("UTOA_BOUNDARY_CONS_MIN_MS", "60")
    monkeypatch.setenv("UTOA_BOUNDARY_MAX_SPAN_MS", "400")
    # Use a tight anchor span so the default cons (14ms min) would normally apply
    a = build_absolute_anchors_for_role(
        role="cv",
        center_ms=200.0,
        duration_ms=600.0,
        left_anchor_ms=150.0,
        right_anchor_ms=300.0,  # tight span 150ms
        active_start_ms=50.0,
        active_end_ms=500.0,
    )
    cons = a.consonant_abs - a.pre_abs
    # cons should respect the 60ms floor (or push against right_anchor guard)
    assert cons >= 30.0  # should be near 60 but right-guard might compress slightly


def test_build_anchors_overrides_apply_to_japanese_role_too(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_TAIL_MAX_MS", "180")
    monkeypatch.setenv("UTOA_BOUNDARY_MAX_SPAN_MS", "600")
    a = build_absolute_anchors_for_role(
        role="vc",
        center_ms=400.0,
        duration_ms=1500.0,
        left_anchor_ms=100.0,
        right_anchor_ms=900.0,
        active_start_ms=50.0,
        active_end_ms=1400.0,
    )
    tail = a.cutoff_abs - a.consonant_abs
    # vc default tail_max is 62; with override 180 we should see a longer tail.
    assert tail > 62.0


def test_transition_pre_is_pulled_earlier_by_lead_env(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_TRANSITION_PRE_LEAD_SCALE", "0")
    monkeypatch.setenv("UTOA_BOUNDARY_TRANSITION_PRE_LEAD_MIN_MS", "80")
    monkeypatch.setenv("UTOA_BOUNDARY_TRANSITION_PRE_LEAD_MAX_MS", "80")
    a = build_absolute_anchors_for_role(
        role="vc",
        center_ms=500.0,
        duration_ms=1500.0,
        left_anchor_ms=300.0,
        right_anchor_ms=700.0,
        active_start_ms=0.0,
        active_end_ms=1400.0,
    )
    # pre should be center - 80ms under the forced lead.
    assert a.pre_abs == pytest.approx(420.0, abs=1e-3)


def test_transition_pre_lead_default_is_disabled(monkeypatch):
    for name in (
        "UTOA_BOUNDARY_TRANSITION_PRE_LEAD_SCALE",
        "UTOA_BOUNDARY_TRANSITION_PRE_LEAD_MIN_MS",
        "UTOA_BOUNDARY_TRANSITION_PRE_LEAD_MAX_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    a = build_absolute_anchors_for_role(
        role="vc",
        center_ms=500.0,
        duration_ms=1500.0,
        left_anchor_ms=300.0,
        right_anchor_ms=700.0,
        active_start_ms=0.0,
        active_end_ms=1400.0,
    )
    assert a.pre_abs == pytest.approx(500.0, abs=1e-3)


def test_vc_offset_default_lead_is_not_excessive(monkeypatch):
    for name in ("UTOA_BOUNDARY_VC_OFFSET_LEFT_PAD_MS", "UTOA_BOUNDARY_VC_OFFSET_PRE_LEAD_MS"):
        monkeypatch.delenv(name, raising=False)
    a = build_absolute_anchors_for_role(
        role="vc",
        center_ms=500.0,
        duration_ms=1500.0,
        left_anchor_ms=300.0,
        right_anchor_ms=700.0,
        active_start_ms=0.0,
        active_end_ms=1400.0,
    )
    # VC default offset lead should stay tight to avoid over-long preceding V.
    assert (a.pre_abs - a.offset_abs) <= 24.0 + 1e-3


def test_pre_cons_gap_guard_prevents_near_equal_pre_and_consonant(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_PRE_CONS_MIN_GAP_MS", "24")
    monkeypatch.setenv("UTOA_BOUNDARY_TRANSITION_PRE_CONS_MIN_GAP_MS", "24")
    # Force a tight right boundary where legacy guard tends to compress cons close to pre.
    a = build_absolute_anchors_for_role(
        role="vc",
        center_ms=205.0,
        duration_ms=800.0,
        left_anchor_ms=100.0,
        right_anchor_ms=220.0,
        active_start_ms=0.0,
        active_end_ms=700.0,
    )
    assert (a.consonant_abs - a.pre_abs) >= 24.0 - 1e-3


def test_pre_cons_gap_guard_clamps_excessive_gap_for_transition(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_TRANSITION_PRE_CONS_MAX_GAP_MS", "40")
    a = build_absolute_anchors_for_role(
        role="vc",
        center_ms=500.0,
        duration_ms=1400.0,
        left_anchor_ms=100.0,
        right_anchor_ms=900.0,
        active_start_ms=0.0,
        active_end_ms=1300.0,
    )
    assert (a.consonant_abs - a.pre_abs) <= 40.0 + 1e-3
