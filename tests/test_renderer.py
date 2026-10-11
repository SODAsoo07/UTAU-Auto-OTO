from pathlib import Path

from universal_auto_oto.domain import PlannedAlias, Slot, SourceOtoRow, Timeline
from universal_auto_oto.renderer import render_alias


def _timeline() -> Timeline:
    return Timeline(
        wav_path=Path("sample.wav"),
        wav_sha256="0" * 64,
        sample_rate=44100,
        duration_ms=2400.0,
        active_start_ms=200.0,
        active_end_ms=2200.0,
        slots=(
            Slot(0, 200.0, 1200.0, 700.0, 780.0, 0.8, 0.8),
            Slot(1, 1200.0, 2200.0, 1700.0, 1780.0, 0.8, 0.8),
        ),
        confidence=0.8,
    )


def test_cvc_vc_uses_current_transition_not_following_slot_start():
    plan = PlannedAlias(SourceOtoRow("sample.wav", "a ch4", 0), 0, 0, 1, "cvc_vc")

    row = render_alias("sample.wav", _timeline(), plan, review_threshold=0.4, format_hint="cvc")

    assert row.absolute["pre_abs"] == 700.0
    assert row.absolute["pre_abs"] < _timeline().slots[1].start_ms
    assert row.absolute["offset_abs"] == 645.0


def test_compact_cvc_keeps_all_anchors_local_to_current_transition():
    plan = PlannedAlias(SourceOtoRow("sample.wav", "ak4", 0), 0, 0, 1, "cvc")

    row = render_alias("sample.wav", _timeline(), plan, review_threshold=0.4, format_hint="cvc")

    assert row.absolute["offset_abs"] == 620.0
    assert row.absolute["pre_abs"] == 700.0
    assert row.absolute["cutoff_abs"] <= 840.0


def test_partial_profile_changes_only_selected_anchor():
    plan = PlannedAlias(SourceOtoRow("sample.wav", "a", 0), 0, 0, 1, "cv")
    baseline = render_alias("sample.wav", _timeline(), plan, review_threshold=0.4, format_hint="cv")
    profile = {"apply_fields": ["cutoff_abs"], "ratios": {"cutoff_abs": 0.2}}

    profiled = render_alias(
        "sample.wav",
        _timeline(),
        plan,
        review_threshold=0.4,
        timing_profile=profile,
        format_hint="cv",
    )

    for field in ("offset_abs", "overlap_abs", "pre_abs", "fixed_end_abs"):
        assert profiled.absolute[field] == baseline.absolute[field]
    assert profiled.absolute["cutoff_abs"] == baseline.absolute["fixed_end_abs"]
