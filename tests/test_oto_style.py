from __future__ import annotations

from pathlib import Path

import pytest

from universal_auto_oto.domain import RenderedOtoRow
from universal_auto_oto.oto_style import OtoStyle, apply_oto_style, load_oto_style, save_oto_style


def _row(role: str = "cv") -> RenderedOtoRow:
    # offset 100, overlap 140, pre 180, fixed 240, cutoff (absolute) 600
    return RenderedOtoRow(
        wav_name="a.wav", alias="ka", offset=100.0, consonant=140.0, cutoff=-500.0, preutterance=80.0,
        overlap=40.0, role=role, slot_index=0, confidence=1.0, split="review", reasons=(), source_index=0,
        absolute={"offset_abs": 100.0, "overlap_abs": 140.0, "pre_abs": 180.0, "fixed_end_abs": 240.0,
                  "cutoff_abs": 600.0},
    )


def test_default_style_keeps_row() -> None:
    row = _row()
    assert apply_oto_style(row, OtoStyle()) is row
    assert apply_oto_style(row, None) is row


def test_overlap_ratio_is_share_of_preutterance() -> None:
    styled = apply_oto_style(_row(), OtoStyle(cv_overlap_ratio=0.5))
    assert styled.preutterance == 80.0
    assert styled.overlap == 40.0
    styled = apply_oto_style(_row(), OtoStyle(cv_overlap_ratio=0.25))
    assert styled.overlap == 20.0


def test_style_applies_per_alias_group() -> None:
    style = OtoStyle(vc_overlap_ratio=0.0, vc_pre_shift_ms=10.0)
    assert apply_oto_style(_row("cv"), style).preutterance == 80.0
    vc = apply_oto_style(_row("vc"), style)
    assert vc.preutterance == 90.0
    assert vc.overlap == 0.0
    assert apply_oto_style(_row("release"), style).preutterance == 80.0


def test_negative_shift_before_offset_moves_offset() -> None:
    styled = apply_oto_style(_row(), OtoStyle(cv_pre_shift_ms=-120.0))
    assert styled.offset == 60.0
    assert styled.preutterance == 0.0
    assert styled.overlap == 0.0


def test_fixed_extension_and_cutoff_margin_stay_inside_template_end() -> None:
    styled = apply_oto_style(_row(), OtoStyle(consonant_extra_ms=20.0, cutoff_margin_ms=50.0))
    assert styled.consonant == 160.0
    assert styled.cutoff == -450.0
    styled = apply_oto_style(_row(), OtoStyle(consonant_extra_ms=500.0))
    assert styled.offset + styled.consonant == 600.0
    assert -styled.cutoff + styled.offset == 600.0


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        OtoStyle(cv_overlap_ratio=1.5)
    with pytest.raises(ValueError):
        OtoStyle(cutoff_margin_ms=-1.0)
    with pytest.raises(ValueError):
        OtoStyle.from_dict({"unknown": 1})


def test_preset_round_trip(tmp_path: Path) -> None:
    style = OtoStyle(cv_overlap_ratio=0.5, vc_pre_shift_ms=-5.0)
    save_oto_style(tmp_path / "style.json", style)
    assert load_oto_style(tmp_path / "style.json") == style


def test_korean_vcv_window_keeps_pre_and_draws_hand_otos_shape():
    from universal_auto_oto.domain import RenderedOtoRow
    from universal_auto_oto.oto_style import apply_korean_vcv_window

    # Acoustic offset at the previous vowel start, 400 ms before pre.
    row = RenderedOtoRow("x.wav", "a ga", 1000.0, 450.0, -900.0, 400.0, 330.0, "vcv", 1, 0.9, "clean", (), 0, {})
    out = apply_korean_vcv_window(row)
    assert out.offset + out.preutterance == 1400.0
    assert (out.offset, out.offset + out.overlap, out.offset + out.consonant, out.offset - out.cutoff) == (
        1205.0, 1325.0, 1510.0, 1695.0)
    # A short acoustic vowel keeps its earlier end; the offset never moves earlier.
    short = apply_korean_vcv_window(RenderedOtoRow("x.wav", "a ga", 1300.0, 150.0, -200.0, 100.0, 50.0, "vcv", 1, 0.9,
                                                   "clean", (), 0, {}))
    assert short.offset == 1300.0 and short.offset - short.cutoff == 1500.0
    # Rows that are not VCV transitions are untouched.
    other = RenderedOtoRow("x.wav", "br", 0.0, 50.0, -100.0, 40.0, 10.0, "breath", 0, 0.9, "clean", (), 0, {})
    assert apply_korean_vcv_window(other) is other


def test_korean_cvc_window_by_row_kind():
    from universal_auto_oto.domain import RenderedOtoRow
    from universal_auto_oto.oto_style import apply_korean_cvc_window, korean_cvc_window_kind

    def row(alias, role):
        return RenderedOtoRow("x.wav", alias, 1000.0, 450.0, -900.0, 400.0, 330.0, role, 1, 0.9, "clean", (), 0, {})

    assert [korean_cvc_window_kind(row(a, r)) for a, r in (
        ("ga", "cv"), ("- ga", "cv_head"), ("a g", "vc"), ("ang", "vc"), ("a i", "vv"), ("n R", "release"),
        ("か", "cv"), ("br", "breath"))] == ["cv", "head", "vc", "coda", "vv", "release", None, None]
    cv = apply_korean_cvc_window(row("ga", "cv"))
    assert cv.offset + cv.preutterance == 1400.0
    assert (cv.offset, cv.offset + cv.overlap, cv.offset + cv.consonant, cv.offset - cv.cutoff) == (
        1315.0, 1352.0, 1489.0, 1632.0)
    vc = apply_korean_cvc_window(row("a g", "vc"))
    assert vc.offset == 1290.0 and vc.offset - vc.cutoff == 1476.0
