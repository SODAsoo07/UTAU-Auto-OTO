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
