from __future__ import annotations

import os
import wave

from core.coarse_crnn.boundary_targets import build_boundary_target_map, build_phone_aware_target_map, load_row_specs_from_source_oto
from core.coarse_crnn import boundary_targets as bt
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors, BoundaryCandidate, OtoRowSpec
from core.coarse_crnn.wav_decoder import (
    _auto_anchor_timeline_mode,
    _pb_assist_role_format_term,
    _phone_aux_posterior_term,
    decode_wav_rows,
)


def _spec(role: str, *, slot_index: int, line_index: int) -> OtoRowSpec:
    return OtoRowSpec(
        wav_name="ga_gi_gu.wav",
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        alias=f"{role}_{line_index}",
        role=role,
        slot_index=slot_index,
        slot_count=3,
        prev_alias="",
        next_alias="",
        language="korean",
        format_type="cvvc",
        line_index=line_index,
        duration_ms=1500.0,
        source_params={},
    )


def _write_wav(path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)


def test_load_row_specs_ignores_source_param_values(tmp_path):
    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    wav_path = wav_dir / "a.wav"
    _write_wav(wav_path)
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text("a.wav=a,123,456,-789,321,111\n", encoding="utf-8")
    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=str(source_oto),
        wav_dir=str(wav_dir),
        language="japanese",
        format_type="cvvc",
    )
    assert len(rows_by_wav) == 1
    specs = next(iter(rows_by_wav.values()))
    assert len(specs) == 1
    assert specs[0].source_params == {}


def test_boundary_target_map_contains_vc_labels():
    spec = _spec("vc", slot_index=0, line_index=0)
    anchors = AbsoluteOtoAnchors(
        offset_abs=140.0,
        overlap_abs=165.0,
        pre_abs=190.0,
        consonant_abs=230.0,
        cutoff_abs=320.0,
    )
    _times, target = build_boundary_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    assert target.shape == (64, 8)
    # vc row should drive vowel_end / transition / next_onset labels.
    assert float(target[:, 4].max()) > 0.5
    assert float(target[:, 5].max()) > 0.5
    assert float(target[:, 6].max()) > 0.5


def test_phone_aware_target_map_marks_vc_transition_identity():
    spec = OtoRowSpec(
        wav_name="ga_gi.wav",
        wav_path=r"C:\tmp\ga_gi.wav",
        alias="a g",
        role="vc",
        slot_index=0,
        slot_count=2,
        prev_alias="",
        next_alias="",
        language="korean",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={"left_vowel": "a", "right_consonant": "g"},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=120.0,
        overlap_abs=150.0,
        pre_abs=190.0,
        consonant_abs=230.0,
        cutoff_abs=320.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    assert int(phone.cvs_mask.sum()) > 0
    assert int(phone.consonant_mask.sum()) > 0
    assert int(phone.vowel_mask.sum()) > 0
    assert int(phone.consonant_family_mask.sum()) > 0
    assert int(phone.vowel_nucleus_mask.sum()) > 0
    assert int(phone.vowel_glide_mask.sum()) > 0
    from core.coarse_crnn.boundary_types import (
        PHONE_AWARE_CONSONANT_FAMILY_LABELS,
        PHONE_AWARE_VOWEL_GLIDE_LABELS,
        PHONE_AWARE_VOWEL_NUCLEUS_LABELS,
    )

    assert PHONE_AWARE_CONSONANT_FAMILY_LABELS[int(phone.consonant_family_target[23])] == "plosive"
    assert PHONE_AWARE_VOWEL_NUCLEUS_LABELS[int(phone.vowel_nucleus_target[18])] == "a"
    assert PHONE_AWARE_VOWEL_GLIDE_LABELS[int(phone.vowel_glide_target[18])] == "none"


def test_phone_aware_target_map_splits_korean_cv_syllable_identity():
    spec = OtoRowSpec(
        wav_name="bwa.wav",
        wav_path=r"C:\tmp\bwa.wav",
        alias="bwa",
        role="cv",
        slot_index=0,
        slot_count=1,
        prev_alias="",
        next_alias="",
        language="korean",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=120.0,
        overlap_abs=140.0,
        pre_abs=180.0,
        consonant_abs=230.0,
        cutoff_abs=340.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    assert int(phone.consonant_mask.sum()) > 0
    assert int(phone.vowel_mask.sum()) > 0
    from core.coarse_crnn.boundary_types import PHONE_AWARE_CONSONANT_LABELS, PHONE_AWARE_VOWEL_LABELS

    assert PHONE_AWARE_CONSONANT_LABELS[int(phone.consonant_target[12])] == "b"
    assert PHONE_AWARE_VOWEL_LABELS[int(phone.vowel_target[18])] == "wa"


def test_phone_aware_target_map_covers_foreign_transition_consonants():
    spec = OtoRowSpec(
        wav_name="a_v.wav",
        wav_path=r"C:\tmp\a_v.wav",
        alias="a v",
        role="v-cv",
        slot_index=0,
        slot_count=2,
        prev_alias="a",
        next_alias="va",
        language="korean",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=100.0,
        overlap_abs=140.0,
        pre_abs=180.0,
        consonant_abs=240.0,
        cutoff_abs=340.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    from core.coarse_crnn.boundary_types import PHONE_AWARE_CONSONANT_LABELS

    assert int(phone.consonant_mask.sum()) > 0
    assert PHONE_AWARE_CONSONANT_LABELS[int(phone.consonant_target[24])] == "v"


def test_phone_aware_target_map_normalizes_eui_vowel_identity():
    spec = OtoRowSpec(
        wav_name="beui.wav",
        wav_path=r"C:\tmp\beui.wav",
        alias="beui",
        role="cv",
        slot_index=0,
        slot_count=1,
        prev_alias="",
        next_alias="",
        language="korean",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=120.0,
        overlap_abs=140.0,
        pre_abs=180.0,
        consonant_abs=230.0,
        cutoff_abs=340.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    from core.coarse_crnn.boundary_types import PHONE_AWARE_VOWEL_LABELS

    assert int(phone.vowel_mask.sum()) > 0
    assert PHONE_AWARE_VOWEL_LABELS[int(phone.vowel_target[18])] == "ui"


def test_phone_aware_identity_defaults_to_korean_and_japanese():
    spec = OtoRowSpec(
        wav_name="a_ka.wav",
        wav_path=r"C:\tmp\a_ka.wav",
        alias="a kC4P",
        role="vc",
        slot_index=0,
        slot_count=2,
        prev_alias="a",
        next_alias="ka",
        language="japanese",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={"left_vowel": "a", "right_consonant": "k"},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=100.0,
        overlap_abs=140.0,
        pre_abs=180.0,
        consonant_abs=240.0,
        cutoff_abs=340.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    assert int(phone.cvs_mask.sum()) > 0
    assert int(phone.consonant_mask.sum()) > 0


def test_phone_aware_identity_language_gate_can_restrict_japanese(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS", "korean")
    spec = OtoRowSpec(
        wav_name="a_ka.wav",
        wav_path=r"C:\tmp\a_ka.wav",
        alias="a kC4P",
        role="vc",
        slot_index=0,
        slot_count=2,
        prev_alias="a",
        next_alias="ka",
        language="japanese",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=100.0,
        overlap_abs=140.0,
        pre_abs=180.0,
        consonant_abs=240.0,
        cutoff_abs=340.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    assert int(phone.cvs_mask.sum()) > 0
    assert int(phone.consonant_mask.sum()) == 0


def test_phone_aware_japanese_pitch_suffix_kana_cv_identity():
    spec = OtoRowSpec(
        wav_name="ki.wav",
        wav_path=r"C:\tmp\ki.wav",
        alias="きC4P",
        role="cv",
        slot_index=0,
        slot_count=1,
        prev_alias="",
        next_alias="",
        language="japanese",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=120.0,
        overlap_abs=140.0,
        pre_abs=180.0,
        consonant_abs=230.0,
        cutoff_abs=340.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    from core.coarse_crnn.boundary_types import PHONE_AWARE_CONSONANT_LABELS, PHONE_AWARE_VOWEL_LABELS

    assert int(phone.consonant_mask.sum()) > 0
    assert int(phone.vowel_mask.sum()) > 0
    assert PHONE_AWARE_CONSONANT_LABELS[int(phone.consonant_target[12])] == "k"
    assert PHONE_AWARE_VOWEL_LABELS[int(phone.vowel_target[18])] == "i"


def test_phone_aware_japanese_pitch_suffix_palatalized_identity():
    spec = OtoRowSpec(
        wav_name="kyu.wav",
        wav_path=r"C:\tmp\kyu.wav",
        alias="きゅC4P",
        role="cv",
        slot_index=0,
        slot_count=1,
        prev_alias="",
        next_alias="",
        language="japanese",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=120.0,
        overlap_abs=140.0,
        pre_abs=180.0,
        consonant_abs=230.0,
        cutoff_abs=340.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    from core.coarse_crnn.boundary_types import PHONE_AWARE_CONSONANT_LABELS, PHONE_AWARE_VOWEL_LABELS

    assert int(phone.consonant_mask.sum()) > 0
    assert int(phone.vowel_mask.sum()) > 0
    assert PHONE_AWARE_CONSONANT_LABELS[int(phone.consonant_target[12])] == "ky"
    assert PHONE_AWARE_VOWEL_LABELS[int(phone.vowel_target[18])] == "u"


def test_phone_aware_pitch_suffix_strips_for_all_identity_languages(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS", "all")
    spec = OtoRowSpec(
        wav_name="a_b.wav",
        wav_path=r"C:\tmp\a_b.wav",
        alias="a bC4P",
        role="vc",
        slot_index=0,
        slot_count=1,
        prev_alias="a",
        next_alias="b",
        language="english",
        format_type="cvvc",
        line_index=0,
        duration_ms=500.0,
        source_params={},
        meta={},
    )
    anchors = AbsoluteOtoAnchors(
        offset_abs=100.0,
        overlap_abs=140.0,
        pre_abs=180.0,
        consonant_abs=240.0,
        cutoff_abs=340.0,
    )
    _times, phone = build_phone_aware_target_map([(spec, anchors)], duration_ms=500.0, hop_ms=10.0, frame_count=64)
    from core.coarse_crnn.boundary_types import PHONE_AWARE_CONSONANT_LABELS

    assert int(phone.consonant_mask.sum()) > 0
    painted = {PHONE_AWARE_CONSONANT_LABELS[int(idx)] for idx in phone.consonant_target[phone.consonant_mask > 0]}
    assert "b" in painted


def test_boundary_scorer_phone_aux_heads_are_optional():
    torch = __import__("torch")
    from core.coarse_crnn.boundary_scorer_model import BoundaryScorerConfig, build_boundary_scorer

    off = build_boundary_scorer(BoundaryScorerConfig(enable_phone_aux_heads=False))
    off_out = off(torch.randn(2, 12, 64))
    assert off_out["cvs_logits"] is None

    on = build_boundary_scorer(BoundaryScorerConfig(enable_phone_aux_heads=True))
    on_out = on(torch.randn(2, 12, 64))
    assert on_out["cvs_logits"].shape[:2] == (2, 12)
    assert on_out["consonant_logits"].shape[:2] == (2, 12)
    assert on_out["vowel_logits"].shape[:2] == (2, 12)
    assert on_out["consonant_family_logits"] is None

    family = build_boundary_scorer(BoundaryScorerConfig(enable_phone_aux_heads=True, enable_phone_family_heads=True))
    family_out = family(torch.randn(2, 12, 64))
    assert family_out["consonant_family_logits"].shape[:2] == (2, 12)
    assert family_out["vowel_nucleus_logits"].shape[:2] == (2, 12)
    assert family_out["vowel_glide_logits"].shape[:2] == (2, 12)


def test_phone_aware_debug_plot_renders_target_only(tmp_path):
    import pytest

    pytest.importorskip("matplotlib")

    from ml.scripts.coarse_crnn import plot_phone_aware_debug

    wav_path = tmp_path / "ba.wav"
    _write_wav(wav_path)
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "audio": str(wav_path),
            "wav": wav_path.name,
            "alias": "ba",
            "alias_role": "cv",
            "format_type": "cvvc",
            "language": "korean",
            "line_index": 0,
            "duration_ms": 500.0,
            "target_offset_ms": 40.0,
            "target_preutterance_ms": 40.0,
            "target_consonant_ms": 80.0,
            "target_overlap_ms": 20.0,
            "target_cutoff": -140.0,
        },
        {
            "audio": str(wav_path),
            "wav": wav_path.name,
            "alias": "a b",
            "alias_role": "vc",
            "format_type": "cvvc",
            "language": "korean",
            "line_index": 1,
            "duration_ms": 500.0,
            "target_offset_ms": 160.0,
            "target_preutterance_ms": 30.0,
            "target_consonant_ms": 70.0,
            "target_overlap_ms": 10.0,
            "target_cutoff": -160.0,
        },
        {
            "audio": str(wav_path),
            "wav": wav_path.name,
            "alias": "be",
            "alias_role": "cv",
            "format_type": "cvvc",
            "language": "korean",
            "line_index": 2,
            "duration_ms": 500.0,
            "target_offset_ms": 300.0,
            "target_preutterance_ms": 40.0,
            "target_consonant_ms": 80.0,
            "target_overlap_ms": 20.0,
            "target_cutoff": -140.0,
        },
    ]
    manifest.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")
    out = tmp_path / "debug.png"
    rc = plot_phone_aware_debug.main(
        [
            "--manifest",
            str(manifest),
            "--wav",
            str(wav_path),
            "--alias",
            "a b",
            "--line-index",
            "1",
            "--out",
            str(out),
            "--dpi",
            "72",
        ]
    )
    assert rc == 0
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_wav_decoder_keeps_vc_inside_anchor_pair():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.93, source="model"),
        BoundaryCandidate(time_ms=420.0, kind="transition_peak", score=0.90, source="model"),
        BoundaryCandidate(time_ms=455.0, kind="next_onset", score=0.82, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert len(decoded.rows) == 3
    left = decoded.anchor_timeline_ms[0]
    right = decoded.anchor_timeline_ms[1]
    vc_row = decoded.rows[1]
    assert left - 28.0 <= vc_row.selected_time_ms <= right + 42.0


def test_wav_decoder_vc_prefers_late_transition_to_reduce_front_v():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=320.0, kind="transition_peak", score=0.92, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="transition_peak", score=0.92, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert len(decoded.rows) == 3
    vc_row = decoded.rows[1]
    # VC target is intentionally right-leaning to avoid over-long preceding vowel.
    assert float(vc_row.selected_time_ms) >= 480.0


def test_wav_decoder_derives_internal_vc_from_neighbor_primary_rows(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_DERIVE_INTERNAL_VC_FROM_PRIMARY", "1")
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="transition_peak", score=0.92, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    prev_cv, vc_row, next_cv = decoded.rows
    assert vc_row.reason == "derived:internal_vc_primary_pair"
    assert vc_row.fallback_used is False
    assert float(vc_row.anchors.offset_abs) == float(prev_cv.anchors.cutoff_abs)
    assert float(vc_row.anchors.cutoff_abs) == float(next_cv.anchors.offset_abs)
    assert float(vc_row.anchors.offset_abs) < float(vc_row.anchors.overlap_abs)
    assert float(vc_row.anchors.overlap_abs) <= float(vc_row.anchors.pre_abs)
    assert float(vc_row.anchors.pre_abs) <= float(vc_row.anchors.consonant_abs)
    assert float(vc_row.anchors.consonant_abs) < float(vc_row.anchors.cutoff_abs)


def test_wav_decoder_keeps_internal_vc_derivation_disabled_by_default():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="transition_peak", score=0.92, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert decoded.rows[1].reason != "derived:internal_vc_primary_pair"


def test_wav_decoder_keeps_final_vc_on_existing_policy():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="transition_peak", score=0.92, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert decoded.rows[1].reason != "derived:internal_vc_primary_pair"


def test_wav_decoder_keeps_vc_before_cv_tail_on_existing_policy():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv-", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="transition_peak", score=0.92, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert decoded.rows[1].reason != "derived:internal_vc_primary_pair"


def test_wav_decoder_vc_penalizes_early_vowel_end_even_when_score_is_higher():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("vc", slot_index=0, line_index=1),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.95, source="model"),
        # Early vowel_end has a higher confidence score, but should not win for VC.
        BoundaryCandidate(time_ms=360.0, kind="vowel_end", score=0.99, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="transition_peak", score=0.90, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert len(decoded.rows) == 3
    vc_row = decoded.rows[1]
    assert float(vc_row.selected_time_ms) >= 500.0


def test_wav_decoder_uses_posterior_channel_for_n_like_transition(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_POSTERIOR_CONS_WEIGHT", "2.0")
    monkeypatch.setenv("UTOA_BOUNDARY_POSTERIOR_VOWEL_WEIGHT", "1.0")
    monkeypatch.setenv("UTOA_BOUNDARY_NLIKE_POSTERIOR_CONS_BONUS", "0.8")
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        OtoRowSpec(
            wav_name="ga_gi_gu.wav",
            wav_path=r"C:\tmp\ga_gi_gu.wav",
            alias="a n",
            role="vc",
            slot_index=0,
            slot_count=3,
            prev_alias="a",
            next_alias="na",
            language="korean",
            format_type="cvvc",
            line_index=1,
            duration_ms=1500.0,
            source_params={},
            meta={"right_consonant": "n"},
        ),
        _spec("cv", slot_index=1, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=200.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.95, source="model"),
        # Raw score favors earlier vowel_end.
        BoundaryCandidate(time_ms=430.0, kind="vowel_end", score=0.99, source="model"),
        BoundaryCandidate(time_ms=520.0, kind="next_onset", score=0.70, source="model"),
    ]
    times = [360.0, 430.0, 500.0, 520.0, 560.0]
    posterior = {
        "consonant_onset": [0.10, 0.20, 0.82, 0.90, 0.50],
        "next_onset": [0.12, 0.20, 0.86, 0.95, 0.62],
        "vowel_start": [0.76, 0.78, 0.30, 0.22, 0.20],
        "vowel_stable": [0.70, 0.74, 0.25, 0.20, 0.18],
        "vowel_end": [0.84, 0.90, 0.32, 0.26, 0.20],
        "transition_peak": [0.30, 0.40, 0.72, 0.80, 0.52],
    }
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\ga_gi_gu.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
        posterior_scores=posterior,
        posterior_times_ms=times,
    )
    vc_row = decoded.rows[1]
    assert float(vc_row.selected_time_ms) >= 500.0


def test_phone_aux_posterior_uses_cvs_but_gates_identity_by_default(monkeypatch):
    monkeypatch.delenv("UTOA_BOUNDARY_PHONE_AUX_ID_WEIGHT", raising=False)
    term = _phone_aux_posterior_term(
        role="vc",
        time_ms=100.0,
        left_anchor=60.0,
        right_anchor=160.0,
        cvs_scores={"consonant": [0.90], "vowel": [0.10]},
        consonant_scores={"b": [0.95]},
        vowel_scores={"a": [0.95]},
        posterior_times_ms=[100.0],
        row_meta={"right_consonant": "b", "left_vowel": "a"},
        n_like=False,
        weak_audio=False,
    )
    assert term > 0.0

    identity_only = _phone_aux_posterior_term(
        role="vc",
        time_ms=100.0,
        left_anchor=60.0,
        right_anchor=160.0,
        cvs_scores={"consonant": [0.0], "vowel": [0.0]},
        consonant_scores={"b": [0.95]},
        vowel_scores={"a": [0.95]},
        posterior_times_ms=[100.0],
        row_meta={"right_consonant": "b", "left_vowel": "a"},
        n_like=False,
        weak_audio=False,
    )
    assert identity_only == 0.0

    monkeypatch.setenv("UTOA_BOUNDARY_PHONE_AUX_ID_WEIGHT", "1.0")
    identity_enabled = _phone_aux_posterior_term(
        role="vc",
        time_ms=100.0,
        left_anchor=60.0,
        right_anchor=160.0,
        cvs_scores={"consonant": [0.0], "vowel": [0.0]},
        consonant_scores={"b": [0.95]},
        vowel_scores={"a": [0.95]},
        posterior_times_ms=[100.0],
        row_meta={"right_consonant": "b", "left_vowel": "a"},
        n_like=False,
        weak_audio=False,
    )
    assert identity_enabled > 0.0


def test_phone_aux_posterior_uses_family_head_by_default(monkeypatch):
    monkeypatch.delenv("UTOA_BOUNDARY_PHONE_AUX_ID_WEIGHT", raising=False)
    monkeypatch.setenv("UTOA_BOUNDARY_PHONE_AUX_WEIGHT", "1.0")
    monkeypatch.setenv("UTOA_BOUNDARY_PHONE_AUX_CVS_WEIGHT", "0.0")
    monkeypatch.setenv("UTOA_BOUNDARY_PHONE_AUX_FAMILY_WEIGHT", "1.0")
    term = _phone_aux_posterior_term(
        role="vc",
        time_ms=100.0,
        left_anchor=60.0,
        right_anchor=160.0,
        cvs_scores={"consonant": [0.0], "vowel": [0.0]},
        consonant_scores={"b": [0.95]},
        vowel_scores={"a": [0.95]},
        consonant_family_scores={"plosive": [0.95]},
        vowel_nucleus_scores={"a": [0.80]},
        vowel_glide_scores={"none": [0.70]},
        posterior_times_ms=[100.0],
        row_meta={"right_consonant": "b", "left_vowel": "a"},
        n_like=False,
        weak_audio=False,
    )
    assert term > 0.0


def test_pb_assist_guard_allows_cvvc_transition_and_blocks_v_vcv(monkeypatch):
    monkeypatch.delenv("UTOA_BOUNDARY_PB_ASSIST_FORMATS", raising=False)
    monkeypatch.delenv("UTOA_BOUNDARY_PB_ASSIST_ROLES", raising=False)

    assert _pb_assist_role_format_term(
        role="vc",
        format_type="cvvc",
        source="merged:model:phoneme_boundary:vowel_end+audio",
    ) > 0.0
    assert _pb_assist_role_format_term(
        role="v",
        format_type="cvvc",
        source="merged:model:phoneme_boundary:vowel_onset+audio",
    ) < 0.0
    assert _pb_assist_role_format_term(
        role="vc",
        format_type="vcv",
        source="merged:model:phoneme_boundary:vowel_end+audio",
    ) < 0.0


def test_pb_assist_guard_can_be_overridden(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_PB_ASSIST_FORMATS", "all")
    monkeypatch.setenv("UTOA_BOUNDARY_PB_ASSIST_ROLES", "v")

    assert _pb_assist_role_format_term(
        role="v",
        format_type="vcv",
        source="model:phoneme_boundary:vowel_onset",
    ) > 0.0


def test_wav_decoder_right_guard_limits_cutoff_for_dense_cv():
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
    ]
    cands = [
        BoundaryCandidate(time_ms=210.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=280.0, kind="syllable_onset", score=0.96, source="model"),
        BoundaryCandidate(time_ms=214.0, kind="vowel_start", score=0.90, source="model"),
        BoundaryCandidate(time_ms=284.0, kind="vowel_start", score=0.92, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\dense_cv.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=980.0,
    )
    assert len(decoded.rows) == 2
    right_anchor = float(decoded.anchor_timeline_ms[1])
    row0 = decoded.rows[0]
    # cutoff must stay before the next slot boundary (guarded by right-anchor margin).
    assert float(row0.anchors.cutoff_abs) <= right_anchor - 4.0


def test_wav_decoder_ignores_source_params():
    rows = [
        _spec("vc", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
    ]
    rows_with_bad_source = [
        OtoRowSpec(**{**rows[0].__dict__, "source_params": {"offset": 999.0, "preutterance": 888.0, "consonant": 777.0, "cutoff": -5.0, "overlap": 666.0}}),
        rows[1],
    ]
    cands = [
        BoundaryCandidate(time_ms=220.0, kind="syllable_onset", score=0.95, source="model"),
        BoundaryCandidate(time_ms=340.0, kind="syllable_onset", score=0.96, source="model"),
        BoundaryCandidate(time_ms=280.0, kind="transition_peak", score=0.92, source="model"),
        BoundaryCandidate(time_ms=300.0, kind="next_onset", score=0.90, source="model"),
    ]
    decoded_plain = decode_wav_rows(
        wav_path=r"C:\tmp\ignore_source_plain.wav",
        duration_ms=1000.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=100.0,
        active_end_ms=900.0,
    )
    decoded_bad = decode_wav_rows(
        wav_path=r"C:\tmp\ignore_source_bad.wav",
        duration_ms=1000.0,
        row_specs=rows_with_bad_source,
        candidates=cands,
        active_start_ms=100.0,
        active_end_ms=900.0,
    )
    assert len(decoded_plain.rows) == len(decoded_bad.rows) == 2
    assert float(decoded_plain.rows[0].selected_time_ms) == float(decoded_bad.rows[0].selected_time_ms)
    assert float(decoded_plain.rows[0].anchors.pre_abs) == float(decoded_bad.rows[0].anchors.pre_abs)
    assert float(decoded_plain.rows[0].anchors.cutoff_abs) == float(decoded_bad.rows[0].anchors.cutoff_abs)


def test_wav_decoder_active_end_pad_extends_anchor_timeline(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ACTIVE_END_PAD_MS", "200")
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
        _spec("cv", slot_index=2, line_index=2),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\pad.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=[],
        active_start_ms=100.0,
        active_end_ms=700.0,
    )
    assert len(decoded.anchor_timeline_ms) == 3
    assert decoded.anchor_timeline_ms[-1] > 700.0


def test_cv_slot_segmenter_assigns_filename_order_without_phone_identity():
    from core.coarse_crnn.cv_slot_segmenter import segment_cv_slots

    cands = [
        BoundaryCandidate(time_ms=205.0, kind="syllable_onset", score=0.91, source="model"),
        BoundaryCandidate(time_ms=240.0, kind="vowel_start", score=0.72, source="model"),
        BoundaryCandidate(time_ms=505.0, kind="consonant_onset", score=0.88, source="model"),
        BoundaryCandidate(time_ms=545.0, kind="vowel_stable", score=0.66, source="model"),
        BoundaryCandidate(time_ms=810.0, kind="syllable_onset", score=0.93, source="model"),
        BoundaryCandidate(time_ms=850.0, kind="vowel_start", score=0.71, source="model"),
    ]
    slots = segment_cv_slots(
        wav_name="slot_order.wav",
        slot_count=3,
        duration_ms=1000.0,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=900.0,
    )
    assert len(slots) == 3
    assert [slot.index for slot in slots] == [0, 1, 2]
    assert [slot.start_ms for slot in slots] == sorted(slot.start_ms for slot in slots)
    assert abs(slots[0].start_ms - 205.0) <= 1.0
    assert abs(slots[1].start_ms - 505.0) <= 1.0
    assert abs(slots[2].start_ms - 810.0) <= 1.0


def test_wav_decoder_cv_slot_timeline_mode_uses_cv_segment_starts(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_TIMELINE", "cv_slot")
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
        _spec("cv", slot_index=2, line_index=2),
    ]
    cands = [
        BoundaryCandidate(time_ms=205.0, kind="syllable_onset", score=0.91, source="model"),
        BoundaryCandidate(time_ms=505.0, kind="consonant_onset", score=0.88, source="model"),
        BoundaryCandidate(time_ms=810.0, kind="syllable_onset", score=0.93, source="model"),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\cv_slot.wav",
        duration_ms=1000.0,
        row_specs=rows,
        candidates=cands,
        active_start_ms=120.0,
        active_end_ms=900.0,
    )
    assert len(decoded.anchor_timeline_ms) == 3
    assert abs(decoded.anchor_timeline_ms[0] - 205.0) <= 1.0
    assert abs(decoded.anchor_timeline_ms[1] - 505.0) <= 1.0
    assert abs(decoded.anchor_timeline_ms[2] - 810.0) <= 1.0


def test_wav_decoder_auto_timeline_policy_is_format_scoped(monkeypatch):
    monkeypatch.delenv("UTOA_BOUNDARY_ANCHOR_TIMELINE_AUTO_KOREAN_CV", raising=False)
    ko_cv = OtoRowSpec(**{**_spec("cv", slot_index=0, line_index=0).__dict__, "language": "korean", "format_type": "cv"})
    jp_cv = OtoRowSpec(**{**_spec("cv", slot_index=0, line_index=0).__dict__, "language": "japanese", "format_type": "cv"})
    jp_vcv = OtoRowSpec(**{**_spec("v-cv", slot_index=0, line_index=0).__dict__, "language": "japanese", "format_type": "vcv"})
    jp_cvvc = OtoRowSpec(**{**_spec("vc", slot_index=0, line_index=0).__dict__, "language": "japanese", "format_type": "cvvc"})

    assert _auto_anchor_timeline_mode(row_specs=[ko_cv]) == "filename"
    assert _auto_anchor_timeline_mode(row_specs=[jp_cv]) == "hybrid"
    assert _auto_anchor_timeline_mode(row_specs=[jp_vcv]) == "linear"
    assert _auto_anchor_timeline_mode(row_specs=[jp_cvvc]) == "hybrid"


def test_wav_decoder_auto_timeline_policy_has_env_override(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_TIMELINE_AUTO_KOREAN_CV", "cv_slot")
    ko_cv = OtoRowSpec(**{**_spec("cv", slot_index=0, line_index=0).__dict__, "language": "korean", "format_type": "cv"})
    assert _auto_anchor_timeline_mode(row_specs=[ko_cv]) == "cv_slot"


def test_wav_decoder_anchor_shift_moves_timeline_forward(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_SHIFT_MS", "120")
    rows = [
        _spec("cv", slot_index=0, line_index=0),
        _spec("cv", slot_index=1, line_index=1),
        _spec("cv", slot_index=2, line_index=2),
    ]
    decoded = decode_wav_rows(
        wav_path=r"C:\tmp\shift.wav",
        duration_ms=1200.0,
        row_specs=rows,
        candidates=[],
        active_start_ms=100.0,
        active_end_ms=700.0,
    )
    assert len(decoded.anchor_timeline_ms) == 3
    assert decoded.anchor_timeline_ms[0] >= 220.0


def test_predictor_dispatch_prefers_boundary_engine(monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as mod

    called = {"boundary": 0}

    def _fake_boundary(**kwargs):
        called["boundary"] += 1
        return 1, 1, []

    monkeypatch.setattr(mod, "generate_oto_with_boundary_decoder", _fake_boundary)

    processed, total, errors = mod.generate_oto_with_crnn_predictor(
        wav_dir=".",
        out_path=".",
        source_oto_path=".",
        language="korean",
        engine="boundary_decoder",
    )
    assert (processed, total, errors) == (1, 1, [])
    assert called["boundary"] == 1


def test_predictor_dispatch_stage1_heuristic_disables_stage2(monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as mod

    captured = {}

    def _fake_boundary(**kwargs):
        captured.update(kwargs)
        return 1, 1, []

    monkeypatch.setattr(mod, "generate_oto_with_boundary_decoder", _fake_boundary)
    monkeypatch.setattr(mod, "resolve_boundary_scorer_model_path", lambda _path="": "boundary.pt")

    processed, total, errors = mod.generate_oto_with_crnn_predictor(
        wav_dir=".",
        out_path=".",
        source_oto_path=".",
        language="korean",
        engine="stage1_heuristic",
        stage2_model_path="stage2.pt",
        stage2_enable=True,
    )
    assert (processed, total, errors) == (1, 1, [])
    assert captured["heuristic_only"] is True
    assert captured["stage2_enable"] is False


def test_predictor_dispatch_rejects_legacy_direct_engine():
    from core.coarse_crnn import oto_predictor_generator as mod

    processed, total, errors = mod.generate_oto_with_crnn_predictor(
        wav_dir=".",
        out_path=".",
        source_oto_path=".",
        language="korean",
        engine="direct",
    )
    assert processed == 0
    assert total == 0
    assert errors
    assert "폐기" in errors[0]


def test_boundary_model_resolver_skips_experimental_auto_pick(tmp_path, monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as mod

    stable = tmp_path / "oto_boundary_scorer_v5.pt"
    phone_aux = tmp_path / "oto_boundary_scorer_phone_aux.pt"
    stable.write_bytes(b"stable")
    phone_aux.write_bytes(b"phone")
    os.utime(stable, (1000, 1000))
    os.utime(phone_aux, (2000, 2000))
    monkeypatch.setattr(mod, "_boundary_model_roots", lambda: [str(tmp_path)])
    monkeypatch.delenv("UTOA_BOUNDARY_SCORER_ALLOW_EXPERIMENTAL", raising=False)
    assert mod.resolve_boundary_scorer_model_path() == str(stable.resolve())

    monkeypatch.setenv("UTOA_BOUNDARY_SCORER_ALLOW_EXPERIMENTAL", "1")
    assert mod.resolve_boundary_scorer_model_path() == str(phone_aux.resolve())
    monkeypatch.delenv("UTOA_BOUNDARY_SCORER_ALLOW_EXPERIMENTAL", raising=False)
    assert mod.resolve_boundary_scorer_model_path(str(phone_aux)) == str(phone_aux.resolve())


def test_slot_count_fallback_uses_anchor_roles_when_filename_slots_missing():
    slot_count = bt._resolve_slot_count(
        filename_slot_count=1,
        row_roles=["cv", "vc", "cv", "vc", "other"],
        row_count=5,
    )
    assert slot_count >= 2


def test_slot_count_fallback_handles_vv_chain_when_filename_slots_missing():
    slot_count = bt._resolve_slot_count(
        filename_slot_count=1,
        row_roles=["v", "vv", "vv", "vv", "vv", "vv", "vv"],
        row_count=7,
    )
    assert slot_count >= 6


def test_slot_count_fallback_handles_vcv_other_mix_when_filename_slots_missing():
    slot_count = bt._resolve_slot_count(
        filename_slot_count=1,
        row_roles=["other", "v-cv", "other", "v-cv", "other", "v-cv", "other", "v-cv", "other"],
        row_count=9,
    )
    assert slot_count >= 5


def test_project_row_index_to_slot_spreads_sparse_rows():
    projected = [
        bt._project_row_index_to_slot(row_index=i, row_count=4, slot_count=8)
        for i in range(4)
    ]
    assert projected == [0, 2, 5, 7]


def test_project_row_index_to_slot_is_monotonic():
    projected = [
        bt._project_row_index_to_slot(row_index=i, row_count=10, slot_count=8)
        for i in range(10)
    ]
    assert projected[0] == 0
    assert projected[-1] == 7
    assert projected == sorted(projected)


def test_low_anchor_mode_projects_slot_indices_monotonic(tmp_path):
    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    wav_path = wav_dir / "_chain.wav"
    _write_wav(wav_path)
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text(
        "\n".join(
            [
                "_chain.wav=あ,0,100,-200,80,40",
                "_chain.wav=a あ,0,100,-200,80,40",
                "_chain.wav=a い,0,100,-200,80,40",
                "_chain.wav=i あ,0,100,-200,80,40",
                "_chain.wav=a う,0,100,-200,80,40",
                "_chain.wav=u え,0,100,-200,80,40",
                "_chain.wav=e あ,0,100,-200,80,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=str(source_oto),
        wav_dir=str(wav_dir),
        language="japanese",
        format_type="cvvc",
    )
    assert len(rows_by_wav) == 1
    specs = next(iter(rows_by_wav.values()))
    assert len(specs) == 7
    assert int(specs[0].slot_count) > 1
    unique_slots = sorted({int(spec.slot_index) for spec in specs})
    assert len(unique_slots) > 2


def test_low_anchor_mode_shifts_korean_coda_bridge_rows_left_when_filename_tokens_exist(tmp_path):
    wav_dir = tmp_path / "bank"
    wav_dir.mkdir()
    wav_name = "_ssung'ssun'ssum'ssul'ssu'sseu'ssa'sseu.wav"
    wav_path = wav_dir / wav_name
    _write_wav(wav_path)
    source_oto = wav_dir / "baseoto.ini"
    source_oto.write_text(
        "\n".join(
            [
                f"{wav_name}=- ssw,0,100,-200,80,40",
                f"{wav_name}=NG ssw,0,100,-200,80,40",
                f"{wav_name}=N ssw,0,100,-200,80,40",
                f"{wav_name}=M ssw,0,100,-200,80,40",
                f"{wav_name}=L ssw,0,100,-200,80,40",
                f"{wav_name}=u ss,0,100,-200,80,40",
                f"{wav_name}=a ss,0,100,-200,80,40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=str(source_oto),
        wav_dir=str(wav_dir),
        language="korean",
        format_type="cvvc",
    )
    assert len(rows_by_wav) == 1
    specs = next(iter(rows_by_wav.values()))
    assert len(specs) == 7
    # base projection for row_count=7, slot_count=8 is [0,1,2,4,5,6,7].
    # KO coda-bridge transition rows (NG/N/M/L + onset) should shift left by
    # one slot even when filename tokens exist, while plain VC rows stay put.
    assert [int(spec.slot_index) for spec in specs] == [0, 0, 1, 3, 4, 6, 7]


def test_japanese_vowel_space_alias_is_not_other():
    alias_type = bt._infer_alias_type("a あ", language="japanese")
    transition_type = bt._infer_transition_type("a あ", language="japanese")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "japanese",
            "a あ",
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )
    assert role in {"vv", "v-cv"}


def test_korean_v_plus_c_space_alias_is_vc():
    alias = "a g"
    alias_type = bt._infer_alias_type(alias, language="korean")
    transition_type = bt._infer_transition_type(alias, language="korean")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "korean",
            alias,
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )
    assert alias_type == "vc"
    assert transition_type == "vc"
    assert role == "vc"


def test_korean_coda_bridge_alias_routes_to_vc_by_default(monkeypatch):
    """KO coda-bridge aliases (``NG g``/``L d``) route to 'vc' by default
    after 2026-05-17: the old 'v-cv' lane contributed 31% of total |Δ| in
    raw-model output. Compound-token slot matcher handles their slot mapping
    now. Set ``UTOA_BOUNDARY_KO_CODA_BRIDGE_ROLE_VCV=on`` to restore the
    legacy lane for diagnostic comparison."""
    monkeypatch.delenv("UTOA_BOUNDARY_KO_CODA_BRIDGE_ROLE_VCV", raising=False)
    alias = "NG g"
    alias_type = bt._infer_alias_type(alias, language="korean")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "korean",
            alias,
            alias_type=alias_type,
            is_special=False,
        )
    )
    assert alias_type == "vc"
    assert role == "vc"


def test_korean_coda_bridge_alias_opt_in_vcv(monkeypatch):
    """Opt-in env restores legacy v-cv routing for coda-bridge."""
    monkeypatch.setenv("UTOA_BOUNDARY_KO_CODA_BRIDGE_ROLE_VCV", "on")
    alias = "NG g"
    alias_type = bt._infer_alias_type(alias, language="korean")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "korean",
            alias,
            alias_type=alias_type,
            is_special=False,
        )
    )
    assert alias_type == "vcv"
    assert role == "v-cv"


def test_normalize_ko_coda_uppercase_to_lowercase():
    """P1-a + P4-a: KO uppercase coda tokens in 2-token aliases normalize
    to lowercase so the 377 uppercase coda-bridge rows merge into the
    13.7k lowercase supervision lane (2026-05-17 wall analysis)."""
    from core.coarse_crnn.boundary_targets import normalize_ko_coda_in_alias
    assert normalize_ko_coda_in_alias("NG g") == "ng g"
    assert normalize_ko_coda_in_alias("L H") == "l H"  # right preserved
    assert normalize_ko_coda_in_alias("M b") == "m b"
    assert normalize_ko_coda_in_alias("K p") == "k p"
    # Suffix on left token preserved (color tag e.g. ``L_W p``)
    assert normalize_ko_coda_in_alias("L_W p") == "l_W p"
    # Already-lowercase passes through
    assert normalize_ko_coda_in_alias("ng g") == "ng g"
    # Single token unchanged
    assert normalize_ko_coda_in_alias("NG") == "NG"
    assert normalize_ko_coda_in_alias("ga") == "ga"
    # Vowel left unchanged
    assert normalize_ko_coda_in_alias("a bw") == "a bw"
    # 3-token unchanged
    assert normalize_ko_coda_in_alias("a ki a") == "a ki a"
    # Empty
    assert normalize_ko_coda_in_alias("") == ""


def test_uppercase_coda_bridge_alias_normalizes_at_inference():
    """P4-a: ``NG g`` at inference is normalized to ``ng g`` before role
    classification, so it lands in the same lane as the lowercase
    training supervision."""
    alias_type = bt._infer_alias_type("NG g", language="korean")
    # After normalization, it's 'ng g' which routes to 'vc' (post-2026-05-17
    # coda-bridge re-routing; opt-in env restores 'vcv').
    assert alias_type == "vc"


def test_coda_bridge_onset_bonus_plosive_only():
    """P2-c: plosive/aspirate right-onset on KO coda-bridge gets +1.5 hard
    score; sonorant/fricative right-onset gets 0."""
    from core.coarse_crnn.boundary_scorer_training import _coda_bridge_onset_bonus
    # Plosive: g/b/d/p/k/t (+ tense + aspirated) → +1.5
    assert _coda_bridge_onset_bonus("ng g") == 1.5
    assert _coda_bridge_onset_bonus("l p") == 1.5
    assert _coda_bridge_onset_bonus("m b") == 1.5
    assert _coda_bridge_onset_bonus("n kk") == 1.5
    assert _coda_bridge_onset_bonus("NG H") == 1.5  # aspirate
    assert _coda_bridge_onset_bonus("L R") == 1.5
    # Sonorant/fricative right-onset → 0
    assert _coda_bridge_onset_bonus("ng s") == 0.0
    assert _coda_bridge_onset_bonus("l m") == 0.0
    assert _coda_bridge_onset_bonus("n l") == 0.0
    # Non-coda left → 0
    assert _coda_bridge_onset_bonus("a bw") == 0.0
    assert _coda_bridge_onset_bonus("eu g") == 0.0
    # Single token → 0
    assert _coda_bridge_onset_bonus("ga") == 0.0
    assert _coda_bridge_onset_bonus("") == 0.0


def test_korean_v_plus_glide_consonant_alias_is_vc():
    alias = "a gy"
    alias_type = bt._infer_alias_type(alias, language="korean")
    transition_type = bt._infer_transition_type(alias, language="korean")
    role = bt.normalize_role(
        bt.classify_alias_role(
            "korean",
            alias,
            alias_type=alias_type,
            transition_type=transition_type,
            is_special=False,
        )
    )
    assert alias_type == "vc"
    assert transition_type == "vc"
    assert role == "vc"
