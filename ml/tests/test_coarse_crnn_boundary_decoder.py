from __future__ import annotations

from core.coarse_crnn.boundary_targets import build_boundary_target_map
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors, BoundaryCandidate, OtoRowSpec
from core.coarse_crnn.wav_decoder import decode_wav_rows


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


def test_predictor_dispatch_prefers_boundary_engine(monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as mod

    called = {"boundary": 0, "legacy": 0}

    def _fake_boundary(**kwargs):
        called["boundary"] += 1
        return 1, 1, []

    def _fake_legacy(**kwargs):
        called["legacy"] += 1
        return 0, 1, ["legacy called"]

    monkeypatch.setattr(mod, "generate_oto_with_boundary_decoder", _fake_boundary)
    monkeypatch.setattr(mod._legacy, "generate_oto_with_crnn_predictor", _fake_legacy)

    processed, total, errors = mod.generate_oto_with_crnn_predictor(
        wav_dir=".",
        out_path=".",
        source_oto_path=".",
        language="korean",
        engine="boundary_decoder",
    )
    assert (processed, total, errors) == (1, 1, [])
    assert called["boundary"] == 1
    assert called["legacy"] == 0

