from __future__ import annotations

import pytest

from core.coarse_crnn.boundary_types import (
    AbsoluteOtoAnchors,
    BoundaryCandidate,
    BoundaryDecodeResult,
    DecodedOtoRow,
    OtoRowSpec,
)


def _row() -> DecodedOtoRow:
    spec = OtoRowSpec(
        wav_name="a.wav",
        wav_path="a.wav",
        alias="a",
        role="cv",
        slot_index=0,
        slot_count=1,
        prev_alias="",
        next_alias="",
        language="japanese",
        format_type="cv",
        line_index=0,
        duration_ms=1000.0,
        source_params={},
        meta={"alias_type": "cv"},
    )
    return DecodedOtoRow(
        spec=spec,
        anchors=AbsoluteOtoAnchors(
            offset_abs=100.0,
            overlap_abs=120.0,
            pre_abs=180.0,
            consonant_abs=260.0,
            cutoff_abs=600.0,
            confidence=0.8,
            reason="base",
        ),
        selected_time_ms=180.0,
        fallback_used=False,
        reason="base",
        quality_score=0.7,
    )


def test_stage2_feature_batch_is_source_param_independent():
    from core.coarse_crnn.stage2_oto.features import build_stage2_feature_batch

    row = _row()
    decoded = BoundaryDecodeResult(
        wav_path="a.wav",
        duration_ms=1000.0,
        rows=[row],
        anchor_timeline_ms=[100.0],
        fallback_count=0,
    )
    batch = build_stage2_feature_batch(
        decoded=decoded,
        candidates=[BoundaryCandidate(time_ms=180.0, kind="vowel_start", score=0.9, source="model")],
        active_start_ms=50.0,
        active_end_ms=850.0,
        model_quality=0.7,
        audio_reliability=0.6,
    )
    assert batch.numeric_dim == len(batch.rows[0].numeric)
    assert row.spec.source_params == {}
    assert batch.rows[0].role_id > 0


def test_stage2_apply_falls_back_on_low_confidence():
    pytest.importorskip("torch")
    import torch

    from core.coarse_crnn.stage2_oto.inference import Stage2Bundle, apply_stage2_to_decode
    from core.coarse_crnn.stage2_oto.model import build_stage2_model
    from core.coarse_crnn.stage2_oto.features import build_stage2_feature_batch
    from core.coarse_crnn.stage2_oto.types import Stage2ModelConfig

    row = _row()
    decoded = BoundaryDecodeResult(
        wav_path="a.wav",
        duration_ms=1000.0,
        rows=[row],
        anchor_timeline_ms=[100.0],
        fallback_count=0,
    )
    candidates = [BoundaryCandidate(time_ms=180.0, kind="vowel_start", score=0.9, source="model")]
    feature_batch = build_stage2_feature_batch(
        decoded=decoded,
        candidates=candidates,
        active_start_ms=50.0,
        active_end_ms=850.0,
    )
    config = Stage2ModelConfig(numeric_dim=feature_batch.numeric_dim, hidden_dim=16, gru_layers=1)
    model = build_stage2_model(config).eval()
    for param in model.parameters():
        torch.nn.init.constant_(param, 0.0)
    bundle = Stage2Bundle(model=model, config=config, path="", device="cpu", meta={})
    result = apply_stage2_to_decode(
        decoded=decoded,
        candidates=candidates,
        bundle=bundle,
        active_start_ms=50.0,
        active_end_ms=850.0,
        min_confidence=0.9,
    )
    assert result.accepted_rows == 0
    assert result.fallback_rows == 1
    assert result.decoded.rows[0].anchors.pre_abs == row.anchors.pre_abs


def test_predictor_dispatch_passes_stage2_options(monkeypatch):
    from core.coarse_crnn import oto_predictor_generator as mod

    captured = {}

    def _fake_boundary(**kwargs):
        captured.update(kwargs)
        return 1, 1, []

    monkeypatch.setattr(mod, "generate_oto_with_boundary_decoder", _fake_boundary)
    monkeypatch.setattr(mod, "resolve_boundary_scorer_model_path", lambda _path="": "boundary.pt")

    processed, total, errors = mod.generate_oto_with_crnn_predictor(
        wav_dir=".",
        out_path="out.ini",
        source_oto_path="oto.ini",
        language="japanese",
        engine="boundary_decoder",
        stage2_model_path="stage2.pt",
        stage2_enable=True,
        phoneme_boundary_model_path="pb.pt",
    )
    assert (processed, total, errors) == (1, 1, [])
    assert captured["stage2_model_path"] == "stage2.pt"
    assert captured["stage2_enable"] is True
    assert captured["phoneme_boundary_model_path"] == "pb.pt"


def test_phoneme_boundary_events_convert_to_stage2_candidates(monkeypatch):
    from core.coarse_crnn import boundary_generator as mod
    from core.phoneme_boundary.types import BoundaryEvent, BoundaryFrameScores

    def _fake_infer(**_kwargs):
        return BoundaryFrameScores(
            wav_path="a.wav",
            times_ms=[0.0, 10.0],
            scores={"vowel_onset": [0.1, 0.9]},
            quality_scores=[],
        )

    def _fake_peaks(_score_map, **_kwargs):
        return [BoundaryEvent(label="vowel_onset", time_ms=123.0, confidence=0.8)]

    import core.phoneme_boundary.inference as pb_inf

    monkeypatch.setattr(pb_inf, "infer_boundary_scores_with_model", _fake_infer)
    monkeypatch.setattr(pb_inf, "peak_events_from_scores", _fake_peaks)
    candidates = mod._phoneme_boundary_candidates_for_wav(
        model=object(),
        config=object(),
        wav_path="a.wav",
        device="cpu",
    )
    assert len(candidates) == 1
    assert candidates[0].kind == "vowel_start"
    assert candidates[0].source.startswith("model:phoneme_boundary")


def test_stage2_pb_merge_weight_can_override_boundary_scorer():
    from core.coarse_crnn.boundary_candidates import merge_candidates

    merged = merge_candidates(
        model_candidates=[
            BoundaryCandidate(time_ms=100.0, kind="vowel_start", score=0.8, source="model"),
            BoundaryCandidate(
                time_ms=140.0,
                kind="vowel_start",
                score=0.8,
                source="model:phoneme_boundary:vowel_onset",
            ),
        ],
        audio_candidates=[],
        merge_ms=50.0,
        model_quality=0.8,
        source_weight_overrides={
            "model:phoneme_boundary": 1.15,
            "model": 0.45,
        },
    )
    assert len(merged) == 1
    assert merged[0].time_ms > 125.0
    assert "model:phoneme_boundary" in merged[0].source


def test_stage2_nearest_candidate_prefers_pb_for_pre_anchor():
    from core.coarse_crnn.stage2_oto.features import nearest_candidate_for_field

    picked = nearest_candidate_for_field(
        field="pre_abs",
        role="cv",
        anchor_ms=190.0,
        duration_ms=1000.0,
        candidates=[
            BoundaryCandidate(time_ms=180.0, kind="vowel_start", score=0.95, source="model"),
            BoundaryCandidate(
                time_ms=206.0,
                kind="vowel_start",
                score=0.80,
                source="model:phoneme_boundary:vowel_onset",
            ),
        ],
    )
    assert picked is not None
    assert picked.source.startswith("model:phoneme_boundary")


def test_phoneme_boundary_silence_score_is_damped(monkeypatch):
    from core.coarse_crnn import boundary_generator as mod
    from core.phoneme_boundary.types import BoundaryEvent, BoundaryFrameScores

    def _fake_infer(**_kwargs):
        return BoundaryFrameScores(
            wav_path="a.wav",
            times_ms=[0.0, 10.0],
            scores={},
            quality_scores=[],
        )

    def _fake_peaks(_score_map, **_kwargs):
        return [
            BoundaryEvent(label="silence_boundary", time_ms=10.0, confidence=0.8),
            BoundaryEvent(label="vowel_onset", time_ms=120.0, confidence=0.8),
        ]

    import core.phoneme_boundary.inference as pb_inf

    monkeypatch.setattr(pb_inf, "infer_boundary_scores_with_model", _fake_infer)
    monkeypatch.setattr(pb_inf, "peak_events_from_scores", _fake_peaks)
    candidates = mod._phoneme_boundary_candidates_for_wav(
        model=object(),
        config=object(),
        wav_path="a.wav",
        device="cpu",
    )
    by_kind = {item.kind: item for item in candidates}
    assert by_kind["silence_boundary"].score < by_kind["vowel_start"].score
