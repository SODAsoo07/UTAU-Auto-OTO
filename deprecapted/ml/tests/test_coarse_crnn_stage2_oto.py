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


def test_phoneme_boundary_assist_can_run_without_stage2():
    from core.coarse_crnn import boundary_generator as mod

    assert mod._phoneme_boundary_assist_requested("pb.pt") is True


def test_phoneme_boundary_state_scores_adjust_candidate_confidence(monkeypatch):
    from core.coarse_crnn import boundary_generator as mod
    from core.phoneme_boundary.types import BoundaryEvent, BoundaryFrameScores

    def _fake_infer(**_kwargs):
        return BoundaryFrameScores(
            wav_path="a.wav",
            times_ms=[100.0, 200.0],
            scores={"vowel_onset": [0.90, 0.90]},
            quality_scores=[],
            phone_state_scores={
                "silence": [0.02, 0.02],
                "consonant": [0.05, 0.05],
                "vowel": [0.95, 0.10],
            },
        )

    def _fake_peaks(_score_map, **_kwargs):
        return [
            BoundaryEvent(label="vowel_onset", time_ms=100.0, confidence=0.8),
            BoundaryEvent(label="vowel_onset", time_ms=200.0, confidence=0.8),
        ]

    import core.phoneme_boundary.inference as pb_inf

    monkeypatch.setattr(pb_inf, "infer_boundary_scores_with_model", _fake_infer)
    monkeypatch.setattr(pb_inf, "peak_events_from_scores", _fake_peaks)
    monkeypatch.setenv("UTOA_BOUNDARY_PB_STATE_SCORE_WEIGHT", "0.5")

    candidates = mod._phoneme_boundary_candidates_for_wav(
        model=object(),
        config=object(),
        wav_path="a.wav",
        device="cpu",
    )

    assert len(candidates) == 2
    assert candidates[0].score > candidates[1].score
    assert "state=" in candidates[0].source


def test_stage2_candidate_distance_policy_precedence():
    from core.coarse_crnn.stage2_oto.constraints import resolve_max_candidate_distance_ms

    policy = {
        "default": 250.0,
        "by_role": {"vv": 170.0},
        "by_format": {"vcv": 140.0},
        "by_format_role": {("vcv", "vv"): 90.0},
    }
    assert resolve_max_candidate_distance_ms(role="vv", format_type="vcv", policy=policy) == pytest.approx(90.0)
    assert resolve_max_candidate_distance_ms(role="vv", format_type="cv", policy=policy) == pytest.approx(170.0)
    assert resolve_max_candidate_distance_ms(role="cv", format_type="vcv", policy=policy) == pytest.approx(140.0)
    assert resolve_max_candidate_distance_ms(role="cv", format_type="cv", policy=policy) == pytest.approx(250.0)


def test_stage2_candidate_distance_policy_from_env(monkeypatch):
    from core.coarse_crnn.stage2_oto.constraints import max_candidate_distance_policy_from_env, resolve_max_candidate_distance_ms

    monkeypatch.setenv("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_MS", "300")
    monkeypatch.setenv("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_BY_ROLE", "cv=200;vv=150")
    monkeypatch.setenv("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_BY_FORMAT", "cv=240,vcv=130")
    monkeypatch.setenv("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_BY_FORMAT_ROLE", "vcv/vv=85,cv:cv=210")
    policy = max_candidate_distance_policy_from_env()
    assert resolve_max_candidate_distance_ms(role="vv", format_type="vcv", policy=policy) == pytest.approx(85.0)
    assert resolve_max_candidate_distance_ms(role="cv", format_type="cv", policy=policy) == pytest.approx(210.0)
    assert resolve_max_candidate_distance_ms(role="cv", format_type="other", policy=policy) == pytest.approx(200.0)
    assert resolve_max_candidate_distance_ms(role="other", format_type="vcv", policy=policy) == pytest.approx(130.0)
    assert resolve_max_candidate_distance_ms(role="other", format_type="other", policy=policy) == pytest.approx(300.0)


def test_stage2_projection_keeps_model_as_boundary_residual(monkeypatch):
    from core.coarse_crnn.stage2_oto.constraints import project_anchors_to_boundary_candidates

    row = _row()
    monkeypatch.setenv("UTOA_STAGE2_OTO_MAX_RESIDUAL_BY_FIELD", "pre_abs=40,cutoff_abs=200")
    monkeypatch.setenv("UTOA_STAGE2_OTO_PB_RESIDUAL_SCALE", "0.5")
    projected = project_anchors_to_boundary_candidates(
        row=row,
        anchors_ms=(100.0, 120.0, 300.0, 330.0, 620.0),
        duration_ms=1000.0,
        candidates=[
            BoundaryCandidate(
                time_ms=200.0,
                kind="vowel_start",
                score=0.9,
                source="model:phoneme_boundary:vowel_onset",
            ),
            BoundaryCandidate(time_ms=610.0, kind="vowel_end", score=0.9, source="model"),
        ],
    )
    assert projected[2] == pytest.approx(220.0)
    assert projected[4] == pytest.approx(620.0)


def test_stage2_gate_accepts_projected_boundary_adapter_reason(monkeypatch):
    from core.coarse_crnn.stage2_oto.constraints import apply_stage2_gate

    row = _row()
    monkeypatch.setenv(
        "UTOA_STAGE2_OTO_MAX_RESIDUAL_BY_FIELD",
        "offset_abs=40,overlap_abs=40,pre_abs=40,consonant_abs=40,cutoff_abs=80",
    )
    decision = apply_stage2_gate(
        row=row,
        predicted_anchors_ms=(100.0, 130.0, 300.0, 360.0, 650.0),
        confidence=0.95,
        candidates=[
            BoundaryCandidate(time_ms=100.0, kind="syllable_onset", score=0.9, source="model:phoneme_boundary:phone_start"),
            BoundaryCandidate(time_ms=130.0, kind="consonant_onset", score=0.9, source="model:phoneme_boundary:consonant_onset"),
            BoundaryCandidate(time_ms=200.0, kind="vowel_start", score=0.9, source="model:phoneme_boundary:vowel_onset"),
            BoundaryCandidate(time_ms=270.0, kind="transition_peak", score=0.9, source="model:phoneme_boundary:vowel_nucleus"),
            BoundaryCandidate(time_ms=610.0, kind="vowel_end", score=0.9, source="model:phoneme_boundary:vowel_end"),
        ],
        min_confidence=0.5,
        max_candidate_distance_ms=90.0,
    )
    assert decision.accepted is True
    assert decision.reason == "stage2:boundary_adapter"
    assert decision.anchors_ms[2] == pytest.approx(226.0)


def test_stage2_train_balance_plan_upsamples_minor_group():
    from core.coarse_crnn.stage2_oto.types import Stage2FeatureBatch, Stage2FeatureRow
    from ml.scripts.coarse_crnn.train_stage2_oto import _build_balance_plan

    def _batch(role_id: int, format_id: int, row_count: int) -> Stage2FeatureBatch:
        rows = []
        for _ in range(row_count):
            rows.append(
                Stage2FeatureRow(
                    numeric=(0.1, 0.2, 0.3),
                    role_id=int(role_id),
                    prev_role_id=0,
                    next_role_id=0,
                    alias_type_id=0,
                    language_id=0,
                    format_id=int(format_id),
                    base_anchors_ms=(10.0, 20.0, 30.0, 40.0, 50.0),
                )
            )
        return Stage2FeatureBatch(rows=tuple(rows), numeric_dim=3)

    sequences = [
        {"features": _batch(role_id=6, format_id=4, row_count=12), "target": [], "seq_role_id": 6, "seq_format_id": 4},
        {"features": _batch(role_id=6, format_id=4, row_count=12), "target": [], "seq_role_id": 6, "seq_format_id": 4},
        {"features": _batch(role_id=2, format_id=1, row_count=4), "target": [], "seq_role_id": 2, "seq_format_id": 1},
    ]
    plan = _build_balance_plan(
        sequences,
        sampling_power=1.0,
        sampling_max_weight=3.0,
        role_loss_power=1.0,
        format_loss_power=1.0,
        row_min_weight=0.35,
        row_max_weight=3.0,
    )
    assert len(plan["sequence_weights"]) == 3
    assert plan["sequence_weights"][2] > plan["sequence_weights"][0]
    assert plan["role_weights"][2] > plan["role_weights"][6]
    assert plan["format_weights"][1] > plan["format_weights"][4]


def test_stage2_confidence_target_tracks_improvement_over_base():
    pytest.importorskip("torch")
    import torch

    from ml.scripts.coarse_crnn.train_stage2_oto import _confidence_target_from_improvement

    numeric = torch.zeros((1, 2, 41), dtype=torch.float32)
    numeric[..., 0] = 0.1  # 1000 ms duration after train script denormalization.
    numeric[0, 0, 12:17] = torch.tensor([0.10, 0.12, 0.18, 0.26, 0.60])
    numeric[0, 1, 12:17] = torch.tensor([0.10, 0.12, 0.18, 0.26, 0.60])
    target = torch.tensor(
        [
            [
                [0.20, 0.22, 0.28, 0.36, 0.70],
                [0.10, 0.12, 0.18, 0.26, 0.60],
            ]
        ],
        dtype=torch.float32,
    )
    pred = torch.tensor(
        [
            [
                [0.19, 0.21, 0.27, 0.35, 0.69],
                [0.20, 0.22, 0.28, 0.36, 0.70],
            ]
        ],
        dtype=torch.float32,
    )
    conf_target = _confidence_target_from_improvement(
        torch,
        pred=pred,
        target_t=target,
        numeric=numeric,
        margin_ms=10.0,
        scale_ms=20.0,
    )
    assert conf_target[0, 0] > 0.95
    assert conf_target[0, 1] < 0.20


def test_evaluate_stage2_anchor_timeline_mode_parser():
    from ml.scripts.coarse_crnn.evaluate_stage2_oto import _parse_anchor_timeline_modes

    assert _parse_anchor_timeline_modes("") == []
    assert _parse_anchor_timeline_modes("all") == ["linear", "filename", "hybrid", "cv_slot"]
    assert _parse_anchor_timeline_modes("legacy,tokens,cv-segment,hybrid,hybrid") == [
        "linear",
        "filename",
        "cv_slot",
        "hybrid",
    ]
    assert _parse_anchor_timeline_modes("auto") == ["auto"]


def test_evaluate_stage2_anchor_timeline_stats_finalize():
    from ml.scripts.coarse_crnn.evaluate_stage2_oto import (
        _add_anchor_timeline_stats,
        _finalize_anchor_timeline_stats,
        _new_anchor_timeline_stats,
    )

    stats = {"cv_slot": _new_anchor_timeline_stats()}
    _add_anchor_timeline_stats(
        stats["cv_slot"],
        role="cv",
        mode_error_ms=70.0,
        base_error_ms=100.0,
        improved=True,
        worse=False,
        neutral=False,
    )
    _add_anchor_timeline_stats(
        stats["cv_slot"],
        role="vc",
        mode_error_ms=130.0,
        base_error_ms=100.0,
        improved=False,
        worse=True,
        neutral=False,
    )
    out = _finalize_anchor_timeline_stats(stats)
    assert out["cv_slot"]["rows"] == 2
    assert out["cv_slot"]["delta_ms"] == pytest.approx(0.0)
    assert out["cv_slot"]["improved_rows"] == 1
    assert out["cv_slot"]["worse_rows"] == 1
    assert out["cv_slot"]["by_role"]["cv"]["delta_ms"] == pytest.approx(-30.0)
    assert out["cv_slot"]["by_role"]["vc"]["delta_ms"] == pytest.approx(30.0)


def test_evaluate_stage2_pb_assist_stats_finalize():
    from ml.scripts.coarse_crnn.evaluate_stage2_oto import (
        _add_pb_assist_stats,
        _finalize_pb_assist_stats,
        _new_pb_assist_stats,
    )

    stats = _new_pb_assist_stats()
    _add_pb_assist_stats(
        stats,
        role="cv",
        base_error_ms=100.0,
        pb_base_error_ms=70.0,
        stage2_error_ms=90.0,
        pb_stage2_error_ms=60.0,
        improved=True,
        worse=False,
        neutral=False,
    )
    _add_pb_assist_stats(
        stats,
        role="vc",
        base_error_ms=100.0,
        pb_base_error_ms=130.0,
        stage2_error_ms=90.0,
        pb_stage2_error_ms=120.0,
        improved=False,
        worse=True,
        neutral=False,
    )
    out = _finalize_pb_assist_stats(stats)
    assert out["rows"] == 2
    assert out["pb_base_delta_ms"] == pytest.approx(0.0)
    assert out["pb_stage2_delta_ms"] == pytest.approx(-10.0)
    assert out["improved_rows"] == 1
    assert out["worse_rows"] == 1
    assert out["by_role"]["cv"]["pb_base_delta_ms"] == pytest.approx(-30.0)
    assert out["by_role"]["vc"]["pb_base_delta_ms"] == pytest.approx(30.0)


def test_pb_assist_format_gate_defaults_to_transition_formats(monkeypatch):
    from core.coarse_crnn import boundary_generator
    from ml.scripts.coarse_crnn import evaluate_stage2_oto

    monkeypatch.delenv("UTOA_BOUNDARY_PB_ASSIST_FORMATS", raising=False)
    assert boundary_generator._pb_assist_format_enabled("cvvc") is True
    assert boundary_generator._pb_assist_format_enabled("cvc") is True
    assert boundary_generator._pb_assist_format_enabled("cmpx") is True
    assert boundary_generator._pb_assist_format_enabled("cv") is False
    assert boundary_generator._pb_assist_format_enabled("vcv") is False
    assert evaluate_stage2_oto._pb_assist_format_enabled("cvvc") is True
    assert evaluate_stage2_oto._pb_assist_format_enabled("cv") is False


def test_pb_assist_format_gate_can_be_overridden(monkeypatch):
    from core.coarse_crnn import boundary_generator
    from ml.scripts.coarse_crnn import evaluate_stage2_oto

    monkeypatch.setenv("UTOA_BOUNDARY_PB_ASSIST_FORMATS", "all")
    assert boundary_generator._pb_assist_format_enabled("cv") is True
    assert evaluate_stage2_oto._pb_assist_format_enabled("vcv") is True

    monkeypatch.setenv("UTOA_BOUNDARY_PB_ASSIST_FORMATS", "cv")
    assert boundary_generator._pb_assist_format_enabled("cv") is True
    assert boundary_generator._pb_assist_format_enabled("cvvc") is False
    assert evaluate_stage2_oto._pb_assist_format_enabled("cv") is True
    assert evaluate_stage2_oto._pb_assist_format_enabled("cvvc") is False
