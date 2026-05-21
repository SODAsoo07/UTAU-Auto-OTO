from __future__ import annotations


def test_boundary_label_pos_weight_tensor_boosts_onset_labels():
    import torch

    from core.coarse_crnn.boundary_scorer_training import _boundary_pos_weight_tensor

    labels = ("syllable_onset", "vowel_start", "transition_peak")
    weights = _boundary_pos_weight_tensor(
        torch,
        labels=labels,
        base_pos_weight=2.0,
        label_weights=("syllable_onset=1.5", "vowel_start=1.25"),
        device="cpu",
    )

    assert tuple(weights.shape) == (1, 1, 3)
    flat = weights.reshape(-1).tolist()
    assert flat == [3.0, 2.5, 2.0]


def test_peak_candidates_default_keeps_low_onset_peak(monkeypatch):
    from core.coarse_crnn.boundary_candidates import peak_candidates_from_scores
    from core.coarse_crnn.boundary_types import BoundaryFrameScores

    monkeypatch.delenv("UTOA_BOUNDARY_PEAK_MIN_SCORE", raising=False)
    score_map = BoundaryFrameScores(
        wav_path="x.wav",
        times_ms=[0.0, 10.0, 20.0, 30.0, 40.0],
        scores={
            "syllable_onset": [0.01, 0.10, 0.21, 0.10, 0.01],
            "transition_peak": [0.01, 0.10, 0.21, 0.10, 0.01],
        },
        quality_scores=[0.5, 0.5, 0.5, 0.5, 0.5],
    )

    candidates = peak_candidates_from_scores(score_map)
    kinds = [item.kind for item in candidates]
    assert "syllable_onset" in kinds
    assert "transition_peak" not in kinds


def test_peak_candidate_env_override_can_restore_higher_floor(monkeypatch):
    from core.coarse_crnn.boundary_candidates import peak_candidates_from_scores
    from core.coarse_crnn.boundary_types import BoundaryFrameScores

    monkeypatch.setenv("UTOA_BOUNDARY_PEAK_MIN_SCORE", "0.45")
    score_map = BoundaryFrameScores(
        wav_path="x.wav",
        times_ms=[0.0, 10.0, 20.0, 30.0, 40.0],
        scores={"syllable_onset": [0.01, 0.10, 0.27, 0.10, 0.01]},
        quality_scores=[0.5, 0.5, 0.5, 0.5, 0.5],
    )

    assert peak_candidates_from_scores(score_map) == []
