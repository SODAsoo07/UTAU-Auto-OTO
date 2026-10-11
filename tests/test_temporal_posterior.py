from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from universal_auto_oto.acoustic_evidence import AcousticEvidence
from universal_auto_oto.temporal_posterior import (
    TEMPORAL_POSTERIOR_HEADS,
    TemporalPosteriorError,
    TemporalPosteriorSequence,
    apply_temporal_posterior_model,
    load_temporal_posterior_model,
    train_temporal_posterior_model,
)


FEATURE_NAMES = ("signal", "distractor")


def _context_sequence(sequence_id: str, speaker_id: str, seed: int, frames: int = 240):
    rng = np.random.default_rng(seed)
    signal = rng.choice((-1.0, 1.0), size=frames)
    features = np.column_stack((signal, rng.normal(size=frames)))
    target = np.empty(frames, dtype=np.float64)
    target[0] = np.nan
    target[1:] = signal[:-1] > 0.0
    target[::17] = np.nan
    targets = {
        head: (target.copy() if index % 2 == 0 else 1.0 - target.copy())
        for index, head in enumerate(TEMPORAL_POSTERIOR_HEADS)
    }
    return TemporalPosteriorSequence(
        sequence_id=sequence_id,
        speaker_id=speaker_id,
        features=features,
        targets=targets,
        source_sha256=f"source-{seed}",
    )


@pytest.fixture(scope="module")
def trained_pair():
    train = (_context_sequence("train-a", "speaker-train", 1, 360),)
    validation = (_context_sequence("validation-a", "speaker-validation", 2, 240),)
    locked = (_context_sequence("locked-a", "speaker-locked", 3, 180),)
    common = dict(
        feature_names=FEATURE_NAMES,
        feature_backend="synthetic_context_v1",
        hidden_channels=6,
        kernel_size=3,
        epochs=220,
        learning_rate=0.025,
        l2=1e-5,
        early_stopping_patience=35,
        seed=73,
    )
    model = train_temporal_posterior_model(train, validation, locked, **common)

    changed_locked = _context_sequence("locked-a", "speaker-locked", 33, 180)
    changed_targets = {
        head: 1.0 - values for head, values in changed_locked.targets.items()
    }
    changed_locked = TemporalPosteriorSequence(
        sequence_id=changed_locked.sequence_id,
        speaker_id=changed_locked.speaker_id,
        features=changed_locked.features * 20.0 + 100.0,
        targets=changed_targets,
        source_sha256="different-locked-source",
    )
    changed_model = train_temporal_posterior_model(
        train,
        validation,
        (changed_locked,),
        **common,
    )
    return model, changed_model, validation[0]


def test_temporal_convolution_learns_previous_frame_context(trained_pair) -> None:
    model, _changed_model, validation = trained_pair
    predictions = model.predict(validation.features)["boundary"]
    target = validation.targets["boundary"]
    mask = np.isfinite(target)
    accuracy = np.mean((predictions[mask] >= 0.5) == (target[mask] >= 0.5))

    current_frame_baseline = validation.features[mask, 0] > 0.0
    current_frame_accuracy = np.mean(current_frame_baseline == (target[mask] >= 0.5))
    assert current_frame_accuracy < 0.62
    assert accuracy > 0.9
    assert model.conv_weights.shape == (6, 3, 2)
    assert set(model.predict(validation.features)) == set(TEMPORAL_POSTERIOR_HEADS)
    assert all(values.shape == (validation.features.shape[0],) for values in model.predict(validation.features).values())
    support = model.training_provenance["target_support"]
    assert support["train"]["boundary"]["ignored_nan"] > 0


def test_artifact_and_inference_are_stable(trained_pair, tmp_path: Path) -> None:
    model, _changed_model, validation = trained_pair
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    model.save(first)
    model.save(second)

    assert first.read_bytes() == second.read_bytes()
    loaded = load_temporal_posterior_model(first)
    assert loaded.to_artifact() == model.to_artifact()
    expected = model.predict(validation.features)
    actual = loaded.predict(validation.features)
    for head in TEMPORAL_POSTERIOR_HEADS:
        np.testing.assert_array_equal(actual[head], expected[head])


def test_temporal_model_blends_only_trained_runtime_heads(trained_pair, tmp_path: Path) -> None:
    model, _changed_model, validation = trained_pair
    model_path = tmp_path / "model.json"
    model.save(model_path)
    frame_count = validation.features.shape[0]
    tracks = {
        "signal": validation.features[:, 0],
        "distractor": validation.features[:, 1],
        **{head: np.full(frame_count, 0.25) for head in TEMPORAL_POSTERIOR_HEADS},
    }
    evidence = AcousticEvidence(
        wav_path=tmp_path / "fixture.wav",
        wav_sha256="a" * 64,
        source_sample_rate=44100,
        analysis_sample_rate=44100,
        duration_ms=float(frame_count * 5),
        frame_times_ms=np.arange(frame_count, dtype=np.float64) * 5.0,
        tracks=tracks,
        active_start_ms=0.0,
        active_end_ms=float(frame_count * 5),
        feature_config_sha256="b" * 64,
        warnings=(),
        metadata={"feature_backend": model.feature_backend},
    )

    applied = apply_temporal_posterior_model(
        evidence,
        model,
        model_path=model_path,
        posterior_blend=0.75,
    )

    assert applied.metadata["posterior_model"]["temporal_context"] is True
    assert applied.metadata["posterior_model"]["production_promotion_claimed"] is False
    assert not np.array_equal(applied.tracks["boundary"], evidence.tracks["boundary"])
    np.testing.assert_array_equal(applied.tracks["signal"], evidence.tracks["signal"])


def test_locked_test_does_not_affect_normalization_training_or_selection(trained_pair) -> None:
    model, changed_model, _validation = trained_pair
    np.testing.assert_array_equal(model.means, changed_model.means)
    np.testing.assert_array_equal(model.scales, changed_model.scales)
    np.testing.assert_array_equal(model.conv_weights, changed_model.conv_weights)
    np.testing.assert_array_equal(model.conv_bias, changed_model.conv_bias)
    np.testing.assert_array_equal(model.head_weights, changed_model.head_weights)
    np.testing.assert_array_equal(model.head_bias, changed_model.head_bias)
    assert model.training_provenance["best_epoch"] == changed_model.training_provenance["best_epoch"]
    assert model.training_provenance["input_sha256"]["locked_test"] != changed_model.training_provenance["input_sha256"]["locked_test"]
    assert model.training_provenance["locked_test_policy"] == "final_evaluation_only_after_model_selection"
    assert model.evaluation["locked_test"] != changed_model.evaluation["locked_test"]


def test_rejects_speaker_split_leakage() -> None:
    train = (_context_sequence("train-a", "same-speaker", 1, 30),)
    validation = (_context_sequence("validation-a", "same-speaker", 2, 30),)
    with pytest.raises(TemporalPosteriorError, match="speaker_ids=.*same-speaker"):
        train_temporal_posterior_model(
            train,
            validation,
            (),
            feature_names=FEATURE_NAMES,
            feature_backend="synthetic_context_v1",
            epochs=2,
        )
