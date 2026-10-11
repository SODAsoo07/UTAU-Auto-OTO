from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from universal_auto_oto.acoustic_evidence import _frame_analysis, _multiband_pitch_tracks
from universal_auto_oto.neural_evidence import (
    NEURAL_INPUT_FEATURES,
    load_neural_evidence_manifest,
    neural_input_features,
    write_neural_evidence_manifest,
)


def test_neural_manifest_is_hash_bound_and_feature_contract_is_stable(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-onnx-payload")
    manifest = write_neural_evidence_manifest(
        model,
        training_manifest_sha256="a" * 64,
        training_summary={"train_recordings": 2},
    )
    loaded = load_neural_evidence_manifest(model, provider="cuda")
    assert loaded.manifest_path == manifest
    assert loaded.feature_names == NEURAL_INPUT_FEATURES
    model.write_bytes(b"changed")
    try:
        load_neural_evidence_manifest(model, provider="cuda")
    except Exception as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("changed model was accepted")


def test_neural_input_features_match_frame_count_and_are_finite(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    times = np.arange(44100, dtype=np.float64) / 44100.0
    sf.write(wav, 0.2 * np.sin(2.0 * np.pi * 220.0 * times), 44100)
    analysis = _frame_analysis(
        wav,
        internal_sample_rate=44100,
        convert_to_mono=True,
        resampler="soxr_hq",
        frame_ms=30.0,
        hop_ms=5.0,
    )
    features = neural_input_features(analysis, _multiband_pitch_tracks(analysis))
    assert features.shape == (len(analysis.frame_times_ms), len(NEURAL_INPUT_FEATURES))
    assert np.all(np.isfinite(features))
