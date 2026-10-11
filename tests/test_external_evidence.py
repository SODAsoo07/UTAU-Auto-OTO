from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from universal_auto_oto.acoustic_evidence import AcousticEvidence
from universal_auto_oto.audio import file_sha256
from universal_auto_oto.external_evidence import ExternalEvidenceError, fuse_external_evidence, load_external_evidence


def _artifact(path: Path, wav_sha256: str) -> None:
    np.savez(
        path,
        schema_version=np.asarray(1),
        wav_sha256=np.asarray(wav_sha256),
        frame_times_ms=np.asarray([0.0, 5.0, 10.0]),
        track_names=np.asarray(["boundary", "transition"]),
        scores=np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        confidence=np.asarray([0.0, 1.0, 0.0]),
        metadata_json=np.asarray(
            json.dumps({"provider": "test_ssl", "model_sha256": "a" * 64, "encoder_kind": "ssl_frame_head"})
        ),
    )


def test_external_evidence_is_hash_checked_and_confidence_gated(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"test-wave-identity")
    artifact = tmp_path / "external.npz"
    _artifact(artifact, file_sha256(wav))
    external = load_external_evidence(artifact, expected_wav_sha256=file_sha256(wav))
    evidence = AcousticEvidence(
        wav_path=wav,
        wav_sha256=file_sha256(wav),
        source_sample_rate=44100,
        analysis_sample_rate=44100,
        duration_ms=10.0,
        frame_times_ms=np.asarray([0.0, 5.0, 10.0]),
        tracks={"boundary": np.zeros(3), "transition": np.zeros(3)},
        active_start_ms=0.0,
        active_end_ms=10.0,
        feature_config_sha256="b" * 64,
        warnings=(),
        metadata={"feature_backend": "numpy_multiband_pitch_v2", "timebase": {}},
    )

    fused = fuse_external_evidence(evidence, external, maximum_weight=0.5)

    assert fused.tracks["boundary"].tolist() == pytest.approx([0.0, 0.5, 0.0])
    assert fused.tracks["transition"].tolist() == pytest.approx([0.0, 0.5, 0.0])
    assert fused.metadata["external_evidence"]["confidence_gated"] is True
    assert fused.metadata["external_evidence"]["heuristic_residual_preserved"] is True
    with pytest.raises(ExternalEvidenceError, match="WAV SHA-256 mismatch"):
        load_external_evidence(artifact, expected_wav_sha256="0" * 64)
