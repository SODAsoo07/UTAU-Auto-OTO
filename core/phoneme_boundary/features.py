from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.phoneme_boundary.audio import load_wav_mono, log_mel_spectrogram


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 64
    frame_ms: float = 25.0
    hop_ms: float = 10.0


@dataclass(frozen=True)
class BoundaryFeatures:
    wav_path: str
    samples: np.ndarray
    sample_rate: int
    duration_ms: float
    features: np.ndarray
    hop_ms: float


def load_boundary_features(wav_path: str, config: FeatureConfig) -> BoundaryFeatures:
    samples, sr, duration_sec = load_wav_mono(str(wav_path), target_sr=int(config.sample_rate))
    features, hop_sec = log_mel_spectrogram(
        samples,
        sr,
        n_mels=int(config.n_mels),
        frame_ms=float(config.frame_ms),
        hop_ms=float(config.hop_ms),
    )
    return BoundaryFeatures(
        wav_path=str(wav_path),
        samples=samples,
        sample_rate=int(sr),
        duration_ms=float(duration_sec) * 1000.0,
        features=features.astype(np.float32, copy=False),
        hop_ms=float(hop_sec) * 1000.0,
    )


__all__ = ["BoundaryFeatures", "FeatureConfig", "load_boundary_features"]
