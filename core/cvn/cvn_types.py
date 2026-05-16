from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameFeatures:
    sample_rate: int
    frame_ms: float
    hop_ms: float
    times_ms: np.ndarray
    log_mel: np.ndarray
    f0_hz: np.ndarray
    voiced_mask: np.ndarray
    rms: np.ndarray
    zcr: np.ndarray
    spectral_centroid: np.ndarray
    spectral_flatness: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.times_ms.shape[0])


@dataclass(frozen=True)
class CvnPosterior:
    times_ms: np.ndarray
    probs: np.ndarray  # shape=(num_frames, num_labels), default order: C,V
    labels: np.ndarray  # shape=(num_frames,), dtype=U1
    label_order: tuple[str, ...] = ("C", "V")

    @property
    def num_frames(self) -> int:
        return int(self.times_ms.shape[0])


__all__ = ["FrameFeatures", "CvnPosterior"]
