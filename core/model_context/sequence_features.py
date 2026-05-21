from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BasicFrameFeatures:
    rms: np.ndarray
    zcr: np.ndarray
    times_ms: np.ndarray
    hop_ms: float
    frame_ms: float

    @property
    def num_frames(self) -> int:
        return int(np.asarray(self.times_ms).size)


def extract_basic_frame_features(
    wav_path: str,
    *,
    hop_ms: float = 10.0,
    frame_ms: float = 25.0,
) -> BasicFrameFeatures:
    samples, sr = _read_wav_mono(wav_path)
    hop = max(1, int(round(float(sr) * float(hop_ms) / 1000.0)))
    frame = max(hop, int(round(float(sr) * float(frame_ms) / 1000.0)))
    if samples.size <= 0:
        return BasicFrameFeatures(
            rms=np.zeros((0,), dtype=np.float32),
            zcr=np.zeros((0,), dtype=np.float32),
            times_ms=np.zeros((0,), dtype=np.float32),
            hop_ms=float(hop_ms),
            frame_ms=float(frame_ms),
        )
    frames = []
    zcr = []
    times = []
    for start in range(0, max(1, samples.size - frame + 1), hop):
        chunk = samples[start : start + frame]
        if chunk.size < frame:
            chunk = np.pad(chunk, (0, frame - chunk.size))
        frames.append(float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64))))
        signs = np.signbit(chunk)
        zcr.append(float(np.mean(signs[1:] != signs[:-1])) if chunk.size > 1 else 0.0)
        times.append(1000.0 * float(start) / float(max(1, sr)))
        if start + frame >= samples.size:
            break
    return BasicFrameFeatures(
        rms=np.asarray(frames, dtype=np.float32),
        zcr=np.asarray(zcr, dtype=np.float32),
        times_ms=np.asarray(times, dtype=np.float32),
        hop_ms=float(hop_ms),
        frame_ms=float(frame_ms),
    )


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = int(handle.getnchannels() or 1)
        width = int(handle.getsampwidth() or 2)
        sr = int(handle.getframerate() or 16000)
        frames = int(handle.getnframes() or 0)
        raw = handle.readframes(frames)
    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.zeros((0,), dtype=np.float32)
    if channels > 1 and data.size:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return np.asarray(data, dtype=np.float32), sr
