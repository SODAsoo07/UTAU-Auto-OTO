from __future__ import annotations

import math
import wave
from typing import Tuple

import numpy as np

from core.cvn_types import FrameFeatures


_EPS = 1e-10


def _read_wav_mono_float(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav_file:
        channels = int(wav_file.getnchannels() or 1)
        sampwidth = int(wav_file.getsampwidth() or 2)
        sample_rate = int(wav_file.getframerate() or 16000)
        frame_count = int(wav_file.getnframes() or 0)
        raw = wav_file.readframes(frame_count)

    if frame_count <= 0:
        return np.zeros((0,), dtype=np.float32), sample_rate

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        data_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data_i32 = (
            data_u8[:, 0].astype(np.int32)
            | (data_u8[:, 1].astype(np.int32) << 8)
            | (data_u8[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        data_i32 = (data_i32 ^ sign_bit) - sign_bit
        data = data_i32.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        usable = (data.shape[0] // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)

    return np.clip(data.astype(np.float32), -1.0, 1.0), sample_rate


def _resample_linear(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
        return signal.astype(np.float32)
    if signal.size <= 1:
        return signal.astype(np.float32)

    src_len = int(signal.size)
    dst_len = max(1, int(round(src_len * float(dst_sr) / float(src_sr))))
    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float64)
    out = np.interp(dst_x, src_x, signal.astype(np.float64)).astype(np.float32)
    return out


def _frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> tuple[np.ndarray, np.ndarray]:
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_len and hop_len must be positive")

    if signal.size == 0:
        return np.zeros((1, frame_len), dtype=np.float32), np.array([0], dtype=np.int64)

    max_start = max(signal.size - frame_len, 0)
    starts = list(range(0, max_start + 1, hop_len))
    if not starts:
        starts = [0]
    if starts[-1] != max_start:
        starts.append(max_start)

    frames = np.zeros((len(starts), frame_len), dtype=np.float32)
    for i, start in enumerate(starts):
        end = min(start + frame_len, signal.size)
        chunk = signal[start:end]
        frames[i, : chunk.size] = chunk
    return frames, np.asarray(starts, dtype=np.int64)


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_min = 0.0
    f_max = float(sample_rate) * 0.5
    mel_min = _hz_to_mel(np.array([f_min], dtype=np.float64))[0]
    mel_max = _hz_to_mel(np.array([f_max], dtype=np.float64))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / float(sample_rate)).astype(np.int64)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, (n_fft // 2) + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)

        for k in range(left, center):
            fb[m - 1, k] = (k - left) / float(max(center - left, 1))
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / float(max(right - center, 1))
    return fb


def _estimate_f0_autocorr(
    frame: np.ndarray,
    sample_rate: int,
    f0_min: float,
    f0_max: float,
    voiced_corr_th: float = 0.30,
) -> tuple[float, bool]:
    x = frame.astype(np.float64)
    x = x - np.mean(x)
    power = float(np.dot(x, x))
    if power < 1e-8:
        return 0.0, False

    acf = np.correlate(x, x, mode="full")[x.size - 1 :]
    if acf.size <= 2:
        return 0.0, False

    lag_min = max(1, int(sample_rate / float(max(f0_max, 1e-3))))
    lag_max = min(acf.size - 1, int(sample_rate / float(max(f0_min, 1e-3))))
    if lag_max <= lag_min:
        return 0.0, False

    segment = acf[lag_min : lag_max + 1]
    best_idx = int(np.argmax(segment))
    best_lag = lag_min + best_idx
    periodicity = float(segment[best_idx] / max(acf[0], _EPS))
    if periodicity < voiced_corr_th:
        return 0.0, False
    return float(sample_rate / float(best_lag)), True


def extract_frame_features_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_mels: int = 80,
    f0_min: float = 50.0,
    f0_max: float = 600.0,
) -> FrameFeatures:
    sig = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    if sample_rate != target_sr:
        sig = _resample_linear(sig, int(sample_rate), int(target_sr))
    sr = int(target_sr)

    frame_len = max(1, int(round(float(frame_ms) * sr / 1000.0)))
    hop_len = max(1, int(round(float(hop_ms) * sr / 1000.0)))
    frames, starts = _frame_signal(sig, frame_len, hop_len)
    n_frames = int(frames.shape[0])

    times_ms = ((starts + (frame_len // 2)) / float(sr) * 1000.0).astype(np.float32)

    window = np.hanning(frame_len).astype(np.float32)
    n_fft = int(2 ** math.ceil(math.log2(max(8, frame_len))))
    mel_fb = _build_mel_filterbank(sr, n_fft, int(n_mels))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr)).astype(np.float32)

    log_mel = np.zeros((n_frames, int(n_mels)), dtype=np.float32)
    rms = np.zeros((n_frames,), dtype=np.float32)
    zcr = np.zeros((n_frames,), dtype=np.float32)
    centroid = np.zeros((n_frames,), dtype=np.float32)
    flatness = np.zeros((n_frames,), dtype=np.float32)
    f0_hz = np.zeros((n_frames,), dtype=np.float32)
    voiced = np.zeros((n_frames,), dtype=bool)

    for i in range(n_frames):
        frame = frames[i]
        rms[i] = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))

        signs = np.sign(frame)
        zcr[i] = float(np.mean(np.abs(np.diff(signs)) > 0))

        windowed = frame * window
        spectrum = np.fft.rfft(windowed, n=n_fft)
        mag = np.abs(spectrum).astype(np.float32)
        power = np.square(mag, dtype=np.float32)

        mel_energy = mel_fb @ power
        log_mel[i] = np.log10(np.maximum(mel_energy, _EPS)).astype(np.float32)

        mag_sum = float(np.sum(mag) + _EPS)
        centroid[i] = float(np.sum(freqs * mag) / mag_sum)
        flatness[i] = float(np.exp(np.mean(np.log(np.maximum(mag, _EPS)))) / (np.mean(mag) + _EPS))

        f0, is_voiced = _estimate_f0_autocorr(frame, sr, f0_min=f0_min, f0_max=f0_max)
        f0_hz[i] = float(f0)
        voiced[i] = bool(is_voiced)

    return FrameFeatures(
        sample_rate=sr,
        frame_ms=float(frame_ms),
        hop_ms=float(hop_ms),
        times_ms=times_ms.astype(np.float32),
        log_mel=log_mel.astype(np.float32),
        f0_hz=f0_hz.astype(np.float32),
        voiced_mask=voiced.astype(bool),
        rms=rms.astype(np.float32),
        zcr=zcr.astype(np.float32),
        spectral_centroid=centroid.astype(np.float32),
        spectral_flatness=flatness.astype(np.float32),
    )


def extract_frame_features(
    wav_path: str,
    *,
    target_sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_mels: int = 80,
    f0_min: float = 50.0,
    f0_max: float = 600.0,
) -> FrameFeatures:
    audio, sr = _read_wav_mono_float(wav_path)
    return extract_frame_features_from_audio(
        audio,
        sr,
        target_sr=target_sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        n_mels=n_mels,
        f0_min=f0_min,
        f0_max=f0_max,
    )


__all__ = [
    "extract_frame_features",
    "extract_frame_features_from_audio",
]

