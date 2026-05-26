from __future__ import annotations

import math
import wave

import numpy as np


def load_wav_mono(path: str, *, target_sr: int = 16000) -> tuple[np.ndarray, int, float]:
    try:
        with wave.open(str(path), "rb") as wf:
            channels = int(wf.getnchannels() or 1)
            sampwidth = int(wf.getsampwidth() or 2)
            sr = int(wf.getframerate() or target_sr)
            frames = int(wf.getnframes() or 0)
            raw = wf.readframes(frames)
    except wave.Error as exc:
        return _load_wav_mono_scipy(path, target_sr=target_sr, reason=str(exc))

    if frames <= 0:
        return np.zeros((0,), dtype=np.float32), int(target_sr), 0.0
    if sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = u8[:, 0].astype(np.int32) | (u8[:, 1].astype(np.int32) << 8) | (u8[:, 2].astype(np.int32) << 16)
        sign = 1 << 23
        data = ((i32 ^ sign) - sign).astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        n = int(data.shape[0] // channels)
        data = data[: n * channels].reshape(n, channels).mean(axis=1)
    data = np.clip(data.astype(np.float32), -1.0, 1.0)
    if int(sr) != int(target_sr) and data.size > 1:
        data = _linear_resample(data, int(sr), int(target_sr))
        sr = int(target_sr)
    return data.astype(np.float32), int(sr), float(data.shape[0]) / float(max(1, sr))


def _load_wav_mono_scipy(path: str, *, target_sr: int, reason: str = "") -> tuple[np.ndarray, int, float]:
    try:
        from scipy.io import wavfile  # type: ignore
    except Exception as exc:
        raise wave.Error(f"{reason}; scipy fallback unavailable: {exc}") from exc
    sr, data = wavfile.read(str(path))
    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if arr.size <= 0:
        return np.zeros((0,), dtype=np.float32), int(target_sr), 0.0
    if np.issubdtype(arr.dtype, np.floating):
        x = arr.astype(np.float32)
    elif arr.dtype == np.uint8:
        x = (arr.astype(np.float32) - 128.0) / 128.0
    elif arr.dtype == np.int16:
        x = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        x = arr.astype(np.float32) / float(1 << 31)
    else:
        peak = float(np.max(np.abs(arr.astype(np.float64)))) or 1.0
        x = arr.astype(np.float32) / peak
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    if int(sr) != int(target_sr) and x.size > 1:
        x = _linear_resample(x, int(sr), int(target_sr))
        sr = int(target_sr)
    return x.astype(np.float32), int(sr), float(x.shape[0]) / float(max(1, sr))


def _linear_resample(samples: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr <= 0 or target_sr <= 0 or sr == target_sr:
        return samples.astype(np.float32, copy=False)
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size <= 1:
        return x
    out_len = max(1, int(round(float(x.size) * float(target_sr) / float(sr))))
    old_idx = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
    new_idx = np.linspace(0.0, 1.0, num=out_len, endpoint=True)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def _hz_to_mel(freq: float) -> float:
    return 2595.0 * math.log10(1.0 + float(freq) / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (float(mel) / 2595.0) - 1.0)


def _mel_filterbank(*, sr: int, n_fft: int, n_mels: int, fmin: float = 60.0) -> np.ndarray:
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(sr * 0.5), int(n_mels) + 2)
    hz_points = np.asarray([_mel_to_hz(m) for m in mel_points], dtype=np.float32)
    bins = np.clip(np.floor((int(n_fft) + 1) * hz_points / float(sr)).astype(int), 0, int(n_fft // 2))
    fb = np.zeros((int(n_mels), int(n_fft // 2) + 1), dtype=np.float32)
    for m in range(1, int(n_mels) + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / float(max(1, center - left))
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / float(max(1, right - center))
    return fb


def log_mel_spectrogram(
    samples: np.ndarray,
    sr: int,
    *,
    n_mels: int = 64,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, float]:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size <= 0:
        return np.zeros((0, int(n_mels)), dtype=np.float32), float(hop_ms) / 1000.0
    x = x - float(np.mean(x, dtype=np.float64))
    win = max(16, int(round(float(sr) * float(frame_ms) / 1000.0)))
    hop = max(1, int(round(float(sr) * float(hop_ms) / 1000.0)))
    n_fft = 1
    while n_fft < win:
        n_fft *= 2
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    frame_count = 1 + int((x.size - win) // hop)
    frames = np.lib.stride_tricks.as_strided(x, shape=(frame_count, win), strides=(x.strides[0] * hop, x.strides[0]))
    spec = np.fft.rfft(frames * np.hanning(win).astype(np.float32)[None, :], n=n_fft, axis=1)
    power = np.abs(spec).astype(np.float32) ** 2
    mel = np.maximum(np.matmul(power, _mel_filterbank(sr=int(sr), n_fft=int(n_fft), n_mels=int(n_mels)).T), 1e-10)
    logmel = np.log(mel).astype(np.float32)
    med = np.median(logmel, axis=0, keepdims=True)
    mad = np.maximum(np.median(np.abs(logmel - med), axis=0, keepdims=True), 0.5)
    normalized = (logmel - med) / (1.4826 * mad)
    np.clip(normalized, -10.0, 10.0, out=normalized)
    return normalized.astype(np.float32), float(hop) / float(sr)


__all__ = ["load_wav_mono", "log_mel_spectrogram"]
