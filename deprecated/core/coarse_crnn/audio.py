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
        # Some public singing datasets store IEEE-float WAV (format tag 3) or
        # WAVE_FORMAT_EXTENSIBLE files. Python's stdlib wave reader only handles
        # PCM, so fall back to scipy when available.
        return _load_wav_mono_scipy(path, target_sr=target_sr, reason=str(exc))

    if frames <= 0:
        return np.zeros((0,), dtype=np.float32), int(target_sr), 0.0

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
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
    duration = float(data.shape[0]) / float(max(1, sr))
    return data.astype(np.float32), int(sr), float(duration)


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
        max_abs = float(np.max(np.abs(arr.astype(np.float64)))) or 1.0
        x = arr.astype(np.float32) / max_abs
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    x = np.clip(x.astype(np.float32), -1.0, 1.0)
    if int(sr) != int(target_sr) and x.size > 1:
        x = _linear_resample(x, int(sr), int(target_sr))
        sr = int(target_sr)
    duration = float(x.shape[0]) / float(max(1, int(sr)))
    return x.astype(np.float32), int(sr), float(duration)


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


def mel_filterbank(*, sr: int, n_fft: int, n_mels: int, fmin: float = 60.0, fmax: float | None = None) -> np.ndarray:
    hi = float(fmax if fmax is not None else sr * 0.5)
    lo_mel = _hz_to_mel(float(fmin))
    hi_mel = _hz_to_mel(hi)
    mel_points = np.linspace(lo_mel, hi_mel, int(n_mels) + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points], dtype=np.float32)
    bins = np.floor((int(n_fft) + 1) * hz_points / float(sr)).astype(int)
    bins = np.clip(bins, 0, int(n_fft // 2))

    fb = np.zeros((int(n_mels), int(n_fft // 2) + 1), dtype=np.float32)
    for m in range(1, int(n_mels) + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
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
    # Pre-STFT peak normalize so voicebank-level gain/mic differences are absorbed
    # before mel computation. 95th-percentile target avoids being dragged by
    # clipping spikes; -3 dBFS leaves headroom for residual peaks.
    peak = float(np.quantile(np.abs(x), 0.95))
    if peak > 1e-6:
        x = (x * (0.7079457843841379 / peak)).astype(np.float32)
    win = max(16, int(round(float(sr) * float(frame_ms) / 1000.0)))
    hop = max(1, int(round(float(sr) * float(hop_ms) / 1000.0)))
    n_fft = 1
    while n_fft < win:
        n_fft *= 2
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    frame_count = 1 + int((x.size - win) // hop)
    shape = (frame_count, win)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    window = np.hanning(win).astype(np.float32)
    spec = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = np.abs(spec).astype(np.float32) ** 2
    fb = mel_filterbank(sr=int(sr), n_fft=int(n_fft), n_mels=int(n_mels))
    mel = np.maximum(np.matmul(power, fb.T), 1e-10)
    logmel = np.log(mel).astype(np.float32)
    # Robust per-utterance CMVN: median + MAD instead of mean + std.
    # MAD is insensitive to silence-dominated mean shifts and clipping outliers
    # that would otherwise inflate std and flatten boundary signals. The 1.4826
    # factor makes MAD an unbiased std estimator under a Gaussian assumption.
    med = np.median(logmel, axis=0, keepdims=True)
    mad = np.median(np.abs(logmel - med), axis=0, keepdims=True)
    # Floor MAD well above zero. Silence-dominated clips have per-bin MAD ≈ 0,
    # so a tiny epsilon floor would divide brief non-silent frames by ~1e-5 and
    # explode the features to ~1e6 → fp16 overflow → NaN loss under AMP. 0.5 sits
    # below the temporal MAD of any genuinely active mel bin, so normal clips are
    # unaffected; the output clip is a hard safety net for residual edge cases.
    mad = np.maximum(mad, 0.5)
    normalized = (logmel - med) / (1.4826 * mad)
    np.clip(normalized, -10.0, 10.0, out=normalized)
    return normalized.astype(np.float32), float(hop) / float(sr)


def add_log_mel_deltas(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return arr
    d1 = np.zeros_like(arr)
    d2 = np.zeros_like(arr)
    if arr.shape[0] >= 2:
        d1[1:] = arr[1:] - arr[:-1]
        d2[1:] = d1[1:] - d1[:-1]
    return np.concatenate([arr, d1, d2], axis=1).astype(np.float32)


def augment_log_mel(
    features: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    gain_db_range: float = 6.0,
    pitch_shift_max_bins: int = 2,
    freq_mask_count: int = 2,
    freq_mask_max_bins: int = 8,
    time_mask_count: int = 1,
    time_mask_max_frames: int = 5,
) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.size <= 0:
        return arr
    r = rng if rng is not None else np.random.default_rng()
    out = arr.copy()
    # Gain in log space: log(10**(g/20)) = g * ln(10)/20 ≈ g * 0.11513
    if gain_db_range > 0.0:
        g_db = float(r.uniform(-float(gain_db_range), float(gain_db_range)))
        out = (out + np.float32(g_db * 0.11512925464970229)).astype(np.float32)
    # Pitch shift approximation via mel-axis roll. Cheap surrogate; for finer
    # control move to waveform-domain pitch_shift later.
    if pitch_shift_max_bins > 0 and out.shape[1] > 2 * int(pitch_shift_max_bins):
        shift = int(r.integers(-int(pitch_shift_max_bins), int(pitch_shift_max_bins) + 1))
        if shift != 0:
            out = np.roll(out, shift, axis=1)
            if shift > 0:
                out[:, :shift] = 0.0
            else:
                out[:, shift:] = 0.0
    n_mels = int(out.shape[1])
    for _ in range(max(0, int(freq_mask_count))):
        if freq_mask_max_bins <= 0 or n_mels <= 2:
            break
        width = int(r.integers(0, int(freq_mask_max_bins) + 1))
        if width <= 0:
            continue
        start = int(r.integers(0, max(1, n_mels - width)))
        out[:, start : start + width] = 0.0
    t_len = int(out.shape[0])
    # Time-mask width capped tight (≈50 ms at 10 ms hop) so labels stay visible.
    for _ in range(max(0, int(time_mask_count))):
        if time_mask_max_frames <= 0 or t_len <= 2:
            break
        width = int(r.integers(0, int(time_mask_max_frames) + 1))
        if width <= 0:
            continue
        start = int(r.integers(0, max(1, t_len - width)))
        out[start : start + width, :] = 0.0
    return out.astype(np.float32)


def estimate_active_region_seconds(
    samples: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pad_sec: float = 0.080,
) -> tuple[float, float] | None:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size <= 0 or sr <= 0:
        return None
    win = max(16, int(round(float(sr) * float(frame_ms) / 1000.0)))
    hop = max(1, int(round(float(sr) * float(hop_ms) / 1000.0)))
    if x.size < win:
        return (0.0, float(x.size) / float(sr))
    frame_count = 1 + int((x.size - win) // hop)
    rms = np.zeros((frame_count,), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * hop
        frame = x[start : start + win]
        rms[idx] = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64) + 1e-12))
    if not np.any(rms > 0.0):
        return None
    p20 = float(np.quantile(rms, 0.20))
    p80 = float(np.quantile(rms, 0.80))
    p95 = float(np.quantile(rms, 0.95))
    threshold = max(p20 + (p95 - p20) * 0.12, p80 * 0.35, 1e-5)
    active = rms >= threshold
    max_gap = max(0, int(round(0.080 / (float(hop) / float(sr)))))
    if max_gap > 0:
        i = 0
        while i < active.size:
            if active[i]:
                i += 1
                continue
            s = i
            while i < active.size and not active[i]:
                i += 1
            e = i
            if s > 0 and e < active.size and active[s - 1] and active[e] and (e - s) <= max_gap:
                active[s:e] = True
    min_run = max(1, int(round(0.030 / (float(hop) / float(sr)))))
    i = 0
    while i < active.size:
        if not active[i]:
            i += 1
            continue
        s = i
        while i < active.size and active[i]:
            i += 1
        e = i
        if e - s < min_run:
            active[s:e] = False
    idxs = np.where(active)[0]
    if idxs.size <= 0:
        return None
    start_sec = max(0.0, (float(idxs[0]) * float(hop) / float(sr)) - float(pad_sec))
    end_sec = min(float(x.size) / float(sr), (float(idxs[-1] + 1) * float(hop) / float(sr)) + float(pad_sec))
    if end_sec <= start_sec:
        return None
    return start_sec, end_sec


def estimate_active_region_for_wav(path: str, *, target_sr: int = 16000) -> tuple[float, float] | None:
    samples, sr, _duration = load_wav_mono(path, target_sr=target_sr)
    return estimate_active_region_seconds(samples, sr)


__all__ = [
    "add_log_mel_deltas",
    "augment_log_mel",
    "estimate_active_region_for_wav",
    "estimate_active_region_seconds",
    "load_wav_mono",
    "log_mel_spectrogram",
    "mel_filterbank",
]
