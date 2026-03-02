from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from core.oto_generator import _read_wav_mono_np
from core.oto_ml_features import CATEGORICAL_FEATURES, FEATURE_NAMES, canonicalize_feature_row

DEFAULT_MEL_BINS = 80
DEFAULT_WINDOW_FRAMES = 160
DEFAULT_HOP_MS = 5.0
DEFAULT_WIN_MS = 25.0
DEFAULT_NFFT = 1024


def _default_torch_cache_dir() -> str:
    configured = os.environ.get("UTOA_OTO_TORCH_CACHE_DIR", "").strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".cache", "oto_torch_windows")
    os.makedirs(path, exist_ok=True)
    return path


def _wav_signature(path: str, mel_bins: int, hop_ms: float, win_ms: float, n_fft: int) -> str:
    st = os.stat(path)
    payload = {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "mel_bins": int(mel_bins),
        "hop_ms": float(hop_ms),
        "win_ms": float(win_ms),
        "n_fft": int(n_fft),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_min = 0.0
    f_max = sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]
        if right <= left:
            continue
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            fb[i - 1, j] = (j - left) / float(center - left)
        for j in range(center, right):
            fb[i - 1, j] = (right - j) / float(right - center)
    return fb


def _compute_full_log_mel(audio: np.ndarray, sr: int, mel_bins: int = DEFAULT_MEL_BINS, hop_ms: float = DEFAULT_HOP_MS, win_ms: float = DEFAULT_WIN_MS, n_fft: int = DEFAULT_NFFT) -> Dict[str, np.ndarray]:
    hop = max(1, int(sr * (hop_ms / 1000.0)))
    win = min(n_fft, max(256, int(sr * (win_ms / 1000.0))))
    window = np.hanning(win).astype(np.float32)
    fb = _mel_filterbank(sr, n_fft, mel_bins)

    frames = []
    db_vals = []
    times_ms = []
    for st in range(0, max(len(audio) - win + 1, 1), hop):
        fr_raw = audio[st:st + win]
        if len(fr_raw) < win:
            fr_raw = np.concatenate([fr_raw, np.zeros(win - len(fr_raw), dtype=np.float32)], axis=0)
        rms = float(np.sqrt(np.mean(fr_raw.astype(np.float64) ** 2) + 1e-12))
        db_vals.append(20.0 * np.log10(max(rms, 1e-7)))
        fr = fr_raw.astype(np.float32) * window
        if n_fft > win:
            fr = np.pad(fr, (0, n_fft - win))
        spec = np.fft.rfft(fr.astype(np.float64))
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel = fb @ power
        frames.append(np.log1p(np.maximum(mel, 0.0)).astype(np.float32))
        times_ms.append(st * 1000.0 / sr)

    if not frames:
        frames = [np.zeros((mel_bins,), dtype=np.float32)]
        db_vals = [-80.0]
        times_ms = [0.0]

    mel = np.stack(frames, axis=1).astype(np.float32)
    times = np.array(times_ms, dtype=np.float32)
    db_arr = np.array(db_vals, dtype=np.float32)
    return {
        "mel": mel,
        "times_ms": times,
        "db": db_arr,
    }


def get_or_build_full_mel(wav_path: str, mel_bins: int = DEFAULT_MEL_BINS, hop_ms: float = DEFAULT_HOP_MS, win_ms: float = DEFAULT_WIN_MS, n_fft: int = DEFAULT_NFFT) -> Dict[str, np.ndarray]:
    cache_dir = _default_torch_cache_dir()
    key = _wav_signature(wav_path, mel_bins, hop_ms, win_ms, n_fft)
    cache_path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.isfile(cache_path):
        data = np.load(cache_path)
        return {
            "mel": data["mel"],
            "times_ms": data["times_ms"],
            "db": data["db"],
        }
    audio, sr = _read_wav_mono_np(wav_path)
    if audio is None or sr is None or sr <= 0:
        raise RuntimeError(f"Failed to read wav: {wav_path}")
    data = _compute_full_log_mel(audio, sr, mel_bins=mel_bins, hop_ms=hop_ms, win_ms=win_ms, n_fft=n_fft)
    np.savez_compressed(cache_path, mel=data["mel"].astype(np.float16), times_ms=data["times_ms"], db=data["db"])
    return {
        "mel": data["mel"],
        "times_ms": data["times_ms"],
        "db": data["db"],
    }


def extract_centered_mel_window(wav_path: str, center_ms: float, window_frames: int = DEFAULT_WINDOW_FRAMES, mel_bins: int = DEFAULT_MEL_BINS) -> Tuple[np.ndarray, Dict[str, float]]:
    data = get_or_build_full_mel(wav_path, mel_bins=mel_bins)
    mel = data["mel"].astype(np.float32)
    times_ms = data["times_ms"].astype(np.float32)
    db = data["db"].astype(np.float32)
    if mel.ndim != 2:
        raise RuntimeError("Unexpected mel shape")
    frame_count = mel.shape[1]
    if frame_count <= 0:
        return np.zeros((mel_bins, window_frames), dtype=np.float32), {
            "local_peak_db": -80.0,
            "local_valley_db": -80.0,
            "mel_window_energy_mean": 0.0,
            "mel_window_silence_ratio": 1.0,
        }
    center_idx = int(np.argmin(np.abs(times_ms - float(center_ms)))) if len(times_ms) else 0
    half = window_frames // 2
    start = center_idx - half
    end = start + window_frames
    pad_left = max(0, -start)
    pad_right = max(0, end - frame_count)
    start = max(0, start)
    end = min(frame_count, end)
    window = mel[:, start:end]
    if pad_left or pad_right:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")
    window = window[:, :window_frames].astype(np.float32)
    db_slice = db[start:end] if len(db) else np.array([], dtype=np.float32)
    if pad_left or pad_right:
        db_slice = np.pad(db_slice, (pad_left, pad_right), mode="constant", constant_values=-80.0)
    db_slice = db_slice[:window_frames]
    energy = np.mean(window, axis=0) if window.size else np.zeros((window_frames,), dtype=np.float32)
    stats = {
        "local_peak_db": float(np.max(db_slice)) if len(db_slice) else -80.0,
        "local_valley_db": float(np.min(db_slice)) if len(db_slice) else -80.0,
        "mel_window_energy_mean": float(np.mean(energy)) if len(energy) else 0.0,
        "mel_window_silence_ratio": float(np.mean(db_slice <= -42.0)) if len(db_slice) else 1.0,
    }
    return window, stats


def get_numeric_feature_names() -> List[str]:
    return [name for name in FEATURE_NAMES if name not in CATEGORICAL_FEATURES]


def build_tabular_parts(feature_row: Dict[str, object], categorical_features: List[str] | None = None) -> Tuple[np.ndarray, Dict[str, str]]:
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    canonical = canonicalize_feature_row(feature_row)
    numeric_names = get_numeric_feature_names()
    numeric = np.array([float(canonical.get(name, 0.0) or 0.0) for name in numeric_names], dtype=np.float32)
    categorical = {name: str(feature_row.get(name, "") or "") for name in categorical_features}
    return numeric, categorical
