"""
Raw mel patch extraction and cache helpers for coupled v2 (weak end-to-end).

This module provides:
- patch spec normalization + hashing
- stable cache key generation
- log-mel computation
- onset/tail patch extraction
- cache writer/index helpers
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from core.oto_normalization import normalize_wav_key

logger = logging.getLogger(__name__)

MEL_PATCH_CACHE_VERSION = "v1"
DEFAULT_MEL_BINS = 80
DEFAULT_FRAME_HOP_MS = 5.0
DEFAULT_WIN_MS = 25.0
DEFAULT_ONSET_WINDOW_MS = (-80.0, 160.0)
DEFAULT_TAIL_WINDOW_MS = (-160.0, 80.0)
DEFAULT_PADDING_POLICY = "zero"
DEFAULT_NORMALIZATION = "per_patch_log_mel_standardize"


def _ensure_numpy():
    if np is None:
        raise RuntimeError("numpy is required for raw mel patches")


def _float_tuple(val) -> Tuple[float, float]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), float(val[1])
    raise ValueError("window must be a tuple/list of length 2")


def _normalize_component(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\\", "/")
    return text


def make_mel_patch_key(
    *,
    language: str,
    format_type: str,
    voicebank_id: str,
    wav_norm: str,
    alias_norm: str,
    occurrence_index: int,
    row_index_in_wav: int,
) -> str:
    parts = [
        _normalize_component(language),
        _normalize_component(format_type),
        _normalize_component(voicebank_id),
        _normalize_component(wav_norm),
        _normalize_component(alias_norm),
        str(int(occurrence_index)),
        str(int(row_index_in_wav)),
    ]
    return "|".join(parts)


def make_mel_patch_debug_key(
    *,
    voicebank_id: str,
    wav: str,
    alias: str,
    alias_norm: str,
    occurrence_index: int,
    row_index_in_wav: int,
) -> str:
    return (
        f"{voicebank_id} | {wav} | {alias} | {alias_norm} "
        f"| occ={int(occurrence_index)} | row={int(row_index_in_wav)}"
    )


def resolve_mel_patch_anchors(feature_row: Dict[str, object]) -> Tuple[float, float, str]:
    onset = float(feature_row.get("mel_offset_candidate_ms", 0.0) or 0.0)
    tail = float(feature_row.get("mel_cutoff_candidate_ms", 0.0) or 0.0)
    onset_src = "mel_candidate" if onset > 0.0 else "base_fallback"
    tail_src = "mel_candidate" if tail > 0.0 else "base_fallback"
    if onset <= 0.0:
        onset = float(feature_row.get("base_offset", 0.0) or 0.0)
    if tail <= 0.0:
        tail = float(feature_row.get("base_cutoff_abs", 0.0) or 0.0)
    source = "mel_candidate" if onset_src == "mel_candidate" and tail_src == "mel_candidate" else "base_fallback"
    return float(onset), float(tail), source


def build_patch_spec(
    *,
    mel_bins: int = DEFAULT_MEL_BINS,
    frame_hop_ms: float = DEFAULT_FRAME_HOP_MS,
    win_ms: float = DEFAULT_WIN_MS,
    onset_window_ms: Tuple[float, float] = DEFAULT_ONSET_WINDOW_MS,
    tail_window_ms: Tuple[float, float] = DEFAULT_TAIL_WINDOW_MS,
    normalization: str = DEFAULT_NORMALIZATION,
    onset_anchor_policy: str = "mel_offset_then_base_offset",
    tail_anchor_policy: str = "mel_cutoff_then_base_cutoff_abs",
    padding_policy: str = DEFAULT_PADDING_POLICY,
) -> Dict[str, object]:
    return {
        "mel_bins": int(mel_bins),
        "frame_hop_ms": float(frame_hop_ms),
        "win_ms": float(win_ms),
        "onset_window_ms": list(_float_tuple(onset_window_ms)),
        "tail_window_ms": list(_float_tuple(tail_window_ms)),
        "normalization": str(normalization),
        "onset_anchor_policy": str(onset_anchor_policy),
        "tail_anchor_policy": str(tail_anchor_policy),
        "padding_policy": str(padding_policy),
    }


def patch_spec_hash(patch_spec: Dict[str, object]) -> str:
    raw = json.dumps(patch_spec or {}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:8]


def default_patch_cache_root() -> str:
    configured = os.environ.get("UTOA_ML_MEL_PATCH_CACHE_DIR", "").strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_dir, ".cache", "oto_ml_mel_patches")
    os.makedirs(path, exist_ok=True)
    return path


def patch_cache_dir(
    language: str,
    format_type: str,
    patch_spec: Dict[str, object],
    cache_version: str = MEL_PATCH_CACHE_VERSION,
    root_dir: Optional[str] = None,
) -> str:
    root = root_dir or default_patch_cache_root()
    spec_hash = patch_spec_hash(patch_spec)
    return os.path.join(
        root,
        _normalize_component(language),
        _normalize_component(format_type),
        str(cache_version),
        spec_hash,
    )


def _mel_filterbank(sr: int, n_fft: int, mel_bins: int) -> "np.ndarray":
    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_points = np.linspace(mel_min, mel_max, mel_bins + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((mel_bins, n_fft // 2 + 1), dtype=np.float64)
    for i in range(1, mel_bins + 1):
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


def compute_log_mel(
    audio: "np.ndarray",
    sr: int,
    *,
    mel_bins: int,
    frame_hop_ms: float,
    win_ms: float,
    n_fft: int = 1024,
) -> Tuple["np.ndarray", float]:
    _ensure_numpy()
    if audio is None or sr <= 0:
        raise ValueError("audio/sr required")
    hop = max(1, int(sr * (float(frame_hop_ms) / 1000.0)))
    win = max(256, int(sr * (float(win_ms) / 1000.0)))
    win = min(n_fft, win)
    window = np.hanning(win).astype(np.float64)
    fb = _mel_filterbank(sr, n_fft, mel_bins)
    frames: List["np.ndarray"] = []
    for st in range(0, max(len(audio) - win + 1, 1), hop):
        fr_raw = audio[st:st + win]
        if len(fr_raw) < win:
            fr_raw = np.concatenate([fr_raw, np.zeros(win - len(fr_raw), dtype=np.float32)], axis=0)
        fr = fr_raw.astype(np.float64) * window
        if n_fft > win:
            fr = np.pad(fr, (0, n_fft - win))
        spec = np.fft.rfft(fr)
        power = (spec.real ** 2 + spec.imag ** 2)
        mel = fb @ power
        frames.append(np.log1p(np.maximum(mel, 0.0)))
    if not frames:
        raise ValueError("empty mel frames")
    mel_arr = np.stack(frames, axis=0).astype(np.float32)
    return mel_arr, float(frame_hop_ms)


def _window_frames(window_ms: Tuple[float, float], hop_ms: float) -> int:
    start, end = _float_tuple(window_ms)
    span = float(end) - float(start)
    return max(1, int(round(span / float(hop_ms))) + 1)


def extract_patch_from_mel(
    mel: "np.ndarray",
    *,
    anchor_ms: float,
    window_ms: Tuple[float, float],
    hop_ms: float,
    mel_bins: int,
) -> "np.ndarray":
    _ensure_numpy()
    if mel is None or mel.size == 0:
        raise ValueError("mel is empty")
    frames = int(mel.shape[0])
    if frames <= 0:
        raise ValueError("mel has no frames")
    if mel.shape[1] != int(mel_bins):
        raise ValueError("mel bin mismatch")
    start_offset = int(round(float(window_ms[0]) / float(hop_ms)))
    frame_len = _window_frames(window_ms, hop_ms)
    anchor_idx = int(round(float(anchor_ms) / float(hop_ms)))
    start_idx = anchor_idx + start_offset
    end_idx = start_idx + frame_len
    out = np.zeros((frame_len, mel_bins), dtype=np.float32)
    src_start = max(start_idx, 0)
    src_end = min(end_idx, frames)
    if src_end > src_start:
        dst_start = max(0, -start_idx)
        dst_end = dst_start + (src_end - src_start)
        out[dst_start:dst_end] = mel[src_start:src_end]
    return out


def normalize_patch(patch: "np.ndarray", normalization: str) -> "np.ndarray":
    _ensure_numpy()
    if normalization != DEFAULT_NORMALIZATION:
        return patch.astype(np.float32)
    mean = float(np.mean(patch))
    std = float(np.std(patch))
    if std < 1e-6:
        std = 1.0
    return ((patch - mean) / std).astype(np.float32)


def read_wav_mono_np(path: str) -> Tuple[Optional["np.ndarray"], Optional[int]]:
    from core.oto_generator import _read_wav_mono_np

    audio, sr = _read_wav_mono_np(path)
    return audio, sr


def find_wav_path(wav_name: str, wav_dir: str, wav_index: Optional[Dict[str, str]] = None) -> str:
    key = normalize_wav_key(wav_name)
    if wav_index and key in wav_index:
        return wav_index[key]
    if wav_dir and os.path.isdir(wav_dir):
        candidate = os.path.join(wav_dir, wav_name)
        if os.path.isfile(candidate):
            return candidate
    return ""


def build_wav_index(wav_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not wav_dir or not os.path.isdir(wav_dir):
        return index
    for dp, _dns, fns in os.walk(wav_dir):
        for fn in fns:
            if fn.lower().endswith(".wav"):
                index[normalize_wav_key(fn)] = os.path.join(dp, fn)
    return index


def extract_patches_for_row(
    *,
    wav_path: str,
    onset_anchor_ms: float,
    tail_anchor_ms: float,
    patch_spec: Dict[str, object],
    mel_cache: Optional[Dict[str, Tuple["np.ndarray", float]]] = None,
) -> Tuple["np.ndarray", "np.ndarray"]:
    _ensure_numpy()
    if not wav_path or not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)
    mel_cache = mel_cache if mel_cache is not None else {}
    mel_bins = int(patch_spec["mel_bins"])
    hop_ms = float(patch_spec["frame_hop_ms"])
    win_ms = float(patch_spec["win_ms"])
    onset_win = tuple(patch_spec["onset_window_ms"])
    tail_win = tuple(patch_spec["tail_window_ms"])
    normalization = str(patch_spec["normalization"])
    if wav_path in mel_cache:
        mel, cached_hop = mel_cache[wav_path]
    else:
        audio, sr = read_wav_mono_np(wav_path)
        if audio is None or sr is None:
            raise ValueError("failed to read wav")
        mel, cached_hop = compute_log_mel(
            audio,
            sr,
            mel_bins=mel_bins,
            frame_hop_ms=hop_ms,
            win_ms=win_ms,
        )
        mel_cache[wav_path] = (mel, cached_hop)
    onset_patch = extract_patch_from_mel(
        mel,
        anchor_ms=onset_anchor_ms,
        window_ms=onset_win,
        hop_ms=hop_ms,
        mel_bins=mel_bins,
    )
    tail_patch = extract_patch_from_mel(
        mel,
        anchor_ms=tail_anchor_ms,
        window_ms=tail_win,
        hop_ms=hop_ms,
        mel_bins=mel_bins,
    )
    onset_patch = normalize_patch(onset_patch, normalization)
    tail_patch = normalize_patch(tail_patch, normalization)
    return onset_patch, tail_patch


def _patch_quality_metrics(patch: "np.ndarray") -> Dict[str, float]:
    if patch is None or patch.size == 0:
        return {"mean": 0.0, "std": 0.0, "low_ratio": 1.0}
    mean = float(np.mean(patch))
    std = float(np.std(patch))
    if std < 1e-6:
        low_ratio = 1.0 if mean <= 0.0 else 0.0
    else:
        threshold = mean - (0.65 * std)
        low_ratio = float(np.mean(patch <= threshold))
    return {"mean": mean, "std": std, "low_ratio": low_ratio}


def extract_patches_for_row_with_quality(
    *,
    wav_path: str,
    onset_anchor_ms: float,
    tail_anchor_ms: float,
    patch_spec: Dict[str, object],
    mel_cache: Optional[Dict[str, Tuple["np.ndarray", float]]] = None,
) -> Tuple["np.ndarray", "np.ndarray", Dict[str, float]]:
    _ensure_numpy()
    if not wav_path or not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)
    mel_cache = mel_cache if mel_cache is not None else {}
    mel_bins = int(patch_spec["mel_bins"])
    hop_ms = float(patch_spec["frame_hop_ms"])
    win_ms = float(patch_spec["win_ms"])
    onset_win = tuple(patch_spec["onset_window_ms"])
    tail_win = tuple(patch_spec["tail_window_ms"])
    normalization = str(patch_spec["normalization"])
    if wav_path in mel_cache:
        mel, cached_hop = mel_cache[wav_path]
    else:
        audio, sr = read_wav_mono_np(wav_path)
        if audio is None or sr is None:
            raise ValueError("failed to read wav")
        mel, cached_hop = compute_log_mel(
            audio,
            sr,
            mel_bins=mel_bins,
            frame_hop_ms=hop_ms,
            win_ms=win_ms,
        )
        mel_cache[wav_path] = (mel, cached_hop)
    onset_raw = extract_patch_from_mel(
        mel,
        anchor_ms=onset_anchor_ms,
        window_ms=onset_win,
        hop_ms=hop_ms,
        mel_bins=mel_bins,
    )
    tail_raw = extract_patch_from_mel(
        mel,
        anchor_ms=tail_anchor_ms,
        window_ms=tail_win,
        hop_ms=hop_ms,
        mel_bins=mel_bins,
    )
    quality = {}
    quality.update({f"onset_{k}": v for k, v in _patch_quality_metrics(onset_raw).items()})
    quality.update({f"tail_{k}": v for k, v in _patch_quality_metrics(tail_raw).items()})
    onset_patch = normalize_patch(onset_raw, normalization)
    tail_patch = normalize_patch(tail_raw, normalization)
    return onset_patch, tail_patch, quality


@dataclass
class MelPatchCacheIndex:
    cache_dir: str
    manifest: Dict[str, object]
    patch_spec: Dict[str, object]
    shard_files: List[str]
    key_index: Dict[str, Tuple[int, int]]
    max_shard_cache: int = 2

    _shard_cache: Dict[int, Tuple["np.ndarray", "np.ndarray"]] = None
    _shard_order: List[int] = None

    @classmethod
    def load(cls, cache_dir: str, max_shard_cache: int = 2) -> "MelPatchCacheIndex":
        _ensure_numpy()
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        patch_spec = manifest.get("patch_spec") or {}
        shard_files = list(manifest.get("shards") or [])
        key_index: Dict[str, Tuple[int, int]] = {}
        for shard_idx, shard_name in enumerate(shard_files):
            shard_path = os.path.join(cache_dir, shard_name)
            data = np.load(shard_path, allow_pickle=True)
            keys = data.get("keys")
            if keys is None:
                continue
            for row_idx, key in enumerate(keys.tolist()):
                key_index[str(key)] = (shard_idx, int(row_idx))
        return cls(
            cache_dir=cache_dir,
            manifest=manifest,
            patch_spec=patch_spec,
            shard_files=shard_files,
            key_index=key_index,
            max_shard_cache=max_shard_cache,
            _shard_cache={},
            _shard_order=[],
        )

    def has_key(self, key: str) -> bool:
        return str(key) in self.key_index

    def _load_shard(self, shard_idx: int) -> Tuple["np.ndarray", "np.ndarray"]:
        if shard_idx in self._shard_cache:
            if shard_idx in self._shard_order:
                self._shard_order.remove(shard_idx)
            self._shard_order.append(shard_idx)
            return self._shard_cache[shard_idx]
        shard_name = self.shard_files[shard_idx]
        shard_path = os.path.join(self.cache_dir, shard_name)
        data = np.load(shard_path, allow_pickle=True)
        onset = data["onset_patches"]
        tail = data["tail_patches"]
        self._shard_cache[shard_idx] = (onset, tail)
        self._shard_order.append(shard_idx)
        while len(self._shard_order) > int(self.max_shard_cache):
            drop = self._shard_order.pop(0)
            self._shard_cache.pop(drop, None)
        return onset, tail

    def get(self, key: str) -> Tuple["np.ndarray", "np.ndarray"]:
        if str(key) not in self.key_index:
            raise KeyError(str(key))
        shard_idx, row_idx = self.key_index[str(key)]
        onset, tail = self._load_shard(shard_idx)
        return onset[row_idx], tail[row_idx]

    def get_batch(self, keys: Iterable[str]) -> Tuple["np.ndarray", "np.ndarray"]:
        onset_list = []
        tail_list = []
        for key in keys:
            o, t = self.get(str(key))
            onset_list.append(o)
            tail_list.append(t)
        return np.stack(onset_list, axis=0), np.stack(tail_list, axis=0)


class MelPatchCacheWriter:
    def __init__(
        self,
        cache_dir: str,
        patch_spec: Dict[str, object],
        *,
        cache_version: str = MEL_PATCH_CACHE_VERSION,
        max_rows_per_shard: int = 2000,
    ):
        _ensure_numpy()
        self.cache_dir = cache_dir
        self.patch_spec = patch_spec
        self.cache_version = cache_version
        self.max_rows_per_shard = int(max_rows_per_shard)
        self.keys: List[str] = []
        self.onset: List["np.ndarray"] = []
        self.tail: List["np.ndarray"] = []
        self.shards: List[str] = []
        os.makedirs(self.cache_dir, exist_ok=True)

    def append(self, key: str, onset_patch: "np.ndarray", tail_patch: "np.ndarray") -> None:
        self.keys.append(str(key))
        self.onset.append(onset_patch.astype(np.float32))
        self.tail.append(tail_patch.astype(np.float32))
        if len(self.keys) >= self.max_rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self.keys:
            return
        shard_idx = len(self.shards) + 1
        shard_name = f"patches_{shard_idx:06d}.npz"
        shard_path = os.path.join(self.cache_dir, shard_name)
        np.savez_compressed(
            shard_path,
            keys=np.asarray(self.keys, dtype=object),
            onset_patches=np.stack(self.onset, axis=0),
            tail_patches=np.stack(self.tail, axis=0),
        )
        self.shards.append(shard_name)
        self.keys = []
        self.onset = []
        self.tail = []

    def finalize(self, total_rows: int) -> str:
        self._flush()
        manifest = {
            "cache_version": self.cache_version,
            "backend": "coupled_nn_v2_rawmel",
            "patch_spec": dict(self.patch_spec),
            "patch_spec_hash": patch_spec_hash(self.patch_spec),
            "mel_bins": int(self.patch_spec["mel_bins"]),
            "frame_hop_ms": float(self.patch_spec["frame_hop_ms"]),
            "onset_window_ms": list(self.patch_spec["onset_window_ms"]),
            "tail_window_ms": list(self.patch_spec["tail_window_ms"]),
            "normalization": str(self.patch_spec["normalization"]),
            "onset_anchor_policy": str(self.patch_spec.get("onset_anchor_policy", "")),
            "tail_anchor_policy": str(self.patch_spec.get("tail_anchor_policy", "")),
            "padding_policy": str(self.patch_spec.get("padding_policy", "")),
            "source_feature_version": str(self.patch_spec.get("source_feature_version", "")),
            "row_count": int(total_rows),
            "shard_count": int(len(self.shards)),
            "shards": list(self.shards),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest_path
