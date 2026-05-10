"""파일 단위 오디오 후보 분석 — 14-2 §2단계

각 WAV에 대해 한 번만 계산해서 캐시에 저장한다.
VC/VV row 처리 시 이 캐시를 참조해 vowel segment 매칭과
consonant onset 탐지에 사용한다.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from core.coarse_crnn.audio import estimate_active_region_seconds, load_wav_mono, log_mel_spectrogram


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------

@dataclass
class OnsetPeak:
    time_ms: float
    strength: float   # 0~1 정규화


@dataclass
class VowelSegmentCandidate:
    segment_id: int
    start_ms: float
    end_ms: float
    center_ms: float
    vowel_id: str           # "a"/"i"/"u"/"e"/"o"/"eo"/"eu"/"other"
    vowel_confidence: float # 0~1
    voiced_score: float     # voiced region 비율
    stability_score: float  # 낮은 mel delta = 안정
    energy_mean: float      # 정규화 에너지
    mel_delta_mean: float   # 프레임 간 mel 변화량 평균


@dataclass
class MelDeltaPeak:
    time_ms: float
    strength: float   # VV transition 힌트


@dataclass
class AudioCandidates:
    wav_path: str
    duration_ms: float
    active_start_ms: float
    active_end_ms: float
    onset_peaks: list[OnsetPeak]
    stable_vowel_segments: list[VowelSegmentCandidate]
    mel_delta_peaks: list[MelDeltaPeak]


# ---------------------------------------------------------------------------
# 핵심 분석 함수
# ---------------------------------------------------------------------------

def compute_audio_candidates(
    wav_path: str,
    *,
    target_sr: int = 16000,
    n_mels: int = 64,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> AudioCandidates:
    samples, sr, duration_sec = load_wav_mono(wav_path, target_sr=target_sr)
    duration_ms = duration_sec * 1000.0
    hop_sec = hop_ms / 1000.0

    # active region
    active = estimate_active_region_seconds(samples, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if active is None:
        active_start_ms, active_end_ms = 0.0, duration_ms
    else:
        active_start_ms, active_end_ms = active[0] * 1000.0, active[1] * 1000.0

    if samples.size <= 0:
        return AudioCandidates(
            wav_path=wav_path,
            duration_ms=duration_ms,
            active_start_ms=active_start_ms,
            active_end_ms=active_end_ms,
            onset_peaks=[],
            stable_vowel_segments=[],
            mel_delta_peaks=[],
        )

    logmel, actual_hop_sec = log_mel_spectrogram(
        samples, sr, n_mels=n_mels, frame_ms=frame_ms, hop_ms=hop_ms
    )
    actual_hop_ms = actual_hop_sec * 1000.0

    rms = _compute_rms(samples, sr, hop_ms=hop_ms, frame_ms=frame_ms)
    n_frames = min(logmel.shape[0], rms.shape[0])
    logmel = logmel[:n_frames]
    rms = rms[:n_frames]

    onset_peaks = _estimate_onset_peaks(logmel, rms, actual_hop_ms)
    voiced_mask = _estimate_voiced_mask(rms, samples, sr, hop_ms=hop_ms)
    voiced_mask = voiced_mask[:n_frames]
    mel_delta = _compute_mel_delta(logmel)
    mel_delta_peaks = _estimate_mel_delta_peaks(mel_delta, actual_hop_ms)
    stable_vowel_segments = _extract_stable_vowel_segments(
        logmel, mel_delta, voiced_mask, rms,
        actual_hop_ms=actual_hop_ms,
        active_start_ms=active_start_ms,
        active_end_ms=active_end_ms,
        onset_peaks=onset_peaks,
    )

    return AudioCandidates(
        wav_path=wav_path,
        duration_ms=duration_ms,
        active_start_ms=active_start_ms,
        active_end_ms=active_end_ms,
        onset_peaks=onset_peaks,
        stable_vowel_segments=stable_vowel_segments,
        mel_delta_peaks=mel_delta_peaks,
    )


# ---------------------------------------------------------------------------
# 내부 분석 함수
# ---------------------------------------------------------------------------

def _compute_rms(samples: np.ndarray, sr: int, *, hop_ms: float, frame_ms: float) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    win = max(16, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    frame_count = 1 + (x.size - win) // hop
    rms = np.zeros(frame_count, dtype=np.float32)
    for i in range(frame_count):
        s = i * hop
        rms[i] = float(np.sqrt(np.mean(np.square(x[s:s + win]), dtype=np.float64) + 1e-12))
    return rms


def _compute_mel_delta(logmel: np.ndarray) -> np.ndarray:
    """프레임 간 mel 변화량 (L1 norm). shape: (T,)"""
    if logmel.shape[0] < 2:
        return np.zeros(logmel.shape[0], dtype=np.float32)
    delta = np.zeros(logmel.shape[0], dtype=np.float32)
    delta[1:] = np.mean(np.abs(np.diff(logmel, axis=0)), axis=1)
    delta[0] = delta[1]
    return delta


def _estimate_onset_peaks(
    logmel: np.ndarray,
    rms: np.ndarray,
    hop_ms: float,
    *,
    min_gap_ms: float = 60.0,
    top_k: int = 64,
) -> list[OnsetPeak]:
    """Spectral flux 기반 onset 탐지."""
    n = min(logmel.shape[0], rms.shape[0])
    if n < 3:
        return []
    flux = np.zeros(n, dtype=np.float32)
    # half-wave rectified spectral flux
    diff = np.diff(logmel[:n], axis=0)
    flux[1:] = np.sum(np.maximum(diff, 0.0), axis=1)
    # energy 가중: 낮은 에너지 구간 억제
    rms_norm = rms[:n] / (np.max(rms[:n]) + 1e-8)
    flux = flux * np.sqrt(rms_norm)

    if np.max(flux) < 1e-8:
        return []
    flux_norm = flux / (np.max(flux) + 1e-8)

    min_gap_frames = max(1, int(round(min_gap_ms / hop_ms)))
    threshold = float(np.quantile(flux_norm, 0.85))

    peaks: list[tuple[float, int]] = []  # (strength, frame_idx)
    for i in range(1, n - 1):
        if flux_norm[i] >= threshold and flux_norm[i] >= flux_norm[i - 1] and flux_norm[i] >= flux_norm[i + 1]:
            peaks.append((float(flux_norm[i]), i))

    # NMS
    peaks.sort(key=lambda x: -x[0])
    selected: list[tuple[float, int]] = []
    used = np.zeros(n, dtype=bool)
    for strength, idx in peaks:
        lo = max(0, idx - min_gap_frames)
        hi = min(n, idx + min_gap_frames + 1)
        if not np.any(used[lo:hi]):
            selected.append((strength, idx))
            used[lo:hi] = True
        if len(selected) >= top_k:
            break

    selected.sort(key=lambda x: x[1])
    return [OnsetPeak(time_ms=idx * hop_ms, strength=s) for s, idx in selected]


def _estimate_voiced_mask(
    rms: np.ndarray,
    samples: np.ndarray,
    sr: int,
    *,
    hop_ms: float,
) -> np.ndarray:
    """voiced frame mask — 에너지 + ZCR 기반."""
    n = rms.shape[0]
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    win = hop * 2

    # ZCR per frame
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    zcr = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        frame = x[s: s + win]
        if frame.size > 1:
            zcr[i] = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)

    rms_norm = rms / (np.max(rms) + 1e-8)
    energy_threshold = float(np.quantile(rms_norm, 0.30))
    zcr_threshold = float(np.quantile(zcr[rms_norm > energy_threshold] if np.any(rms_norm > energy_threshold) else zcr, 0.70))

    voiced = (rms_norm >= energy_threshold) & (zcr <= zcr_threshold)
    # 짧은 gap 메우기 (80ms)
    gap_frames = max(1, int(round(80.0 / hop_ms)))
    i = 0
    voiced_list = voiced.tolist()
    while i < n:
        if not voiced_list[i]:
            i += 1
            continue
        j = i
        while j < n and voiced_list[j]:
            j += 1
        k = j
        while k < n and not voiced_list[k]:
            k += 1
        if k < n and voiced_list[k] and (k - j) <= gap_frames:
            for m in range(j, k):
                voiced_list[m] = True
        i = j + 1
    return np.array(voiced_list, dtype=bool)


def _estimate_mel_delta_peaks(
    mel_delta: np.ndarray,
    hop_ms: float,
    *,
    min_gap_ms: float = 40.0,
    top_k: int = 64,
) -> list[MelDeltaPeak]:
    """VV transition 힌트용 mel delta peak."""
    n = mel_delta.shape[0]
    if n < 3:
        return []
    d = mel_delta / (np.max(mel_delta) + 1e-8)
    threshold = float(np.quantile(d, 0.80))
    min_gap = max(1, int(round(min_gap_ms / hop_ms)))

    peaks: list[tuple[float, int]] = []
    for i in range(1, n - 1):
        if d[i] >= threshold and d[i] >= d[i - 1] and d[i] >= d[i + 1]:
            peaks.append((float(d[i]), i))

    peaks.sort(key=lambda x: -x[0])
    selected: list[tuple[float, int]] = []
    used = np.zeros(n, dtype=bool)
    for strength, idx in peaks:
        lo = max(0, idx - min_gap)
        hi = min(n, idx + min_gap + 1)
        if not np.any(used[lo:hi]):
            selected.append((strength, idx))
            used[lo:hi] = True
        if len(selected) >= top_k:
            break

    selected.sort(key=lambda x: x[1])
    return [MelDeltaPeak(time_ms=idx * hop_ms, strength=s) for s, idx in selected]


def _extract_stable_vowel_segments(
    logmel: np.ndarray,
    mel_delta: np.ndarray,
    voiced_mask: np.ndarray,
    rms: np.ndarray,
    *,
    actual_hop_ms: float,
    active_start_ms: float,
    active_end_ms: float,
    onset_peaks: list[OnsetPeak],
    min_duration_ms: float = 60.0,
    max_segments: int = 32,
) -> list[VowelSegmentCandidate]:
    n = min(logmel.shape[0], mel_delta.shape[0], voiced_mask.shape[0], rms.shape[0])
    if n <= 0:
        return []

    # active 구간 mask
    active_start_frame = max(0, int(active_start_ms / actual_hop_ms))
    active_end_frame = min(n, int(active_end_ms / actual_hop_ms) + 1)

    # onset 시간 집합 (onset 직후 짧은 구간은 자음 → vowel segment 분리 경계)
    onset_times_ms = {p.time_ms for p in onset_peaks}

    # voiced + active mask
    mask = voiced_mask[:n].copy()
    mask[:active_start_frame] = False
    mask[active_end_frame:] = False

    # voiced run 추출
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        runs.append((start, i))

    min_frames = max(1, int(round(min_duration_ms / actual_hop_ms)))
    segments: list[VowelSegmentCandidate] = []
    seg_id = 0

    for run_start, run_end in runs:
        if run_end - run_start < min_frames:
            continue

        # onset 위치에서 subrun 분리
        sub_starts = [run_start]
        for onset_ms in sorted(onset_times_ms):
            onset_frame = int(onset_ms / actual_hop_ms)
            # onset 직후 30ms는 자음 — 그 이후부터 새 sub-run
            split_frame = onset_frame + max(1, int(30.0 / actual_hop_ms))
            if run_start < split_frame < run_end:
                sub_starts.append(split_frame)
        sub_starts.append(run_end)

        for si in range(len(sub_starts) - 1):
            s, e = sub_starts[si], sub_starts[si + 1]
            if e - s < min_frames:
                continue

            mel_chunk = logmel[s:e]
            delta_chunk = mel_delta[s:e]
            rms_chunk = rms[s:e]

            mel_mean = np.mean(mel_chunk, axis=0)
            stability = float(1.0 / (1.0 + np.mean(delta_chunk)))
            voiced_score = float(np.mean(voiced_mask[s:e].astype(np.float32)))
            energy_mean = float(np.mean(rms_chunk) / (np.max(rms[:n]) + 1e-8))
            mel_delta_mean = float(np.mean(delta_chunk))

            vowel_id, vowel_conf = _classify_vowel_mel(mel_mean)

            segments.append(VowelSegmentCandidate(
                segment_id=seg_id,
                start_ms=float(s * actual_hop_ms),
                end_ms=float(e * actual_hop_ms),
                center_ms=float((s + e) / 2.0 * actual_hop_ms),
                vowel_id=vowel_id,
                vowel_confidence=vowel_conf,
                voiced_score=voiced_score,
                stability_score=stability,
                energy_mean=energy_mean,
                mel_delta_mean=mel_delta_mean,
            ))
            seg_id += 1

        if len(segments) >= max_segments:
            break

    return segments[:max_segments]


# ---------------------------------------------------------------------------
# 모음 분류 (rule-based heuristic, n_mels=64 기준)
# ---------------------------------------------------------------------------

# n_mels=64 기준 대략적인 band 구분 (SR=16kHz)
# mel band 0~10:  0~700 Hz  (F1 저역)
# mel band 11~25: 700~2000 Hz (F1 고역/F2 저역)
# mel band 26~40: 2000~4000 Hz (F2 고역/F3)
# mel band 41~63: 4000+ Hz

def _classify_vowel_mel(mel_mean: np.ndarray) -> tuple[str, float]:
    """log-mel 평균 벡터에서 모음 ID를 heuristic으로 추정."""
    m = np.asarray(mel_mean, dtype=np.float32)
    if m.size < 20:
        return "other", 0.0

    n = m.size
    # softmax-normalize to [0,1]
    m_exp = np.exp(m - np.max(m))
    m_prob = m_exp / (m_exp.sum() + 1e-8)

    # 대역별 에너지 비율
    b0 = int(n * 0.00); b1 = int(n * 0.18)  # F1 저역 (~700Hz)
    b2 = int(n * 0.40)                        # F2 저역 (~2kHz)
    b3 = int(n * 0.60)                        # F2 고역 (~3.5kHz)

    e_f1_lo = float(m_prob[b0:b1].sum())   # F1 저역 에너지
    e_f1_f2 = float(m_prob[b1:b2].sum())   # F1-F2 중간
    e_f2_hi = float(m_prob[b2:b3].sum())   # F2 고역
    e_hi    = float(m_prob[b3:].sum())      # 고주파

    # 간단한 decision rule (Japanese 5모음 + 한국어 eo/eu 우선)
    # /a/ : F1 높고 넓게 분포 → e_f1_lo + e_f1_f2 높음
    # /i/ : F1 낮고 F2 높음 → e_f2_hi 높음, e_f1_lo 낮음
    # /u/ : F1/F2 둘다 낮음 → e_f1_lo 높지만 e_f2_hi 낮음
    # /e/ : F1 중간 F2 중간-높음
    # /o/ : F1 중간-높음 F2 낮음

    scores: dict[str, float] = {
        "a":  e_f1_lo * 1.5 + e_f1_f2 * 1.2 - e_f2_hi * 0.8,
        "i":  e_f2_hi * 2.0 + e_hi * 0.5 - e_f1_lo * 1.5,
        "u":  e_f1_lo * 1.2 - e_f2_hi * 1.0 - e_hi * 0.5,
        "e":  e_f1_f2 * 1.5 + e_f2_hi * 0.8 - e_f1_lo * 0.5,
        "o":  e_f1_lo * 0.8 + e_f1_f2 * 0.5 - e_f2_hi * 1.2,
        "eo": e_f1_f2 * 1.3 + e_f1_lo * 0.5 - e_f2_hi * 0.5,
        "eu": e_f1_lo * 1.0 - e_f2_hi * 0.8 - e_hi * 0.3,
    }

    best_vowel = max(scores, key=lambda k: scores[k])
    best_score = scores[best_vowel]
    # softmax confidence over scores
    arr = np.array(list(scores.values()), dtype=np.float32)
    arr_exp = np.exp(arr - np.max(arr))
    probs = arr_exp / (arr_exp.sum() + 1e-8)
    confidence = float(probs[list(scores.keys()).index(best_vowel)])

    if confidence < 0.25:
        return "other", confidence
    return best_vowel, confidence


# ---------------------------------------------------------------------------
# 직렬화 / 역직렬화
# ---------------------------------------------------------------------------

def _candidates_to_dict(c: AudioCandidates) -> dict[str, Any]:
    return {
        "wav_path": c.wav_path,
        "duration_ms": c.duration_ms,
        "active_start_ms": c.active_start_ms,
        "active_end_ms": c.active_end_ms,
        "onset_peaks": [asdict(p) for p in c.onset_peaks],
        "stable_vowel_segments": [asdict(s) for s in c.stable_vowel_segments],
        "mel_delta_peaks": [asdict(p) for p in c.mel_delta_peaks],
    }


def _candidates_from_dict(d: dict[str, Any]) -> AudioCandidates:
    return AudioCandidates(
        wav_path=str(d.get("wav_path", "")),
        duration_ms=float(d.get("duration_ms", 0.0)),
        active_start_ms=float(d.get("active_start_ms", 0.0)),
        active_end_ms=float(d.get("active_end_ms", 0.0)),
        onset_peaks=[OnsetPeak(**p) for p in d.get("onset_peaks", [])],
        stable_vowel_segments=[VowelSegmentCandidate(**s) for s in d.get("stable_vowel_segments", [])],
        mel_delta_peaks=[MelDeltaPeak(**p) for p in d.get("mel_delta_peaks", [])],
    )


# ---------------------------------------------------------------------------
# 캐시 I/O
# ---------------------------------------------------------------------------

def _wav_cache_key(wav_path: str) -> str:
    """파일 경로 + mtime 기반 캐시 키 (SHA256 축약)."""
    try:
        mtime = str(os.path.getmtime(wav_path))
    except OSError:
        mtime = "0"
    raw = f"{os.path.abspath(wav_path)}|{mtime}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def load_or_compute_candidates(
    wav_path: str,
    cache_dir: str,
    *,
    readonly: bool = False,
    target_sr: int = 16000,
    n_mels: int = 64,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> AudioCandidates:
    cache_key = _wav_cache_key(wav_path)
    cache_path = os.path.join(cache_dir, cache_key + ".json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return _candidates_from_dict(json.load(f))
        except Exception:
            pass  # 캐시 손상 → 재계산

    candidates = compute_audio_candidates(
        wav_path,
        target_sr=target_sr,
        n_mels=n_mels,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    if not readonly:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(_candidates_to_dict(candidates), f, ensure_ascii=False)
        except Exception:
            pass

    return candidates


# ---------------------------------------------------------------------------
# VC/VV 매칭 유틸
# ---------------------------------------------------------------------------

_VOWEL_CLASS_NORM = {
    "a": 1 / 6, "i": 2 / 6, "u": 3 / 6, "e": 4 / 6, "o": 5 / 6,
    "eo": 4 / 6, "eu": 3 / 6,
}


def find_best_vowel_segment(
    candidates: AudioCandidates,
    target_vowel_id_norm: float,
    *,
    hint_center_ms: float | None = None,
    hint_weight: float = 0.3,
) -> VowelSegmentCandidate | None:
    """left_vowel_id_norm에 가장 잘 맞는 vowel segment를 반환."""
    if not candidates.stable_vowel_segments:
        return None

    segs = candidates.stable_vowel_segments

    def score(seg: VowelSegmentCandidate) -> float:
        seg_norm = _VOWEL_CLASS_NORM.get(seg.vowel_id, 0.0)
        vowel_match = 1.0 - abs(seg_norm - target_vowel_id_norm) * 3.0
        vowel_match = max(-1.0, min(1.0, vowel_match))
        s = (2.0 * vowel_match * seg.vowel_confidence
             + 0.7 * seg.stability_score
             + 0.5 * seg.voiced_score
             - 0.5 * seg.mel_delta_mean)
        if hint_center_ms is not None and hint_weight > 0.0:
            dist_ms = abs(seg.center_ms - hint_center_ms)
            proximity = max(0.0, 1.0 - dist_ms / 2000.0)
            s += hint_weight * proximity
        return s

    best = max(segs, key=score)
    best_score = score(best)
    if best_score < -0.5:
        return None
    return best


def find_nearest_onset_after(
    candidates: AudioCandidates,
    after_ms: float,
    *,
    search_window_ms: float = 500.0,
) -> OnsetPeak | None:
    """after_ms 이후 가장 가까운 onset peak."""
    peaks = [p for p in candidates.onset_peaks
             if after_ms <= p.time_ms <= after_ms + search_window_ms]
    if not peaks:
        return None
    return min(peaks, key=lambda p: p.time_ms)


__all__ = [
    "AudioCandidates",
    "MelDeltaPeak",
    "OnsetPeak",
    "VowelSegmentCandidate",
    "compute_audio_candidates",
    "find_best_vowel_segment",
    "find_nearest_onset_after",
    "load_or_compute_candidates",
]
