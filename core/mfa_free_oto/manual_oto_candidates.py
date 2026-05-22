from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .features import extract_features


SCORABLE_ANCHORS = ("offset", "overlap", "preutterance", "fixed_end")
ALL_ANCHORS = SCORABLE_ANCHORS + ("cutoff",)
ROLE_VALUES = ("cv", "vc", "vcv", "v", "vv", "br", "sil", "other", "invalid")
FORMAT_VALUES = ("cv", "cvvc", "vcv", "cvc", "cmpx", "vccv", "cv-vc")
LANGUAGE_VALUES = ("japanese", "korean", "english", "ja", "jp", "ko", "kr", "en")
ALIAS_FAMILY_VALUES = (
    "plain",
    "pitch_suffix",
    "weak_suffix",
    "power_suffix",
    "leading_n",
    "vowel_transition",
    "punct_transition",
    "breath_silence",
    "unknown",
)


@dataclass(frozen=True)
class ManualOtoCandidateTracks:
    times_ms: np.ndarray
    candidate_indices: Mapping[str, tuple[int, ...]]
    anchor_scores: Mapping[str, np.ndarray]
    tracks: Mapping[str, np.ndarray]
    duration_ms: float
    encoder: str

    def candidate_times_ms(self, anchor: str) -> list[float]:
        indices = self.candidate_indices.get(anchor, ())
        return [float(self.times_ms[idx]) for idx in indices if 0 <= idx < self.times_ms.size]


def _track(scores: Mapping[str, np.ndarray], key: str, size: int) -> np.ndarray:
    arr = np.asarray(scores.get(key, np.zeros((size,), dtype=np.float32)), dtype=np.float32)
    if arr.size != size:
        return np.resize(arr, (size,)).astype(np.float32)
    return arr


def _unit(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 5.0))
    hi = float(np.percentile(arr, 95.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _local_peak_indices(values: np.ndarray, *, min_score: float) -> list[int]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    out: list[int] = []
    for idx in range(arr.size):
        left = arr[idx - 1] if idx > 0 else -1.0
        right = arr[idx + 1] if idx + 1 < arr.size else -1.0
        if arr[idx] >= min_score and arr[idx] >= left and arr[idx] >= right:
            out.append(idx)
    return out


def _top_indices(values: np.ndarray, *, k: int) -> list[int]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    k = max(1, min(int(k), arr.size))
    return [int(idx) for idx in np.argsort(arr)[-k:]]


def extract_manual_oto_candidate_tracks(
    wav_path: str,
    *,
    encoder: str = "acoustic",
    min_score: float = 0.28,
    top_k_fallback: int = 24,
) -> ManualOtoCandidateTracks:
    batch = extract_features(wav_path, encoder=encoder)
    times = np.asarray(batch.times_ms, dtype=np.float32)
    size = int(times.size)
    scores = {key: np.asarray(value, dtype=np.float32) for key, value in dict(batch.acoustic_scores).items()}

    transition = _unit(
        0.55 * _track(scores, "transition_likelihood", size)
        + 0.20 * _track(scores, "flux_likelihood", size)
        + 0.15 * _track(scores, "sonorant_onset_likelihood", size)
        + 0.10 * _track(scores, "onset_strength", size)
    )
    nucleus = _unit(
        0.45 * _track(scores, "nucleus_likelihood", size)
        + 0.25 * _track(scores, "voicing", size)
        + 0.20 * _track(scores, "world_spectral_stability", size)
        + 0.10 * _track(scores, "rms", size)
    )
    rms = _unit(_track(scores, "rms", size))
    activity_edge = _unit(np.abs(np.gradient(rms)) if rms.size else rms)
    tail = _unit(0.55 * (1.0 - _track(scores, "silence", size)) + 0.45 * activity_edge)
    anchor_scores = {
        "offset": _unit(0.65 * transition + 0.35 * activity_edge),
        "overlap": _unit(0.50 * transition + 0.30 * nucleus + 0.20 * rms),
        "preutterance": _unit(0.60 * transition + 0.25 * nucleus + 0.15 * rms),
        "fixed_end": _unit(0.45 * transition + 0.40 * nucleus + 0.15 * rms),
        "cutoff": tail,
    }
    candidate_indices: dict[str, tuple[int, ...]] = {}
    for anchor, values in anchor_scores.items():
        indices = _local_peak_indices(values, min_score=float(min_score))
        if len(indices) < max(1, int(top_k_fallback)):
            indices = sorted(set(indices) | set(_top_indices(values, k=int(top_k_fallback))))
        candidate_indices[anchor] = tuple(sorted(idx for idx in indices if 0 <= idx < size))

    tracks = {
        "transition": transition,
        "nucleus": nucleus,
        "rms": rms,
        "activity_edge": activity_edge,
        "silence": _track(scores, "silence", size),
        "voicing": _track(scores, "voicing", size),
        "flux": _track(scores, "flux_likelihood", size),
        "onset": _track(scores, "onset_strength", size),
        "sonorant_onset": _track(scores, "sonorant_onset_likelihood", size),
        "world_stability": _track(scores, "world_spectral_stability", size),
    }
    return ManualOtoCandidateTracks(
        times_ms=times,
        candidate_indices=candidate_indices,
        anchor_scores=anchor_scores,
        tracks=tracks,
        duration_ms=float(batch.duration_ms),
        encoder=str(batch.encoder),
    )


def manual_oto_candidate_feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"anchor={value}" for value in SCORABLE_ANCHORS)
    names.extend(f"role={value}" for value in ROLE_VALUES)
    names.extend(f"format={value}" for value in FORMAT_VALUES)
    names.extend(f"language={value}" for value in LANGUAGE_VALUES)
    names.extend(f"alias_family={value}" for value in ALIAS_FAMILY_VALUES)
    names.extend(
        [
            "time_norm",
            "remaining_norm",
            "slot_pos_norm",
            "slot_prior_delta_norm",
            "slot_prior_abs_delta_norm",
            "slot_index_norm",
            "slot_count_norm",
            "anchor_score",
            "anchor_score_left",
            "anchor_score_right",
            "anchor_score_local_mean",
            "anchor_rank_norm",
            "candidate_time_order_norm",
            "candidate_time_order_delta_norm",
            "candidate_time_order_abs_delta_norm",
            "candidate_count_norm",
            "transition",
            "nucleus",
            "rms",
            "activity_edge",
            "silence",
            "voicing",
            "flux",
            "onset",
            "sonorant_onset",
            "world_stability",
        ]
    )
    return names


def _one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    text = str(value or "").strip().lower()
    return [1.0 if text == choice else 0.0 for choice in choices]


_PITCH_SUFFIX_RE = re.compile(r"(?:^|[\s_+\-]|[^\x00-\x7f])(?:[a-g](?:#|b)?-?\d+)(?:$|[\s_+\-])", re.IGNORECASE)
_POWER_SUFFIX_RE = re.compile(r"[a-g](?:#|b)?-?\d+p$", re.IGNORECASE)
_LEADING_N_RE = re.compile(r"^(?:n|ng|N|ん|ン)(?:\s|[_+\-]|$)")
_VOWEL_TOKEN_RE = re.compile(r"^(?:[aeiou]|a|i|u|e|o|aa|ii|uu|ee|oo)(?:\s|$)", re.IGNORECASE)


def manual_oto_alias_family(alias: object) -> str:
    text = str(alias or "").strip()
    if not text:
        return "unknown"
    lower = text.lower()
    if lower in {"sil", "pau", "rest", "br", "breath", "exh"} or lower.startswith(("br", "息", "吸")):
        return "breath_silence"
    if "weak" in lower or "soft" in lower or "弱" in text:
        return "weak_suffix"
    if "power" in lower or "strong" in lower or "強" in text or _POWER_SUFFIX_RE.search(text):
        return "power_suffix"
    if _LEADING_N_RE.search(text):
        return "leading_n"
    if "・" in text or "/" in text or "\\" in text:
        return "punct_transition"
    parts = [part for part in re.split(r"\s+", lower) if part]
    if len(parts) >= 2 and (_VOWEL_TOKEN_RE.match(parts[0]) or _VOWEL_TOKEN_RE.match(parts[-1])):
        return "vowel_transition"
    if _PITCH_SUFFIX_RE.search(text):
        return "pitch_suffix"
    return "plain"


def _value_at(values: np.ndarray, idx: int) -> float:
    if values.size == 0:
        return 0.0
    idx = min(max(0, int(idx)), int(values.size) - 1)
    return float(values[idx])


def manual_oto_candidate_features(
    row: Mapping[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    anchor: str,
    candidate_index: int,
) -> np.ndarray:
    duration = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    candidate_time = _value_at(tracks.times_ms, candidate_index)
    slot_pos = float(row.get("slot_pos_norm", 0.0) or 0.0)
    slot_index = float(row.get("slot_index", 0.0) or 0.0)
    slot_count = max(1.0, float(row.get("slot_count", 1.0) or 1.0))
    slot_prior_ms = slot_pos * duration
    anchor_values = np.asarray(tracks.anchor_scores.get(anchor, np.zeros_like(tracks.times_ms)), dtype=np.float32)
    anchor_indices = list(tracks.candidate_indices.get(anchor, ()))
    scores_for_anchor = [(_value_at(anchor_values, idx), idx) for idx in anchor_indices]
    scores_for_anchor.sort(reverse=True)
    rank_lookup = {idx: rank for rank, (_, idx) in enumerate(scores_for_anchor)}
    rank_norm = float(rank_lookup.get(candidate_index, len(scores_for_anchor))) / max(1.0, float(len(scores_for_anchor) - 1))
    ordered_indices = sorted(anchor_indices, key=lambda idx: float(tracks.times_ms[idx]) if 0 <= idx < tracks.times_ms.size else 0.0)
    order_lookup = {idx: order for order, idx in enumerate(ordered_indices)}
    order_norm = float(order_lookup.get(candidate_index, len(ordered_indices))) / max(1.0, float(len(ordered_indices) - 1))

    values: list[float] = []
    values.extend(_one_hot(anchor, SCORABLE_ANCHORS))
    values.extend(_one_hot(str(row.get("alias_role", "") or ""), ROLE_VALUES))
    values.extend(_one_hot(str(row.get("format_type", "") or ""), FORMAT_VALUES))
    values.extend(_one_hot(str(row.get("language", "") or ""), LANGUAGE_VALUES))
    values.extend(_one_hot(manual_oto_alias_family(row.get("alias", "")), ALIAS_FAMILY_VALUES))
    values.extend(
        [
            candidate_time / duration,
            max(0.0, duration - candidate_time) / duration,
            slot_pos,
            (candidate_time - slot_prior_ms) / duration,
            abs(candidate_time - slot_prior_ms) / duration,
            slot_index / max(1.0, slot_count - 1.0),
            min(slot_count, 64.0) / 64.0,
            _value_at(anchor_values, candidate_index),
            _value_at(anchor_values, candidate_index - 1),
            _value_at(anchor_values, candidate_index + 1),
            float(
                np.mean(
                    [
                        _value_at(anchor_values, candidate_index - 1),
                        _value_at(anchor_values, candidate_index),
                        _value_at(anchor_values, candidate_index + 1),
                    ]
                )
            ),
            rank_norm,
            order_norm,
            order_norm - slot_pos,
            abs(order_norm - slot_pos),
            min(len(anchor_indices), 96) / 96.0,
        ]
    )
    for key in (
        "transition",
        "nucleus",
        "rms",
        "activity_edge",
        "silence",
        "voicing",
        "flux",
        "onset",
        "sonorant_onset",
        "world_stability",
    ):
        values.append(_value_at(np.asarray(tracks.tracks.get(key, np.zeros_like(tracks.times_ms)), dtype=np.float32), candidate_index))
    return np.asarray(values, dtype=np.float32)


__all__ = [
    "ALL_ANCHORS",
    "ALIAS_FAMILY_VALUES",
    "SCORABLE_ANCHORS",
    "ManualOtoCandidateTracks",
    "extract_manual_oto_candidate_tracks",
    "manual_oto_alias_family",
    "manual_oto_candidate_feature_names",
    "manual_oto_candidate_features",
]
