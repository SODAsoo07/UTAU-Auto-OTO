from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BOUNDARY_LABELS: tuple[str, ...] = (
    "phone_start",
    "phone_end",
    "consonant_onset",
    "vowel_onset",
    "vowel_nucleus",
    "vowel_end",
    "silence_boundary",
)

BOUNDARY_LABEL_SIGMA_MS: dict[str, float] = {
    "phone_start": 14.0,
    "phone_end": 14.0,
    "consonant_onset": 14.0,
    "vowel_onset": 20.0,
    "vowel_nucleus": 30.0,
    "vowel_end": 22.0,
    "silence_boundary": 30.0,
}

PHONE_STATE_LABELS: tuple[str, ...] = ("silence", "consonant", "vowel")
CONSONANT_LABELS: tuple[str, ...] = (
    "b",
    "bb",
    "ch",
    "d",
    "dd",
    "f",
    "g",
    "gg",
    "h",
    "j",
    "jj",
    "k",
    "kh",
    "l",
    "m",
    "n",
    "ng",
    "p",
    "ph",
    "r",
    "s",
    "sh",
    "ss",
    "t",
    "th",
    "ts",
    "v",
    "w",
    "y",
    "z",
    "other",
)
VOWEL_LABELS: tuple[str, ...] = (
    "a",
    "ae",
    "e",
    "eo",
    "eu",
    "i",
    "o",
    "oe",
    "u",
    "ui",
    "wa",
    "wae",
    "we",
    "weo",
    "wi",
    "wo",
    "ya",
    "yae",
    "ye",
    "yeo",
    "yo",
    "yu",
    "other",
)
IGNORE_INDEX = -100

_LABEL_ALIASES: dict[str, str] = {
    "cv_boundary": "vowel_onset",
    "cv_vowel_onset": "vowel_onset",
    "vowel_start": "vowel_onset",
    "v_start": "vowel_onset",
    "v_onset": "vowel_onset",
    "vv_nucleus": "vowel_nucleus",
    "target_vowel_nucleus": "vowel_nucleus",
    "nucleus": "vowel_nucleus",
    "c_start": "consonant_onset",
    "consonant_start": "consonant_onset",
    "c_onset": "consonant_onset",
    "silence_start": "silence_boundary",
    "silence_end": "silence_boundary",
    "active_start": "silence_boundary",
    "active_end": "silence_boundary",
}


@dataclass(frozen=True)
class PhoneToken:
    text: str
    index: int
    alias_index: int
    coarse: str = ""


@dataclass(frozen=True)
class AliasPlan:
    wav_name: str
    alias: str
    alias_index: int
    phones: tuple[PhoneToken, ...]
    language: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhonemePlan:
    wav_path: str
    aliases: tuple[AliasPlan, ...]
    duration_ms: float = 0.0
    language: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryEvent:
    label: str
    time_ms: float
    alias_index: int = -1
    phone_index: int = -1
    phone: str = ""
    confidence: float = 1.0
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryFrameScores:
    wav_path: str
    times_ms: list[float]
    scores: dict[str, list[float]]
    quality_scores: list[float] | None = None
    phone_state_scores: dict[str, list[float]] | None = None
    consonant_scores: dict[str, list[float]] | None = None
    vowel_scores: dict[str, list[float]] | None = None
    heuristic_scores: dict[str, list[float]] | None = None


@dataclass(frozen=True)
class DecodedBoundaryEvent:
    expected_label: str
    selected_time_ms: float
    score: float
    alias_index: int = -1
    phone_index: int = -1
    phone: str = ""
    source: str = "model"
    fallback_used: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def normalize_boundary_label(label: object) -> str:
    text = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in BOUNDARY_LABELS:
        return text
    return _LABEL_ALIASES.get(text, text)


def label_sigma_ms(label: object) -> float:
    return float(BOUNDARY_LABEL_SIGMA_MS.get(normalize_boundary_label(label), 20.0))


def event_from_mapping(row: dict[str, Any]) -> BoundaryEvent | None:
    label = normalize_boundary_label(row.get("label", row.get("kind", row.get("event", ""))))
    if label not in BOUNDARY_LABELS:
        return None
    raw_time = row.get("time_ms", row.get("time", row.get("center_ms", None)))
    if raw_time is None:
        return None
    try:
        time_ms = float(raw_time)
    except Exception:
        return None
    return BoundaryEvent(
        label=label,
        time_ms=max(0.0, time_ms),
        alias_index=_safe_int(row.get("alias_index", row.get("row_index", -1)), -1),
        phone_index=_safe_int(row.get("phone_index", -1), -1),
        phone=str(row.get("phone", "") or ""),
        confidence=max(0.0, min(1.0, _safe_float(row.get("confidence", 1.0), 1.0))),
        source=str(row.get("source", "") or ""),
        meta={k: v for k, v in row.items() if k not in {"label", "kind", "event", "time_ms", "time", "center_ms"}},
    )


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


__all__ = [
    "AliasPlan",
    "BOUNDARY_LABELS",
    "BOUNDARY_LABEL_SIGMA_MS",
    "BoundaryEvent",
    "BoundaryFrameScores",
    "CONSONANT_LABELS",
    "DecodedBoundaryEvent",
    "IGNORE_INDEX",
    "PHONE_STATE_LABELS",
    "PhoneToken",
    "PhonemePlan",
    "VOWEL_LABELS",
    "event_from_mapping",
    "label_sigma_ms",
    "normalize_boundary_label",
]
