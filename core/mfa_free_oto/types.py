from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

FRAME_LABELS: tuple[str, ...] = ("silence", "consonant", "vowel", "other")
EVENT_LABELS: tuple[str, ...] = ("cv_boundary", "vowel_nucleus", "phone_change")

FRAME_LABEL_TO_ID = {label: idx for idx, label in enumerate(FRAME_LABELS)}
EVENT_LABEL_TO_ID = {label: idx for idx, label in enumerate(EVENT_LABELS)}

MANUAL_SOURCE = "manual_gold"
REJECTED_LABEL_SOURCES = {
    "mfa",
    "mfa_textgrid",
    "textgrid",
    "oto",
    "oto_ini",
    "oto_pseudo",
    "source_oto",
    "pseudo_label",
}


@dataclass(frozen=True)
class SuccessCriteria:
    frame_macro_f1_min: float = 0.90
    cv_boundary_median_error_ms_max: float = 25.0
    cv_boundary_p90_error_ms_max: float = 70.0
    catastrophic_row_mismatch_rate_max: float = 0.03
    hard_failure_rate_max: float = 0.03


@dataclass(frozen=True)
class FrameLabel:
    label: str
    start_ms: float
    end_ms: float
    confidence: float = 1.0


@dataclass(frozen=True)
class EventLabel:
    label: str
    time_ms: float
    confidence: float = 1.0
    phone: str | None = None


@dataclass(frozen=True)
class DecodedEvent:
    label: str
    selected_time_ms: float
    score: float
    expected_phone: str | None = None
    frame_index: int | None = None


@dataclass(frozen=True)
class FramePosterior:
    wav_path: str
    times_ms: Sequence[float]
    class_probs: Mapping[str, Sequence[float]]
    event_scores: Mapping[str, Sequence[float]]
    acoustic_scores: Mapping[str, Sequence[float]] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def frame_count(self) -> int:
        return len(self.times_ms)


_VOWEL_PHONES: frozenset[str] = frozenset(
    {
        "a",
        "e",
        "i",
        "o",
        "u",
        "eo",
        "eu",
        "ae",
        "ya",
        "ye",
        "yo",
        "yu",
        "wa",
        "we",
        "wi",
        "wo",
        "ui",
        "ang",
        "eng",
        "ing",
        "ong",
        "ung",
        "aa",
        "ii",
        "uu",
        "ee",
        "oo",
        "ɯ",
        "ə",
        "ɑ",
    }
)


def is_vowel_phone(phone: str, language: str = "") -> bool:
    normalized = phone.strip().lower()
    if normalized in _VOWEL_PHONES:
        return True
    if normalized == "n":
        # Bare 'n' is the syllabic moraic nasal in Japanese (vowel-like for
        # CV-boundary purposes), but a consonant onset/coda in Korean.
        lang = (language or "").strip().lower()
        return not (lang.startswith("ko") or lang.startswith("kr"))
    return False
