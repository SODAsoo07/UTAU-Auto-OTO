from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BOUNDARY_LABELS: tuple[str, ...] = (
    "syllable_onset",
    "consonant_onset",
    "vowel_start",
    "vowel_stable",
    "vowel_end",
    "transition_peak",
    "next_onset",
    "silence_boundary",
)

BOUNDARY_LABEL_SIGMA_MS: dict[str, float] = {
    "syllable_onset": 14.0,
    "consonant_onset": 14.0,
    "vowel_start": 22.0,
    "vowel_stable": 26.0,
    "vowel_end": 22.0,
    "transition_peak": 18.0,
    "next_onset": 16.0,
    "silence_boundary": 30.0,
}

PHONE_AWARE_CVS_LABELS: tuple[str, ...] = ("silence", "consonant", "vowel")
PHONE_AWARE_CONSONANT_LABELS: tuple[str, ...] = (
    "b",
    "bb",
    "by",
    "c",
    "ch",
    "cl",
    "d",
    "dd",
    "dy",
    "dz",
    "f",
    "g",
    "gg",
    "gy",
    "gw",
    "h",
    "hy",
    "j",
    "jj",
    "k",
    "kh",
    "ky",
    "kk",
    "l",
    "m",
    "my",
    "n",
    "ng",
    "ny",
    "p",
    "ph",
    "pp",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "ss",
    "t",
    "th",
    "tt",
    "ts",
    "ty",
    "v",
    "w",
    "y",
    "z",
)
PHONE_AWARE_VOWEL_LABELS: tuple[str, ...] = (
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
)
PHONE_AWARE_CONSONANT_FAMILY_LABELS: tuple[str, ...] = (
    "plosive",
    "aspirate",
    "nasal",
    "liquid",
    "fricative",
    "affricate",
    "glide",
    "closure",
    "other",
)
PHONE_AWARE_VOWEL_NUCLEUS_LABELS: tuple[str, ...] = (
    "a",
    "e",
    "i",
    "o",
    "u",
    "eo",
    "eu",
    "other",
)
PHONE_AWARE_VOWEL_GLIDE_LABELS: tuple[str, ...] = (
    "none",
    "y",
    "w",
    "ui",
    "other",
)
PHONE_AWARE_IGNORE_INDEX = -100

ANCHOR_ROLES: frozenset[str] = frozenset({"-cv", "cv", "v", "special"})
TRANSITION_ROLES: frozenset[str] = frozenset({"vc", "vv", "v-cv"})

ROLE_LABEL_PREFERENCES: dict[str, tuple[str, ...]] = {
    "-cv": ("syllable_onset", "vowel_start"),
    "cv": ("consonant_onset", "vowel_start"),
    "v": ("vowel_start", "vowel_stable"),
    "special": ("consonant_onset", "vowel_start"),
    "vc": ("vowel_end", "transition_peak", "next_onset"),
    "vv": ("transition_peak", "vowel_end", "vowel_start"),
    "v-cv": ("transition_peak", "next_onset", "consonant_onset"),
    "v-": ("vowel_end", "silence_boundary"),
    "cv-": ("vowel_end", "silence_boundary"),
    "br": ("silence_boundary",),
    "endbr": ("silence_boundary",),
    "other": ("vowel_start", "transition_peak"),
}


@dataclass(frozen=True)
class BoundaryFrameScores:
    wav_path: str
    times_ms: list[float]
    scores: dict[str, list[float]]
    quality_scores: list[float] | None = None
    cvs_scores: dict[str, list[float]] | None = None
    consonant_scores: dict[str, list[float]] | None = None
    vowel_scores: dict[str, list[float]] | None = None
    consonant_family_scores: dict[str, list[float]] | None = None
    vowel_nucleus_scores: dict[str, list[float]] | None = None
    vowel_glide_scores: dict[str, list[float]] | None = None


@dataclass(frozen=True)
class BoundaryCandidate:
    time_ms: float
    kind: str
    score: float
    source: str


@dataclass(frozen=True)
class OtoRowSpec:
    wav_name: str
    wav_path: str
    alias: str
    role: str
    slot_index: int
    slot_count: int
    prev_alias: str
    next_alias: str
    language: str
    format_type: str
    line_index: int
    duration_ms: float
    source_params: dict[str, float]
    alias_suffix: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AbsoluteOtoAnchors:
    offset_abs: float
    overlap_abs: float
    pre_abs: float
    consonant_abs: float
    cutoff_abs: float
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class DecodedOtoRow:
    spec: OtoRowSpec
    anchors: AbsoluteOtoAnchors
    selected_time_ms: float
    fallback_used: bool
    reason: str
    quality_score: float = 0.0


@dataclass(frozen=True)
class BoundaryDecodeResult:
    wav_path: str
    duration_ms: float
    rows: list[DecodedOtoRow]
    anchor_timeline_ms: list[float]
    fallback_count: int


def normalize_boundary_label(label: object) -> str:
    text = str(label or "").strip().lower()
    if text in BOUNDARY_LABELS:
        return text
    aliases = {
        "syllable": "syllable_onset",
        "onset": "consonant_onset",
        "vowel_onset": "vowel_start",
        "stable_vowel": "vowel_stable",
        "transition": "transition_peak",
        "next": "next_onset",
        "silence": "silence_boundary",
    }
    return aliases.get(text, "vowel_start")


def role_label_preferences(role: object) -> tuple[str, ...]:
    key = str(role or "").strip().lower()
    return ROLE_LABEL_PREFERENCES.get(key, ROLE_LABEL_PREFERENCES["other"])


def label_sigma_ms(label: object) -> float:
    key = normalize_boundary_label(label)
    return float(BOUNDARY_LABEL_SIGMA_MS.get(key, 20.0))


__all__ = [
    "ANCHOR_ROLES",
    "BOUNDARY_LABELS",
    "BOUNDARY_LABEL_SIGMA_MS",
    "PHONE_AWARE_CONSONANT_LABELS",
    "PHONE_AWARE_CONSONANT_FAMILY_LABELS",
    "PHONE_AWARE_CVS_LABELS",
    "PHONE_AWARE_VOWEL_GLIDE_LABELS",
    "PHONE_AWARE_IGNORE_INDEX",
    "PHONE_AWARE_VOWEL_LABELS",
    "PHONE_AWARE_VOWEL_NUCLEUS_LABELS",
    "TRANSITION_ROLES",
    "AbsoluteOtoAnchors",
    "BoundaryCandidate",
    "BoundaryDecodeResult",
    "BoundaryFrameScores",
    "DecodedOtoRow",
    "OtoRowSpec",
    "label_sigma_ms",
    "normalize_boundary_label",
    "role_label_preferences",
]
