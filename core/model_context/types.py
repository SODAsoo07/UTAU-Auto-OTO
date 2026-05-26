from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class WavIdentity:
    wav_path: str
    wav_name: str
    wav_stem: str
    voicebank_id: str = ""
    language: str = ""
    format_type: str = ""


@dataclass(frozen=True)
class FilenameTokenContext:
    raw: str
    tokens: tuple[str, ...]
    canonical_tokens: tuple[str, ...]
    token_count: int
    slot_count: int
    parse_confidence: float = 1.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReclistMatchContext:
    recording_index: int = -1
    recording_text: str = ""
    match_status: str = "unknown"
    match_ratio: float = 0.0
    mismatch_reason: str = ""
    expected_tokens: tuple[str, ...] = ()
    observed_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class AliasRowContext:
    alias: str
    alias_norm: str = ""
    alias_role: str = "other"
    alias_type: str = "other"
    slot_index: int = 0
    slot_count: int = 1
    prev_alias: str = ""
    next_alias: str = ""
    line_index: int = 0


@dataclass(frozen=True)
class OtoParamContext:
    raw_params: Mapping[str, float]
    repaired_params: Mapping[str, float]
    anchors_ms: tuple[float, float, float, float, float]
    duration_ms: float
    repair_warnings: tuple[str, ...] = ()
    usable_as_target: bool = True
    reason: str = ""

    def anchor_mapping(self) -> dict[str, float]:
        return {
            "offset_abs": float(self.anchors_ms[0]),
            "overlap_abs": float(self.anchors_ms[1]),
            "pre_abs": float(self.anchors_ms[2]),
            "consonant_abs": float(self.anchors_ms[3]),
            "cutoff_abs": float(self.anchors_ms[4]),
        }

    def to_absolute_anchors(self):
        values = self.anchor_mapping()
        return AbsoluteOtoAnchors(
            offset_abs=values["offset_abs"],
            overlap_abs=values["overlap_abs"],
            pre_abs=values["pre_abs"],
            consonant_abs=values["consonant_abs"],
            cutoff_abs=values["cutoff_abs"],
            confidence=1.0 if self.usable_as_target else 0.0,
            reason=self.reason or ("source" if self.usable_as_target else "unusable_source"),
        )


@dataclass(frozen=True)
class AudioContext:
    wav_path: str
    duration_ms: float
    active_start_ms: float = 0.0
    active_end_ms: float = 0.0
    audio_reliability: float = 0.5
    onset_peaks: tuple[Mapping[str, Any], ...] = ()
    stable_vowel_segments: tuple[Mapping[str, Any], ...] = ()
    mel_delta_peaks: tuple[Mapping[str, Any], ...] = ()
    cache_key: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSampleContext:
    wav: WavIdentity
    filename: FilenameTokenContext
    alias: AliasRowContext
    oto: OtoParamContext
    audio: AudioContext | None = None
    reclist: ReclistMatchContext | None = None
    warnings: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)


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
    source_params: Mapping[str, float]
    alias_suffix: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AbsoluteOtoAnchors:
    offset_abs: float
    overlap_abs: float
    pre_abs: float
    consonant_abs: float
    cutoff_abs: float
    confidence: float = 0.0
    reason: str = ""
    syllable_audio_start_abs: float | None = None
    silence_audio_end_abs: float | None = None


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
