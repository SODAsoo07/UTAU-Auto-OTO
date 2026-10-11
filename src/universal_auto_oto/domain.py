from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Slot:
    index: int
    start_ms: float
    end_ms: float
    transition_ms: float
    stable_ms: float
    boundary_confidence: float
    transition_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Timeline:
    wav_path: Path
    wav_sha256: str
    sample_rate: int
    duration_ms: float
    active_start_ms: float
    active_end_ms: float
    slots: tuple[Slot, ...]
    confidence: float
    decoder_version: str = "local_boundary_v1"
    warnings: tuple[str, ...] = ()
    feature_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "wav": str(self.wav_path),
            "wav_sha256": self.wav_sha256,
            "sample_rate": self.sample_rate,
            "duration_ms": self.duration_ms,
            "active_start_ms": self.active_start_ms,
            "active_end_ms": self.active_end_ms,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "feature_summary": dict(self.feature_summary),
            "slots": [slot.to_dict() for slot in self.slots],
            "provenance": {
                "transcript_source": "original_filename_only",
                "label_source": "acoustic_heuristic_v1",
                "timing_source": "audio_only_no_base_oto_timing",
                "feature_backend": "numpy_spectral_v1",
                "decoder_version": self.decoder_version,
            },
        }


@dataclass(frozen=True)
class PhoneStateDefinition:
    id: str
    acoustic_role: str
    min_ms: float
    max_ms: float
    mode_ms: float | None = None


@dataclass(frozen=True)
class PhoneDefinition:
    id: str
    display_name: str
    features: Mapping[str, object]
    states: tuple[PhoneStateDefinition, ...]
    aliases: tuple[str, ...] = ()
    prototype_ids: tuple[str, ...] = ()


class StructuralRole(str, Enum):
    V = "V"
    V_BLEND = "V_BLEND"
    V_END = "V_END"
    CV = "CV"
    CCV = "CCV"
    VV = "VV"
    VCV = "VCV"
    VC = "VC"
    CC = "CC"
    C = "C"
    BREATH = "BREATH"
    SILENCE = "SILENCE"
    SPECIAL = "SPECIAL"
    UNKNOWN = "UNKNOWN"


class AcousticSubtype(str, Enum):
    PLAIN = "plain"
    HARD = "hard"
    SOFTER_HARD = "softer_hard"
    SOFT_VOICED = "soft_voiced"
    SOFT_UNVOICED = "soft_unvoiced"
    GLIDE = "glide"
    BREATH = "breath"
    SPECIAL = "special"
    UNKNOWN = "unknown"


class TransitionType(str, Enum):
    PLAIN = "plain"
    GLIDE = "glide"
    DIPHTHONG = "diphthong"
    CLUSTER = "cluster"
    RELEASE = "release"
    BREATH = "breath"
    UNKNOWN = "unknown"


class EdgeRole(str, Enum):
    HEAD = "head"
    MEDIAL = "medial"
    TAIL = "tail"
    ISOLATED = "isolated"


@dataclass(frozen=True)
class AliasType:
    structural_role: StructuralRole
    acoustic_subtype: AcousticSubtype
    transition_type: TransitionType
    edge_role: EdgeRole
    confidence: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "structural_role": self.structural_role.value,
            "acoustic_subtype": self.acoustic_subtype.value,
            "transition_type": self.transition_type.value,
            "edge_role": self.edge_role.value,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AliasPlan:
    alias: str
    phone_indices: tuple[int, ...]
    template: str
    role: str
    overrides: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedAliasPlan:
    source: AliasPlan
    alias_type: AliasType
    template_id: str
    selection_source: str
    recommended_template_id: str | None
    warnings: tuple[str, ...]
    rule_set_sha256: str


@dataclass(frozen=True)
class GuideAnchor:
    label: str
    time_ms: float
    phone_index: int | None = None
    confidence: float = 1.0
    source: str = "project_manifest"


@dataclass(frozen=True)
class RecordingSpec:
    wav: str
    phones: tuple[str, ...]
    aliases: tuple[AliasPlan, ...]
    transcript_source: str
    label_source: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    guide_anchors: tuple[GuideAnchor, ...] = ()


@dataclass(frozen=True)
class CompiledState:
    instance_id: str
    phone_index: int
    phone_id: str
    state_id: str
    acoustic_role: str
    min_ms: float
    max_ms: float
    mode_ms: float | None = None
    occurrence_index: int = 0
    occurrence_count: int = 1


@dataclass(frozen=True)
class StateSegment:
    instance_id: str
    phone_index: int
    state_id: str
    acoustic_role: str
    start_ms: float
    end_ms: float
    confidence: float
    raw_start_ms: float | None = None
    raw_end_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhoneSegment:
    phone_index: int
    phone_id: str
    start_ms: float
    end_ms: float
    confidence: float
    state_segments: tuple[StateSegment, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state_segments"] = [segment.to_dict() for segment in self.state_segments]
        return payload


@dataclass(frozen=True)
class Landmark:
    label: str
    time_ms: float
    phone_index: int | None
    confidence: float
    source: str
    raw_time_ms: float | None = None
    refinement_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LandmarkTimeline:
    schema_version: int
    wav: str
    wav_sha256: str
    recording_sha256: str
    inventory_sha256: str
    feature_config_sha256: str
    aligner_config_sha256: str
    model_sha256: str | None
    phone_segments: tuple[PhoneSegment, ...]
    landmarks: tuple[Landmark, ...]
    usable_start_ms: float
    usable_end_ms: float
    confidence: float
    warnings: tuple[str, ...]
    provenance: Mapping[str, object]
    sample_rate: int | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "wav": self.wav,
            "wav_sha256": self.wav_sha256,
            "recording_sha256": self.recording_sha256,
            "inventory_sha256": self.inventory_sha256,
            "feature_config_sha256": self.feature_config_sha256,
            "aligner_config_sha256": self.aligner_config_sha256,
            "model_sha256": self.model_sha256,
            "sample_rate": self.sample_rate,
            "duration_ms": self.duration_ms,
            "phone_segments": [segment.to_dict() for segment in self.phone_segments],
            "landmarks": [landmark.to_dict() for landmark in self.landmarks],
            "usable_start_ms": self.usable_start_ms,
            "usable_end_ms": self.usable_end_ms,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }

    def landmarks_for_phone(self, phone_index: int) -> tuple[Landmark, ...]:
        return tuple(item for item in self.landmarks if item.phone_index == phone_index)

    def landmark_time(self, phone_index: int, label: str) -> float | None:
        candidates = (
            item.time_ms
            for item in self.landmarks
            if item.phone_index == phone_index and item.label == label
        )
        return next(candidates, None)


def landmark_timeline_from_dict(payload: Mapping[str, object]) -> LandmarkTimeline:
    """Load a persisted timeline without silently filling missing boundaries.

    Timeline artifacts are an input contract for template-only OTO generation.
    Be deliberately strict here: a malformed artifact must be repaired or
    regenerated rather than turning into a plausible but unverifiable OTO.
    """
    def required(name: str) -> object:
        if name not in payload:
            raise ValueError(f"timeline field is required: {name}")
        return payload[name]

    def text(name: str) -> str:
        value = required(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"timeline field must be a non-empty string: {name}")
        return value

    def number(name: str, source: Mapping[str, object]) -> float:
        value = source.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"timeline field must be numeric: {name}")
        return float(value)

    raw_phones = required("phone_segments")
    raw_landmarks = required("landmarks")
    if not isinstance(raw_phones, list) or not isinstance(raw_landmarks, list):
        raise ValueError("timeline phone_segments and landmarks must be arrays")
    phones: list[PhoneSegment] = []
    for raw_phone in raw_phones:
        if not isinstance(raw_phone, Mapping):
            raise ValueError("timeline phone segment must be an object")
        raw_states = raw_phone.get("state_segments")
        if not isinstance(raw_states, list):
            raise ValueError("timeline phone segment state_segments must be an array")
        states: list[StateSegment] = []
        for raw_state in raw_states:
            if not isinstance(raw_state, Mapping):
                raise ValueError("timeline state segment must be an object")
            raw_start = raw_state.get("raw_start_ms")
            raw_end = raw_state.get("raw_end_ms")
            if raw_start is not None and (isinstance(raw_start, bool) or not isinstance(raw_start, (int, float))):
                raise ValueError("timeline state raw_start_ms must be numeric or null")
            if raw_end is not None and (isinstance(raw_end, bool) or not isinstance(raw_end, (int, float))):
                raise ValueError("timeline state raw_end_ms must be numeric or null")
            states.append(
                StateSegment(
                    instance_id=_timeline_text(raw_state, "instance_id"),
                    phone_index=_timeline_integer(raw_state, "phone_index"),
                    state_id=_timeline_text(raw_state, "state_id"),
                    acoustic_role=_timeline_text(raw_state, "acoustic_role"),
                    start_ms=number("start_ms", raw_state),
                    end_ms=number("end_ms", raw_state),
                    confidence=number("confidence", raw_state),
                    raw_start_ms=None if raw_start is None else float(raw_start),
                    raw_end_ms=None if raw_end is None else float(raw_end),
                )
            )
        phones.append(
            PhoneSegment(
                phone_index=_timeline_integer(raw_phone, "phone_index"),
                phone_id=_timeline_text(raw_phone, "phone_id"),
                start_ms=number("start_ms", raw_phone),
                end_ms=number("end_ms", raw_phone),
                confidence=number("confidence", raw_phone),
                state_segments=tuple(states),
            )
        )
    landmarks: list[Landmark] = []
    for raw_landmark in raw_landmarks:
        if not isinstance(raw_landmark, Mapping):
            raise ValueError("timeline landmark must be an object")
        phone_index = raw_landmark.get("phone_index")
        raw_time = raw_landmark.get("raw_time_ms")
        refinement_reason = raw_landmark.get("refinement_reason")
        if phone_index is not None and (isinstance(phone_index, bool) or not isinstance(phone_index, int)):
            raise ValueError("timeline landmark phone_index must be an integer or null")
        if raw_time is not None and (isinstance(raw_time, bool) or not isinstance(raw_time, (int, float))):
            raise ValueError("timeline landmark raw_time_ms must be numeric or null")
        if refinement_reason is not None and not isinstance(refinement_reason, str):
            raise ValueError("timeline landmark refinement_reason must be a string or null")
        landmarks.append(
            Landmark(
                label=_timeline_text(raw_landmark, "label"),
                time_ms=number("time_ms", raw_landmark),
                phone_index=phone_index,
                confidence=number("confidence", raw_landmark),
                source=_timeline_text(raw_landmark, "source"),
                raw_time_ms=None if raw_time is None else float(raw_time),
                refinement_reason=refinement_reason,
            )
        )
    warnings = required("warnings")
    provenance = required("provenance")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise ValueError("timeline warnings must be an array of strings")
    if not isinstance(provenance, Mapping):
        raise ValueError("timeline provenance must be an object")
    sample_rate = payload.get("sample_rate")
    duration_ms = payload.get("duration_ms")
    model_sha256 = payload.get("model_sha256")
    if sample_rate is not None and (isinstance(sample_rate, bool) or not isinstance(sample_rate, int)):
        raise ValueError("timeline sample_rate must be an integer or null")
    if duration_ms is not None and (isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float))):
        raise ValueError("timeline duration_ms must be numeric or null")
    if model_sha256 is not None and not isinstance(model_sha256, str):
        raise ValueError("timeline model_sha256 must be a string or null")
    schema_version = required("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("timeline schema_version must be an integer")
    return LandmarkTimeline(
        schema_version=schema_version,
        wav=text("wav"),
        wav_sha256=text("wav_sha256"),
        recording_sha256=text("recording_sha256"),
        inventory_sha256=text("inventory_sha256"),
        feature_config_sha256=text("feature_config_sha256"),
        aligner_config_sha256=text("aligner_config_sha256"),
        model_sha256=model_sha256,
        phone_segments=tuple(phones),
        landmarks=tuple(landmarks),
        usable_start_ms=number("usable_start_ms", payload),
        usable_end_ms=number("usable_end_ms", payload),
        confidence=number("confidence", payload),
        warnings=tuple(warnings),
        provenance=dict(provenance),
        sample_rate=sample_rate,
        duration_ms=None if duration_ms is None else float(duration_ms),
    )


def _timeline_text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"timeline field must be a non-empty string: {name}")
    return value


def _timeline_integer(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"timeline field must be an integer: {name}")
    return value


def landmark_timeline_from_legacy(
    timeline: Timeline,
    *,
    phone_ids: tuple[str, ...] | None = None,
    recording_sha256: str = "",
    inventory_sha256: str = "",
) -> LandmarkTimeline:
    """Convert the acoustic MVP timeline without claiming phonetic gold.

    The legacy slot boundaries are preserved as pseudo landmarks.  This adapter
    exists so renderer/evaluation code can migrate independently from alignment.
    """
    resolved_phone_ids = phone_ids or tuple(f"slot_{slot.index}" for slot in timeline.slots)
    if len(resolved_phone_ids) != len(timeline.slots):
        raise ValueError("phone_ids 수와 legacy slot 수가 다릅니다")
    phone_segments: list[PhoneSegment] = []
    landmarks: list[Landmark] = []
    for slot, phone_id in zip(timeline.slots, resolved_phone_ids):
        instance_prefix = f"p{slot.index:04d}.{phone_id}"
        transition = StateSegment(
            instance_id=f"{instance_prefix}.transition",
            phone_index=slot.index,
            state_id="transition",
            acoustic_role="vowel_transition",
            start_ms=slot.start_ms,
            end_ms=slot.stable_ms,
            confidence=slot.transition_confidence,
            raw_start_ms=slot.start_ms,
            raw_end_ms=slot.stable_ms,
        )
        stable = StateSegment(
            instance_id=f"{instance_prefix}.stable",
            phone_index=slot.index,
            state_id="stable",
            acoustic_role="vowel_stable",
            start_ms=slot.stable_ms,
            end_ms=slot.end_ms,
            confidence=min(slot.boundary_confidence, slot.transition_confidence),
            raw_start_ms=slot.stable_ms,
            raw_end_ms=slot.end_ms,
        )
        phone_segments.append(
            PhoneSegment(
                phone_index=slot.index,
                phone_id=phone_id,
                start_ms=slot.start_ms,
                end_ms=slot.end_ms,
                confidence=min(slot.boundary_confidence, slot.transition_confidence),
                state_segments=(transition, stable),
            )
        )
        landmarks.extend(
            (
                Landmark("phone_boundary", slot.start_ms, slot.index, slot.boundary_confidence, "legacy_slot_pseudo"),
                Landmark("vowel_transition", slot.transition_ms, slot.index, slot.transition_confidence, "legacy_slot_pseudo"),
                Landmark("vowel_stable_start", slot.stable_ms, slot.index, slot.transition_confidence, "legacy_slot_pseudo"),
                Landmark("vowel_stable_end", slot.end_ms, slot.index, slot.boundary_confidence, "legacy_slot_pseudo"),
            )
        )
    return LandmarkTimeline(
        schema_version=2,
        wav=str(timeline.wav_path),
        wav_sha256=timeline.wav_sha256,
        recording_sha256=recording_sha256,
        inventory_sha256=inventory_sha256,
        feature_config_sha256="numpy_spectral_v1",
        aligner_config_sha256=timeline.decoder_version,
        model_sha256=None,
        phone_segments=tuple(phone_segments),
        landmarks=tuple(landmarks),
        usable_start_ms=timeline.active_start_ms,
        usable_end_ms=timeline.active_end_ms,
        confidence=timeline.confidence,
        warnings=timeline.warnings,
        provenance={
            "transcript_source": "original_filename_only",
            "label_source": "acoustic_heuristic_v1_pseudo",
            "feature_backend": "numpy_spectral_v1",
            "decoder_version": timeline.decoder_version,
            "legacy_timeline_schema_version": 1,
        },
        sample_rate=timeline.sample_rate,
        duration_ms=timeline.duration_ms,
    )


@dataclass(frozen=True)
class SourceOtoRow:
    source_wav: str
    alias: str
    source_index: int


@dataclass(frozen=True)
class PlannedAlias:
    source: SourceOtoRow
    slot_index: int
    variant_index: int
    variants_per_slot: int
    role: str


@dataclass(frozen=True)
class RenderedOtoRow:
    wav_name: str
    alias: str
    offset: float
    consonant: float
    cutoff: float
    preutterance: float
    overlap: float
    role: str
    slot_index: int
    confidence: float
    split: str
    reasons: tuple[str, ...]
    source_index: int
    absolute: dict[str, float]

    def to_line(self) -> str:
        return (
            f"{self.wav_name}={self.alias},"
            f"{self.offset:.3f},{self.consonant:.3f},{self.cutoff:.3f},"
            f"{self.preutterance:.3f},{self.overlap:.3f}"
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "wav": self.wav_name,
            "alias": self.alias,
            "source_index": self.source_index,
            "slot_index": self.slot_index,
            "role": self.role,
            "offset": self.offset,
            "consonant": self.consonant,
            "cutoff": self.cutoff,
            "preutterance": self.preutterance,
            "overlap": self.overlap,
            "absolute": self.absolute,
            "confidence": self.confidence,
            "split": self.split,
            "reasons": list(self.reasons),
            "timing_source": "audio_only_no_base_oto_timing",
        }
