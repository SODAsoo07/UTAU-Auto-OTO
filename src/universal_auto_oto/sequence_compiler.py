from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

from .domain import CompiledState, PhoneDefinition, PhoneStateDefinition, RecordingSpec


_SYMBOLIC_INTERVAL_STATES = {
    "s": PhoneStateDefinition("frication", "frication", 30.0, 350.0, 90.0),
    "ss": PhoneStateDefinition("frication", "frication", 30.0, 350.0, 90.0),
}
_VOCAL_FRY_STATES = (
    PhoneStateDefinition("onset", "irregular_voicing_onset", 8.0, 120.0, 30.0),
    PhoneStateDefinition("fry", "aperiodic", 8.0, 700.0, 70.0),
)
STATE_INFERENCE_VERSION = "symbolic_phone_state_v1"


def compile_recording_states(
    recording: RecordingSpec,
    inventory: Mapping[str, PhoneDefinition],
) -> tuple[CompiledState, ...]:
    """Compile a recording into a linear occurrence-aware state sequence."""
    compiled: list[CompiledState] = []
    instance_ids: set[str] = set()
    occurrence_counts = Counter(recording.phones)
    occurrence_indices: defaultdict[str, int] = defaultdict(int)
    for phone_index, phone_id in enumerate(recording.phones):
        try:
            phone = inventory[phone_id]
        except KeyError as exc:
            raise ValueError(f"inventory에 없는 음소입니다: {phone_id}") from exc
        occurrence_index = occurrence_indices[phone_id]
        occurrence_indices[phone_id] += 1
        states = _resolved_phone_states(phone_id, phone)
        for state in states:
            instance_id = f"p{phone_index:04d}.{_instance_token(phone_id)}.{_instance_token(state.id)}"
            if instance_id in instance_ids:
                raise ValueError(f"중복 compiled state instance_id입니다: {instance_id}")
            instance_ids.add(instance_id)
            compiled.append(
                CompiledState(
                    instance_id=instance_id,
                    phone_index=phone_index,
                    phone_id=phone_id,
                    state_id=state.id,
                    acoustic_role=state.acoustic_role,
                    min_ms=state.min_ms,
                    max_ms=state.max_ms,
                    mode_ms=state.mode_ms,
                    occurrence_index=occurrence_index,
                    occurrence_count=occurrence_counts[phone_id],
                )
            )
    if not compiled:
        raise ValueError("컴파일된 상태열이 비어 있습니다")
    return tuple(compiled)


def _resolved_phone_states(
    phone_id: str,
    phone: PhoneDefinition,
) -> tuple[PhoneStateDefinition, ...]:
    if (
        str(phone_id).casefold() == "vf"
        and len(phone.states) == 1
        and phone.states[0].acoustic_role == "irregular_periodic"
    ):
        return _VOCAL_FRY_STATES
    if (
        phone.features.get("manual_review_interval") is True
        and len(phone.states) == 1
        and phone.states[0].acoustic_role == "phone_interval"
    ):
        inferred = _SYMBOLIC_INTERVAL_STATES.get(str(phone_id).casefold())
        if inferred is not None:
            return (inferred,)
    return phone.states


def compiled_state_inference_provenance(
    recording: RecordingSpec,
    inventory: Mapping[str, PhoneDefinition],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for phone_id in recording.phones:
        if phone_id in seen:
            continue
        seen.add(phone_id)
        phone = inventory[phone_id]
        resolved = _resolved_phone_states(phone_id, phone)
        if resolved == phone.states:
            continue
        rows.append(
            {
                "inference_version": STATE_INFERENCE_VERSION,
                "phone_id": phone_id,
                "source_states": [_phone_state_dict(state) for state in phone.states],
                "resolved_states": [_phone_state_dict(state) for state in resolved],
            }
        )
    return tuple(rows)


def _phone_state_dict(state: PhoneStateDefinition) -> Mapping[str, object]:
    return {
        "id": state.id,
        "acoustic_role": state.acoustic_role,
        "min_ms": state.min_ms,
        "mode_ms": state.mode_ms,
        "max_ms": state.max_ms,
    }


def _instance_token(value: str) -> str:
    encoded = []
    for character in value:
        if character.isalnum() or character in {"_", "-"}:
            encoded.append(character)
        else:
            encoded.append(f"x{ord(character):04x}")
    return "".join(encoded) or "empty"
