from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .audio import file_sha256
from .oto_io import write_lines_atomic
from .project import ProjectSpec


ANNOTATION_SCHEMA_VERSION = 1
LABEL_KINDS = {"manual_gold", "manual_pending", "weak", "silver", "pseudo"}
CORE_MARKER_LABELS = {
    "usable_start",
    "usable_end",
    "phone_boundary",
    "consonant_onset",
    "vowel_transition",
    "vowel_stable_start",
    "vowel_stable_end",
}
OPTIONAL_MARKER_LABELS = {
    "subphone_transition",
    "closure_start",
    "release",
    "frication_start",
    "frication_end",
    "voicing_onset",
    "sonorant_start",
    "breath_start",
    "breath_end",
    "special_core_start",
    "special_core_end",
    "nucleus",
}


class AnnotationValidationError(ValueError):
    pass


@dataclass(frozen=True)
class AnnotationMarker:
    label: str
    time_ms: float
    confidence: float
    annotator: str
    source: str
    phone_index: int | None = None
    left_phone_index: int | None = None
    right_phone_index: int | None = None
    state_id: str | None = None
    state_instance_id: str | None = None
    note: str = ""


@dataclass(frozen=True)
class AnnotationDocument:
    annotation_id: str
    revision: int
    wav: str
    wav_sha256: str
    inventory_sha256: str
    transcript_source: str
    label_source: str
    phones: tuple[str, ...]
    markers: tuple[AnnotationMarker, ...]
    schema_version: int = ANNOTATION_SCHEMA_VERSION
    parent_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "annotation_id": self.annotation_id,
            "revision": self.revision,
            "wav": self.wav,
            "wav_sha256": self.wav_sha256,
            "schema_version": self.schema_version,
            "inventory_sha256": self.inventory_sha256,
            "transcript_source": self.transcript_source,
            "label_source": self.label_source,
            "phones": list(self.phones),
            "parent_sha256": self.parent_sha256,
            "markers": [asdict(marker) for marker in self.markers],
        }


def load_annotation(path: Path, *, project: ProjectSpec | None = None) -> AnnotationDocument:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AnnotationValidationError("annotation root는 mapping이어야 합니다")
    allowed = {
        "annotation_id",
        "revision",
        "wav",
        "wav_sha256",
        "schema_version",
        "inventory_sha256",
        "transcript_source",
        "label_source",
        "phones",
        "parent_sha256",
        "markers",
    }
    required = allowed - {"parent_sha256"}
    _strict_keys(payload, allowed, required, "annotation")
    if payload["schema_version"] != ANNOTATION_SCHEMA_VERSION:
        raise AnnotationValidationError(f"지원하지 않는 annotation schema_version: {payload['schema_version']}")
    phones = _string_tuple(payload["phones"], "annotation.phones")
    label_source = _choice(payload["label_source"], LABEL_KINDS, "annotation.label_source")
    raw_markers = payload["markers"]
    if isinstance(raw_markers, (str, bytes)) or not isinstance(raw_markers, Sequence):
        raise AnnotationValidationError("annotation.markers는 목록이어야 합니다")
    markers = tuple(_parse_marker(item, index, phones, label_source) for index, item in enumerate(raw_markers))
    document = AnnotationDocument(
        annotation_id=_string(payload["annotation_id"], "annotation.annotation_id"),
        revision=_integer(payload["revision"], "annotation.revision", minimum=1),
        wav=_string(payload["wav"], "annotation.wav"),
        wav_sha256=_sha256(payload["wav_sha256"], "annotation.wav_sha256"),
        inventory_sha256=_sha256(payload["inventory_sha256"], "annotation.inventory_sha256"),
        transcript_source=_string(payload["transcript_source"], "annotation.transcript_source"),
        label_source=label_source,
        phones=phones,
        markers=markers,
        parent_sha256=None if payload.get("parent_sha256") is None else _sha256(payload["parent_sha256"], "annotation.parent_sha256"),
    )
    _validate_marker_order(document)
    if project is not None:
        _validate_against_project(document, project)
        _validate_subphone_transition_markers(document, project)
    return document


def save_annotation(
    document: AnnotationDocument,
    path: Path,
    *,
    previous: Path | None = None,
) -> Path:
    resolved = path.resolve()
    if previous is not None:
        previous_document = load_annotation(previous)
        if document.annotation_id != previous_document.annotation_id:
            raise AnnotationValidationError("revision 저장 시 annotation_id를 바꿀 수 없습니다")
        if document.revision != previous_document.revision + 1:
            raise AnnotationValidationError("revision은 이전 revision보다 정확히 1 커야 합니다")
        expected_parent = file_sha256(previous.resolve())
        if document.parent_sha256 != expected_parent:
            raise AnnotationValidationError("parent_sha256가 이전 annotation 파일과 일치하지 않습니다")
    text = json.dumps(document.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(
        resolved,
        text.splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=resolved.exists(),
    )
    return resolved


def initialize_annotation(
    project: ProjectSpec,
    recording_index: int,
    timeline_path: Path,
    output: Path,
) -> AnnotationDocument:
    if recording_index < 0 or recording_index >= len(project.recordings):
        raise IndexError("recording_index가 프로젝트 범위를 벗어났습니다")
    recording = project.recordings[recording_index]
    timeline_payload = json.loads(timeline_path.resolve().read_text(encoding="utf-8"))
    if not isinstance(timeline_payload, Mapping) or timeline_payload.get("schema_version") != 2:
        raise AnnotationValidationError("LandmarkTimeline v2 파일이 필요합니다")
    if timeline_payload.get("wav_sha256") != file_sha256(project.wav_path(recording)):
        raise AnnotationValidationError("timeline과 현재 WAV hash가 다릅니다")
    suggested: list[AnnotationMarker] = []
    for raw in timeline_payload.get("landmarks", []):
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("label", ""))
        if label not in CORE_MARKER_LABELS | OPTIONAL_MARKER_LABELS:
            continue
        phone_index = raw.get("phone_index")
        resolved_phone_index = None if phone_index is None else int(phone_index)
        if label == "phone_boundary" and (resolved_phone_index is None or resolved_phone_index <= 0):
            continue
        suggested.append(
            AnnotationMarker(
                label=label,
                time_ms=float(raw.get("time_ms", 0.0) or 0.0),
                confidence=float(raw.get("confidence", 0.0) or 0.0),
                annotator="",
                source="model_suggestion_pseudo",
                phone_index=None if label == "phone_boundary" else resolved_phone_index,
                left_phone_index=resolved_phone_index - 1 if label == "phone_boundary" else None,
                right_phone_index=resolved_phone_index if label == "phone_boundary" else None,
                note="수동 확인 전에는 manual_gold가 아닙니다",
            )
        )
    suggested.extend(
        (
            AnnotationMarker(
                label="usable_start",
                time_ms=float(timeline_payload.get("usable_start_ms", 0.0) or 0.0),
                confidence=float(timeline_payload.get("confidence", 0.0) or 0.0),
                annotator="",
                source="model_suggestion_pseudo",
                note="수동 확인 전에는 manual_gold가 아닙니다",
            ),
            AnnotationMarker(
                label="usable_end",
                time_ms=float(timeline_payload.get("usable_end_ms", 0.0) or 0.0),
                confidence=float(timeline_payload.get("confidence", 0.0) or 0.0),
                annotator="",
                source="model_suggestion_pseudo",
                note="수동 확인 전에는 manual_gold가 아닙니다",
            ),
        )
    )
    annotation_id = hashlib.sha256(
        f"{project.project_sha256}:{recording_index}:{recording.wav}".encode("utf-8")
    ).hexdigest()[:20]
    document = AnnotationDocument(
        annotation_id=f"annotation_{annotation_id}",
        revision=1,
        wav=recording.wav,
        wav_sha256=file_sha256(project.wav_path(recording)),
        inventory_sha256=project.inventory_sha256,
        transcript_source=recording.transcript_source,
        label_source="manual_pending",
        phones=recording.phones,
        markers=tuple(sorted(suggested, key=lambda item: (item.time_ms, item.label))),
    )
    save_annotation(document, output)
    return document


def build_annotation_queue(
    project: ProjectSpec,
    alignment_report: Path,
    output_csv: Path,
    *,
    target_count: int = 500,
) -> dict[str, object]:
    if target_count < 1:
        raise ValueError("target_count는 1 이상이어야 합니다")
    report = json.loads(alignment_report.resolve().read_text(encoding="utf-8"))
    if not isinstance(report, Mapping) or report.get("project_sha256") != project.project_sha256:
        raise AnnotationValidationError("alignment report가 현재 project hash와 일치하지 않습니다")
    raw_rows = report.get("rows")
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
        raise AnnotationValidationError("alignment report rows가 없습니다")
    by_index: dict[int, Mapping[str, object]] = {}
    for raw in raw_rows:
        if isinstance(raw, Mapping):
            by_index[int(raw.get("recording_index", -1))] = raw
    candidates: list[dict[str, object]] = []
    for index, recording in enumerate(project.recordings):
        row = by_index.get(index)
        if row is None:
            continue
        roles = {alias.role.casefold() for alias in recording.aliases}
        cv_priority = 1 if roles & {"cv", "cv_head"} else 0
        confidence = float(row.get("confidence", 0.0) or 0.0)
        warnings = row.get("warnings")
        warning_count = len(warnings) if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)) else 0
        metadata = recording.metadata
        speaker = str(metadata.get("root_speaker_id") or metadata.get("speaker_id") or f"unknown_{index:05d}")
        bank = str(metadata.get("bank_id") or Path(recording.wav).parent.as_posix())
        language = str(metadata.get("language_hint") or project.language_hint)
        priority = 3.0 * cv_priority + 2.0 * (1.0 - confidence) + min(2.0, 0.5 * warning_count)
        candidates.append(
            {
                "recording_index": index,
                "wav": recording.wav,
                "timeline": str(row.get("timeline", "")),
                "speaker_id": speaker,
                "bank_id": bank,
                "language_hint": language,
                "pitch": str(metadata.get("pitch", "")),
                "expression": str(metadata.get("expression", "")),
                "roles": ";".join(sorted(roles)),
                "confidence": confidence,
                "warning_count": warning_count,
                "priority": priority,
                "status": "pending_manual_annotation",
                "label_source": "manual_pending",
            }
        )
    selected = _round_robin_by_speaker(candidates, target_count)
    stream = io.StringIO(newline="")
    fieldnames = [
        "recording_index",
        "wav",
        "timeline",
        "speaker_id",
        "bank_id",
        "language_hint",
        "pitch",
        "expression",
        "roles",
        "confidence",
        "warning_count",
        "priority",
        "status",
        "label_source",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)
    write_lines_atomic(
        output_csv.resolve(),
        stream.getvalue().splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=output_csv.exists(),
    )
    return {
        "schema_version": 1,
        "project_sha256": project.project_sha256,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "target_count": target_count,
        "cv_priority_count": sum(1 for row in selected if "cv" in str(row["roles"]).split(";") or "cv_head" in str(row["roles"]).split(";")),
        "speaker_count": len({str(row["speaker_id"]) for row in selected}),
        "output_csv": str(output_csv.resolve()),
        "provenance": {
            "selection_source": "alignment_report_uncertainty_and_cv_priority",
            "label_source": "manual_pending",
            "automatic_gold_created": False,
        },
    }


def _round_robin_by_speaker(candidates: Sequence[dict[str, object]], target_count: int) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in sorted(candidates, key=lambda item: (-float(item["priority"]), int(item["recording_index"]))):
        groups.setdefault(str(row["speaker_id"]), []).append(row)
    selected: list[dict[str, object]] = []
    speakers = sorted(groups)
    while speakers and len(selected) < target_count:
        remaining: list[str] = []
        for speaker in speakers:
            rows = groups[speaker]
            if rows and len(selected) < target_count:
                selected.append(rows.pop(0))
            if rows:
                remaining.append(speaker)
        speakers = remaining
    return selected


def _parse_marker(raw: object, index: int, phones: Sequence[str], label_source: str) -> AnnotationMarker:
    path = f"annotation.markers[{index}]"
    if not isinstance(raw, Mapping):
        raise AnnotationValidationError(f"{path}: mapping이어야 합니다")
    allowed = {
        "label",
        "time_ms",
        "confidence",
        "annotator",
        "source",
        "phone_index",
        "left_phone_index",
        "right_phone_index",
        "state_id",
        "state_instance_id",
        "note",
    }
    required = {"label", "time_ms", "confidence", "annotator", "source"}
    _strict_keys(raw, allowed, required, path)
    label = _choice(raw["label"], CORE_MARKER_LABELS | OPTIONAL_MARKER_LABELS, f"{path}.label")
    phone_index = _optional_index(raw.get("phone_index"), len(phones), f"{path}.phone_index")
    left = _optional_index(raw.get("left_phone_index"), len(phones), f"{path}.left_phone_index")
    right = _optional_index(raw.get("right_phone_index"), len(phones), f"{path}.right_phone_index")
    state_id = _optional_string(raw.get("state_id"), f"{path}.state_id")
    state_instance_id = _optional_string(raw.get("state_instance_id"), f"{path}.state_instance_id")
    if label == "phone_boundary":
        if left is None or right is None:
            raise AnnotationValidationError(f"{path}: phone_boundary에는 left/right_phone_index가 필요합니다")
        if right != left + 1:
            raise AnnotationValidationError(f"{path}: phone boundary index는 인접해야 합니다")
    elif phone_index is None and label not in {"usable_start", "usable_end"}:
        raise AnnotationValidationError(f"{path}: {label}에는 phone_index가 필요합니다")
    if label == "subphone_transition":
        if phone_index is None or state_id is None or state_instance_id is None:
            raise AnnotationValidationError(
                f"{path}: subphone_transition requires phone_index, state_id, and state_instance_id"
            )
        if left is not None or right is not None:
            raise AnnotationValidationError(
                f"{path}: subphone_transition is anchored to one phone occurrence, not a phone boundary"
            )
        expected_instance_id = _state_instance_id(phone_index, phones[phone_index], state_id)
        if state_instance_id != expected_instance_id:
            raise AnnotationValidationError(
                f"{path}: subphone_transition state_instance_id must be {expected_instance_id}"
            )
    annotator = _string_allow_empty(raw["annotator"], f"{path}.annotator")
    source = _string(raw["source"], f"{path}.source")
    if label_source == "manual_gold" and (not annotator or source.startswith("model_")):
        raise AnnotationValidationError(f"{path}: manual_gold에는 annotator와 수동 source가 필요합니다")
    return AnnotationMarker(
        label=label,
        time_ms=_number(raw["time_ms"], f"{path}.time_ms", minimum=0.0),
        confidence=_number(raw["confidence"], f"{path}.confidence", minimum=0.0, maximum=1.0),
        annotator=annotator,
        source=source,
        phone_index=phone_index,
        left_phone_index=left,
        right_phone_index=right,
        state_id=state_id,
        state_instance_id=state_instance_id,
        note=_string_allow_empty(raw.get("note", ""), f"{path}.note"),
    )


def _validate_marker_order(document: AnnotationDocument) -> None:
    previous = -1.0
    for index, marker in enumerate(document.markers):
        if marker.time_ms < previous:
            raise AnnotationValidationError(f"annotation.markers[{index}]: marker 시간은 단조 증가해야 합니다")
        previous = marker.time_ms


def _validate_against_project(document: AnnotationDocument, project: ProjectSpec) -> None:
    if document.inventory_sha256 != project.inventory_sha256:
        raise AnnotationValidationError("annotation inventory hash가 현재 project와 다릅니다")
    matches = [recording for recording in project.recordings if recording.wav == document.wav]
    if len(matches) != 1:
        raise AnnotationValidationError("annotation WAV가 project에서 유일하게 매칭되지 않습니다")
    recording = matches[0]
    if document.phones != recording.phones:
        raise AnnotationValidationError("annotation 음소열이 project recording과 다릅니다")
    wav_path = project.wav_path(recording)
    if file_sha256(wav_path) != document.wav_sha256:
        raise AnnotationValidationError("annotation WAV hash가 현재 파일과 다릅니다")


def _validate_subphone_transition_markers(document: AnnotationDocument, project: ProjectSpec) -> None:
    for marker in document.markers:
        if marker.label != "subphone_transition":
            continue
        assert marker.phone_index is not None
        assert marker.state_id is not None
        assert marker.state_instance_id is not None
        phone_id = document.phones[marker.phone_index]
        definition = project.inventory.get(phone_id)
        if definition is None:
            raise AnnotationValidationError(f"subphone_transition phone is absent from inventory: {phone_id}")
        state = next((item for item in definition.states if item.id == marker.state_id), None)
        if state is None:
            raise AnnotationValidationError(
                f"subphone_transition state is absent from inventory: {phone_id}.{marker.state_id}"
            )
        if state.acoustic_role not in {"transition", "vowel_transition"}:
            raise AnnotationValidationError(
                f"subphone_transition must target a transition acoustic role: {phone_id}.{marker.state_id}"
            )
        expected_instance_id = _state_instance_id(marker.phone_index, phone_id, marker.state_id)
        if marker.state_instance_id != expected_instance_id:
            raise AnnotationValidationError("subphone_transition state occurrence does not match annotation phones")


def _strict_keys(data: Mapping[str, object], allowed: set[str], required: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    missing = sorted(required - set(data))
    if unknown:
        raise AnnotationValidationError(f"{path}: 지원하지 않는 필드: {', '.join(unknown)}")
    if missing:
        raise AnnotationValidationError(f"{path}: 필수 필드 누락: {', '.join(missing)}")


def _optional_index(value: object, count: int, path: str) -> int | None:
    if value is None:
        return None
    index = _integer(value, path, minimum=0)
    if index >= count:
        raise AnnotationValidationError(f"{path}: phone index 범위를 벗어났습니다")
    return index


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnnotationValidationError(f"{path}: 목록이어야 합니다")
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationValidationError(f"{path}: 비어 있지 않은 문자열이어야 합니다")
    return value.strip()


def _string_allow_empty(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise AnnotationValidationError(f"{path}: 문자열이어야 합니다")
    return value.strip()


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _state_instance_id(phone_index: int, phone_id: str, state_id: str) -> str:
    return f"p{phone_index:04d}.{phone_id}.{state_id}"


def _integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnnotationValidationError(f"{path}: {minimum} 이상의 정수여야 합니다")
    return value


def _number(value: object, path: str, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationValidationError(f"{path}: 숫자여야 합니다")
    number = float(value)
    if not (-float("inf") < number < float("inf")) or number < minimum or (maximum is not None and number > maximum):
        raise AnnotationValidationError(f"{path}: 범위 밖 숫자입니다")
    return number


def _choice(value: object, choices: set[str], path: str) -> str:
    text = _string(value, path)
    if text not in choices:
        raise AnnotationValidationError(f"{path}: 지원값은 {', '.join(sorted(choices))}입니다")
    return text


def _sha256(value: object, path: str) -> str:
    text = _string(value, path).casefold()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AnnotationValidationError(f"{path}: SHA-256 hex가 필요합니다")
    return text
