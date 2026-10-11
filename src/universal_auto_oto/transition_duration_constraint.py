from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .annotations import AnnotationDocument, load_annotation
from .audio import file_sha256
from .domain import RecordingSpec
from .oto_io import write_lines_atomic
from .project import ProjectSpec, load_project
from .sequence_compiler import compile_recording_states


CONSTRAINT_PURPOSE = "development_train_phone_state_duration_constraint"
TRANSITION_ROLES = frozenset({"transition", "vowel_transition"})


class TransitionDurationConstraintError(ValueError):
    pass


@dataclass(frozen=True)
class PhoneStateDurationConstraint:
    phone_id: str
    state_id: str
    acoustic_roles: tuple[str, ...]
    min_duration_ms: float
    mode_duration_ms: float
    max_duration_ms: float


def fit_transition_duration_constraint(
    items_csv: Path,
    output_json: Path,
    *,
    target_phone: str,
    target_state: str = "transition",
    minimum_markers: int = 5,
    minimum_recordings: int = 2,
    minimum_speakers: int = 2,
) -> dict[str, object]:
    """Fit one development-only phone/state duration constraint from train Gold.

    A transition marker is the start of the target state. Its duration is the
    distance from that marker to the reviewed end edge of the same phone.
    Validation and locked-test annotations are counted in the manifest but are
    deliberately not opened or used by this fitter.
    """

    phone_id = target_phone.strip()
    state_id = target_state.strip()
    if not phone_id or not state_id:
        raise TransitionDurationConstraintError("target phone/state must not be empty")
    if minimum_markers < 1 or minimum_recordings < 1 or minimum_speakers < 1:
        raise TransitionDurationConstraintError("minimum support values must be positive")

    items_path = items_csv.resolve()
    rows = _read_csv(items_path)
    split_counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split", "")).strip() or "unassigned"
        split_counts[split] = split_counts.get(split, 0) + 1
    if split_counts.get("train", 0) == 0:
        raise TransitionDurationConstraintError("items CSV has no train rows")

    projects: dict[Path, ProjectSpec] = {}
    observations: list[dict[str, object]] = []
    eligible_recordings: set[str] = set()
    marked_recordings: set[str] = set()
    marked_speakers: set[str] = set()
    eligible_state_count = 0

    for row in rows:
        if str(row.get("split", "")).strip() != "train":
            continue
        project_path = Path(_required_text(row.get("project"), "items.project")).resolve()
        project = projects.get(project_path)
        if project is None:
            project = load_project(project_path)
            projects[project_path] = project
        recording_index = int(_required_text(row.get("recording_index"), "items.recording_index"))
        if not 0 <= recording_index < len(project.recordings):
            raise TransitionDurationConstraintError(f"recording index outside project: {project_path}/{recording_index}")
        source_recording = project.recordings[recording_index]
        annotation_path = Path(_required_text(row.get("annotation"), "items.annotation")).resolve()
        expected_annotation_sha = str(row.get("annotation_sha256", "")).strip().casefold()
        actual_annotation_sha = file_sha256(annotation_path)
        if expected_annotation_sha and expected_annotation_sha != actual_annotation_sha:
            raise TransitionDurationConstraintError(f"annotation hash mismatch: {annotation_path}")
        annotation = load_annotation(annotation_path, project=project)
        if annotation.label_source != "manual_gold":
            raise TransitionDurationConstraintError(f"annotation is not manual_gold: {annotation_path}")
        if source_recording.wav != annotation.wav or file_sha256(project.wav_path(source_recording)) != annotation.wav_sha256:
            raise TransitionDurationConstraintError(f"recording/annotation WAV mismatch: {annotation_path}")

        evaluation_recording = RecordingSpec(
            wav=source_recording.wav,
            phones=tuple(annotation.phones),
            aliases=source_recording.aliases,
            transcript_source=annotation.transcript_source,
            label_source=annotation.label_source,
            metadata=source_recording.metadata,
            guide_anchors=source_recording.guide_anchors,
        )
        compiled = compile_recording_states(evaluation_recording, project.inventory)
        targets = [
            state
            for state in compiled
            if state.phone_id == phone_id
            and state.state_id == state_id
            and state.acoustic_role in TRANSITION_ROLES
        ]
        if not targets:
            continue
        recording_id = str(row.get("recording_id", "")).strip() or f"{project_path}:{recording_index}"
        speaker_id = _required_text(row.get("root_speaker_id"), "items.root_speaker_id")
        eligible_recordings.add(recording_id)
        eligible_state_count += len(targets)
        edges = _gold_edges(annotation)
        markers_by_instance: dict[str, list[object]] = {}
        final_state_by_phone = {state.phone_index: state.instance_id for state in compiled}
        for marker in annotation.markers:
            if marker.label == "subphone_transition" and marker.state_instance_id:
                markers_by_instance.setdefault(marker.state_instance_id, []).append(marker)
        for state in targets:
            # A reviewed phone end is a state end only for the final state of
            # that occurrence. A vowel-transition followed by stable vowel
            # cannot use this duration target without a reviewed internal end.
            if final_state_by_phone[state.phone_index] != state.instance_id:
                raise TransitionDurationConstraintError(
                    f"{recording_id}: target state is not phone-final; reviewed state end required: {state.instance_id}"
                )
            markers = markers_by_instance.get(state.instance_id, [])
            if len(markers) != 1:
                raise TransitionDurationConstraintError(
                    f"{recording_id}: expected exactly one manual marker for {state.instance_id}, got {len(markers)}"
                )
            marker = markers[0]
            if (not str(marker.source).startswith("manual_") or not str(marker.annotator).strip()
                    or not math.isfinite(float(marker.confidence)) or float(marker.confidence) < 1.0):
                raise TransitionDurationConstraintError(
                    f"{recording_id}: marker is not confirmed manual Gold: {state.instance_id}"
                )
            start_ms = float(marker.time_ms)
            phone_start_ms = float(edges[state.phone_index])
            phone_end_ms = float(edges[state.phone_index + 1])
            if not phone_start_ms < start_ms < phone_end_ms:
                raise TransitionDurationConstraintError(
                    f"{recording_id}: marker is outside its reviewed phone interval: {state.instance_id}"
                )
            observations.append(
                {
                    "recording_id": recording_id,
                    "bank_id": str(row.get("bank_id", "")).strip(),
                    "root_speaker_id": speaker_id,
                    "split": "train",
                    "phone_id": state.phone_id,
                    "phone_index": state.phone_index,
                    "state_id": state.state_id,
                    "state_instance_id": state.instance_id,
                    "acoustic_role": state.acoustic_role,
                    "phone_start_ms": phone_start_ms,
                    "transition_start_ms": start_ms,
                    "phone_end_ms": phone_end_ms,
                    "duration_ms": phone_end_ms - start_ms,
                    "marker_source": marker.source,
                    "marker_annotator": marker.annotator,
                    "annotation": str(annotation_path),
                    "annotation_sha256": actual_annotation_sha,
                }
            )
            marked_recordings.add(recording_id)
            marked_speakers.add(speaker_id)

    blockers: list[str] = []
    if len(observations) < minimum_markers:
        blockers.append(f"manual_marker_count<{minimum_markers}")
    if len(marked_recordings) < minimum_recordings:
        blockers.append(f"marked_recording_count<{minimum_recordings}")
    if len(marked_speakers) < minimum_speakers:
        blockers.append(f"marked_speaker_count<{minimum_speakers}")
    if len(observations) != eligible_state_count:
        blockers.append("not_all_eligible_train_states_have_manual_markers")
    if blockers:
        raise TransitionDurationConstraintError("insufficient train-only support: " + ", ".join(blockers))

    values = np.asarray([float(row["duration_ms"]) for row in observations], dtype=np.float64)
    quantiles = {name: float(np.quantile(values, q)) for name, q in (
        ("q10", 0.10), ("q25", 0.25), ("median", 0.50), ("q75", 0.75), ("q90", 0.90)
    )}
    min_ms = _floor_step(quantiles["q10"], 5.0)
    mode_ms = _round_step(quantiles["median"], 2.5)
    max_ms = _ceil_step(float(np.max(values)), 5.0)
    min_ms = max(5.0, min(min_ms, mode_ms))
    max_ms = max(mode_ms, max_ms)

    output_path = output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations_path = output_path.with_name(f"{output_path.stem}_observations.csv")
    _write_csv(observations_path, observations)
    payload = {
        "schema_version": 1,
        "purpose": CONSTRAINT_PURPOSE,
        "target_phone": phone_id,
        "target_state": state_id,
        "source_items_csv": str(items_path),
        "source_items_csv_sha256": file_sha256(items_path),
        "fit_split": "train",
        "duration_target_policy": "reviewed_start_to_same_phone_final_edge_v1",
        "excluded_from_fit": ["validation", "locked_test"],
        "source_row_counts_by_split": dict(sorted(split_counts.items())),
        "train_rows_loaded": split_counts.get("train", 0),
        "non_train_annotations_loaded": 0,
        "support": {
            "manual_marker_count": len(observations),
            "eligible_state_count": eligible_state_count,
            "eligible_recording_count": len(eligible_recordings),
            "marked_recording_count": len(marked_recordings),
            "marked_speaker_count": len(marked_speakers),
            "recording_ids": sorted(marked_recordings),
            "speaker_ids": sorted(marked_speakers),
        },
        "distribution_ms": {
            "count": len(values),
            "min": float(np.min(values)),
            **quantiles,
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        },
        "constraints": [
            {
                "phone_id": phone_id,
                "state_id": state_id,
                "acoustic_roles": sorted({str(row["acoustic_role"]) for row in observations}),
                "min_duration_ms": min_ms,
                "mode_duration_ms": mode_ms,
                "max_duration_ms": max_ms,
            }
        ],
        "fit_rule": {
            "min_duration_ms": "floor(train_q10_ms / 5) * 5",
            "mode_duration_ms": "round_half_up(train_median_ms / 2.5) * 2.5",
            "max_duration_ms": "ceil(train_max_ms / 5) * 5",
        },
        "observations_csv": str(observations_path),
        "observations_csv_sha256": file_sha256(observations_path),
        "validation_labels_used": False,
        "locked_test_labels_used": False,
        "release_promotion_evidence": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_path, payload)
    return {**payload, "output_json": str(output_path), "output_json_sha256": file_sha256(output_path)}


def load_phone_state_duration_constraints(
    path: Path,
) -> tuple[dict[tuple[str, str], PhoneStateDurationConstraint], dict[str, object]]:
    artifact_path = path.resolve()
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TransitionDurationConstraintError(f"cannot load duration constraint: {artifact_path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise TransitionDurationConstraintError("duration constraint schema_version must be 1")
    if payload.get("purpose") != CONSTRAINT_PURPOSE:
        raise TransitionDurationConstraintError("duration constraint purpose is not supported")
    if payload.get("fit_split") != "train":
        raise TransitionDurationConstraintError("duration constraint must be fitted from train only")
    if payload.get("validation_labels_used") is not False or payload.get("locked_test_labels_used") is not False:
        raise TransitionDurationConstraintError("duration constraint contains validation/locked-test fit evidence")
    if payload.get("release_promotion_evidence") is not False:
        raise TransitionDurationConstraintError("development duration constraint must not claim release evidence")
    _verify_bound_file(
        payload.get("source_items_csv"),
        payload.get("source_items_csv_sha256"),
        "source_items_csv",
    )
    _verify_bound_file(
        payload.get("observations_csv"),
        payload.get("observations_csv_sha256"),
        "observations_csv",
    )
    raw_constraints = payload.get("constraints")
    if isinstance(raw_constraints, (str, bytes)) or not isinstance(raw_constraints, Sequence) or not raw_constraints:
        raise TransitionDurationConstraintError("duration constraint list is empty")
    constraints: dict[tuple[str, str], PhoneStateDurationConstraint] = {}
    for index, raw in enumerate(raw_constraints):
        if not isinstance(raw, Mapping):
            raise TransitionDurationConstraintError(f"constraints[{index}] must be a mapping")
        phone_id = _required_text(raw.get("phone_id"), f"constraints[{index}].phone_id")
        state_id = _required_text(raw.get("state_id"), f"constraints[{index}].state_id")
        roles = raw.get("acoustic_roles")
        if isinstance(roles, (str, bytes)) or not isinstance(roles, Sequence) or not roles:
            raise TransitionDurationConstraintError(f"constraints[{index}].acoustic_roles must be a list")
        parsed = PhoneStateDurationConstraint(
            phone_id=phone_id,
            state_id=state_id,
            acoustic_roles=tuple(str(role) for role in roles),
            min_duration_ms=_number(raw.get("min_duration_ms"), f"constraints[{index}].min_duration_ms"),
            mode_duration_ms=_number(raw.get("mode_duration_ms"), f"constraints[{index}].mode_duration_ms"),
            max_duration_ms=_number(raw.get("max_duration_ms"), f"constraints[{index}].max_duration_ms"),
        )
        if not 0.0 < parsed.min_duration_ms <= parsed.mode_duration_ms <= parsed.max_duration_ms:
            raise TransitionDurationConstraintError(f"constraints[{index}] has invalid min/mode/max ordering")
        key = (phone_id, state_id)
        if key in constraints:
            raise TransitionDurationConstraintError(f"duplicate duration constraint: {phone_id}.{state_id}")
        constraints[key] = parsed
    provenance = {
        "artifact": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "purpose": payload["purpose"],
        "fit_split": payload["fit_split"],
        "source_items_csv": payload.get("source_items_csv"),
        "source_items_csv_sha256": payload.get("source_items_csv_sha256"),
        "validation_labels_used": False,
        "locked_test_labels_used": False,
        "release_promotion_evidence": False,
        "constraints": [
            {
                "phone_id": item.phone_id,
                "state_id": item.state_id,
                "acoustic_roles": list(item.acoustic_roles),
                "min_duration_ms": item.min_duration_ms,
                "mode_duration_ms": item.mode_duration_ms,
                "max_duration_ms": item.max_duration_ms,
            }
            for item in constraints.values()
        ],
    }
    return constraints, provenance


def _gold_edges(annotation: AnnotationDocument) -> list[float]:
    starts = [marker.time_ms for marker in annotation.markers if marker.label == "usable_start"]
    ends = [marker.time_ms for marker in annotation.markers if marker.label == "usable_end"]
    boundaries = {
        int(marker.right_phone_index): float(marker.time_ms)
        for marker in annotation.markers
        if marker.label == "phone_boundary" and marker.right_phone_index is not None
    }
    expected = set(range(1, len(annotation.phones)))
    if len(starts) != 1 or len(ends) != 1 or set(boundaries) != expected:
        raise TransitionDurationConstraintError(f"{annotation.annotation_id}: incomplete reviewed phone edges")
    edges = [float(starts[0]), *(boundaries[index] for index in range(1, len(annotation.phones))), float(ends[0])]
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise TransitionDurationConstraintError(f"{annotation.annotation_id}: non-increasing reviewed phone edges")
    return edges


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise TransitionDurationConstraintError(f"items CSV is empty: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise TransitionDurationConstraintError("cannot write empty duration observations")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(path, text.splitlines(), encoding="utf-8", newline="\n", backup_existing=path.exists())


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransitionDurationConstraintError(f"{field} is empty")
    return value.strip()


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransitionDurationConstraintError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise TransitionDurationConstraintError(f"{field} must be finite")
    return number


def _verify_bound_file(raw_path: object, raw_sha256: object, field: str) -> None:
    path = Path(_required_text(raw_path, field)).resolve()
    expected = _required_text(raw_sha256, f"{field}_sha256").casefold()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise TransitionDurationConstraintError(f"{field}_sha256 must be a SHA-256 hex digest")
    if not path.is_file():
        raise TransitionDurationConstraintError(f"bound duration constraint input is missing: {path}")
    if file_sha256(path) != expected:
        raise TransitionDurationConstraintError(f"bound duration constraint input hash mismatch: {path}")


def _floor_step(value: float, step: float) -> float:
    return float(math.floor(value / step) * step)


def _ceil_step(value: float, step: float) -> float:
    return float(math.ceil(value / step) * step)


def _round_step(value: float, step: float) -> float:
    return float(math.floor(value / step + 0.5) * step)


__all__ = [
    "PhoneStateDurationConstraint",
    "TransitionDurationConstraintError",
    "fit_transition_duration_constraint",
    "load_phone_state_duration_constraints",
]
