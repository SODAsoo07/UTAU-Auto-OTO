from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.alignment import _state_spec
from universal_auto_oto.annotations import AnnotationDocument, AnnotationMarker, save_annotation
from universal_auto_oto.domain import CompiledState
from universal_auto_oto.project import load_project
from universal_auto_oto.transition_duration_constraint import (
    PhoneStateDurationConstraint,
    TransitionDurationConstraintError,
    fit_transition_duration_constraint,
    load_phone_state_duration_constraints,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> Path:
    (tmp_path / "phones.yaml").write_text(
        """format_version: 1
phones:
  r:
    display_name: r
    features: {voiced: true, manner: liquid}
    states:
      - {id: consonant, acoustic_role: consonant, min_ms: 8, mode_ms: 30, max_ms: 120}
      - {id: transition, acoustic_role: transition, min_ms: 8, mode_ms: 35, max_ms: 160}
""",
        encoding="utf-8",
    )
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    for name in ("train_a.wav", "train_b.wav", "validation.wav"):
        sf.write(wav_dir / name, np.zeros(3200, dtype=np.float32), 16000)
    (tmp_path / "recordings.yaml").write_text(
        """format_version: 1
recordings:
  - wav: wav/train_a.wav
    phones: [r]
    transcript_source: manual_fixture
    label_source: manual_gold
    metadata: {root_speaker_id: speaker_a, bank_id: bank_a, split: train}
    aliases: [{alias: r_a, template: builtin:v, phones: [0]}]
  - wav: wav/train_b.wav
    phones: [r]
    transcript_source: manual_fixture
    label_source: manual_gold
    metadata: {root_speaker_id: speaker_b, bank_id: bank_b, split: train}
    aliases: [{alias: r_b, template: builtin:v, phones: [0]}]
  - wav: wav/validation.wav
    phones: [r]
    transcript_source: manual_fixture
    label_source: manual_gold
    metadata: {root_speaker_id: speaker_c, bank_id: bank_c, split: validation}
    aliases: [{alias: r_c, template: builtin:v, phones: [0]}]
""",
        encoding="utf-8",
    )
    (tmp_path / "project.yaml").write_text(
        """format_version: 1
project: {name: duration_fixture, language_hint: ko}
inventory: phones.yaml
recordings: recordings.yaml
audio: {recursive: false, convert_to_mono: true, internal_sample_rate: 16000, resampler: soxr_hq}
alignment: {model: null, backend: hsmm, device: cpu, feature_backend: numpy_multiband_pitch_v2, use_ctc_prior: false, use_hsmm: true, use_prototypes: false}
generation: {templates: [builtin:v], output: generated/oto.ini, output_layout: single_root, output_encoding: utf-8, atomic_write: true, backup_existing: true, write_debug_timeline: false, write_textgrid: false, split_review: true}
calibration: {enabled: false}
""",
        encoding="utf-8",
    )
    project = load_project(tmp_path / "project.yaml")
    rows: list[dict[str, object]] = []
    for index, transition_start in enumerate((50.0, 60.0)):
        recording = project.recordings[index]
        annotation_path = tmp_path / f"train_{index}.annotation.json"
        document = AnnotationDocument(
            annotation_id=f"duration_{index}",
            revision=1,
            wav=recording.wav,
            wav_sha256=_sha256(project.wav_path(recording)),
            inventory_sha256=project.inventory_sha256,
            transcript_source="manual_fixture",
            label_source="manual_gold",
            phones=("r",),
            markers=(
                AnnotationMarker("usable_start", 0.0, 1.0, "reviewer", "manual_fixture"),
                AnnotationMarker(
                    "subphone_transition",
                    transition_start,
                    1.0,
                    "reviewer",
                    "manual_vlabeler_subphone_transition",
                    phone_index=0,
                    state_id="transition",
                    state_instance_id="p0000.r.transition",
                ),
                AnnotationMarker("usable_end", 100.0, 1.0, "reviewer", "manual_fixture"),
            ),
        )
        save_annotation(document, annotation_path)
        rows.append(
            {
                "recording_id": f"train_{index}",
                "bank_id": f"bank_{index}",
                "root_speaker_id": f"speaker_{index}",
                "split": "train",
                "recording_index": index,
                "project": str(tmp_path / "project.yaml"),
                "annotation": str(annotation_path),
                "annotation_sha256": _sha256(annotation_path),
            }
        )
    rows.append(
        {
            "recording_id": "validation",
            "bank_id": "validation_bank",
            "root_speaker_id": "validation_speaker",
            "split": "validation",
            "recording_index": 2,
            "project": str(tmp_path / "project.yaml"),
            "annotation": str(tmp_path / "must_not_be_loaded.annotation.json"),
            "annotation_sha256": "missing-on-purpose",
        }
    )
    items = tmp_path / "items.csv"
    _write_csv(items, rows)
    return items


def test_fit_transition_duration_constraint_uses_train_only_and_is_loadable(tmp_path: Path) -> None:
    items = _fixture(tmp_path)
    output = tmp_path / "r_duration.json"

    result = fit_transition_duration_constraint(items, output, target_phone="r", minimum_markers=2)

    assert result["support"]["manual_marker_count"] == 2
    assert result["support"]["marked_speaker_count"] == 2
    assert result["distribution_ms"]["min"] == 40.0
    assert result["distribution_ms"]["max"] == 50.0
    assert result["non_train_annotations_loaded"] == 0
    assert result["validation_labels_used"] is False
    assert result["constraints"] == [
        {
            "phone_id": "r",
            "state_id": "transition",
            "acoustic_roles": ["transition"],
            "min_duration_ms": 40.0,
            "mode_duration_ms": 45.0,
            "max_duration_ms": 50.0,
        }
    ]
    constraints, provenance = load_phone_state_duration_constraints(output)
    assert constraints[("r", "transition")].mode_duration_ms == 45.0
    assert provenance["artifact_sha256"] == _sha256(output)


def test_state_spec_applies_only_exact_phone_state_constraint() -> None:
    state = CompiledState("p0000.r.transition", 0, "r", "transition", "transition", 8.0, 160.0, 35.0)
    constraint = PhoneStateDurationConstraint("r", "transition", ("transition",), 30.0, 42.5, 90.0)

    spec = _state_spec(state, {("r", "transition"): constraint})

    assert spec.min_duration_ms == 30.0
    assert spec.mode_duration_ms == 42.5
    assert spec.max_duration_ms == 90.0


def test_duration_fit_rejects_nonfinal_state_without_reviewed_end(tmp_path, monkeypatch):
    from universal_auto_oto import transition_duration_constraint as module
    items = _fixture(tmp_path)
    original_compile = module.compile_recording_states

    def with_stable_tail(recording, inventory):
        return (*original_compile(recording, inventory),
                CompiledState("p0000.r.stable", 0, "r", "stable", "vowel_stable", 5., 100., 40.))

    monkeypatch.setattr(module, "compile_recording_states", with_stable_tail)
    with pytest.raises(TransitionDurationConstraintError, match="reviewed state end required"):
        fit_transition_duration_constraint(items, tmp_path / "invalid.json", target_phone="r", minimum_markers=2)
    assert not (tmp_path / "invalid.json").exists()


def test_loader_rejects_constraint_that_used_validation_labels(tmp_path: Path) -> None:
    artifact = tmp_path / "bad.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "development_train_phone_state_duration_constraint",
                "fit_split": "train",
                "validation_labels_used": True,
                "locked_test_labels_used": False,
                "constraints": [
                    {
                        "phone_id": "r",
                        "state_id": "transition",
                        "acoustic_roles": ["transition"],
                        "min_duration_ms": 30.0,
                        "mode_duration_ms": 40.0,
                        "max_duration_ms": 90.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TransitionDurationConstraintError, match="validation/locked-test"):
        load_phone_state_duration_constraints(artifact)
