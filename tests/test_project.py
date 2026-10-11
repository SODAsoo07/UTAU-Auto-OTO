from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.domain import PhoneDefinition, PhoneStateDefinition, RecordingSpec
from universal_auto_oto.project import ProjectValidationError, load_project, save_project
from universal_auto_oto.sequence_compiler import (
    compile_recording_states,
    compiled_state_inference_provenance,
)


def _write_project(root: Path, *, extra_state_field: str = "") -> Path:
    (root / "wav").mkdir()
    sf.write(root / "wav" / "ka.wav", np.zeros(4410, dtype=np.float32), 44100)
    (root / "project.yaml").write_text(
        """format_version: 1
project:
  name: Test Bank
  language_hint: mixed
inventory: phones.yaml
recordings: recordings.yaml
audio:
  recursive: false
  convert_to_mono: true
  internal_sample_rate: 44100
  resampler: soxr_hq
alignment:
  model: null
  backend: hsmm
  device: cpu
generation:
  templates: [builtin:cv]
  output: output/oto.ini
  output_layout: preserve_source_directories
  output_encoding: utf-8
  atomic_write: true
  backup_existing: true
calibration:
  enabled: false
""",
        encoding="utf-8",
    )
    (root / "phones.yaml").write_text(
        f"""format_version: 1
phones:
  k:
    display_name: K
    features: {{manner: stop, voiced: false}}
    states:
      - id: closure
        acoustic_role: closure
        min_ms: 5
        max_ms: 80
        mode_ms: 30
        {extra_state_field}
  a:
    display_name: A
    features: {{syllabic: true, voiced: true}}
    states:
      - id: transition
        acoustic_role: vowel_transition
        min_ms: 10
        max_ms: 80
      - id: stable
        acoustic_role: vowel_stable
        min_ms: 20
        max_ms: 500
""",
        encoding="utf-8",
    )
    (root / "recordings.yaml").write_text(
        """format_version: 1
recordings:
  - wav: wav/ka.wav
    phones: [k, a]
    transcript_source: project_manifest
    label_source: none
    aliases:
      - alias: ka
        template: builtin:cv
        phones: [0, 1]
""",
        encoding="utf-8",
    )
    return root / "project.yaml"


def test_project_load_compile_and_roundtrip_hash(tmp_path: Path) -> None:
    project_path = _write_project(tmp_path)
    project = load_project(project_path)
    assert project.name == "Test Bank"
    assert project.alignment.feature_backend == "numpy_multiband_pitch_v2"
    assert tuple(project.inventory) == ("k", "a")
    assert project.recordings[0].aliases[0].role == "cv"
    assert project.calibration.alias_type_enabled is False
    assert project.calibration.alias_type_profile is None

    states = compile_recording_states(project.recordings[0], project.inventory)
    assert [state.instance_id for state in states] == [
        "p0000.k.closure",
        "p0001.a.transition",
        "p0001.a.stable",
    ]
    assert len(set(state.instance_id for state in states)) == len(states)

    original_hashes = (project.project_sha256, project.inventory_sha256, project.recordings_sha256)
    save_project(project, tmp_path, force=True)
    reloaded = load_project(project_path)
    assert (reloaded.project_sha256, reloaded.inventory_sha256, reloaded.recordings_sha256) == original_hashes


def test_unconfigured_new_alignment_and_calibration_fields_preserve_v1_project_hash(tmp_path: Path) -> None:
    project = load_project(_write_project(tmp_path))

    # This fixture has no opt-in neural/scorer/renderer setting.  Its semantic
    # identity must remain compatible with Gold review contracts sealed before
    # those optional defaults were introduced.
    assert project.project_sha256 == "d00371e78d63dbbc4f4829d032a18f6738b4bf9ea16e3340022c2f87702ebc3b"


def test_project_requires_an_explicit_model_for_neural_evidence_backend(tmp_path: Path) -> None:
    project_path = _write_project(tmp_path)
    original = project_path.read_text(encoding="utf-8")
    project_path.write_text(
        original.replace("  device: cpu", "  device: cuda\n  feature_backend: onnx_neural_evidence_v1"),
        encoding="utf-8",
    )
    with pytest.raises(ProjectValidationError, match="evidence_model"):
        load_project(project_path)
    project_path.write_text(
        original.replace(
            "  device: cpu",
            "  device: cuda\n  feature_backend: onnx_neural_evidence_v1\n  evidence_model: models/evidence.onnx\n  evidence_provider: cuda",
        ),
        encoding="utf-8",
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "evidence.onnx").write_bytes(b"test model placeholder")
    project = load_project(project_path)
    assert project.alignment.evidence_model == "models/evidence.onnx"
    assert project.alignment.evidence_provider == "cuda"


def test_project_requires_hash_bound_model_for_trained_subphone_backend(tmp_path: Path) -> None:
    project_path = _write_project(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "trained_subphone.json").write_text("{}", encoding="utf-8")
    original = project_path.read_text(encoding="utf-8")
    project_path.write_text(
        original.replace(
            "  device: cpu",
            "  device: cpu\n  scorer_backend: trained_subphone_v1\n  scorer_model: models/trained_subphone.json",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectValidationError, match="scorer_model"):
        load_project(project_path)

    project_path.write_text(
        original.replace(
            "  device: cpu",
            "  device: cpu\n  scorer_backend: trained_subphone_v1\n"
            "  scorer_model: models/trained_subphone.json\n  scorer_model_sha256: 'not-a-hash'",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectValidationError, match="SHA-256"):
        load_project(project_path)


def test_compiled_states_record_repeated_phone_occurrences(tmp_path: Path) -> None:
    project = load_project(_write_project(tmp_path))
    recording = replace(project.recordings[0], phones=("k", "a", "k"))

    states = compile_recording_states(recording, project.inventory)

    k_states = [state for state in states if state.phone_id == "k"]
    assert [state.occurrence_index for state in k_states] == [0, 1]
    assert [state.occurrence_count for state in k_states] == [2, 2]
    assert [state.instance_id for state in k_states] == ["p0000.k.closure", "p0002.k.closure"]


def test_compiler_records_symbolic_interval_and_legacy_vf_state_inference() -> None:
    interval = PhoneStateDefinition("interval", "phone_interval", 0.1, 60000.0, 100.0)
    inventory = {
        "s": PhoneDefinition("s", "s", {"manual_review_interval": True}, (interval,)),
        "vf": PhoneDefinition(
            "vf",
            "vocal fry",
            {"manner": "vocal_fry", "voiced": True},
            (PhoneStateDefinition("fry", "irregular_periodic", 8.0, 700.0, 70.0),),
        ),
    }
    recording = RecordingSpec("sample.wav", ("s", "vf"), (), "project_manifest", "none")

    states = compile_recording_states(recording, inventory)
    inference = compiled_state_inference_provenance(recording, inventory)

    assert [(state.phone_id, state.state_id, state.acoustic_role) for state in states] == [
        ("s", "frication", "frication"),
        ("vf", "onset", "irregular_voicing_onset"),
        ("vf", "fry", "aperiodic"),
    ]
    assert [row["phone_id"] for row in inference] == ["s", "vf"]
    assert all(row["inference_version"] == "symbolic_phone_state_v1" for row in inference)


def test_project_rejects_unsupported_topology_field(tmp_path: Path) -> None:
    project_path = _write_project(tmp_path, extra_state_field="optional: true")
    with pytest.raises(ProjectValidationError, match="지원하지 않는 필드"):
        load_project(project_path)


def test_project_rejects_out_of_range_alias_phone_index(tmp_path: Path) -> None:
    project_path = _write_project(tmp_path)
    recordings = tmp_path / "recordings.yaml"
    recordings.write_text(recordings.read_text(encoding="utf-8").replace("phones: [0, 1]", "phones: [0, 2]"), encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="범위를 벗어났습니다"):
        load_project(project_path)


def test_project_requires_alias_type_profile_when_confidence_calibration_is_enabled(tmp_path: Path) -> None:
    project_path = _write_project(tmp_path)
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "calibration:\n  enabled: false",
            "calibration:\n  enabled: false\n  alias_type_enabled: true",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectValidationError, match="alias_type_profile"):
        load_project(project_path)
