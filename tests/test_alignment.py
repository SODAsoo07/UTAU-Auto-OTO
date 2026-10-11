import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.alignment import (
    AlignmentError,
    _compile_frame_score_compilation,
    _compile_frame_scores,
    _compile_frame_scores_legacy_reference,
    _diagnostic_warnings,
    _lexical_coverage_diagnostics,
    _normalized_ranker_start_bonus,
    _occurrence_ranker_start_scores,
    _supports_acoustic_rhythm_prior,
    align_recording,
    load_alignment_profile,
)
from universal_auto_oto.acoustic_evidence import extract_acoustic_evidence
from universal_auto_oto.frame_scorer import FrameScoreResult
from universal_auto_oto.trained_frame_scorer import (
    TrainedSubphoneScorerError,
    _feature_schema_sha256,
    load_trained_subphone_scorer,
)
from universal_auto_oto.annotations import build_annotation_queue, initialize_annotation, load_annotation
from universal_auto_oto.audio import file_sha256
from universal_auto_oto.ctc_prior import known_sequence_ctc_posteriors
from universal_auto_oto.domain import AliasPlan, RecordingSpec
from universal_auto_oto.confidence import (
    build_confidence_examples,
    build_confidence_examples_from_candidate_items,
    load_confidence_examples,
)
from universal_auto_oto.project import CalibrationConfig, load_project
from universal_auto_oto.project_runtime import TimelineReuseError, align_project, generate_project
from universal_auto_oto.sequence_compiler import compile_recording_states
from universal_auto_oto.posterior_model import (
    PosteriorModelError,
    train_posterior_model,
    train_posterior_model_from_manifest,
)
from universal_auto_oto.renderer_calibration import ANCHOR_FIELDS, fit_renderer_calibration
from universal_auto_oto.template_engine import TemplateError, load_template_set, render_project_alias


def _project(tmp_path: Path, *, backend: str = "hsmm", guide: bool = True) -> Path:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    sample_rate = 44100
    silence = np.zeros(int(0.10 * sample_rate), dtype=np.float32)
    rng = np.random.default_rng(7)
    consonant = (0.08 * rng.standard_normal(int(0.08 * sample_rate))).astype(np.float32)
    times = np.arange(int(0.32 * sample_rate), dtype=np.float64) / sample_rate
    vowel = (0.25 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    tail = np.zeros(int(0.05 * sample_rate), dtype=np.float32)
    sf.write(wav_dir / "ka.wav", np.concatenate((silence, consonant, vowel, tail)), sample_rate)
    (tmp_path / "project.yaml").write_text(
        f"""format_version: 1
project: {{name: Align Test, language_hint: mixed}}
inventory: phones.yaml
recordings: recordings.yaml
audio:
  recursive: false
  convert_to_mono: true
  internal_sample_rate: 44100
  resampler: soxr_hq
alignment:
  model: null
  backend: {backend}
  device: cpu
  use_hsmm: true
generation:
  templates: [builtin:cv]
  output: output/oto.ini
  output_layout: preserve_source_directories
  output_encoding: utf-8
  atomic_write: true
  backup_existing: true
calibration: {{enabled: false}}
""",
        encoding="utf-8",
    )
    (tmp_path / "phones.yaml").write_text(
        """format_version: 1
phones:
  k:
    display_name: K
    features: {manner: stop, voiced: false, continuant: false}
    states:
      - {id: closure, acoustic_role: closure, min_ms: 20, mode_ms: 60, max_ms: 180}
      - {id: release, acoustic_role: burst, min_ms: 5, mode_ms: 15, max_ms: 60}
      - {id: transition, acoustic_role: transition, min_ms: 10, mode_ms: 40, max_ms: 100}
  a:
    display_name: A
    features: {syllabic: true, voiced: true, vowel_like: true}
    states:
      - {id: transition, acoustic_role: vowel_transition, min_ms: 20, mode_ms: 50, max_ms: 150}
      - {id: stable, acoustic_role: vowel_stable, min_ms: 100, mode_ms: 280, max_ms: 600}
""",
        encoding="utf-8",
    )
    guide_block = """
    guide_anchors:
      - {label: vowel_transition, time_ms: 180.0, phone_index: 1, confidence: 1.0, source: manual_guide}
""" if guide else ""
    (tmp_path / "recordings.yaml").write_text(
        """format_version: 1
recordings:
  - wav: wav/ka.wav
    phones: [k, a]
    transcript_source: project_manifest
    label_source: none
    aliases:
      - {alias: ka, template: builtin:cv, phones: [0, 1]}
""" + guide_block,
        encoding="utf-8",
    )
    return tmp_path / "project.yaml"


def test_alignment_profile_and_hsmm_project_path(tmp_path: Path) -> None:
    profile = load_alignment_profile()
    assert profile.profile_id == "acoustic_hsmm_v2"
    assert profile.sha256

    project = load_project(_project(tmp_path))
    result = align_recording(project, project.recordings[0])
    timeline = result.timeline
    assert result.backend_used == "hsmm"
    assert timeline.schema_version == 2
    assert len(timeline.phone_segments) == 2
    assert len({state.instance_id for phone in timeline.phone_segments for state in phone.state_segments}) == 5
    assert timeline.landmark_time(1, "vowel_transition") == 180.0
    assert any(item.source == "manual_guide" for item in timeline.landmarks)
    assert timeline.provenance["label_source"] == "acoustic_heuristic_hsmm_v2_pseudo"
    assert timeline.provenance["confidence_calibrated"] is False
    assert timeline.provenance["timebase"]["frame_time_anchor"] == "center"
    assert timeline.provenance["frame_scorer"]["provider"] == "heuristic_frame_score_v1"
    assert timeline.provenance["frame_scorer"]["fallback"] is True
    assert timeline.provenance["frame_scorer"]["fallback_reason"] == "no_trained_frame_score_provider_configured"
    assert timeline.provenance["frame_scorer"]["neural_evidence_shadow"] == {
        "enabled": False,
        "model_sha256": None,
        "manifest_sha256": None,
        "execution_provider": None,
        "transition_scorer": "disabled_no_reviewed_subphone_transition_labels",
        "fallback_reason": "feature_backend_not_onnx_neural_evidence_v1",
    }


def test_lexical_coverage_collapse_is_reported_for_a_long_uncovered_tail(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    states = compile_recording_states(project.recordings[0], project.inventory)
    decoded = tuple(
        SimpleNamespace(start_ms=float(index * 100), end_ms=float((index + 1) * 100))
        for index in range(len(states))
    )

    diagnostics = _lexical_coverage_diagnostics(
        states,
        decoded,
        usable_start_ms=0.0,
        usable_end_ms=2000.0,
    )

    assert diagnostics["lexical_coverage_available"] is True
    assert diagnostics["lexical_start_ms"] == 0.0
    assert diagnostics["lexical_end_ms"] == 500.0
    assert diagnostics["lexical_coverage_ratio"] == pytest.approx(0.25)
    assert diagnostics["lexical_trailing_uncovered_ms"] == 1500.0
    assert diagnostics["lexical_coverage_collapse"] is True
    assert "hsmm_coverage_collapse" in _diagnostic_warnings(diagnostics)


def test_heuristic_frame_score_provider_matches_legacy_reference(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    recording = project.recordings[0]
    evidence = extract_acoustic_evidence(
        project.root / recording.wav,
        internal_sample_rate=project.audio.internal_sample_rate,
        convert_to_mono=project.audio.convert_to_mono,
        resampler=project.audio.resampler,
    )
    states = compile_recording_states(recording, project.inventory)
    profile = load_alignment_profile()
    provider_scores, provider_warnings, provider_applied, provider_diagnostics = _compile_frame_scores(
        states, project, evidence, profile, recording=recording
    )
    legacy_scores, legacy_warnings, legacy_applied, legacy_diagnostics = _compile_frame_scores_legacy_reference(
        states, project, evidence, profile, recording=recording
    )
    assert provider_warnings == legacy_warnings
    assert provider_applied == legacy_applied
    assert provider_diagnostics == legacy_diagnostics
    assert provider_scores.keys() == legacy_scores.keys()
    for state_id in provider_scores:
        np.testing.assert_array_equal(provider_scores[state_id], legacy_scores[state_id])


def test_frame_score_provider_contract_preserves_optional_transition_starts(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    recording = project.recordings[0]
    evidence = extract_acoustic_evidence(
        project.root / recording.wav,
        internal_sample_rate=project.audio.internal_sample_rate,
        convert_to_mono=project.audio.convert_to_mono,
        resampler=project.audio.resampler,
    )
    states = compile_recording_states(recording, project.inventory)
    frame_count = len(evidence.frame_times_ms)

    class StubProvider:
        def score(self, *, states, project, recording, evidence):
            del project, recording, evidence
            emissions = {state.instance_id: np.full(frame_count, 0.5) for state in states}
            starts = {states[1].instance_id: np.full(frame_count, 0.2)}
            return FrameScoreResult(
                emission_scores=emissions,
                transition_start_scores=starts,
                provenance={"provider": "test_stub"},
            )

    compilation = _compile_frame_score_compilation(
        states,
        project,
        evidence,
        load_alignment_profile(),
        recording=recording,
        frame_score_provider=StubProvider(),
    )
    assert compilation.result.provenance["provider"] == "test_stub"
    assert set(compilation.result.transition_start_scores) == {states[1].instance_id}


def test_frame_score_provider_contract_rejects_invalid_transition_start_shape(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    recording = project.recordings[0]
    evidence = extract_acoustic_evidence(
        project.root / recording.wav,
        internal_sample_rate=project.audio.internal_sample_rate,
        convert_to_mono=project.audio.convert_to_mono,
        resampler=project.audio.resampler,
    )
    states = compile_recording_states(recording, project.inventory)

    class InvalidProvider:
        def score(self, *, states, project, recording, evidence):
            del project, recording, evidence
            return FrameScoreResult(
                emission_scores={state.instance_id: np.full(4, 0.5) for state in states},
                transition_start_scores={states[0].instance_id: np.full(3, 0.2)},
            )

    with pytest.raises(AlignmentError, match="emission_scores is invalid"):
        _compile_frame_score_compilation(
            states,
            project,
            evidence,
            load_alignment_profile(),
            recording=recording,
            frame_score_provider=InvalidProvider(),
        )


def test_hsmm_consumes_provider_transition_start_scores(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))

    class TransitionStartProvider:
        def score(self, *, states, project, recording, evidence):
            del project, recording
            frame_count = len(evidence.frame_times_ms)
            return FrameScoreResult(
                emission_scores={state.instance_id: np.full(frame_count, 0.5) for state in states},
                transition_start_scores={states[1].instance_id: np.full(frame_count, 0.05)},
                provenance={"provider": "transition_start_stub", "fallback": False},
            )

    result = align_recording(
        project,
        project.recordings[0],
        frame_score_provider=TransitionStartProvider(),
    )
    scorer = result.timeline.provenance["frame_scorer"]
    assert scorer["provider"] == "transition_start_stub"
    assert scorer["transition_start_score_state_count"] == 1


def _write_trained_subphone_artifact(
    project,
    path: Path,
    *,
    feature_backend: str | None = None,
    minimum_mean_confidence: float = 0.0,
    include_transition_start: bool = False,
    transition_only: bool = False,
) -> str:
    feature_names = ["periodicity", "vowel_like"]
    state_context_feature_names = [
        "phone_occurrence_ordinal",
        "phone_occurrence_count",
        "normalized_phone_position",
        "normalized_frame_position",
        "is_repeated_phone",
        "is_transition_role",
    ]
    payload = {
        "schema_version": 1,
        "model_kind": "trained_subphone_v1",
        "feature_backend": feature_backend or project.alignment.feature_backend,
        "feature_names": feature_names,
        "state_context_feature_names": state_context_feature_names,
        "feature_schema_sha256": _feature_schema_sha256(feature_names, state_context_feature_names),
        "normalization": {"mean": [0.0] * 8, "scale": [1.0] * 8},
        "emission_heads": (
            {} if transition_only else {"other": [0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        ),
        "runtime": {"minimum_mean_confidence": minimum_mean_confidence},
        "training": {
            "fit_split": "train",
            "validation_labels_used": False,
            "locked_test_labels_used": False,
            "inventory_sha256": project.inventory_sha256,
            "source_table_sha256": "a" * 64,
            "label_sources": (
                ["manual_gold", "manual_confirmed_subphone_transition"]
                if include_transition_start
                else ["manual_gold"]
            ),
        },
    }
    if include_transition_start:
        payload["transition_start_heads"] = {"other": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        payload["training"]["transition_labels"] = {
            "source": "manual_confirmed_subphone_transition",
            "marker_count": 1,
        }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return file_sha256(path)


def _configure_trained_subphone_project(project_path: Path, *, sha256: str, allow_fallback: bool = True) -> None:
    original = project_path.read_text(encoding="utf-8")
    project_path.write_text(
        original.replace(
            "  device: cpu",
            "  device: cpu\n"
            "  scorer_backend: trained_subphone_v1\n"
            "  scorer_model: models/trained_subphone.json\n"
            f"  scorer_model_sha256: '{sha256}'\n"
            f"  allow_heuristic_fallback: {'true' if allow_fallback else 'false'}",
        ),
        encoding="utf-8",
    )


def test_trained_subphone_provider_is_hash_bound_and_opt_in(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    baseline = load_project(project_path)
    artifact = tmp_path / "models" / "trained_subphone.json"
    digest = _write_trained_subphone_artifact(baseline, artifact)
    _configure_trained_subphone_project(project_path, sha256=digest)

    project = load_project(project_path)
    result = align_recording(project, project.recordings[0])

    scorer = result.timeline.provenance["frame_scorer"]
    assert scorer["provider"] == "trained_subphone_v1", scorer.get("fallback_reason")
    assert scorer["artifact_sha256"] == digest
    assert scorer["configured_scorer_backend"] == "trained_subphone_v1"
    assert scorer["fallback"] is False
    assert scorer["transition_start_enabled"] is False


def test_trained_subphone_invalid_hash_has_explicit_fallback_or_fails_closed(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    baseline = load_project(project_path)
    _write_trained_subphone_artifact(baseline, tmp_path / "models" / "trained_subphone.json")
    _configure_trained_subphone_project(project_path, sha256="0" * 64, allow_fallback=True)

    fallback_project = load_project(project_path)
    fallback = align_recording(fallback_project, fallback_project.recordings[0])
    scorer = fallback.timeline.provenance["frame_scorer"]
    assert scorer["provider"] == "heuristic_frame_score_v1"
    assert scorer["fallback_reason"] == "trained_subphone_v1_rejected:artifact_hash_mismatch"

    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "allow_heuristic_fallback: true", "allow_heuristic_fallback: false"
        ),
        encoding="utf-8",
    )
    strict_project = load_project(project_path)
    with pytest.raises(AlignmentError, match="artifact_hash_mismatch"):
        align_recording(strict_project, strict_project.recordings[0])


def test_trained_subphone_runtime_low_confidence_uses_explicit_fallback(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    baseline = load_project(project_path)
    artifact = tmp_path / "models" / "trained_subphone.json"
    digest = _write_trained_subphone_artifact(
        baseline,
        artifact,
        minimum_mean_confidence=1.0,
    )
    _configure_trained_subphone_project(project_path, sha256=digest)

    project = load_project(project_path)
    result = align_recording(project, project.recordings[0])
    assert result.timeline.provenance["frame_scorer"]["fallback_reason"] == "trained_subphone_v1_runtime_failed:low_artifact_confidence"


def test_trained_subphone_transition_start_scores_remain_optional_hsmm_bonus(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    baseline = load_project(project_path)
    artifact = tmp_path / "models" / "trained_subphone.json"
    digest = _write_trained_subphone_artifact(baseline, artifact, include_transition_start=True)
    _configure_trained_subphone_project(project_path, sha256=digest)

    project = load_project(project_path)
    result = align_recording(project, project.recordings[0])
    scorer = result.timeline.provenance["frame_scorer"]
    assert scorer["provider"] == "trained_subphone_v1"
    assert scorer["transition_start_enabled"] is True
    assert scorer["transition_start_score_state_count"] == 5


def test_transition_only_scorer_preserves_heuristic_emissions(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    baseline = load_project(project_path)
    artifact = tmp_path / "models" / "trained_subphone.json"
    digest = _write_trained_subphone_artifact(
        baseline,
        artifact,
        include_transition_start=True,
        transition_only=True,
    )
    _configure_trained_subphone_project(project_path, sha256=digest)

    project = load_project(project_path)
    result = align_recording(project, project.recordings[0])
    scorer = result.timeline.provenance["frame_scorer"]
    assert scorer["provider"] == "trained_subphone_v1"
    assert scorer["emission_mode"] == "heuristic_passthrough"
    assert scorer["emission_provider"] == "heuristic_frame_score_v1"


def test_trained_subphone_artifact_rejects_backend_inventory_and_outside_project(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    artifact = tmp_path / "models" / "trained_subphone.json"
    digest = _write_trained_subphone_artifact(project, artifact, feature_backend="numpy_multiband_pitch_v3")
    with pytest.raises(TrainedSubphoneScorerError, match="feature backend"):
        load_trained_subphone_scorer(artifact, project=project, expected_sha256=digest)

    outside = tmp_path.parent / "outside_trained_subphone.json"
    outside.write_text("{}", encoding="utf-8")
    try:
        with pytest.raises(TrainedSubphoneScorerError, match="inside the project root"):
            load_trained_subphone_scorer(outside, project=project, expected_sha256=file_sha256(outside))
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema_version", 99, "unsupported trained sub-phone scorer schema"),
        ("training.inventory_sha256", "b" * 64, "inventory hash mismatch"),
    ],
)
def test_trained_subphone_artifact_rejects_schema_and_inventory_binding(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    project = load_project(_project(tmp_path))
    artifact = tmp_path / "models" / "trained_subphone.json"
    _write_trained_subphone_artifact(project, artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if field == "training.inventory_sha256":
        payload["training"]["inventory_sha256"] = value
    else:
        payload[field] = value
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrainedSubphoneScorerError, match=expected):
        load_trained_subphone_scorer(artifact, project=project, expected_sha256=file_sha256(artifact))


def test_multiband_v3_profile_and_hsmm_project_path(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "device: cpu", "device: cpu\n  feature_backend: numpy_multiband_pitch_v3"
        ),
        encoding="utf-8",
    )
    project = load_project(project_path)
    profile = load_alignment_profile(project.alignment.feature_backend)
    result = align_recording(project, project.recordings[0])
    assert profile.profile_id == "acoustic_hsmm_v3_pitch_confidence_local_contrast"
    assert result.evidence.metadata["feature_backend"] == "numpy_multiband_pitch_v3"
    assert "pitch_confidence" in result.evidence.tracks
    assert result.timeline.provenance["alignment_profile"] == profile.profile_id


def test_neural_evidence_profile_is_declared_without_replacing_v2_default() -> None:
    profile = load_alignment_profile("onnx_neural_evidence_v1")
    assert profile.profile_id == "acoustic_hsmm_neural_v1"
    assert "neural_boundary_likelihood" in profile.role_weights["burst"]
    assert "neural_transition_likelihood" not in profile.role_weights["transition"]


def test_ranker_bonus_uses_relative_logits_without_sigmoid_offset_collapse() -> None:
    bonus, diagnostics = _normalized_ranker_start_bonus(
        np.asarray([-6.0, -5.0, -4.0, -3.0, -2.0], dtype=np.float64), strength=0.06
    )

    assert bonus[0] < 0.0 < bonus[-1]
    assert bonus[-1] - bonus[0] > 0.12
    assert diagnostics["median_logit"] == pytest.approx(-4.0)
    assert diagnostics["bonus_min"] >= -0.12
    assert diagnostics["bonus_max"] <= 0.12


def test_ranker_bonus_ignores_flat_logits() -> None:
    bonus, diagnostics = _normalized_ranker_start_bonus(np.full(8, -4.0), strength=0.2)

    assert np.all(bonus == 0.0)
    assert diagnostics["robust_half_span"] == 0.0


def test_occurrence_ranker_runtime_only_adds_bounded_start_bonuses_for_repeated_phone_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ranker may guide starts, but never choose an occurrence or duration."""
    project = load_project(_project(tmp_path))
    project = replace(
        project,
        inventory={**project.inventory, "u": project.inventory["k"], "v": project.inventory["k"]},
    )
    recording = replace(project.recordings[0], phones=("a", "k", "a", "u", "v"))
    states = compile_recording_states(recording, project.inventory)

    monkeypatch.setattr(
        "universal_auto_oto.acoustic_evidence._frame_analysis",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "universal_auto_oto.acoustic_evidence._multiband_pitch_tracks",
        lambda _analysis: {"unused": np.zeros(5, dtype=np.float64)},
    )

    def fake_infer(*_args: object, **_kwargs: object) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        return (
            np.asarray(
                [
                    [-4.0, -3.0, -2.0, -1.0, 0.0],
                    [0.0, -1.0, -2.0, -3.0, -4.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0],
                    [4.0, 3.0, 2.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            np.asarray([4, 0, 0, 0], dtype=np.int64),
            {"requested": True, "applied": True, "query_count": 4},
        )

    monkeypatch.setattr(
        "universal_auto_oto.occurrence_ranker_runtime.infer_occurrence_boundary_scores",
        fake_infer,
    )
    bonuses, diagnostics = _occurrence_ranker_start_scores(
        states,
        project,
        recording,
        tmp_path / "ranker.onnx",
        strength=0.20,
        provider="cpu",
    )

    first_state_by_phone = {}
    for state in states:
        first_state_by_phone.setdefault(state.phone_index, state.instance_id)
    # Boundaries 1–3 touch repeated "a"; boundary 4 (u -> v) does not.
    assert set(bonuses) == {
        first_state_by_phone[1],
        first_state_by_phone[2],
        first_state_by_phone[3],
    }
    assert diagnostics["applied_boundary_count"] == 3
    assert diagnostics["requested_provider"] == "cpu"
    assert all(values.shape == (5,) for values in bonuses.values())
    assert all(float(np.max(np.abs(values))) <= 0.40 for values in bonuses.values())


def test_vowel_only_template_uses_stable_start_for_preutterance(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace("templates: [builtin:cv]", "templates: [builtin:v]"),
        encoding="utf-8",
    )
    recordings = tmp_path / "recordings.yaml"
    recordings.write_text(
        recordings.read_text(encoding="utf-8").replace(
            "{alias: ka, template: builtin:cv, phones: [0, 1]}",
            "{alias: a, template: builtin:v, phones: [1]}",
        ),
        encoding="utf-8",
    )
    project = load_project(project_path)
    recording = project.recordings[0]
    timeline = align_recording(project, recording).timeline

    row, _trace = render_project_alias(project, recording, timeline, recording.aliases[0], source_index=0)

    stable_start = timeline.landmark_time(1, "vowel_stable_start")
    assert stable_start is not None
    assert row.absolute["pre_abs"] == stable_start
    assert row.preutterance > 0.0


def test_multiple_landmark_fallbacks_are_traced_and_routed_to_review(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    recording = project.recordings[0]
    timeline = replace(align_recording(project, recording).timeline, landmarks=())

    row, trace = render_project_alias(project, recording, timeline, recording.aliases[0], source_index=0)

    assert trace.fallback_count >= 2
    assert trace.missing_landmarks
    assert "fallback_landmark_multiple" in row.reasons
    assert row.split == "review"


def test_documented_silence_and_special_templates_load_and_render_safely(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "templates: [builtin:cv]",
            "templates: [builtin:cv, builtin:silence, builtin:special_generic]",
        ),
        encoding="utf-8",
    )
    project = load_project(project_path)
    recording = project.recordings[0]
    timeline = align_recording(project, recording).timeline
    templates = load_template_set(project)
    assert {"builtin:silence", "builtin:special_generic"} <= set(templates.templates)

    silence_plan = AliasPlan("sil", (0,), "builtin:silence", "silence")
    silence_row, _ = render_project_alias(project, recording, timeline, silence_plan, source_index=0)
    assert silence_row.cutoff < 0.0
    assert silence_row.overlap <= silence_row.preutterance <= silence_row.consonant

    special_plan = AliasPlan("special", (0,), "builtin:special_generic", "special")
    special_row, special_trace = render_project_alias(
        project,
        recording,
        timeline,
        special_plan,
        source_index=0,
    )
    assert "special_alias" in special_row.reasons
    assert special_row.split == "review"
    assert special_trace.recommended_template_id == "builtin:special_generic"


@pytest.mark.parametrize(
    ("legacy_template", "safe_alias_template", "phones", "phone_indices", "role"),
    [
        ("v", "v_plain", ("k", "a"), (1,), "v"),
        ("v", "v_blend", ("k", "a"), (1,), "v_blend"),
        ("release", "v_end", ("k", "a"), (1,), "v_end"),
        ("vv", "vv_plain", ("a", "a"), (0, 1), "vv"),
        ("vcv", "vcv_plain", ("a", "k", "a"), (0, 1, 2), "vcv"),
    ],
)
def test_safe_v_vv_vcv_template_ids_are_golden_aliases_of_legacy_timing(
    tmp_path: Path,
    legacy_template: str,
    safe_alias_template: str,
    phones: tuple[str, ...],
    phone_indices: tuple[int, ...],
    role: str,
) -> None:
    project_path = _project(tmp_path, guide=False)
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "templates: [builtin:cv]",
            f"templates: [builtin:{legacy_template}, builtin:{safe_alias_template}]",
        ),
        encoding="utf-8",
    )
    recordings_path = tmp_path / "recordings.yaml"
    recordings_path.write_text(
        recordings_path.read_text(encoding="utf-8").replace(
            "phones: [k, a]",
            f"phones: [{', '.join(phones)}]",
        ),
        encoding="utf-8",
    )
    project = load_project(project_path)
    recording = project.recordings[0]
    timeline = align_recording(project, recording).timeline
    template_set = load_template_set(project)
    legacy_row, legacy_trace = render_project_alias(
        project,
        recording,
        timeline,
        AliasPlan("golden", phone_indices, f"builtin:{legacy_template}", role),
        source_index=0,
        template_set=template_set,
    )
    alias_row, alias_trace = render_project_alias(
        project,
        recording,
        timeline,
        AliasPlan("golden", phone_indices, f"builtin:{safe_alias_template}", role),
        source_index=0,
        template_set=template_set,
    )
    assert alias_trace.raw_anchors == legacy_trace.raw_anchors
    assert alias_trace.final_anchors == legacy_trace.final_anchors
    assert alias_row.to_line() == legacy_row.to_line()


def test_acoustic_heuristic_backend_keeps_pseudo_provenance(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path, backend="acoustic_heuristic_v1", guide=False))
    result = align_recording(project, project.recordings[0])
    assert result.backend_used == "acoustic_heuristic_v1"
    assert result.timeline.provenance["label_source"] == "acoustic_heuristic_v1_pseudo"
    assert result.timeline.provenance["backend_requested"] == "acoustic_heuristic_v1"


def test_legacy_hsmm_compat_adds_occurrence_diagnostics(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path, backend="legacy_hsmm_compat", guide=False))
    result = align_recording(project, project.recordings[0])
    assert result.backend_used == "legacy_hsmm_compat"
    diagnostics = result.diagnostics["hsmm_diagnostics"]
    assert diagnostics["states"]
    assert "min_selected_vs_best_local_margin" in diagnostics
    assert result.timeline.provenance["hsmm"]["diagnostics"] == diagnostics


def test_hsmm_development_ablation_records_occurrence_and_gap_policy(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path, guide=False))
    result = align_recording(
        project,
        project.recordings[0],
        use_occurrence_priors=True,
        internal_gap_policy="contiguous",
    )
    provenance = result.timeline.provenance
    assert provenance["occurrence_priors_enabled"] is True
    assert provenance["occurrence_prior_scope"] == "all"
    assert provenance["internal_gap_policy"] == "contiguous"
    assert provenance["hsmm"]["meta"]["allow_internal_gaps"] is False
    assert provenance["hsmm"]["meta"]["internal_gap_count"] == 0
    gated = align_recording(
        project,
        project.recordings[0],
        use_occurrence_priors=True,
        occurrence_prior_scope="supported_vcv",
    )
    assert gated.timeline.provenance["occurrence_priors_requested"] is True
    assert gated.timeline.provenance["occurrence_priors_enabled"] is False
    gated_meta = gated.timeline.provenance["hsmm"]["meta"]
    assert gated_meta["max_internal_gap_ms"] > 30.0
    prior_strict = align_recording(
        project,
        project.recordings[0],
        use_occurrence_priors=True,
        internal_gap_policy="prior_strict_30ms",
    )
    assert prior_strict.timeline.provenance["occurrence_priors_enabled"] is True
    assert prior_strict.timeline.provenance["hsmm"]["meta"]["max_internal_gap_ms"] == 30.0
    with pytest.raises(AlignmentError, match="internal_gap_policy"):
        align_recording(project, project.recordings[0], internal_gap_policy="unknown")


def test_acoustic_rhythm_prior_is_gated_to_non_fortis_vcv() -> None:
    def recording(phones: tuple[str, ...], role: str, *, format_hint: str = "") -> RecordingSpec:
        return RecordingSpec(
            wav="x.wav",
            phones=phones,
            aliases=(AliasPlan("x", (0,), "builtin:v", role),),
            transcript_source="test",
            label_source="none",
            metadata={"format_hint": format_hint} if format_hint else {},
        )

    assert _supports_acoustic_rhythm_prior(recording(("k", "a", "k", "i"), "vcv"))
    assert not _supports_acoustic_rhythm_prior(recording(("bb", "a", "bb", "i"), "vcv"))
    assert not _supports_acoustic_rhythm_prior(recording(("k", "a", "k", "i"), "vc"))
    assert _supports_acoustic_rhythm_prior(
        recording(("k", "a", "k", "i"), "phone_interval_review", format_hint="vcv")
    )


def test_annotation_queue_and_pending_document_never_claim_automatic_gold(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    report = align_project(project, output=tmp_path / "aligned")
    queue_path = tmp_path / "annotation_queue.csv"
    queue = build_annotation_queue(
        project,
        tmp_path / "aligned" / "alignment_report.json",
        queue_path,
        target_count=500,
    )
    assert queue["selected_count"] == 1
    assert queue["provenance"]["automatic_gold_created"] is False
    assert "manual_pending" in queue_path.read_text(encoding="utf-8")

    timeline_path = Path(report["rows"][0]["timeline"])
    annotation_path = tmp_path / "markers" / "ka.annotation.json"
    document = initialize_annotation(project, 0, timeline_path, annotation_path)
    assert document.label_source == "manual_pending"
    assert all(marker.source == "model_suggestion_pseudo" for marker in document.markers)
    loaded = load_annotation(annotation_path, project=project)
    assert loaded.wav_sha256 == document.wav_sha256
    assert any(marker.label == "phone_boundary" for marker in loaded.markers)


def test_project_confidence_profile_adds_calibrated_risks(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    profile_path = tmp_path / "confidence.json"
    targets = {
        name: {"coefficients": [0.0, -1.0, 0.2, 0.2, 0.2, -0.5]}
        for name in ("error_gt_80ms", "error_gt_200ms", "one_slot_shift")
    }
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_kind": "landmark_risk_platt_v1",
                "feature_names": [
                    "raw_confidence",
                    "warning_count",
                    "constraint_repair_count",
                    "duration_z_abs",
                    "boundary_margin",
                ],
                "normalization": {"mean": [0, 0, 0, 0, 0], "scale": [1, 1, 1, 1, 1]},
                "targets": targets,
            }
        ),
        encoding="utf-8",
    )
    calibrated_project = replace(project, calibration=CalibrationConfig(True, "confidence.json"))
    result = align_recording(calibrated_project, calibrated_project.recordings[0])
    assert result.timeline.provenance["confidence_calibrated"] is True
    assert set(result.timeline.provenance["risk_probabilities"]) == {
        "error_gt_80ms",
        "error_gt_200ms",
        "one_slot_shift",
    }


def test_builtin_template_renders_valid_review_row_and_project_outputs(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    aligned = align_recording(project, project.recordings[0])
    row, trace = render_project_alias(
        project,
        project.recordings[0],
        aligned.timeline,
        project.recordings[0].aliases[0],
        source_index=0,
    )
    assert row.split == "review"
    assert "uncalibrated_confidence" in row.reasons
    assert row.offset <= row.overlap + row.offset <= row.preutterance + row.offset
    assert trace.template_id == "cv"
    assert trace.template_selection_source == "explicit"
    assert trace.recommended_template_id == "builtin:cv_hard"
    assert trace.alias_type["acoustic_subtype"] == "hard"
    assert "explicit_template_differs_from_classifier_recommendation" in trace.selection_warnings
    assert len(trace.alias_type_rule_set_sha256) == 64
    assert any("vowel_transition" in name for name in trace.referenced_landmarks)
    assert len(trace.template_source_sha256) == 64
    assert all(set(item) == {"field", "before_ms", "after_ms", "delta_ms", "reason", "severity"} for item in trace.repair_log)

    output = tmp_path / "generated" / "oto.ini"
    report = generate_project(project, output_oto=output)
    assert report["generated_rows"] == 1
    assert report["review_rows"] == 1
    assert report["alias_type_counts"] == {"CV:hard": 1}
    assert report["template_selection_warning_counts"] == {
        "explicit_template_differs_from_classifier_recommendation": 1
    }
    assert output.is_file()
    assert (output.parent / "oto.review.ini").is_file()
    assert (output.parent / "symbolic_alias_type_labels.jsonl").is_file()
    assert (output.parent / "symbolic_alias_type_review_queue.csv").is_file()
    assert (output.parent / "symbolic_alias_type_export_report.json").is_file()
    assert "ka.wav=ka," in output.read_text(encoding="utf-8")
    report_row = json.loads((output.parent / "row_report.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert report_row["alias_type"]["acoustic_subtype"] == "hard"
    assert len(report_row["template_source_sha256"]) == 64
    assert report_row["template_selection_source"] == "explicit"
    symbolic = report["symbolic_alias_type_export"]
    assert symbolic["automatic_row_count"] == 1
    assert symbolic["review_row_count"] == 0
    assert symbolic["automatic_gold_created"] is False
    persisted_report = json.loads((output.parent / "generation_report.json").read_text(encoding="utf-8"))
    assert persisted_report["symbolic_alias_type_export"]["classification_complete"] is True

    with pytest.raises(FileExistsError):
        generate_project(project, output_oto=output)

    generate_project(project, output_oto=output, force=True)
    assert list(output.parent.glob("symbolic_alias_type_labels.jsonl.*.bak"))
    assert list(output.parent.glob("symbolic_alias_type_review_queue.csv.*.bak"))
    assert list(output.parent.glob("symbolic_alias_type_export_report.json.*.bak"))


def test_generate_project_reuses_verified_timelines_without_rerunning_hsmm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_path = _project(tmp_path)
    aligned_project = load_project(project_path)
    timeline_dir = tmp_path / "aligned"
    align_project(aligned_project, output=timeline_dir)

    # Template changes deliberately do not invalidate an acoustic timeline.
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "templates: [builtin:cv]", "templates: [builtin:cv, builtin:cv_hard]"
        ),
        encoding="utf-8",
    )
    template_project = load_project(project_path)

    def unexpected_alignment(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("HSMM must not run when --timeline-dir is supplied")

    monkeypatch.setattr("universal_auto_oto.project_runtime.align_recording", unexpected_alignment)
    output = tmp_path / "reused" / "oto.ini"
    report = generate_project(template_project, output_oto=output, timeline_dir=timeline_dir)

    assert output.is_file()
    assert report["timeline_reuse"]["used"] is True
    assert report["provenance"]["alignment_source"] == "reused_timeline_artifacts"
    assert report["timelines"][0]["timeline_source"] == "reused"
    assert len(report["timelines"][0]["timeline_sha256"]) == 64
    row = json.loads((output.parent / "row_report.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["timeline_source"] == "reused"
    assert len(row["timeline_source_sha256"]) == 64


def test_generate_project_rejects_reused_timeline_when_trained_scorer_file_changes(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    baseline = load_project(project_path)
    artifact = tmp_path / "models" / "trained_subphone.json"
    digest = _write_trained_subphone_artifact(baseline, artifact)
    _configure_trained_subphone_project(project_path, sha256=digest)
    project = load_project(project_path)
    timeline_dir = tmp_path / "trained_aligned"
    align_project(project, output=timeline_dir)

    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(TimelineReuseError, match="does not match configured SHA-256"):
        generate_project(project, output_oto=tmp_path / "stale_scorer" / "oto.ini", timeline_dir=timeline_dir)


def test_generate_project_rejects_stale_timeline_wav_or_backend(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    project = load_project(project_path)
    timeline_dir = tmp_path / "aligned"
    align_project(project, output=timeline_dir)

    wav_path = tmp_path / "wav" / "ka.wav"
    samples, sample_rate = sf.read(wav_path, dtype="float32")
    samples[0] = 0.125
    sf.write(wav_path, samples, sample_rate)
    with pytest.raises(TimelineReuseError, match="WAV hash changed"):
        generate_project(project, output_oto=tmp_path / "stale_wav" / "oto.ini", timeline_dir=timeline_dir)

    # Create a fresh artifact, then change only the backend.
    backend_root = tmp_path / "backend_case"
    backend_root.mkdir()
    backend_project_path = _project(backend_root)
    project = load_project(backend_project_path)
    fresh_timeline_dir = backend_root / "aligned_fresh"
    align_project(project, output=fresh_timeline_dir)
    backend_project_path.write_text(
        backend_project_path.read_text(encoding="utf-8").replace("backend: hsmm", "backend: hybrid"),
        encoding="utf-8",
    )
    with pytest.raises(TimelineReuseError, match="alignment backend changed"):
        generate_project(
            load_project(backend_project_path),
            output_oto=backend_root / "stale_backend" / "oto.ini",
            timeline_dir=fresh_timeline_dir,
        )


def test_alias_type_confidence_profile_is_explicit_and_used_without_gold_at_inference(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    project_path.write_text(
        project_path.read_text(encoding="utf-8")
        .replace("templates: [builtin:cv]", "templates: [builtin:cv_hard]")
        .replace(
            "calibration: {enabled: false}",
            "calibration:\n  enabled: false\n  alias_type_enabled: true\n  alias_type_profile: alias_type_confidence.json",
        ),
        encoding="utf-8",
    )
    recordings = tmp_path / "recordings.yaml"
    recordings.write_text(
        recordings.read_text(encoding="utf-8").replace("template: builtin:cv", "template: builtin:cv_hard"),
        encoding="utf-8",
    )
    profile = {
        "schema_version": 1,
        "profile_kind": "alias_type_empirical_confidence_v1",
        "profile_id": "test-alias-type-confidence",
        "row_report_sha256": "1" * 64,
        "global_probability": 0.97,
        "training_bank_ids": ["train_bank"],
        "training_root_speaker_ids": ["train_speaker"],
        "entries": {
            "CV|hard": {
                "support": 20,
                "calibrated_probability": 0.99,
            }
        },
    }
    (tmp_path / "alias_type_confidence.json").write_text(
        json.dumps(profile),
        encoding="utf-8",
    )
    project = load_project(project_path)
    recording = project.recordings[0]
    timeline = align_recording(project, recording).timeline
    row, trace = render_project_alias(project, recording, timeline, recording.aliases[0], source_index=0)
    assert trace.alias_type_calibration["applied"] is True
    assert trace.alias_type_calibration["selection_key"] == "CV|hard"
    assert trace.alias_type_calibration["gold_labels_used_at_inference"] is False
    assert trace.alias_type_calibration["training_bank_ids"] == ["train_bank"]
    assert trace.alias_type["confidence"] == pytest.approx(0.99)
    assert "alias_type_uncalibrated_confidence" not in row.reasons
    assert "alias_type_low_calibrated_confidence" not in row.reasons
    report = generate_project(project, output_oto=tmp_path / "alias_calibrated" / "oto.ini")
    assert report["alias_type_confidence_calibration"]["enabled"] is True
    assert report["alias_type_confidence_calibration"]["training_root_speaker_ids"] == ["train_speaker"]
    report_row = json.loads(
        (tmp_path / "alias_calibrated" / "row_report.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert report_row["alias_type_calibration"]["profile_id"] == "test-alias-type-confidence"


def test_explicit_subtype_template_renders_and_matches_classifier(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace("templates: [builtin:cv]", "templates: [builtin:cv_hard]"),
        encoding="utf-8",
    )
    recordings = tmp_path / "recordings.yaml"
    recordings.write_text(
        recordings.read_text(encoding="utf-8").replace("template: builtin:cv", "template: builtin:cv_hard"),
        encoding="utf-8",
    )
    project = load_project(project_path)
    recording = project.recordings[0]
    timeline = align_recording(project, recording).timeline
    row, trace = render_project_alias(project, recording, timeline, recording.aliases[0], source_index=0)
    assert trace.template_id == "cv_hard"
    assert trace.recommended_template_id == "builtin:cv_hard"
    assert trace.selection_warnings == ()
    assert "alias_type_requires_review" not in row.reasons
    assert row.offset <= row.overlap + row.offset <= row.preutterance + row.offset <= row.consonant + row.offset


def test_explicit_mismatched_subtype_template_is_routed_to_review(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace("templates: [builtin:cv]", "templates: [builtin:cv_glide]"),
        encoding="utf-8",
    )
    recordings = tmp_path / "recordings.yaml"
    recordings.write_text(
        recordings.read_text(encoding="utf-8").replace("template: builtin:cv", "template: builtin:cv_glide"),
        encoding="utf-8",
    )
    project = load_project(project_path)
    recording = project.recordings[0]
    timeline = align_recording(project, recording).timeline
    row, trace = render_project_alias(project, recording, timeline, recording.aliases[0], source_index=0)
    assert trace.template_id == "cv_glide"
    assert trace.recommended_template_id == "builtin:cv_hard"
    assert row.split == "review"
    assert "alias_type_requires_review" in row.reasons
    assert "alias_type:explicit_template_differs_from_classifier_recommendation" in row.reasons


@pytest.mark.parametrize(
    ("template_id", "phone_order"),
    [
        ("cv_hard", "cv"),
        ("cv_softer_hard", "cv"),
        ("cv_soft", "cv"),
        ("cv_glide", "cv"),
        ("vc_hard", "vc"),
        ("vc_soft_voiced", "vc"),
        ("vc_soft_unvoiced", "vc"),
    ],
)
def test_all_phase_r3_subtype_templates_render_valid_oto(
    tmp_path: Path, template_id: str, phone_order: str
) -> None:
    project_path = _project(tmp_path, guide=False)
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "templates: [builtin:cv]", f"templates: [builtin:{template_id}]"
        ),
        encoding="utf-8",
    )
    recordings = tmp_path / "recordings.yaml"
    text = recordings.read_text(encoding="utf-8").replace("template: builtin:cv", f"template: builtin:{template_id}")
    if phone_order == "vc":
        text = text.replace("phones: [k, a]", "phones: [a, k]")
    recordings.write_text(text, encoding="utf-8")
    project = load_project(project_path)
    recording = project.recordings[0]
    timeline = align_recording(project, recording).timeline
    row, trace = render_project_alias(project, recording, timeline, recording.aliases[0], source_index=0)
    assert trace.template_id == template_id
    assert trace.soft_constraint_log
    assert all(set(item) == {"id", "expression", "observed", "min", "max", "passed", "action"} for item in trace.soft_constraint_log)
    assert row.cutoff < 0.0
    assert row.offset <= row.overlap + row.offset <= row.preutterance + row.offset
    assert row.preutterance + row.offset <= row.consonant + row.offset <= row.absolute["cutoff_abs"]


def test_soft_constraint_violation_routes_to_review_without_repairing_values(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    custom = tmp_path / "templates.yaml"
    custom.write_text(
        """schema_version: 1
templates:
  test_soft_constraint:
    role: cv
    soft_constraints:
      - {id: intentionally_too_long, expression: "duration(overlap_abs, pre_abs)", min: 500}
    anchors:
      offset_abs: phone[0].start
      overlap_abs: phone[0].start + 5
      pre_abs: phone[0].start + 30
      fixed_abs: phone[0].start + 80
      cutoff_abs: phone[1].end - 10
""",
        encoding="utf-8",
    )
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "templates: [builtin:cv]", "templates: [file:templates.yaml]"
        ),
        encoding="utf-8",
    )
    recordings = tmp_path / "recordings.yaml"
    recordings.write_text(
        recordings.read_text(encoding="utf-8").replace(
            "template: builtin:cv", "template: test_soft_constraint"
        ),
        encoding="utf-8",
    )
    project = load_project(project_path)
    recording = project.recordings[0]
    timeline = align_recording(project, recording).timeline
    row, trace = render_project_alias(project, recording, timeline, recording.aliases[0], source_index=0)
    assert row.split == "review"
    assert "soft_constraint_violation" in row.reasons
    assert "soft_constraint:intentionally_too_long" in row.reasons
    assert trace.soft_constraint_log[0]["passed"] is False
    assert trace.soft_constraint_log[0]["action"] == "review_only_no_repair"
    assert trace.calibrated_anchors == trace.final_anchors


def test_template_loader_rejects_arbitrary_function_calls(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    custom = tmp_path / "templates.yaml"
    custom.write_text(
        """schema_version: 1
templates:
  unsafe:
    role: cv
    anchors:
      offset_abs: __import__('os').system('echo unsafe')
      overlap_abs: 0
      pre_abs: 1
      fixed_abs: 2
      cutoff_abs: 3
""",
        encoding="utf-8",
    )
    project_text = project_path.read_text(encoding="utf-8").replace("templates: [builtin:cv]", "templates: [file:templates.yaml]")
    project_path.write_text(project_text, encoding="utf-8")
    project = load_project(project_path)
    with pytest.raises(TemplateError, match="허용되지 않는 함수 호출"):
        load_template_set(project)


def test_posterior_training_rejects_non_gold_annotations(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    project = load_project(project_path)
    report = align_project(project, output=tmp_path / "nongold_aligned")
    initialize_annotation(
        project,
        0,
        Path(report["rows"][0]["timeline"]),
        tmp_path / "nongold" / "ka.annotation.json",
    )
    with pytest.raises(PosteriorModelError, match="manual_gold annotation이 없습니다"):
        train_posterior_model(
            project,
            tmp_path / "nongold",
            tmp_path / "should_not_exist.json",
            minimum_train_recordings=1,
            minimum_validation_recordings=1,
        )


def test_external_ctc_logits_are_hash_checked_and_used_as_known_sequence_prior(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    frame_times = np.arange(2.5, 552.5, 5.0, dtype=np.float64)
    logits = np.full((len(frame_times), 3), -3.0, dtype=np.float32)
    logits[:, 0] = 2.0
    logits[18:48, 1] = 7.0
    logits[18:48, 0] = -2.0
    logits[52:100, 2] = 7.0
    logits[52:100, 0] = -2.0
    np.savez(
        tmp_path / "ka_ctc.npz",
        schema_version=np.asarray(1),
        wav_sha256=np.asarray(file_sha256(tmp_path / "wav" / "ka.wav")),
        frame_times_ms=frame_times,
        logits=logits,
        tokens=np.asarray(["<blank>", "k", "a"]),
        blank_token=np.asarray("<blank>"),
        metadata_json=np.asarray(json.dumps({"provider": "test_ssl_ctc", "model_sha256": "a" * 64, "encoder_kind": "ssl_ctc"})),
    )
    project_path.write_text(
        project_path.read_text(encoding="utf-8").replace(
            "use_hsmm: true",
            "use_hsmm: true\n  use_ctc_prior: true",
        ),
        encoding="utf-8",
    )
    recordings = tmp_path / "recordings.yaml"
    recordings.write_text(
        recordings.read_text(encoding="utf-8").replace(
            "    label_source: none",
            "    label_source: none\n    metadata: {ctc_prior: ka_ctc.npz, ctc_tokens: [k, a], ctc_prior_weight: 0.2}",
        ),
        encoding="utf-8",
    )
    project = load_project(project_path)
    result = align_recording(project, project.recordings[0])
    assert result.backend_used == "hsmm"
    assert result.timeline.provenance["ctc_prior"]["wav_sha256_verified"] is True
    assert result.timeline.provenance["ctc_prior"]["provider"] == "test_ssl_ctc"

    log_probabilities = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    posterior, log_likelihood = known_sequence_ctc_posteriors(log_probabilities, [1, 2], blank_id=0)
    assert posterior.shape == (len(frame_times), 2)
    assert np.isfinite(log_likelihood)
    assert int(np.argmax(posterior[:, 0])) < int(np.argmax(posterior[:, 1]))


def test_external_acoustic_evidence_feature_flag_fuses_tracks_and_records_provenance(tmp_path: Path) -> None:
    project_path = _project(tmp_path)
    baseline_project = load_project(project_path)
    baseline = align_recording(baseline_project, baseline_project.recordings[0])
    np.savez(
        tmp_path / "ka_external.npz",
        schema_version=np.asarray(1),
        wav_sha256=np.asarray(file_sha256(tmp_path / "wav" / "ka.wav")),
        frame_times_ms=np.asarray([0.0, 550.0]),
        track_names=np.asarray(["boundary", "transition"]),
        scores=np.asarray([[1.0, 1.0], [1.0, 1.0]]),
        confidence=np.asarray([1.0, 1.0]),
        metadata_json=np.asarray(
            json.dumps({"provider": "test_ssl_heads", "model_sha256": "c" * 64, "encoder_kind": "ssl_frame_head"})
        ),
    )
    recordings = tmp_path / "recordings.yaml"
    recordings.write_text(
        recordings.read_text(encoding="utf-8").replace(
            "    label_source: none",
            "    label_source: none\n    metadata: {use_external_evidence: true, external_evidence: ka_external.npz, external_evidence_weight: 0.25}",
        ),
        encoding="utf-8",
    )
    external_project = load_project(project_path)
    result = align_recording(external_project, external_project.recordings[0])

    assert result.timeline.provenance["external_evidence"]["provider"] == "test_ssl_heads"
    assert result.timeline.provenance["external_evidence"]["confidence_gated"] is True
    assert not np.allclose(result.evidence.tracks["boundary"], baseline.evidence.tracks["boundary"])
