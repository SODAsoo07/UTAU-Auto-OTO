from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import re
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import find_peaks

from .acoustic_evidence import (
    AcousticEvidence,
    FEATURE_BACKEND_ID,
    LEGACY_FEATURE_BACKEND_ID,
    MULTIBAND_V3_FEATURE_BACKEND_ID,
    PADDING_STABLE_FEATURE_BACKEND_ID,
    NEURAL_FEATURE_BACKEND_ID,
    extract_acoustic_evidence,
)
from . import analysis_cache
from .audio import analyze_wav
from .voicing import frame_voicing
from .confidence import load_confidence_profile, predict_risks, runtime_confidence_features
from .ctc_prior import CTCPrior, load_ctc_prior
from .external_evidence import ExternalEvidenceError, fuse_external_evidence, load_external_evidence
from .frame_scorer import FrameScoreProvider, FrameScoreProviderError, FrameScoreResult, HeuristicFrameScoreProvider
from .trained_frame_scorer import (
    HeuristicEmissionTransitionOverlayProvider,
    TrainedSubphoneScorerError,
    load_trained_subphone_scorer,
)
from .domain import (
    CompiledState,
    Landmark,
    LandmarkTimeline,
    PhoneSegment,
    RecordingSpec,
    StateSegment,
    landmark_timeline_from_legacy,
)
from .hsmm_decoder import (
    HSMMStateSpec,
    decode_segmental_hsmm,
    decode_segmental_hsmm_coarse_to_refine,
    decode_segmental_hsmm_exact,
)
from .project import ProjectSpec
from .posterior_model import apply_posterior_model, load_posterior_model
from .sequence_compiler import compile_recording_states, compiled_state_inference_provenance
from .sonorant_identity import DEFAULT_TEMPERATURE as SONORANT_IDENTITY_TEMPERATURE
from .sonorant_identity import SonorantTemplates, cepstra_at, in_identity_scope
from .temporal_posterior import (
    TEMPORAL_POSTERIOR_MODEL_KIND,
    TemporalPosteriorError,
    apply_temporal_posterior_model,
    load_temporal_posterior_model,
)
from .transition_duration_constraint import PhoneStateDurationConstraint


ALIGNER_VERSION = "known_sequence_hsmm_v4"
PROFILE_RESOURCES = {
    LEGACY_FEATURE_BACKEND_ID: "profiles/acoustic_hsmm_v1.yaml",
    FEATURE_BACKEND_ID: "profiles/acoustic_hsmm_v2.yaml",
    MULTIBAND_V3_FEATURE_BACKEND_ID: "profiles/acoustic_hsmm_v3.yaml",
    PADDING_STABLE_FEATURE_BACKEND_ID: "profiles/acoustic_hsmm_padding_v1.yaml",
    NEURAL_FEATURE_BACKEND_ID: "profiles/acoustic_hsmm_neural_v1.yaml",
    "world_fusion_v2": "profiles/acoustic_hsmm_v2.yaml",
}
LOCAL_MARGIN_SEARCH_WINDOW_MS = 250.0
WRONG_OCCURRENCE_MARGIN_THRESHOLD = -0.25
EXPLICIT_GAP_ROLES = frozenset({"sil", "silence", "pause", "breath", "noise"})
INTERNAL_GAP_POLICIES = frozenset(
    {"adaptive", "strict_30ms", "prior_strict_30ms", "contiguous"}
)
OCCURRENCE_PRIOR_SCOPES = frozenset({"all", "supported_vcv", "acoustic_vcv"})
OCCURRENCE_RHYTHM_TRACKING_MODES = frozenset({"uniform_cells", "adaptive_interval"})
BOUNDARY_COMPETITION_MAX_STRENGTH = 0.35
MANUAL_OCCURRENCE_ANCHOR_START_BONUS = 0.75
# A timeline that leaves a material part of the detected active recording
# unassigned is not safe to render as an automatic OTO. This is deliberately a
# review gate, not an attempt to force a path through genuine tail silence.
MIN_LEXICAL_COVERAGE_RATIO = 0.85
# Leaving trailing frames free makes a long concatenated recording collapse
# onto its first few high-confidence syllables.  This is deliberately smaller
# than an explicit-gap penalty so genuine tail silence remains available, but
# it prevents multi-second active speech from being silently discarded.
TRAILING_GAP_PENALTY_PER_MS = 0.003
# cvvc_slot_rhythm_v2 (development only): nasal codas are scored with
# obstruent-tuned consonant tracks, so a sung murmur is absorbed by the vowel.
CVVC_NASAL_CODAS = frozenset({"n", "m", "ng"})
CVVC_CODA_STEP_MIN_STRENGTH = 0.15
CVVC_CODA_MIN_VOWEL_AFTER_TRANSITION_MS = 40.0
CVVC_CODA_MIN_CONSONANT_MS = 30.0
# Stops and affricates whose closure follows the preceding phone (v7).
CVVC_STOP_ONSETS = frozenset({"b", "bb", "p", "pp", "d", "dd", "t", "tt", "g", "gg", "k", "kk",
                              "j", "jj", "ch", "c", "q", "ts"})
STOP_CLOSURE_DROP_DB = 15.0
# Vowel length before a coda as a share of the syllable period: median of 127
# acoustically placed nasal codas in 42 development-bank files (IQR 0.19-0.36). v9 only.
CVVC_VOWEL_PERIOD_RATIO = 0.255
# Pause-aware gaps (development): skipping a frame costs the speech penalty
# times its activity, with silence at this floor, so pauses between
# syllables and tail silence are skipped instead of stretching phones.
PAUSE_GAP_SPEECH_PENALTY_PER_MS = 0.01
PAUSE_GAP_SILENCE_WEIGHT = 0.02
PAUSE_GAP_MAX_MS = 600.0
# Phones that end voicing: breath, vocal fry (low F0 counts as unvoiced here) and pauses (v8).
CVVC_UNVOICED_ENDINGS = frozenset({"exh", "vf", "sil", "pau"})


class AlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class AlignmentProfile:
    schema_version: int
    profile_id: str
    feature_backend: str
    role_weights: Mapping[str, Mapping[str, float]]
    sha256: str


@dataclass(frozen=True)
class AlignmentResult:
    timeline: LandmarkTimeline
    evidence: AcousticEvidence
    backend_requested: str
    backend_used: str
    diagnostics: Mapping[str, object]


def align_recording(
    project: ProjectSpec,
    recording: RecordingSpec,
    *,
    backend: str | None = None,
    feature_backend_override: str | None = None,
    use_occurrence_priors: bool = False,
    occurrence_prior_scope: str = "all",
    occurrence_prior_min_coverage: float = 0.60,
    occurrence_prior_weight_scale: float = 1.0,
    occurrence_rhythm_tracking: str = "uniform_cells",
    internal_gap_policy: str = "adaptive",
    sustain_isolated_sonorants: bool = False,
    cvvc_slot_rhythm: bool = False,
    cvvc_nasal_coda_boundary: bool = False,
    cvvc_slot_grid_refit: bool = False,
    korean_nasal_coda_onset: bool = False,
    korean_lateral_coda_rhythm: bool = False,
    korean_nasal_coda_rhythm_fallback: bool = False,
    korean_stop_closure_onset: bool = False,
    korean_ending_voicing_offset: bool = False,
    korean_coda_period_ratio: bool = False,
    pause_aware_gaps: bool = False,
    sonorant_identity: SonorantTemplates | None = None,
    sonorant_identity_weight: float = 0.0,
    syllabic_sonorant_holds: bool = False,
    explicit_pause_tokens: bool = False,
    exhaustive_coarse_search: bool = False,
    exact_hsmm_search: bool = False,
    dense_speech_activity_floor: bool = False,
    syllable_period_ms: float | None = None,
    boundary_competition_strength: float = 0.0,
    occurrence_ranker_model: Path | None = None,
    occurrence_ranker_strength: float = 0.0,
    occurrence_ranker_provider: str | None = None,
    manual_occurrence_start_anchors: Mapping[str, float] | None = None,
    manual_occurrence_anchor_provenance: Mapping[str, object] | None = None,
    frame_score_provider: FrameScoreProvider | None = None,
) -> AlignmentResult:
    requested = (backend or project.alignment.backend).strip().casefold()
    if requested not in {"acoustic_heuristic_v1", "hsmm", "hybrid", "posterior_hsmm", "legacy_hsmm_compat"}:
        raise ValueError(f"지원하지 않는 alignment backend입니다: {requested}")
    wav_path = project.wav_path(recording)
    evidence = _cached_evidence(
        wav_path,
        internal_sample_rate=project.audio.internal_sample_rate,
        convert_to_mono=project.audio.convert_to_mono,
        resampler=project.audio.resampler,
        feature_backend=feature_backend_override or project.alignment.feature_backend,
        expected_activity_regions=len(recording.phones),
        evidence_model=(project.root / project.alignment.evidence_model).resolve() if project.alignment.evidence_model else None,
        evidence_provider=project.alignment.evidence_provider,
        preserve_leading_silence=(
            project.inventory[recording.phones[0]].states[0].acoustic_role in EXPLICIT_GAP_ROLES
            if recording.phones and recording.phones[0] in project.inventory else False
        ),
        dense_speech_activity_floor=dense_speech_activity_floor,
    )
    if requested == "acoustic_heuristic_v1":
        return _legacy_alignment(project, recording, evidence, requested=requested)
    try:
        return _hsmm_alignment(
            project,
            recording,
            evidence,
            requested=requested,
            use_occurrence_priors=use_occurrence_priors,
            occurrence_prior_scope=occurrence_prior_scope,
            occurrence_prior_min_coverage=occurrence_prior_min_coverage,
            occurrence_prior_weight_scale=occurrence_prior_weight_scale,
            occurrence_rhythm_tracking=occurrence_rhythm_tracking,
            internal_gap_policy=internal_gap_policy,
            sustain_isolated_sonorants=sustain_isolated_sonorants,
            cvvc_slot_rhythm=cvvc_slot_rhythm,
            cvvc_nasal_coda_boundary=cvvc_nasal_coda_boundary,
            cvvc_slot_grid_refit=cvvc_slot_grid_refit,
            korean_nasal_coda_onset=korean_nasal_coda_onset,
            korean_lateral_coda_rhythm=korean_lateral_coda_rhythm,
            korean_nasal_coda_rhythm_fallback=korean_nasal_coda_rhythm_fallback,
            korean_stop_closure_onset=korean_stop_closure_onset,
            korean_ending_voicing_offset=korean_ending_voicing_offset,
            korean_coda_period_ratio=korean_coda_period_ratio,
            pause_aware_gaps=pause_aware_gaps,
            sonorant_identity=sonorant_identity,
            sonorant_identity_weight=sonorant_identity_weight,
            syllabic_sonorant_holds=syllabic_sonorant_holds,
            explicit_pause_tokens=explicit_pause_tokens,
            exhaustive_coarse_search=exhaustive_coarse_search,
            exact_hsmm_search=exact_hsmm_search,
            syllable_period_ms=syllable_period_ms,
            boundary_competition_strength=boundary_competition_strength,
            occurrence_ranker_model=occurrence_ranker_model,
            occurrence_ranker_strength=occurrence_ranker_strength,
            occurrence_ranker_provider=occurrence_ranker_provider,
            manual_occurrence_start_anchors=manual_occurrence_start_anchors,
            manual_occurrence_anchor_provenance=manual_occurrence_anchor_provenance,
            frame_score_provider=frame_score_provider,
        )
    except AlignmentError as error:
        if requested != "hybrid":
            raise
        fallback = _legacy_alignment(project, recording, evidence, requested=requested)
        timeline = fallback.timeline
        fallback_timeline = LandmarkTimeline(
            **{
                **timeline.__dict__,
                "warnings": tuple(dict.fromkeys((*timeline.warnings, "hsmm_failed_heuristic_fallback"))),
                "provenance": {
                    **dict(timeline.provenance),
                    "backend_requested": requested,
                    "backend_used": "acoustic_heuristic_v1",
                    "hsmm_failure": str(error),
                },
            }
        )
        return AlignmentResult(
            timeline=fallback_timeline,
            evidence=evidence,
            backend_requested=requested,
            backend_used="acoustic_heuristic_v1",
            diagnostics={"fallback_reason": str(error)},
        )


def align_recording_with_posterior_model(
    project: ProjectSpec,
    recording: RecordingSpec,
    posterior_model: Path,
    *,
    use_occurrence_priors: bool = False,
    occurrence_prior_scope: str = "all",
    occurrence_prior_min_coverage: float = 0.60,
    occurrence_prior_weight_scale: float = 1.0,
    occurrence_rhythm_tracking: str = "uniform_cells",
    internal_gap_policy: str = "adaptive",
    boundary_competition_strength: float = 0.0,
    occurrence_ranker_model: Path | None = None,
    occurrence_ranker_strength: float = 0.0,
    occurrence_ranker_provider: str | None = None,
    state_duration_constraints: Mapping[tuple[str, str], PhoneStateDurationConstraint] | None = None,
    state_duration_constraint_provenance: Mapping[str, object] | None = None,
    manual_occurrence_start_anchors: Mapping[str, float] | None = None,
    manual_occurrence_anchor_provenance: Mapping[str, object] | None = None,
    frame_score_provider: FrameScoreProvider | None = None,
) -> AlignmentResult:
    """Evaluate a manifest-level posterior without copying it into every project.

    Normal project runtime still requires a project-local model path.  This
    explicit entry point is for speaker-locked multi-project Gold evaluation.
    """

    model_path = posterior_model.resolve()
    evidence = _cached_evidence(
        project.wav_path(recording),
        internal_sample_rate=project.audio.internal_sample_rate,
        convert_to_mono=project.audio.convert_to_mono,
        resampler=project.audio.resampler,
        feature_backend=project.alignment.feature_backend,
        expected_activity_regions=len(recording.phones),
        evidence_model=(project.root / project.alignment.evidence_model).resolve() if project.alignment.evidence_model else None,
        evidence_provider=project.alignment.evidence_provider,
    )
    evidence = _apply_posterior_artifact(project, evidence, model_path)
    aligned = _hsmm_alignment(
        project,
        recording,
        evidence,
        requested="hsmm",
        use_occurrence_priors=use_occurrence_priors,
        occurrence_prior_scope=occurrence_prior_scope,
        occurrence_prior_min_coverage=occurrence_prior_min_coverage,
        occurrence_prior_weight_scale=occurrence_prior_weight_scale,
        occurrence_rhythm_tracking=occurrence_rhythm_tracking,
        frame_score_provider=frame_score_provider,
        internal_gap_policy=internal_gap_policy,
        boundary_competition_strength=boundary_competition_strength,
        occurrence_ranker_model=occurrence_ranker_model,
        occurrence_ranker_strength=occurrence_ranker_strength,
        occurrence_ranker_provider=occurrence_ranker_provider,
        state_duration_constraints=state_duration_constraints,
        state_duration_constraint_provenance=state_duration_constraint_provenance,
        manual_occurrence_start_anchors=manual_occurrence_start_anchors,
        manual_occurrence_anchor_provenance=manual_occurrence_anchor_provenance,
    )
    model_sha256 = str(evidence.metadata["posterior_model"]["model_sha256"])
    posterior_kind = str(evidence.metadata["posterior_model"].get("model_kind", ""))
    posterior_label_source = (
        "manual_gold_trained_temporal_cnn_hsmm_v1_pseudo"
        if posterior_kind == TEMPORAL_POSTERIOR_MODEL_KIND
        else "manual_gold_trained_posterior_hsmm_v2_pseudo"
    )
    timeline = LandmarkTimeline(
        **{
            **aligned.timeline.__dict__,
            "model_sha256": model_sha256,
            "provenance": {
                **dict(aligned.timeline.provenance),
                "label_source": posterior_label_source,
                "backend_requested": "posterior_hsmm",
                "backend_used": "posterior_hsmm",
                "posterior_model": dict(evidence.metadata["posterior_model"]),
                "manifest_evaluation_override": True,
                "occurrence_priors_requested": bool(use_occurrence_priors),
            },
        }
    )
    return AlignmentResult(
        timeline=timeline,
        evidence=evidence,
        backend_requested="posterior_hsmm",
        backend_used="posterior_hsmm",
        diagnostics={
            **dict(aligned.diagnostics),
            "model_sha256": model_sha256,
            "manifest_evaluation_override": True,
        },
    )


def _apply_posterior_artifact(
    project: ProjectSpec,
    evidence: AcousticEvidence,
    model_path: Path,
) -> AcousticEvidence:
    """Load either posterior generation while preserving one HSMM contract."""

    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlignmentError(f"posterior model을 읽을 수 없습니다: {model_path}") from error
    model_kind = payload.get("model_kind") if isinstance(payload, Mapping) else None
    if model_kind == TEMPORAL_POSTERIOR_MODEL_KIND:
        try:
            temporal = load_temporal_posterior_model(model_path)
            if temporal.feature_backend != project.alignment.feature_backend:
                raise AlignmentError("temporal posterior feature backend가 project와 다릅니다")
            allowlist = temporal.training_provenance.get("inventory_sha256_allowlist", ())
            if project.inventory_sha256 not in allowlist:
                raise AlignmentError("temporal posterior inventory hash가 현재 project와 다릅니다")
            return apply_temporal_posterior_model(evidence, temporal, model_path=model_path)
        except TemporalPosteriorError as error:
            raise AlignmentError(f"temporal posterior model rejected: {error}") from error

    model = load_posterior_model(
        model_path,
        expected_feature_backend=project.alignment.feature_backend,
    )
    training = model.get("training")
    inventory_allowlist = (
        training.get("inventory_sha256_allowlist", ()) if isinstance(training, Mapping) else ()
    )
    if (
        not isinstance(training, Mapping)
        or (
            training.get("inventory_sha256") != project.inventory_sha256
            and project.inventory_sha256 not in inventory_allowlist
        )
    ):
        raise AlignmentError("posterior model inventory hash가 현재 project와 다릅니다")
    return apply_posterior_model(evidence, model, model_path=model_path)


@lru_cache(maxsize=8)  # parsed once per process instead of once per recording
def load_alignment_profile(feature_backend: str = FEATURE_BACKEND_ID) -> AlignmentProfile:
    resource_name = PROFILE_RESOURCES.get(feature_backend)
    if resource_name is None:
        raise AlignmentError(f"지원하지 않는 alignment feature backend입니다: {feature_backend}")
    resource = files("universal_auto_oto").joinpath(resource_name)
    text = resource.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping) or raw.get("schema_version") not in {1, 2}:
        raise AlignmentError("alignment profile schema_version이 유효하지 않습니다")
    profile_id = raw.get("profile_id")
    compatible_backends = raw.get("feature_backends", [raw.get("feature_backend")])
    roles = raw.get("roles")
    if not isinstance(profile_id, str) or not profile_id:
        raise AlignmentError("alignment profile_id가 없습니다")
    if not isinstance(compatible_backends, Sequence) or isinstance(compatible_backends, (str, bytes)):
        raise AlignmentError("alignment profile feature_backends가 유효하지 않습니다")
    if feature_backend not in compatible_backends:
        raise AlignmentError(f"alignment profile feature backend 불일치: {feature_backend}")
    if not isinstance(roles, Mapping) or "other" not in roles:
        raise AlignmentError("alignment profile roles.other가 없습니다")
    parsed: dict[str, dict[str, float]] = {}
    for role, definition in roles.items():
        if not isinstance(role, str) or not isinstance(definition, Mapping):
            raise AlignmentError("alignment role 정의가 유효하지 않습니다")
        weights = definition.get("tracks")
        if not isinstance(weights, Mapping) or not weights:
            raise AlignmentError(f"alignment role track weight가 없습니다: {role}")
        parsed_weights: dict[str, float] = {}
        for track, raw_weight in weights.items():
            if not isinstance(track, str) or isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise AlignmentError(f"alignment role weight가 유효하지 않습니다: {role}.{track}")
            weight = float(raw_weight)
            if not np.isfinite(weight) or weight < 0.0:
                raise AlignmentError(f"alignment role weight는 0 이상의 finite 값이어야 합니다: {role}.{track}")
            parsed_weights[track] = weight
        total = sum(parsed_weights.values())
        if total <= 0.0:
            raise AlignmentError(f"alignment role weight 합이 0입니다: {role}")
        parsed[role] = {track: value / total for track, value in parsed_weights.items()}
    return AlignmentProfile(
        schema_version=int(raw["schema_version"]),
        profile_id=profile_id,
        feature_backend=feature_backend,
        role_weights=parsed,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _legacy_alignment(
    project: ProjectSpec,
    recording: RecordingSpec,
    evidence: AcousticEvidence,
    *,
    requested: str,
) -> AlignmentResult:
    legacy = analyze_wav(project.wav_path(recording), len(recording.phones), decoder_mode="monotonic")
    timeline = landmark_timeline_from_legacy(
        legacy,
        phone_ids=recording.phones,
        recording_sha256=_recording_sha256(recording),
        inventory_sha256=project.inventory_sha256,
    )
    timeline = LandmarkTimeline(
        **{
            **timeline.__dict__,
            "provenance": {
                **dict(timeline.provenance),
                "transcript_source": recording.transcript_source,
                "source_label_kind": recording.label_source,
                "backend_requested": requested,
                "backend_used": "acoustic_heuristic_v1",
                "confidence_calibrated": False,
                "timebase": evidence.metadata["timebase"],
            },
        }
    )
    timeline = _calibrate_timeline(project, timeline)
    return AlignmentResult(
        timeline=timeline,
        evidence=evidence,
        backend_requested=requested,
        backend_used="acoustic_heuristic_v1",
        diagnostics={"legacy_decoder_version": legacy.decoder_version},
    )


def _hsmm_alignment(
    project: ProjectSpec,
    recording: RecordingSpec,
    evidence: AcousticEvidence,
    *,
    requested: str,
    use_occurrence_priors: bool = False,
    occurrence_prior_scope: str = "all",
    occurrence_prior_min_coverage: float = 0.60,
    occurrence_prior_weight_scale: float = 1.0,
    occurrence_rhythm_tracking: str = "uniform_cells",
    internal_gap_policy: str = "adaptive",
    sustain_isolated_sonorants: bool = False,
    cvvc_slot_rhythm: bool = False,
    cvvc_nasal_coda_boundary: bool = False,
    cvvc_slot_grid_refit: bool = False,
    korean_nasal_coda_onset: bool = False,
    korean_lateral_coda_rhythm: bool = False,
    korean_nasal_coda_rhythm_fallback: bool = False,
    korean_stop_closure_onset: bool = False,
    korean_ending_voicing_offset: bool = False,
    korean_coda_period_ratio: bool = False,
    pause_aware_gaps: bool = False,
    sonorant_identity: SonorantTemplates | None = None,
    sonorant_identity_weight: float = 0.0,
    syllabic_sonorant_holds: bool = False,
    explicit_pause_tokens: bool = False,
    exhaustive_coarse_search: bool = False,
    exact_hsmm_search: bool = False,
    syllable_period_ms: float | None = None,
    boundary_competition_strength: float = 0.0,
    occurrence_ranker_model: Path | None = None,
    occurrence_ranker_strength: float = 0.0,
    occurrence_ranker_provider: str | None = None,
    state_duration_constraints: Mapping[tuple[str, str], PhoneStateDurationConstraint] | None = None,
    state_duration_constraint_provenance: Mapping[str, object] | None = None,
    manual_occurrence_start_anchors: Mapping[str, float] | None = None,
    manual_occurrence_anchor_provenance: Mapping[str, object] | None = None,
    frame_score_provider: FrameScoreProvider | None = None,
) -> AlignmentResult:
    if syllable_period_ms is not None:
        syllable_period_ms = float(syllable_period_ms)
        if not 150.0 <= syllable_period_ms <= 2000.0:
            raise AlignmentError("syllable_period_ms must be between 150 and 2000")
        use_occurrence_priors = True
    internal_gap_policy = str(internal_gap_policy).strip().casefold()
    if internal_gap_policy not in INTERNAL_GAP_POLICIES:
        raise AlignmentError(
            f"internal_gap_policy must be one of {', '.join(sorted(INTERNAL_GAP_POLICIES))}"
        )
    occurrence_prior_scope = str(occurrence_prior_scope).strip().casefold()
    if occurrence_prior_scope not in OCCURRENCE_PRIOR_SCOPES:
        raise AlignmentError(
            f"occurrence_prior_scope must be one of {', '.join(sorted(OCCURRENCE_PRIOR_SCOPES))}"
        )
    occurrence_prior_min_coverage = float(occurrence_prior_min_coverage)
    if not 0.30 <= occurrence_prior_min_coverage <= 1.0:
        raise AlignmentError("occurrence_prior_min_coverage must be between 0.30 and 1.0")
    occurrence_prior_weight_scale = float(occurrence_prior_weight_scale)
    if not 0.25 <= occurrence_prior_weight_scale <= 2.0:
        raise AlignmentError("occurrence_prior_weight_scale must be between 0.25 and 2.0")
    occurrence_rhythm_tracking = str(occurrence_rhythm_tracking).strip().casefold()
    if occurrence_rhythm_tracking not in OCCURRENCE_RHYTHM_TRACKING_MODES:
        raise AlignmentError(
            "occurrence_rhythm_tracking must be one of "
            + ", ".join(sorted(OCCURRENCE_RHYTHM_TRACKING_MODES))
        )
    boundary_competition_strength = float(boundary_competition_strength)
    if not 0.0 <= boundary_competition_strength <= BOUNDARY_COMPETITION_MAX_STRENGTH:
        raise AlignmentError(
            "boundary_competition_strength must be between 0 and "
            f"{BOUNDARY_COMPETITION_MAX_STRENGTH:g}"
        )
    if boundary_competition_strength > 0.0 and evidence.metadata.get("feature_backend") != NEURAL_FEATURE_BACKEND_ID:
        raise AlignmentError("boundary competition requires onnx_neural_evidence_v1")
    profile = load_alignment_profile(str(evidence.metadata["feature_backend"]))
    model_sha256: str | None = None
    compatibility_mode = requested == "legacy_hsmm_compat"
    occurrence_priors_requested = compatibility_mode or bool(use_occurrence_priors)
    occurrence_priors_eligible = bool(
        occurrence_priors_requested
        and (
            compatibility_mode
            or occurrence_prior_scope == "all"
            or _supports_acoustic_rhythm_prior(recording)
            or (
                occurrence_prior_scope == "acoustic_vcv"
                and _is_vcv_recording(recording)
            )
        )
    )
    backend_used = "legacy_hsmm_compat" if compatibility_mode else "hsmm"
    label_source = "acoustic_heuristic_hsmm_v2_pseudo"
    if requested == "posterior_hsmm":
        if project.alignment.model is None:
            raise AlignmentError("posterior_hsmm backend에는 alignment.model이 필요합니다")
        model_path = (project.root / project.alignment.model).resolve()
        try:
            model_path.relative_to(project.root.resolve())
        except ValueError as error:
            raise AlignmentError("posterior model은 project 폴더 내부 경로여야 합니다") from error
        evidence = _apply_posterior_artifact(project, evidence, model_path)
        model_sha256 = str(evidence.metadata["posterior_model"]["model_sha256"])
        backend_used = "posterior_hsmm"
        label_source = (
            "manual_gold_trained_temporal_cnn_hsmm_v1_pseudo"
            if evidence.metadata["posterior_model"].get("model_kind") == TEMPORAL_POSTERIOR_MODEL_KIND
            else "manual_gold_trained_posterior_hsmm_v2_pseudo"
        )
    use_external_evidence = recording.metadata.get("use_external_evidence", False)
    if not isinstance(use_external_evidence, bool):
        raise AlignmentError(f"{recording.wav}: metadata.use_external_evidence must be boolean")
    if use_external_evidence:
        raw_external_path = recording.metadata.get("external_evidence")
        if not isinstance(raw_external_path, str) or not raw_external_path.strip():
            raise AlignmentError(f"{recording.wav}: metadata.external_evidence path is required")
        external_path = (project.root / raw_external_path).resolve()
        try:
            external_path.relative_to(project.root.resolve())
        except ValueError as error:
            raise AlignmentError("external evidence must be inside the project directory") from error
        raw_external_weight = recording.metadata.get("external_evidence_weight", 0.25)
        if isinstance(raw_external_weight, bool) or not isinstance(raw_external_weight, (int, float)):
            raise AlignmentError(f"{recording.wav}: external_evidence_weight must be numeric")
        try:
            external = load_external_evidence(external_path, expected_wav_sha256=evidence.wav_sha256)
            evidence = fuse_external_evidence(evidence, external, maximum_weight=float(raw_external_weight))
        except ExternalEvidenceError as error:
            raise AlignmentError(f"{recording.wav}: external evidence rejected: {error}") from error
    ctc_prior: CTCPrior | None = None
    ctc_prior_weight = 0.0
    if project.alignment.use_ctc_prior:
        raw_prior_path = recording.metadata.get("ctc_prior")
        if not isinstance(raw_prior_path, str) or not raw_prior_path.strip():
            raise AlignmentError(f"{recording.wav}: metadata.ctc_prior 경로가 필요합니다")
        prior_path = (project.root / raw_prior_path).resolve()
        try:
            prior_path.relative_to(project.root.resolve())
        except ValueError as error:
            raise AlignmentError("CTC prior는 project 폴더 내부 경로여야 합니다") from error
        raw_tokens = recording.metadata.get("ctc_tokens", recording.phones)
        if isinstance(raw_tokens, (str, bytes)) or not isinstance(raw_tokens, Sequence):
            raise AlignmentError(f"{recording.wav}: metadata.ctc_tokens는 token 목록이어야 합니다")
        tokens = tuple(str(item) for item in raw_tokens)
        if len(tokens) != len(recording.phones):
            raise AlignmentError(f"{recording.wav}: ctc_tokens 수가 phone 수와 다릅니다")
        raw_weight = recording.metadata.get("ctc_prior_weight", 0.20)
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise AlignmentError(f"{recording.wav}: ctc_prior_weight는 숫자여야 합니다")
        ctc_prior_weight = float(raw_weight)
        if not 0.0 < ctc_prior_weight <= 0.50:
            raise AlignmentError(f"{recording.wav}: ctc_prior_weight는 0보다 크고 0.5 이하여야 합니다")
        ctc_prior = load_ctc_prior(
            prior_path,
            expected_wav_sha256=evidence.wav_sha256,
            target_tokens=tokens,
            target_frame_times_ms=evidence.frame_times_ms,
        )
    compiled = compile_recording_states(recording, project.inventory)
    compiled_state_inferences = compiled_state_inference_provenance(recording, project.inventory)
    sonorant_policy = None
    if (sustain_isolated_sonorants and requested in {"hsmm", "legacy_hsmm_compat"}
            and frame_score_provider is None and project.alignment.scorer_backend == "heuristic_fallback"
            and not use_external_evidence and ctc_prior is None
            and not state_duration_constraints and not manual_occurrence_start_anchors
            and not boundary_competition_strength and not occurrence_ranker_strength):
        compiled, sonorant_policy = _isolated_sonorant_states(project, recording, evidence, compiled)
    if sonorant_policy is not None:
        internal_gap_policy = "contiguous"
    held_policy = None
    if syllabic_sonorant_holds and sonorant_policy is None:
        compiled, held_policy = _syllabic_sonorant_holds(project, recording, compiled)
    score_compilation = _compile_frame_score_compilation(
        compiled,
        project,
        evidence,
        profile,
        ctc_prior=ctc_prior,
        ctc_prior_weight=ctc_prior_weight,
        recording=recording,
        use_occurrence_priors=occurrence_priors_eligible,
        require_acoustic_rhythm=(
            not compatibility_mode
            and occurrence_prior_scope in {"supported_vcv", "acoustic_vcv"}
        ),
        allow_fortis_acoustic_rhythm=(occurrence_prior_scope == "acoustic_vcv"),
        occurrence_prior_min_coverage=occurrence_prior_min_coverage,
        occurrence_prior_weight_scale=occurrence_prior_weight_scale,
        occurrence_rhythm_tracking=occurrence_rhythm_tracking,
        frame_score_provider=frame_score_provider,
        cvvc_slot_rhythm=bool(cvvc_slot_rhythm and requested in {"hsmm", "legacy_hsmm_compat"}
            and frame_score_provider is None and project.alignment.scorer_backend == "heuristic_fallback"
            and not use_external_evidence and ctc_prior is None and not state_duration_constraints
            and not manual_occurrence_start_anchors and not boundary_competition_strength
            and not occurrence_ranker_strength),
        cvvc_slot_grid_refit=cvvc_slot_grid_refit,
        syllable_period_ms=syllable_period_ms,
    )
    frame_scores = dict(score_compilation.result.emission_scores)
    identity_policy = None
    if sonorant_identity is not None and sonorant_identity_weight > 0.0:
        frame_scores, identity_policy = _apply_sonorant_identity(
            project, recording, evidence, compiled, frame_scores, sonorant_identity, sonorant_identity_weight)
    score_warnings = score_compilation.result.warnings
    occurrence_priors_enabled = score_compilation.occurrence_priors_applied
    occurrence_prior_diagnostics = score_compilation.occurrence_diagnostics
    slot_policy = (occurrence_prior_diagnostics if occurrence_prior_diagnostics.get("reason") == "cvvc_slot_rhythm_applied" else None)
    if slot_policy is not None:
        internal_gap_policy = "contiguous"
    provider_transition_start_scores = {
        state_id: np.asarray(values, dtype=np.float64)
        for state_id, values in score_compilation.result.transition_start_scores.items()
    }
    neural_evidence_enabled = evidence.metadata.get("feature_backend") == NEURAL_FEATURE_BACKEND_ID
    frame_scorer_provenance = {
        **dict(score_compilation.result.provenance),
        "emission_state_count": len(frame_scores),
        "transition_start_score_state_count": len(provider_transition_start_scores),
        "diagnostics": dict(score_compilation.result.diagnostics),
        "neural_evidence_shadow": {
            "enabled": neural_evidence_enabled,
            "model_sha256": evidence.metadata.get("neural_model_sha256"),
            "manifest_sha256": evidence.metadata.get("neural_manifest_sha256"),
            "execution_provider": evidence.metadata.get("neural_execution_provider"),
            "transition_scorer": "disabled_no_reviewed_subphone_transition_labels",
            "fallback_reason": (
                None if neural_evidence_enabled else "feature_backend_not_onnx_neural_evidence_v1"
            ),
        },
    }
    duration_constraints = state_duration_constraints or {}
    specs = tuple(_state_spec(state, duration_constraints) for state in compiled)
    if sonorant_policy is not None:
        specs = tuple(replace(spec, duration_sigma_ms=sonorant_policy["duration_sigma_ms"])
            if spec.state_id == sonorant_policy["state_instance_id"] else spec for spec in specs)
    manual_anchor_scores = _manual_occurrence_anchor_start_scores(
        compiled,
        evidence.frame_times_ms,
        manual_occurrence_start_anchors,
    )
    boundary_start_scores, boundary_competition_diagnostics = _repeated_phone_boundary_competition_scores(
        compiled,
        recording,
        evidence,
        strength=boundary_competition_strength,
    )
    for state_id, values in provider_transition_start_scores.items():
        prior = boundary_start_scores.get(state_id)
        boundary_start_scores[state_id] = values if prior is None else prior + values
    ranker_scores, ranker_diagnostics = _occurrence_ranker_start_scores(
        compiled,
        project,
        recording,
        occurrence_ranker_model,
        occurrence_ranker_strength,
        occurrence_ranker_provider,
    )
    for state_id, values in ranker_scores.items():
        prior = boundary_start_scores.get(state_id)
        boundary_start_scores[state_id] = values if prior is None else prior + values
    for state_id, values in manual_anchor_scores.items():
        prior = boundary_start_scores.get(state_id)
        boundary_start_scores[state_id] = values if prior is None else prior + values
    applied_duration_constraint_states = [
        state.instance_id for state in compiled if (state.phone_id, state.state_id) in duration_constraints
    ]
    explicit_gap_state_count = sum(state.acoustic_role in EXPLICIT_GAP_ROLES for state in compiled)
    raw_explicit_gap_topology = recording.metadata.get("explicit_gap_topology", False)
    if not isinstance(raw_explicit_gap_topology, bool):
        raise AlignmentError(f"{recording.wav}: metadata.explicit_gap_topology must be boolean")
    explicit_gap_topology = bool(raw_explicit_gap_topology or explicit_gap_state_count)
    if raw_explicit_gap_topology and not explicit_gap_state_count:
        raise AlignmentError(
            f"{recording.wav}: explicit_gap_topology requires at least one silence/pause/breath/noise state"
        )
    if internal_gap_policy == "contiguous":
        allow_internal_gaps = False
        max_internal_gap_ms = 0.0
        internal_gap_penalty_per_ms = 0.0
    elif internal_gap_policy == "strict_30ms" or (
        internal_gap_policy == "prior_strict_30ms" and occurrence_priors_enabled
    ):
        allow_internal_gaps = True
        max_internal_gap_ms = 30.0
        internal_gap_penalty_per_ms = 0.01
    else:
        allow_internal_gaps = True
        max_internal_gap_ms = (
            30.0
            if explicit_gap_topology
            else min(320.0, max(100.0, float(evidence.metadata.get("alignment_duration_ms", evidence.duration_ms)) / max(4.0, len(recording.phones))))
        )
        internal_gap_penalty_per_ms = 0.01 if explicit_gap_topology else 0.0002
    allow_leading_gap = compiled[0].acoustic_role not in {"silence", "sil"}
    analysis_origin_ms = float(evidence.metadata.get("analysis_origin_ms", 0.0))
    max_leading_gap_ms = min(900.0, max(120.0, evidence.active_start_ms - analysis_origin_ms + 120.0)) if allow_leading_gap else 0.0
    trailing_gap_penalty_per_ms = (
        0.01 if explicit_gap_topology else TRAILING_GAP_PENALTY_PER_MS
    )
    timeout_ms = max(5000, min(30000, 1000 + len(compiled) * 350))
    decode_options = {}
    if sonorant_policy is not None or slot_policy is not None:
        timeout_ms = 30000
        decode_options["trailing_gap_frame_weights"] = np.clip(
            evidence.tracks["active"] * (1.0 - evidence.tracks["low_energy"]), 0.0, 1.0)
    internal_gap_weights = None
    if pause_aware_gaps:
        activity = np.clip(np.asarray(evidence.tracks["active"], dtype=np.float64)
                           * (1.0 - np.asarray(evidence.tracks["low_energy"], dtype=np.float64)), 0.0, 1.0)
        timeout_ms = 30000
        decode_options["trailing_gap_frame_weights"] = activity
        if allow_internal_gaps:
            internal_gap_weights = np.maximum(activity, PAUSE_GAP_SILENCE_WEIGHT)
            internal_gap_penalty_per_ms = PAUSE_GAP_SPEECH_PENALTY_PER_MS
            max_internal_gap_ms = max(max_internal_gap_ms, PAUSE_GAP_MAX_MS)
    pause_positions: list[int] = []
    pause_policy = None
    if explicit_pause_tokens:
        specs, frame_scores, pause_positions, pause_policy, pause_start_scores = _insert_pause_token_states(
            recording, compiled, specs, frame_scores, evidence)
        boundary_start_scores.update(pause_start_scores)
    long_input = len(compiled) >= 10 or len(evidence.frame_times_ms) >= 1400
    if exact_hsmm_search:
        # Full search of the same objective (no beam, no coarse pass); vectorised, so it is
        # also faster than the pruned search.
        decoded = decode_segmental_hsmm_exact(
            specs,
            evidence.frame_times_ms,
            frame_scores,
            allow_leading_gap=allow_leading_gap,
            max_leading_gap_ms=max_leading_gap_ms,
            allow_trailing_gap=True,
            trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
            allow_internal_gaps=allow_internal_gaps,
            internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
            max_internal_gap_ms=max_internal_gap_ms,
            internal_gap_frame_weights=internal_gap_weights,
            transition_start_scores=boundary_start_scores,
            # the same decode options the pruned path below would pass
            **(decode_options if not long_input or slot_policy is not None or pause_aware_gaps else {}),
        )
    elif long_input:
        decoded = decode_segmental_hsmm_coarse_to_refine(
            specs,
            evidence.frame_times_ms,
            frame_scores,
            coarse_stride=4,
            refine_window_ms=110.0,
            beam_width_per_state=64,
            timeout_ms=timeout_ms,
            allow_leading_gap=allow_leading_gap,
            max_leading_gap_ms=max_leading_gap_ms,
            allow_trailing_gap=True,
            trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
            allow_internal_gaps=allow_internal_gaps,
            internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
            max_internal_gap_ms=max_internal_gap_ms,
            internal_gap_frame_weights=internal_gap_weights,
            transition_start_scores=boundary_start_scores,
            **(decode_options if slot_policy is not None or pause_aware_gaps else {}),
            **({"coarse_beam_width_per_state": 0} if exhaustive_coarse_search else {}),
        )
    else:
        decoded = decode_segmental_hsmm(
            specs,
            evidence.frame_times_ms,
            frame_scores,
            beam_width_per_state=0 if sonorant_policy is not None else 64,
            timeout_ms=timeout_ms,
            allow_leading_gap=allow_leading_gap,
            max_leading_gap_ms=max_leading_gap_ms,
            allow_trailing_gap=True,
            trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
            allow_internal_gaps=allow_internal_gaps,
            internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
            max_internal_gap_ms=max_internal_gap_ms,
            internal_gap_frame_weights=internal_gap_weights,
            transition_start_scores=boundary_start_scores,
            **decode_options,
        )
    if not decoded.ok:
        reason = decoded.reason or "no_valid_path"
        raise AlignmentError(f"HSMM decode 실패: {reason} ({recording.wav})")
    if pause_positions:
        # The pause states only order the decode; timelines keep the recording's own phones.
        dropped = set(pause_positions)
        pause_policy["decoded_ms"] = [[round(decoded.states[i].start_ms, 3), round(decoded.states[i].end_ms, 3)]
                                      for i in pause_positions]
        decoded = replace(decoded, states=tuple(state for i, state in enumerate(decoded.states) if i not in dropped))
        specs = tuple(spec for i, spec in enumerate(specs) if i not in dropped)
        frame_scores = {key: value for key, value in frame_scores.items() if not key.startswith(PAUSE_TOKEN_STATE_PREFIX)}
    if len(decoded.states) != len(compiled):
        raise AlignmentError("HSMM state 수가 compiled sequence와 다릅니다")

    hsmm_diagnostics = _build_hsmm_diagnostics(
        compiled,
        specs,
        decoded.states,
        frame_scores,
        evidence.frame_times_ms,
        decode_meta=decoded.meta,
        explicit_gap_topology=explicit_gap_topology,
        usable_start_ms=evidence.active_start_ms,
        usable_end_ms=evidence.active_end_ms,
    )

    state_segments: list[StateSegment] = []
    state_confidences: list[float] = []
    for compiled_state, decoded_state in zip(compiled, decoded.states):
        confidence = _segment_confidence(evidence, frame_scores[compiled_state.instance_id], decoded_state.start_ms, decoded_state.end_ms)
        state_confidences.append(confidence)
        state_segments.append(
            StateSegment(
                instance_id=compiled_state.instance_id,
                phone_index=compiled_state.phone_index,
                state_id=compiled_state.state_id,
                acoustic_role=compiled_state.acoustic_role,
                start_ms=decoded_state.start_ms,
                end_ms=min(evidence.duration_ms, decoded_state.end_ms),
                confidence=confidence,
                raw_start_ms=decoded_state.start_ms,
                raw_end_ms=min(evidence.duration_ms, decoded_state.end_ms),
            )
        )
    coda_onset_policy = None
    general_coda_onset = bool(
        korean_nasal_coda_onset and requested in {"hsmm", "legacy_hsmm_compat"}
        and str(recording.metadata.get("language_hint", project.language_hint)).casefold() in {"ko", "kr", "korean"}
        and frame_score_provider is None and project.alignment.scorer_backend == "heuristic_fallback"
        and not use_external_evidence and ctc_prior is None and not state_duration_constraints
        and not manual_occurrence_start_anchors and not boundary_competition_strength
        and not occurrence_ranker_strength)
    nasal_vowel_policy = None
    if general_coda_onset and recording.metadata.get("format_hint") == "vcv":
        state_segments, nasal_vowel_policy = _vcv_nasal_vowel_boundaries(
            recording, state_segments, evidence.frame_times_ms,
            _audio_band_balance(evidence.wav_path, evidence.frame_times_ms))
        nasal_vowel_policy = nasal_vowel_policy if nasal_vowel_policy["applied_count"] else None
    coda_set = CVVC_NASAL_CODAS | ({"l"} if korean_coda_period_ratio and korean_lateral_coda_rhythm else frozenset())
    if general_coda_onset and any(
            slot["vowel"] + 1 <= slot["end"] and recording.phones[slot["vowel"] + 1] in coda_set
            for slot in recording.metadata.get("phone_slots", ())):
        state_segments, coda_onset_policy = _cvvc_nasal_coda_boundaries(
            recording, evidence, state_segments,
            balance=_audio_band_balance(evidence.wav_path, evidence.frame_times_ms),
            lateral_from_rhythm=korean_lateral_coda_rhythm,
            weak_nasals_from_rhythm=korean_nasal_coda_rhythm_fallback,
            period_ratio=CVVC_VOWEL_PERIOD_RATIO if korean_coda_period_ratio else None)
        state_segments = [
            replace(segment, confidence=_segment_confidence(
                evidence, frame_scores[segment.instance_id], segment.start_ms, segment.end_ms))
            if segment.start_ms != segment.raw_start_ms or segment.end_ms != segment.raw_end_ms else segment
            for segment in state_segments
        ]
        state_confidences = [segment.confidence for segment in state_segments]
        coda_onset_policy = {**coda_onset_policy,
                             "profile": _coda_onset_profile(korean_lateral_coda_rhythm, korean_nasal_coda_rhythm_fallback,
                                                            korean_stop_closure_onset, korean_coda_period_ratio),
                             "review_only": True,
                             "promotion_eligible": False}
        if slot_policy is not None:
            slot_policy = {**slot_policy, "profile": coda_onset_policy["profile"], "nasal_coda_boundary": coda_onset_policy}
    elif slot_policy is not None and korean_nasal_coda_onset:
        slot_policy = {**slot_policy,
                       "profile": _coda_onset_profile(korean_lateral_coda_rhythm, korean_nasal_coda_rhythm_fallback,
                                                      korean_stop_closure_onset, korean_coda_period_ratio)}
    elif slot_policy is not None and cvvc_nasal_coda_boundary:
        state_segments, coda_diagnostics = _cvvc_nasal_coda_boundaries(recording, evidence, state_segments)
        state_segments = [
            replace(segment, confidence=_segment_confidence(
                evidence, frame_scores[segment.instance_id], segment.start_ms, segment.end_ms))
            if segment.start_ms != segment.raw_start_ms or segment.end_ms != segment.raw_end_ms else segment
            for segment in state_segments
        ]
        state_confidences = [segment.confidence for segment in state_segments]
        slot_policy = {**slot_policy, "profile": "cvvc_slot_rhythm_v3" if cvvc_slot_grid_refit else "cvvc_slot_rhythm_v2",
                       "nasal_coda_boundary": coda_diagnostics}
    stop_closure_policy = None
    if general_coda_onset and korean_stop_closure_onset and any(
            phone in CVVC_STOP_ONSETS for phone in recording.phones[1:]):
        state_segments, stop_closure_policy = _stop_closure_onsets(recording, evidence.wav_path, state_segments)
        state_segments = [
            replace(segment, confidence=_segment_confidence(
                evidence, frame_scores[segment.instance_id], segment.start_ms, segment.end_ms))
            if segment.start_ms != segment.raw_start_ms or segment.end_ms != segment.raw_end_ms else segment
            for segment in state_segments
        ]
        state_confidences = [segment.confidence for segment in state_segments]
        stop_closure_policy = {**stop_closure_policy, "profile": "cvvc_slot_rhythm_v7", "review_only": True,
                               "promotion_eligible": False}
    ending_policy = None
    if general_coda_onset and korean_ending_voicing_offset:
        state_segments, ending_policy = _ending_voicing_offsets(recording, evidence.wav_path, state_segments)
        state_segments = [
            replace(segment, confidence=_segment_confidence(
                evidence, frame_scores[segment.instance_id], segment.start_ms, segment.end_ms))
            if segment.start_ms != segment.raw_start_ms or segment.end_ms != segment.raw_end_ms else segment
            for segment in state_segments
        ]
        state_confidences = [segment.confidence for segment in state_segments]
        ending_policy = ({**ending_policy, "profile": "cvvc_slot_rhythm_v8", "review_only": True,
                          "promotion_eligible": False} if ending_policy["applied_count"] else None)
    phones = _phone_segments(recording, state_segments)
    landmarks = _build_landmarks(phones, evidence)
    guide_warnings = _guide_agreement_warnings(phones, recording)
    if compatibility_mode and "guide_agreement_failed" in guide_warnings:
        raise AlignmentError("고신뢰 guide anchor와 HSMM 경계가 250ms 넘게 다릅니다")
    landmarks = _apply_guide_anchors(landmarks, recording)
    warnings = [*evidence.warnings, *score_warnings, *guide_warnings, *_diagnostic_warnings(hsmm_diagnostics)]
    if slot_policy is not None:
        warnings.append("cvvc_slot_rhythm_review_only")
    if ctc_prior is not None and ctc_prior.confidence < 0.35:
        warnings.append("low_ctc_prior_confidence")
    confidence = float(np.clip(np.mean(state_confidences) if state_confidences else 0.0, 0.0, 1.0))
    if confidence < 0.45:
        warnings.append("low_alignment_confidence")
    aligner_config = {
        "aligner_version": ALIGNER_VERSION,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "backend": backend_used,
        "model_sha256": model_sha256,
        "ctc_prior_sha256": ctc_prior.provenance["artifact_sha256"] if ctc_prior is not None else None,
        "ctc_prior_weight": ctc_prior_weight,
        "external_evidence_sha256": (
            evidence.metadata["external_evidence"]["artifact_sha256"]
            if "external_evidence" in evidence.metadata
            else None
        ),
        "timeout_ms": timeout_ms,
        "coarse_to_refine": bool(decoded.meta.get("coarse_to_refine")),
        "explicit_gap_topology": explicit_gap_topology,
        "explicit_gap_state_count": explicit_gap_state_count,
        "occurrence_priors_enabled": occurrence_priors_enabled,
        "occurrence_priors_requested": occurrence_priors_requested,
        "occurrence_prior_scope": occurrence_prior_scope,
        "occurrence_prior_min_coverage": occurrence_prior_min_coverage,
        "occurrence_prior_weight_scale": occurrence_prior_weight_scale,
        "occurrence_rhythm_tracking": occurrence_rhythm_tracking,
        "internal_gap_policy": internal_gap_policy,
        "boundary_competition_strength": boundary_competition_strength,
        "boundary_competition": boundary_competition_diagnostics,
        "occurrence_ranker": ranker_diagnostics,
        "frame_scorer": frame_scorer_provenance,
        "allow_internal_gaps": allow_internal_gaps,
        "internal_gap_penalty_per_ms": internal_gap_penalty_per_ms,
        "max_internal_gap_ms": max_internal_gap_ms,
        "trailing_gap_penalty_per_ms": trailing_gap_penalty_per_ms,
        **({"pause_aware_gaps": {"review_only": True, "speech_penalty_per_ms": PAUSE_GAP_SPEECH_PENALTY_PER_MS,
                                 "silence_weight": PAUSE_GAP_SILENCE_WEIGHT,
                                 "internal_gaps_weighted": internal_gap_weights is not None,
                                 "trailing_gap_weighted": True}} if pause_aware_gaps else {}),
        **({"sonorant_identity": identity_policy} if identity_policy is not None else {}),
        **({"syllabic_sonorant_holds": held_policy} if held_policy is not None else {}),
        **({"explicit_pause_tokens": pause_policy} if pause_policy is not None else {}),
        **({"exhaustive_coarse_search": True}
           if exhaustive_coarse_search and decoded.meta.get("coarse_to_refine") else {}),
        **({"exact_hsmm_search": True} if exact_hsmm_search else {}),
        "state_duration_constraint_sha256": (
            state_duration_constraint_provenance.get("artifact_sha256")
            if state_duration_constraint_provenance is not None
            else None
        ),
        "state_duration_constraint_applied_state_count": len(applied_duration_constraint_states),
        "manual_occurrence_anchor_count": len(manual_anchor_scores),
        "manual_occurrence_anchor_start_bonus": (
            MANUAL_OCCURRENCE_ANCHOR_START_BONUS if manual_anchor_scores else None
        ),
        "manual_occurrence_anchor_candidate_sha256": (
            manual_occurrence_anchor_provenance.get("candidate_sha256")
            if manual_occurrence_anchor_provenance is not None
            else None
        ),
        "compiled_state_inferences": compiled_state_inferences,
        **({"isolated_sonorant_policy": sonorant_policy} if sonorant_policy is not None else {}),
        **({"cvvc_slot_rhythm_policy": slot_policy} if slot_policy is not None else {}),
        **({"korean_nasal_coda_onset_policy": coda_onset_policy}
           if coda_onset_policy is not None and slot_policy is None else {}),
        **({"korean_stop_closure_policy": stop_closure_policy} if stop_closure_policy is not None else {}),
        **({"korean_ending_voicing_policy": ending_policy} if ending_policy is not None else {}),
        **({"korean_vcv_nasal_vowel_policy": nasal_vowel_policy} if nasal_vowel_policy is not None else {}),
    }
    aligner_hash = hashlib.sha256(
        json.dumps(aligner_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    timeline = LandmarkTimeline(
        schema_version=2,
        wav=recording.wav,
        wav_sha256=evidence.wav_sha256,
        recording_sha256=_recording_sha256(recording),
        inventory_sha256=project.inventory_sha256,
        feature_config_sha256=evidence.feature_config_sha256,
        aligner_config_sha256=aligner_hash,
        model_sha256=model_sha256,
        phone_segments=phones,
        landmarks=landmarks,
        usable_start_ms=evidence.active_start_ms,
        usable_end_ms=evidence.active_end_ms,
        confidence=confidence,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance={
            "transcript_source": recording.transcript_source,
            "source_label_kind": recording.label_source,
            "label_source": label_source,
            "feature_backend": evidence.metadata["feature_backend"],
            "neural_evidence_model_sha256": evidence.metadata.get("neural_model_sha256"),
            "neural_evidence_manifest_sha256": evidence.metadata.get("neural_manifest_sha256"),
            "neural_evidence_execution_provider": evidence.metadata.get("neural_execution_provider"),
            "alignment_profile": profile.profile_id,
            "alignment_profile_sha256": profile.sha256,
            "decoder_version": ALIGNER_VERSION,
            "backend_requested": requested,
            "backend_used": backend_used,
            "confidence_calibrated": False,
            "occurrence_priors_enabled": occurrence_priors_enabled,
            "occurrence_priors_requested": occurrence_priors_requested,
            "occurrence_prior_scope": occurrence_prior_scope,
            "occurrence_prior_min_coverage": occurrence_prior_min_coverage,
            "occurrence_prior_weight_scale": occurrence_prior_weight_scale,
            "occurrence_rhythm_tracking": occurrence_rhythm_tracking,
            "occurrence_prior_diagnostics": occurrence_prior_diagnostics,
            "internal_gap_policy": internal_gap_policy,
            "boundary_competition_strength": boundary_competition_strength,
            "boundary_competition": boundary_competition_diagnostics,
            "occurrence_ranker": ranker_diagnostics,
            "frame_scorer": frame_scorer_provenance,
            "manual_occurrence_anchors": (
                {
                    **dict(manual_occurrence_anchor_provenance or {}),
                    "applied_state_count": len(manual_anchor_scores),
                    "state_instance_ids": sorted(manual_anchor_scores),
                    "mode": "bounded_start_bonus_40ms_strength_0.75",
                }
                if manual_anchor_scores
                else {"applied_state_count": 0}
            ),
            "compiled_state_inferences": compiled_state_inferences,
            **({"isolated_sonorant_policy": sonorant_policy} if sonorant_policy is not None else {}),
            **({"cvvc_slot_rhythm_policy": slot_policy} if slot_policy is not None else {}),
            **({"korean_nasal_coda_onset_policy": coda_onset_policy}
               if coda_onset_policy is not None and slot_policy is None else {}),
            **({"korean_stop_closure_policy": stop_closure_policy} if stop_closure_policy is not None else {}),
            **({"korean_ending_voicing_policy": ending_policy} if ending_policy is not None else {}),
            **({"korean_vcv_nasal_vowel_policy": nasal_vowel_policy} if nasal_vowel_policy is not None else {}),
            **({"sonorant_identity": identity_policy} if identity_policy is not None else {}),
            **({"syllabic_sonorant_holds": held_policy} if held_policy is not None else {}),
            **({"explicit_pause_tokens": pause_policy} if pause_policy is not None else {}),
            "timebase": evidence.metadata["timebase"],
            "hsmm": {
                "score": decoded.score,
                "timeout": decoded.timeout,
                "pruned_endpoint_count": decoded.pruned_endpoint_count,
                "state_count": len(decoded.states),
                "meta": dict(decoded.meta),
                "diagnostics": hsmm_diagnostics,
            },
            **(
                {"posterior_model": dict(evidence.metadata["posterior_model"])}
                if model_sha256 is not None
                else {}
            ),
            **({"ctc_prior": dict(ctc_prior.provenance)} if ctc_prior is not None else {}),
            **(
                {"external_evidence": dict(evidence.metadata["external_evidence"])}
                if "external_evidence" in evidence.metadata
                else {}
            ),
            **(
                {
                    "state_duration_constraint": {
                        **dict(state_duration_constraint_provenance),
                        "applied_state_count": len(applied_duration_constraint_states),
                        "applied_state_instance_ids": applied_duration_constraint_states,
                    }
                }
                if state_duration_constraint_provenance is not None
                else {}
            ),
        },
        sample_rate=evidence.source_sample_rate,
        duration_ms=evidence.duration_ms,
    )
    timeline = _calibrate_timeline(project, timeline)
    return AlignmentResult(
        timeline=timeline,
        evidence=evidence,
        backend_requested=requested,
        backend_used=backend_used,
        diagnostics={
            "profile_id": profile.profile_id,
            "profile_sha256": profile.sha256,
            "state_count": len(compiled),
            "decode_score": decoded.score,
            "decode_meta": dict(decoded.meta),
            "model_sha256": model_sha256,
            "ctc_prior": dict(ctc_prior.provenance) if ctc_prior is not None else None,
            "hsmm_diagnostics": hsmm_diagnostics,
            "state_duration_constraint": (
                {
                    **dict(state_duration_constraint_provenance),
                    "applied_state_count": len(applied_duration_constraint_states),
                }
                if state_duration_constraint_provenance is not None
                else None
            ),
        },
    )


PAUSE_TOKEN_STATE_PREFIX = "__pause_token__"
PAUSE_TOKEN_MIN_MS = 30.0
PAUSE_TOKEN_MAX_MS = 2000.0
PAUSE_TOKEN_MODE_MS = 300.0
PAUSE_TOKEN_SILENCE_ACTIVITY = 0.15
PAUSE_TOKEN_SILENCE_MIN_MS = 150.0
PAUSE_TOKEN_OFF_SILENCE_START_PENALTY = 5.0


def _insert_pause_token_states(
    recording: RecordingSpec,
    compiled: Sequence[CompiledState],
    specs: Sequence[HSMMStateSpec],
    frame_scores: Mapping[str, np.ndarray],
    evidence: AcousticEvidence,
) -> tuple[tuple[HSMMStateSpec, ...], dict[str, np.ndarray], list[int], dict[str, object] | None,
           dict[str, np.ndarray]]:
    """Development option: a pause written in the file name (``hing'-'al``) becomes a silence
    state before the next syllable, so the syllables cannot slide across the pause.

    Segment scores are frame means, so a short silence state costs little anywhere. When the
    recording has interior silences of at least 150 ms, the pause state may only start cheaply
    inside one of them (start score 0 there, -5 elsewhere).
    """
    from .voicebank_project_adapter import _pause_token_slot_indices

    slots = recording.metadata.get("phone_slots") or ()
    heads = sorted(_pause_token_slot_indices(recording.wav, len(slots))) if slots else []
    first_state = {}
    for position, state in enumerate(compiled):
        first_state.setdefault(state.phone_index, position)
    targets = sorted({first_state[int(slots[k]["start"])] for k in heads if int(slots[k]["start"]) in first_state} - {0})
    if not targets:
        return tuple(specs), dict(frame_scores), [], None, {}
    activity = np.clip(np.asarray(evidence.tracks["active"], dtype=np.float64)
                       * (1.0 - np.asarray(evidence.tracks["low_energy"], dtype=np.float64)), 0.0, 1.0)
    times = np.asarray(evidence.frame_times_ms, dtype=np.float64)
    quiet = activity <= PAUSE_TOKEN_SILENCE_ACTIVITY
    loud = np.flatnonzero(~quiet)
    runs: list[tuple[int, int]] = []
    if len(loud):
        start = None
        for index in range(int(loud[0]), int(loud[-1]) + 1):
            if quiet[index] and start is None:
                start = index
            elif not quiet[index] and start is not None:
                if times[index - 1] - times[start] >= PAUSE_TOKEN_SILENCE_MIN_MS:
                    runs.append((start, index))
                start = None
    start_score = np.zeros(len(times), dtype=np.float64)
    if runs:
        start_score -= PAUSE_TOKEN_OFF_SILENCE_START_PENALTY
        for first, end in runs:
            start_score[first:end] = 0.0
    scores = dict(frame_scores)
    start_scores: dict[str, np.ndarray] = {}
    out: list[HSMMStateSpec] = []
    positions: list[int] = []
    for position, spec in enumerate(specs):
        if position in targets:
            state_id = f"{PAUSE_TOKEN_STATE_PREFIX}{compiled[position].phone_index}"
            positions.append(len(out))
            out.append(HSMMStateSpec(
                state_id=state_id, state_type="silence", min_duration_ms=PAUSE_TOKEN_MIN_MS,
                max_duration_ms=PAUSE_TOKEN_MAX_MS, mode_duration_ms=PAUSE_TOKEN_MODE_MS,
                duration_sigma_ms=(PAUSE_TOKEN_MAX_MS - PAUSE_TOKEN_MIN_MS) / 2.5, over_duration_penalty_scale=0.30))
            scores[state_id] = 1.0 - activity
            start_scores[state_id] = start_score
        out.append(spec)
    return tuple(out), scores, positions, {
        "review_only": True,
        "before_phone_indices": [compiled[position].phone_index for position in targets],
        "min_ms": PAUSE_TOKEN_MIN_MS,
        "max_ms": PAUSE_TOKEN_MAX_MS,
        "score": "1 - clip(active * (1 - low_energy), 0, 1)",
        "interior_silences_ms": [[round(float(times[a]), 3), round(float(times[b - 1]), 3)] for a, b in runs],
        "off_silence_start_penalty": PAUSE_TOKEN_OFF_SILENCE_START_PENALTY if runs else 0.0,
    }, start_scores


SYLLABIC_SONORANT_MAX_MS = 1400.0
SYLLABIC_SONORANT_MODE_MS = 300.0


def _syllabic_sonorant_holds(
    project: ProjectSpec,
    recording: RecordingSpec,
    compiled: Sequence[CompiledState],
) -> tuple[Sequence[CompiledState], dict[str, object] | None]:
    """Development overlay: a nasal or liquid that is a syllable of its own (Korean list '-M',
    '-L', Japanese moraic n) may be held like a vowel instead of a 260 ms consonant."""
    held: set[int] = set()
    for slot in recording.metadata.get("phone_slots") or ():
        if not isinstance(slot, Mapping) or not slot.get("start") == slot.get("end") == slot.get("vowel"):
            continue
        index = int(slot["start"])
        definition = project.inventory.get(recording.phones[index]) if 0 <= index < len(recording.phones) else None
        if definition is not None and definition.features.get("manner") in {"nasal", "liquid"}:
            held.add(index)
    changed: list[str] = []
    states: list[CompiledState] = []
    for state in compiled:
        if state.phone_index in held and state.state_id == "consonant" and state.acoustic_role == "consonant":
            state = replace(state, acoustic_role="periodic", max_ms=max(state.max_ms, SYLLABIC_SONORANT_MAX_MS),
                            mode_ms=SYLLABIC_SONORANT_MODE_MS)
            changed.append(state.instance_id)
        states.append(state)
    if not changed:
        return compiled, None
    return tuple(states), {
        "review_only": True,
        "phone_indices": sorted(held),
        "state_instance_ids": changed,
        "acoustic_role": "periodic",
        "max_ms": SYLLABIC_SONORANT_MAX_MS,
        "mode_ms": SYLLABIC_SONORANT_MODE_MS,
    }


def _apply_sonorant_identity(
    project: ProjectSpec,
    recording: RecordingSpec,
    evidence: AcousticEvidence,
    compiled: Sequence[CompiledState],
    frame_scores: Mapping[str, np.ndarray],
    templates: SonorantTemplates,
    weight: float,
) -> tuple[dict[str, np.ndarray], Mapping[str, object] | None]:
    """Lower sonorant state scores on frames that look like another sonorant of this recording."""
    if not 0.0 < weight <= 0.9:
        raise AlignmentError("sonorant_identity_weight must be in (0, 0.9]")
    phones = [phone for phone in recording.phones if in_identity_scope(project.inventory.get(phone), templates.scope)]
    frames = cepstra_at(project.wav_path(recording), evidence.frame_times_ms)
    factors = templates.factors(phones, frames, weight=weight)
    if not factors:
        return dict(frame_scores), None
    scores = dict(frame_scores)
    touched = 0
    for state in compiled:
        factor = factors.get(state.phone_id)
        if factor is not None:
            scores[state.instance_id] = np.clip(np.asarray(scores[state.instance_id], dtype=np.float64) * factor, 0.0, 1.0)
            touched += 1
    return scores, {
        "review_only": True,
        "weight": float(weight),
        "temperature": SONORANT_IDENTITY_TEMPERATURE,
        "phones": sorted(factors),
        "state_count": touched,
        "templates_sha256": templates.sha256(),
    }


def _cvvc_nasal_coda_boundaries(
    recording: RecordingSpec,
    evidence: AcousticEvidence,
    segments: Sequence[StateSegment],
    balance: np.ndarray | None = None,
    lateral_from_rhythm: bool = False,
    weak_nasals_from_rhythm: bool = False,
    period_ratio: float | None = None,
) -> tuple[list[StateSegment], dict[str, object]]:
    """Development hypothesis: start each nasal coda at the vowel->murmur step.

    Only the vowel/coda boundary inside one rhythm slot moves, to the strongest
    rise of low-band over mid-band energy. Occurrence choice, the coda's end and
    every other state stay as decoded. Laterals lack this cue and are skipped.
    """
    adjusted = list(segments)
    times = evidence.frame_times_ms
    cue = "smoothed(low_band_ratio - mid_band_ratio) step"
    if balance is None:
        balance = np.asarray(evidence.tracks["low_band_ratio"], dtype=np.float64) - np.asarray(
            evidence.tracks["mid_band_ratio"], dtype=np.float64)
    else:
        cue = "smoothed(P<400Hz - P800-3000Hz)/P step from audio"
    balance = np.convolve(balance, np.ones(5) / 5.0, mode="same")
    slots_out: list[dict[str, object]] = []
    for slot in recording.metadata.get("phone_slots", ()):
        vowel, end = slot["vowel"], slot["vowel"] + 1
        # The coda directly follows the vowel; a slot may continue with a breath.
        if end > slot["end"] or recording.phones[end] not in CVVC_NASAL_CODAS:
            continue
        vowel_ids = [i for i, item in enumerate(adjusted) if item.phone_index == vowel]
        coda_ids = [i for i, item in enumerate(adjusted) if item.phone_index == end]
        entry: dict[str, object] = {"vowel_phone_index": vowel, "coda_phone_index": end,
                                    "coda": recording.phones[end]}
        slots_out.append(entry)
        if (not vowel_ids or not coda_ids or adjusted[vowel_ids[-1]].acoustic_role != "vowel_stable"
                or adjusted[coda_ids[0]].state_id != "consonant"):
            entry["applied"], entry["reason"] = False, "unsupported_state_topology"
            continue
        stable, consonant = adjusted[vowel_ids[-1]], adjusted[coda_ids[0]]
        region_start = adjusted[vowel_ids[0]].start_ms + 60.0
        lo = max(region_start, stable.start_ms + CVVC_CODA_MIN_VOWEL_AFTER_TRANSITION_MS)
        hi = consonant.end_ms - CVVC_CODA_MIN_CONSONANT_MS
        region = np.flatnonzero((times >= region_start) & (times < adjusted[coda_ids[-1]].end_ms))
        candidates = [k for k in range(4, len(region) - 4) if lo <= times[region[k]] <= hi]
        entry["decoded_start_ms"] = float(consonant.start_ms)
        if not candidates:
            entry["applied"], entry["reason"] = False, "no_search_window"
            continue
        values = balance[region]
        steps = [float(values[k:].mean() - values[:k].mean()) for k in candidates]
        best = int(np.argmax(steps))
        step_ms = float(times[region[candidates[best]]])
        entry["step_strength"] = steps[best]
        entry["step_ms"] = step_ms
        # VCV codas only close the held final vowel, so the window holds one murmur onset;
        # even weak steps (eu/i + n/m) land within ~60 ms of hand-made OTOs (a Korean VCV bank, 2026-10-11).
        if steps[best] < CVVC_CODA_STEP_MIN_STRENGTH and recording.metadata.get("format_hint") != "vcv":
            entry["applied"], entry["reason"] = False, "weak_band_balance_step"
            continue
        adjusted[vowel_ids[-1]] = replace(stable, end_ms=step_ms)
        adjusted[coda_ids[0]] = replace(consonant, start_ms=step_ms)
        entry["applied"], entry["shift_ms"] = True, step_ms - float(consonant.start_ms)
    if lateral_from_rhythm or weak_nasals_from_rhythm:
        _codas_from_rhythm(recording, adjusted, slots_out,
                           laterals=lateral_from_rhythm, weak_nasals=weak_nasals_from_rhythm,
                           period_ratio=period_ratio)
    return adjusted, {
        "review_only": True, "anchors_are_gold": False, "cue": cue,
        "min_strength": CVVC_CODA_STEP_MIN_STRENGTH, "slots": slots_out,
        "applied_count": sum(1 for item in slots_out if item.get("applied")),
    }


VCV_NASAL_ONSETS = frozenset({"n", "m"})
VCV_BAND_STEP_HALF_WINDOW_MS = 80.0
# Fitted on the nasal files of two Korean VCV banks (2026-10-11): moves up to 150 ms always help; farther
# moves only when the step is strong (held hums that the decoder let a vowel swallow).
VCV_NASAL_FREE_SHIFT_MS = 150.0
VCV_NASAL_FAR_STEP_MIN_STRENGTH = 0.7


def _vcv_nasal_vowel_boundaries(recording: RecordingSpec, segments: Sequence[StateSegment], times: np.ndarray,
                                balance: np.ndarray) -> tuple[list[StateSegment], dict[str, object]]:
    """Development hypothesis: in Korean VCV, a vowel after nasal murmur starts at the murmur->vowel band step.

    Held hums (M_maM_meo, maNmaN) are as loud as the vowels, so the decoder may let a
    vowel swallow the murmur. For each vowel preceded by a run of nasal phones, the
    vowel start moves to the strongest fall of low-minus-mid band balance between the
    preceding vowel's start and this vowel's end; the run's start moves to the strongest
    rise before it. Inner nasal boundaries (coda m | onset m) are not observable and are
    rescaled. Only these boundaries move; no phone is added, removed or reordered.
    """
    adjusted = list(segments)
    smooth = np.convolve(balance, np.ones(5) / 5.0, mode="same")
    phones = recording.phones
    half = VCV_BAND_STEP_HALF_WINDOW_MS

    def span(index: int) -> tuple[float, float] | None:
        items = [item for item in adjusted if item.phone_index == index]
        return (min(item.start_ms for item in items), max(item.end_ms for item in items)) if items else None

    def local_step(lo: float, hi: float, sign: float) -> tuple[float, float] | None:
        best = None
        for k in np.flatnonzero((times >= lo) & (times <= hi)):
            before = smooth[(times >= times[k] - half) & (times < times[k])]
            after = smooth[(times >= times[k]) & (times < times[k] + half)]
            if len(before) < 3 or len(after) < 3:
                continue
            value = sign * float(before.mean() - after.mean())
            if best is None or value > best[1]:
                best = (float(times[k]), value)
        return best

    entries: list[dict[str, object]] = []
    for vowel in range(1, len(phones)):
        # w-glides keep murmur-like low F2 into the vowel; the step there is not the onset.
        if (not _is_vowel_like(phones[vowel]) or phones[vowel].startswith("w")
                or phones[vowel - 1] not in VCV_NASAL_ONSETS):
            continue
        first = vowel - 1
        while first > 0 and phones[first - 1] in CVVC_NASAL_CODAS:
            first -= 1
        vowel_span, run_start = span(vowel), span(first)
        left_vowel = first - 1 if first > 0 and _is_vowel_like(phones[first - 1]) else None
        left_span = span(left_vowel) if left_vowel is not None else None
        if vowel_span is None or run_start is None:
            continue
        lo = (left_span[0] + 60.0) if left_span is not None else run_start[0]
        fall = local_step(lo + 40.0, vowel_span[1] - 60.0, 1.0)
        entry: dict[str, object] = {"vowel_phone_index": vowel, "nasal_phone_indices": list(range(first, vowel)),
                                    "decoded_vowel_start_ms": vowel_span[0]}
        entries.append(entry)
        if fall is None or fall[1] <= 0.0:
            entry["applied"], entry["reason"] = False, "no_murmur_to_vowel_fall"
            continue
        if abs(fall[0] - vowel_span[0]) > VCV_NASAL_FREE_SHIFT_MS and fall[1] < VCV_NASAL_FAR_STEP_MIN_STRENGTH:
            # A far move must be backed by a clear step, or it may land on a neighbouring syllable.
            entry["applied"], entry["reason"], entry["fall_strength"] = False, "weak_far_step", fall[1]
            continue
        start = run_start[0]
        if left_span is not None:
            rise = local_step(lo, fall[0] - 40.0, -1.0)
            if rise is not None and rise[1] > 0.0:
                start = rise[0]
        if fall[0] - start < 30.0:
            entry["applied"], entry["reason"] = False, "murmur_too_short"
            continue
        if left_span is not None:
            _respan_phone(adjusted, left_vowel, left_span[0], start)
        count = vowel - first
        # Inner nasal boundaries are invisible; the onset nasal keeps at most 60 ms.
        onset_len = min(60.0, (fall[0] - start) / 2.0) if count > 1 else fall[0] - start
        edges = [start + (fall[0] - onset_len - start) * i / (count - 1) for i in range(count)] if count > 1 else [start]
        edges.append(fall[0])
        for offset, phone in enumerate(range(first, vowel)):
            _respan_phone(adjusted, phone, edges[offset], edges[offset + 1])
        _respan_phone(adjusted, vowel, fall[0], vowel_span[1])
        entry.update({"applied": True, "vowel_start_ms": fall[0], "fall_strength": fall[1], "murmur_start_ms": start,
                      "shift_ms": fall[0] - vowel_span[0]})
    return adjusted, {"review_only": True, "anchors_are_gold": False, "profile": "korean_vcv_nasal_vowel_v1",
                      "cue": "smoothed(P<400Hz - P800-3000Hz)/P fall (murmur->vowel) and rise (vowel->murmur), 80 ms halves",
                      "slots": entries, "applied_count": sum(1 for item in entries if item.get("applied"))}


def _is_vowel_like(phone: str) -> bool:
    return phone not in CVVC_NASAL_CODAS | {"l", "r"} and phone not in _NON_VOWEL_SLOT_PHONES and any(
        ch in "aeiou" for ch in phone) and phone[0] in "aeiouwy"


def _respan_phone(adjusted: list[StateSegment], phone: int, start: float, end: float) -> None:
    """Linearly map one phone's states into [start, end]."""
    ids = [i for i, item in enumerate(adjusted) if item.phone_index == phone]
    if not ids or end <= start:
        return
    old_start, old_end = adjusted[ids[0]].start_ms, adjusted[ids[-1]].end_ms
    scale = (end - start) / max(old_end - old_start, 1e-6)
    for i in ids:
        item = adjusted[i]
        adjusted[i] = replace(item, start_ms=start + (item.start_ms - old_start) * scale,
                              end_ms=start + (item.end_ms - old_start) * scale)


def _coda_onset_profile(lateral_from_rhythm: bool, weak_nasals_from_rhythm: bool,
                        stop_closure_onset: bool = False, period_ratio: bool = False) -> str:
    if period_ratio:
        return "cvvc_slot_rhythm_v9"
    if stop_closure_onset:
        return "cvvc_slot_rhythm_v7"
    if weak_nasals_from_rhythm:
        return "cvvc_slot_rhythm_v6"
    return "cvvc_slot_rhythm_v5" if lateral_from_rhythm else "cvvc_slot_rhythm_v4"


def _codas_from_rhythm(recording: RecordingSpec, adjusted: list[StateSegment],
                       slots_out: list[dict[str, object]], *, laterals: bool, weak_nasals: bool,
                       period_ratio: float | None = None) -> None:
    """Start codas without a usable band step after the file's usual vowel length.

    A lateral keeps vowel-like formants, so no band step marks it; some nasal
    codas (after eu or i) give only a weak step. The singer holds every syllable
    of one list on the same rhythm, so the median vowel length measured before
    the acoustically found nasal codas is reused. Needs at least two such codas
    in the recording. Weak nasals also need a consonant onset in their slot,
    which anchors the vowel start.
    """
    phone_start = {}
    for segment in adjusted:
        phone_start.setdefault(segment.phone_index, segment.start_ms)
    lengths = [float(entry["step_ms"]) - phone_start[int(entry["vowel_phone_index"])]
               for entry in slots_out if entry.get("applied")]
    fallback_ms = None
    vcv = recording.metadata.get("format_hint") == "vcv"
    if vcv and len(lengths) < 2:
        # VCV lists hold every vowel up to the next consonant, the final one too;
        # the CVVC vowel share of the period would cut the final vowel to ~1/4.
        fallback_ms = _vcv_median_vowel_ms(recording, adjusted)
    elif period_ratio is not None and len(lengths) < 2:
        # Too few acoustic codas: use the bank's vowel share of the syllable period.
        onsets = [phone_start[slot["start"]] for slot in recording.metadata.get("phone_slots", ())
                  if slot["start"] in phone_start]
        if len(onsets) >= 3:
            fallback_ms = period_ratio * float(np.median(np.diff(onsets)))
    if weak_nasals:
        slots = {slot["vowel"]: slot for slot in recording.metadata.get("phone_slots", ())}
        for entry in [item for item in slots_out if item.get("reason") == "weak_band_balance_step"]:
            vowel = int(entry["vowel_phone_index"])
            # A vowel-only VCV syllable starts at the decoded vowel->vowel boundary.
            if slots[vowel]["start"] >= vowel and not (vcv and fallback_ms is not None):
                entry["rhythm_fallback"] = "no_consonant_onset"
                continue
            entry["weak_step_ms"] = entry.pop("step_ms")
            entry["cue"] = "median vowel length before nasal codas in this file (weak band step)"
            del entry["applied"], entry["reason"]
            _start_from_rhythm(adjusted, vowel, vowel + 1, lengths, entry, fallback_ms)
            entry["rhythm_fallback"] = "applied" if entry["applied"] else entry["reason"]
    if not laterals:
        return
    for slot in recording.metadata.get("phone_slots", ()):
        vowel, coda = slot["vowel"], slot["vowel"] + 1
        if coda > slot["end"] or recording.phones[coda] != "l":
            continue
        entry: dict[str, object] = {"vowel_phone_index": vowel, "coda_phone_index": coda, "coda": "l",
                                    "cue": "median vowel length before nasal codas in this file"}
        slots_out.append(entry)
        _start_from_rhythm(adjusted, vowel, coda, lengths, entry, fallback_ms)


_NON_VOWEL_SLOT_PHONES = frozenset({"n", "m", "ng", "l", "br", "breath", "exh", "vf", "pau", "sil", "sp"})


def _vcv_median_vowel_ms(recording: RecordingSpec, adjusted: Sequence[StateSegment]) -> float | None:
    """Median decoded length of the coda-less vowels of a VCV recording (None if fewer than two)."""
    spans: dict[int, list[float]] = {}
    for segment in adjusted:
        spans.setdefault(segment.phone_index, []).extend((segment.start_ms, segment.end_ms))
    lengths = [max(spans[slot["vowel"]]) - min(spans[slot["vowel"]])
               for slot in recording.metadata.get("phone_slots", ())
               if slot["vowel"] == slot["end"] and slot["vowel"] in spans
               and recording.phones[slot["vowel"]] not in _NON_VOWEL_SLOT_PHONES]
    return float(np.median(lengths)) if len(lengths) >= 2 else None


def _start_from_rhythm(adjusted: list[StateSegment], vowel: int, coda: int, lengths: list[float],
                       entry: dict[str, object], fallback_ms: float | None = None) -> None:
    vowel_ids = [i for i, item in enumerate(adjusted) if item.phone_index == vowel]
    coda_ids = [i for i, item in enumerate(adjusted) if item.phone_index == coda]
    if len(lengths) < 2 and fallback_ms is None:
        entry["applied"], entry["reason"] = False, "fewer_than_two_nasal_codas"
        return
    if (not vowel_ids or not coda_ids or adjusted[vowel_ids[-1]].acoustic_role != "vowel_stable"
            or adjusted[coda_ids[0]].state_id != "consonant"):
        entry["applied"], entry["reason"] = False, "unsupported_state_topology"
        return
    stable, consonant = adjusted[vowel_ids[-1]], adjusted[coda_ids[0]]
    vowel_ms = float(np.median(lengths)) if len(lengths) >= 2 else float(fallback_ms)
    if len(lengths) < 2:
        entry["cue"] = "bank vowel share of the syllable period (too few nasal codas in this file)"
    target = adjusted[vowel_ids[0]].start_ms + vowel_ms
    lo = stable.start_ms + CVVC_CODA_MIN_VOWEL_AFTER_TRANSITION_MS
    hi = consonant.end_ms - CVVC_CODA_MIN_CONSONANT_MS
    entry["decoded_start_ms"], entry["rhythm_vowel_ms"] = float(consonant.start_ms), vowel_ms
    if hi < lo:
        entry["applied"], entry["reason"] = False, "no_search_window"
        return
    start = float(min(max(target, lo), hi))
    adjusted[vowel_ids[-1]] = replace(stable, end_ms=start)
    adjusted[coda_ids[0]] = replace(consonant, start_ms=start)
    entry["step_ms"], entry["applied"], entry["shift_ms"] = start, True, start - float(consonant.start_ms)


def _stop_closure_onsets(recording: RecordingSpec, wav_path: Path,
                         segments: Sequence[StateSegment]) -> tuple[list[StateSegment], dict[str, object]]:
    """Development hypothesis: a stop's closure starts where the level drops 15 dB.

    The decoder lets the preceding vowel run into the closure. The level of the
    preceding phone is its median 10 ms RMS (or its maximum when it is shorter
    than 100 ms); the closure starts at the first 5 ms frame at least 15 dB
    below it, searched from 150 ms before to 40 ms after the decoded boundary.
    Only that boundary moves; the preceding phone keeps its last state.
    """
    adjusted = list(segments)
    samples, rate = sf.read(str(wav_path), dtype="float64", always_2d=True)
    mono = samples.mean(axis=1)
    hop, window = max(1, int(round(rate * 0.005))), max(2, int(round(rate * 0.010)))
    frames = np.lib.stride_tricks.sliding_window_view(mono, window)[::hop]
    level = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    times = (np.arange(len(frames)) * hop + window / 2.0) / rate * 1000.0
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, segment in enumerate(adjusted):
        first.setdefault(segment.phone_index, i)
        last[segment.phone_index] = i
    stops: list[dict[str, object]] = []
    for p in range(1, len(recording.phones)):
        if recording.phones[p] not in CVVC_STOP_ONSETS or p - 1 not in last or p not in first:
            continue
        previous, closure = adjusted[last[p - 1]], adjusted[first[p]]
        if closure.state_id != "closure":
            continue
        boundary, phone_start = closure.start_ms, adjusted[first[p - 1]].start_ms
        entry: dict[str, object] = {"stop_phone_index": p, "stop": recording.phones[p], "decoded_start_ms": float(boundary)}
        stops.append(entry)
        if boundary - phone_start > 100.0:
            reference = float(np.median(level[(times >= phone_start + 40.0) & (times <= boundary - 40.0)]))
        else:
            span = level[(times >= phone_start) & (times <= boundary)]
            if not len(span):
                entry["applied"], entry["reason"] = False, "no_previous_phone_frames"
                continue
            reference = float(span.max())
        candidates = np.flatnonzero((times >= max(phone_start + 40.0, boundary - 150.0)) & (times <= boundary + 40.0))
        hits = candidates[level[candidates] <= reference - STOP_CLOSURE_DROP_DB]
        if not len(hits):
            entry["applied"], entry["reason"] = False, "no_level_drop"
            continue
        onset = float(min(max(times[hits[0]], previous.start_ms + 5.0), closure.end_ms - 10.0))
        if abs(onset - boundary) < 10.0:
            entry["applied"], entry["reason"] = False, "within_10ms"
            continue
        adjusted[last[p - 1]] = replace(previous, end_ms=onset)
        adjusted[first[p]] = replace(closure, start_ms=onset)
        entry.update(applied=True, onset_ms=onset, shift_ms=onset - float(boundary))
    return adjusted, {
        "review_only": True, "anchors_are_gold": False, "drop_db": STOP_CLOSURE_DROP_DB,
        "cue": "first 10 ms RMS frame 15 dB below the preceding phone", "stops": stops,
        "applied_count": sum(1 for item in stops if item.get("applied")),
    }


def _ending_voicing_offsets(recording: RecordingSpec, wav_path: Path,
                            segments: Sequence[StateSegment]) -> tuple[list[StateSegment], dict[str, object]]:
    """Development hypothesis: a phrase-final vowel or coda ends where voicing stops.

    Applies to a vowel or sonorant followed by breath, fry or a pause, and to
    the last phone of the file. Voicing is a normalised autocorrelation peak of
    at least 0.5 for F0 70-600 Hz (30 ms window, 5 ms hop); fry falls below
    70 Hz and so counts as unvoiced. The phone ends at the last voiced frame
    before 40 ms without voicing, searched from 40 ms after its start to the
    end of the following phone (or of the file).

    Only the searched frames get the autocorrelation (FFT), shared per file through
    ``voicing.frame_voicing``; the level gate uses the whole file's loudest frame, as before.
    Equal to ``_ending_voicing_offsets_reference``.
    """
    adjusted = list(segments)
    voicing = frame_voicing(wav_path)
    times = voicing.times
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, segment in enumerate(adjusted):
        first.setdefault(segment.phone_index, i)
        last[segment.phone_index] = i
    phones = recording.phones
    searches: list[tuple[int, int | None, float, np.ndarray]] = []
    for p, phone in enumerate(phones):
        if phone in CVVC_UNVOICED_ENDINGS or p not in last:
            continue
        following = p + 1 if p + 1 < len(phones) else None
        if following is not None and phones[following] not in CVVC_UNVOICED_ENDINGS:
            continue
        if following is not None and following not in first:
            continue
        end_limit = adjusted[last[following]].end_ms if following is not None else float(times[-1])
        candidates = np.flatnonzero((times >= adjusted[first[p]].start_ms + 40.0) & (times <= end_limit))
        searches.append((p, following, end_limit, candidates))
    needed = np.unique(np.concatenate([item[3] for item in searches])) if searches else np.zeros(0, dtype=np.int64)
    voiced = voicing.voiced(needed)
    endings: list[dict[str, object]] = []
    for p, following, end_limit, candidates in searches:
        phone = phones[p]
        entry: dict[str, object] = {"phone_index": p, "phone": phone,
                                    "next": phones[following] if following is not None else "file_end",
                                    "decoded_end_ms": float(adjusted[last[p]].end_ms)}
        endings.append(entry)
        offset = None
        run, previous_voiced = 0, None
        for k in candidates:
            if voiced[k]:
                previous_voiced, run = k, 0
            else:
                run += 1
                if run >= 8 and previous_voiced is not None:
                    offset = float(times[previous_voiced])
                    break
        if offset is None:
            entry["applied"], entry["reason"] = False, "no_voicing_offset"
            continue
        tail = adjusted[last[p]]
        upper = adjusted[first[following]].end_ms - 10.0 if following is not None else end_limit
        offset = min(max(offset, tail.start_ms + 5.0), upper)
        if abs(offset - tail.end_ms) < 10.0:
            entry["applied"], entry["reason"] = False, "within_10ms"
            continue
        adjusted[last[p]] = replace(tail, end_ms=offset)
        if following is not None:
            head = adjusted[first[following]]
            adjusted[first[following]] = replace(head, start_ms=offset)
        entry.update(applied=True, offset_ms=offset, shift_ms=offset - float(tail.end_ms))
    return adjusted, {
        "review_only": True, "anchors_are_gold": False,
        "cue": "last voiced frame (autocorrelation >= 0.5, F0 70-600 Hz) before 40 ms unvoiced",
        "endings": endings, "applied_count": sum(1 for item in endings if item.get("applied")),
    }


def _ending_voicing_offsets_reference(recording: RecordingSpec, wav_path: Path,
                            segments: Sequence[StateSegment]) -> tuple[list[StateSegment], dict[str, object]]:
    """Original whole-file version of _ending_voicing_offsets, kept as its reference.

    Development hypothesis: a phrase-final vowel or coda ends where voicing stops.

    Applies to a vowel or sonorant followed by breath, fry or a pause, and to
    the last phone of the file. Voicing is a normalised autocorrelation peak of
    at least 0.5 for F0 70-600 Hz (30 ms window, 5 ms hop); fry falls below
    70 Hz and so counts as unvoiced. The phone ends at the last voiced frame
    before 40 ms without voicing, searched from 40 ms after its start to the
    end of the following phone (or of the file).
    """
    adjusted = list(segments)
    samples, rate = sf.read(str(wav_path), dtype="float64", always_2d=True)
    mono = samples.mean(axis=1)
    hop, window = max(1, int(round(rate * 0.005))), max(4, int(round(rate * 0.030)))
    frames = np.lib.stride_tricks.sliding_window_view(mono, window)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames, n=2 * window, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :window]
    lo, hi = max(1, int(rate / 600.0)), min(window - 1, int(rate / 70.0))
    strength = autocorr[:, lo:hi].max(axis=1) / (autocorr[:, 0] + 1e-12)
    level = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    voiced = (strength >= 0.5) & (level > level.max() - 45.0)
    times = (np.arange(len(frames)) * hop + window / 2.0) / rate * 1000.0
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, segment in enumerate(adjusted):
        first.setdefault(segment.phone_index, i)
        last[segment.phone_index] = i
    endings: list[dict[str, object]] = []
    phones = recording.phones
    for p, phone in enumerate(phones):
        if phone in CVVC_UNVOICED_ENDINGS or p not in last:
            continue
        following = p + 1 if p + 1 < len(phones) else None
        if following is not None and phones[following] not in CVVC_UNVOICED_ENDINGS:
            continue
        if following is not None and following not in first:
            continue
        end_limit = adjusted[last[following]].end_ms if following is not None else float(times[-1])
        entry: dict[str, object] = {"phone_index": p, "phone": phone,
                                    "next": phones[following] if following is not None else "file_end",
                                    "decoded_end_ms": float(adjusted[last[p]].end_ms)}
        endings.append(entry)
        candidates = np.flatnonzero((times >= adjusted[first[p]].start_ms + 40.0) & (times <= end_limit))
        offset = None
        run, previous_voiced = 0, None
        for k in candidates:
            if voiced[k]:
                previous_voiced, run = k, 0
            else:
                run += 1
                if run >= 8 and previous_voiced is not None:
                    offset = float(times[previous_voiced])
                    break
        if offset is None:
            entry["applied"], entry["reason"] = False, "no_voicing_offset"
            continue
        tail = adjusted[last[p]]
        upper = adjusted[first[following]].end_ms - 10.0 if following is not None else end_limit
        offset = min(max(offset, tail.start_ms + 5.0), upper)
        if abs(offset - tail.end_ms) < 10.0:
            entry["applied"], entry["reason"] = False, "within_10ms"
            continue
        adjusted[last[p]] = replace(tail, end_ms=offset)
        if following is not None:
            head = adjusted[first[following]]
            adjusted[first[following]] = replace(head, start_ms=offset)
        entry.update(applied=True, offset_ms=offset, shift_ms=offset - float(tail.end_ms))
    return adjusted, {
        "review_only": True, "anchors_are_gold": False,
        "cue": "last voiced frame (autocorrelation >= 0.5, F0 70-600 Hz) before 40 ms unvoiced",
        "endings": endings, "applied_count": sum(1 for item in endings if item.get("applied")),
    }


def _cached_evidence(path: Path, **options: object) -> AcousticEvidence:
    """extract_acoustic_evidence, computed once per WAV and options during a generation run."""
    key = tuple(sorted((name, str(value)) for name, value in options.items()))
    return analysis_cache.cached("evidence", path, key, lambda: extract_acoustic_evidence(path, **options))


def _audio_band_balance(wav_path: Path, times_ms: np.ndarray) -> np.ndarray:
    times = np.asarray(times_ms, dtype=np.float64)
    return analysis_cache.cached("band_balance", wav_path, (times.tobytes(),),
                                 lambda: _audio_band_balance_uncached(wav_path, times))


def _audio_band_balance_uncached(wav_path: Path, times_ms: np.ndarray) -> np.ndarray:
    """Low (<400 Hz) minus mid (800-3000 Hz) power share at each frame time.

    A vowel keeps F2 energy in the mid band; a nasal murmur moves almost all
    energy below 400 Hz. Raw bands separate i from a murmur better than the
    normalised 0-500/500-3000 evidence tracks, whose low band holds i's F1.
    """
    samples, rate = sf.read(str(wav_path), dtype="float64", always_2d=True)
    mono = samples.mean(axis=1)
    window = max(16, int(round(0.030 * rate)))
    padded = np.concatenate((np.zeros(window), mono, np.zeros(window)))
    starts = np.clip(np.round(np.asarray(times_ms) / 1000.0 * rate).astype(int) - window // 2 + window,
                     0, len(padded) - window)
    frames = padded[starts[:, None] + np.arange(window)[None, :]] * np.hanning(window)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(window, 1.0 / rate)
    total = power.sum(axis=1) + 1e-12
    return (power[:, freqs < 400.0].sum(axis=1) - power[:, (freqs >= 800.0) & (freqs < 3000.0)].sum(axis=1)) / total


def _isolated_sonorant_states(
    project: ProjectSpec,
    recording: RecordingSpec,
    evidence: AcousticEvidence,
    compiled: Sequence[CompiledState],
) -> tuple[Sequence[CompiledState], dict[str, object] | None]:
    """Development overlay for filename-only Korean vf-C-vf holds; never labels."""
    if (project.language_hint.casefold() not in {"ko", "kr", "korean"}
            or recording.transcript_source != "original_filename_only"
            or recording.label_source != "none"
            or recording.guide_anchors or len(recording.phones) != 3
            or recording.phones[0] != "vf" or recording.phones[2] != "vf"
            or recording.phones[1] not in {"l", "m", "n", "ng"}
            or evidence.metadata.get("feature_backend") != FEATURE_BACKEND_ID
            or len(compiled) >= 10 or len(evidence.frame_times_ms) >= 1400
            or evidence.active_end_ms - evidence.active_start_ms > 4000.0):
        return compiled, None
    centers = [s for s in compiled if s.phone_index == 1 and s.state_id == "consonant"
        and s.acoustic_role == "consonant"]
    if len(centers) != 1 or centers[0].min_ms >= 360.0:
        return compiled, None
    center = centers[0]
    capacity = max(1400.0, evidence.active_end_ms - evidence.active_start_ms)
    modified = replace(center, acoustic_role="periodic", max_ms=capacity, mode_ms=360.0)
    policy = {
        "profile": "isolated_sonorant_v1", "review_only": True, "promotion_eligible": False,
        "state_instance_id": center.instance_id,
        "original_state": dict(center.__dict__), "compiled_state": dict(modified.__dict__),
        "duration_sigma_ms": (1400.0 - center.min_ms) / 2.5,
        "beam_width_per_state": 0, "internal_gap_policy": "contiguous",
        "trailing_gap_weights": "clip(active * (1 - low_energy), 0, 1)",
        "review_timestamps_used": False,
    }
    return tuple(modified if s.instance_id == center.instance_id else s for s in compiled), policy


@dataclass(frozen=True)
class _FrameScoreCompilation:
    result: FrameScoreResult
    occurrence_priors_applied: bool
    occurrence_diagnostics: Mapping[str, object]


def _compile_frame_scores(
    states: Sequence[CompiledState],
    project: ProjectSpec,
    evidence: AcousticEvidence,
    profile: AlignmentProfile,
    **kwargs: object,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], bool, Mapping[str, object]]:
    """Backward-compatible view of the internal provider result."""
    compilation = _compile_frame_score_compilation(states, project, evidence, profile, **kwargs)
    return (
        dict(compilation.result.emission_scores),
        compilation.result.warnings,
        compilation.occurrence_priors_applied,
        compilation.occurrence_diagnostics,
    )


def _compile_frame_score_compilation(
    states: Sequence[CompiledState],
    project: ProjectSpec,
    evidence: AcousticEvidence,
    profile: AlignmentProfile,
    *,
    ctc_prior: CTCPrior | None = None,
    ctc_prior_weight: float = 0.0,
    recording: RecordingSpec | None = None,
    use_occurrence_priors: bool = False,
    require_acoustic_rhythm: bool = False,
    occurrence_prior_min_coverage: float = 0.60,
    occurrence_prior_weight_scale: float = 1.0,
    allow_fortis_acoustic_rhythm: bool = False,
    occurrence_rhythm_tracking: str = "uniform_cells",
    frame_score_provider: FrameScoreProvider | None = None,
    cvvc_slot_rhythm: bool = False,
    cvvc_slot_grid_refit: bool = False,
    syllable_period_ms: float | None = None,
) -> _FrameScoreCompilation:
    """Compile scores through the production provider seam.

    Topology and all occurrence-prior construction remain in this module.  The
    provider receives only already-validated frame evidence and returns the
    unchanged ``instance_id -> score`` input expected by the HSMM.
    """
    warnings: list[str] = []
    occurrence_priors, acoustic_rhythm_used, occurrence_diagnostics = (
        _occurrence_priors(
            states,
            evidence,
            recording,
            minimum_coverage=occurrence_prior_min_coverage,
            allow_fortis_acoustic_rhythm=allow_fortis_acoustic_rhythm,
            rhythm_tracking=occurrence_rhythm_tracking,
            cvvc_slot_rhythm=cvvc_slot_rhythm,
            cvvc_slot_grid_refit=cvvc_slot_grid_refit,
            syllable_period_ms=syllable_period_ms,
        )
        if use_occurrence_priors
        else ({}, False, {"requested": False, "reason": "not_requested"})
    )
    if require_acoustic_rhythm and not acoustic_rhythm_used:
        occurrence_priors = {}
        if use_occurrence_priors:
            warnings.append("occurrence_prior_acoustic_gate_not_met")
    occurrence_priors_applied = bool(occurrence_priors)
    if acoustic_rhythm_used:
        warnings.append("acoustic_rhythm_occurrence_prior")
    heuristic_provider = HeuristicFrameScoreProvider(
        profile_id=profile.profile_id,
        profile_sha256=profile.sha256,
        role_weights=profile.role_weights,
        ctc_prior=ctc_prior,
        ctc_prior_weight=ctc_prior_weight,
        occurrence_priors=occurrence_priors,
        acoustic_rhythm_used=acoustic_rhythm_used,
        occurrence_prior_weight_scale=occurrence_prior_weight_scale,
        breath_end_ms=_leading_breath_end_ms(evidence) if occurrence_priors_applied else 0.0,
    )
    provider = frame_score_provider or heuristic_provider
    configured_fallback_reason: str | None = None
    if frame_score_provider is None and project.alignment.scorer_backend == "trained_subphone_v1":
        assert project.alignment.scorer_model is not None
        assert project.alignment.scorer_model_sha256 is not None
        try:
            provider = load_trained_subphone_scorer(
                project.root / project.alignment.scorer_model,
                project=project,
                expected_sha256=project.alignment.scorer_model_sha256,
            )
            if not provider.artifact.emission_heads:
                provider = HeuristicEmissionTransitionOverlayProvider(
                    heuristic_provider=heuristic_provider,
                    trained_provider=provider,
                )
        except TrainedSubphoneScorerError as exc:
            if not project.alignment.allow_heuristic_fallback:
                raise AlignmentError(f"trained_subphone_v1 rejected: {exc.code}: {exc}") from exc
            configured_fallback_reason = f"trained_subphone_v1_rejected:{exc.code}"
            warnings.append(configured_fallback_reason)
    try:
        provider_result = provider.score(
            states=states,
            project=project,
            recording=recording,
            evidence=evidence,
        )
    except FrameScoreProviderError as exc:
        if frame_score_provider is None and project.alignment.scorer_backend == "trained_subphone_v1" and project.alignment.allow_heuristic_fallback:
            configured_fallback_reason = f"trained_subphone_v1_runtime_failed:{getattr(exc, 'code', 'provider_error')}"
            warnings.append(configured_fallback_reason)
            provider_result = heuristic_provider.score(
                states=states,
                project=project,
                recording=recording,
                evidence=evidence,
            )
        else:
            raise AlignmentError(str(exc)) from exc
    occurrence_diagnostics = {
        **dict(occurrence_diagnostics),
        "requested": bool(use_occurrence_priors),
        "require_acoustic_rhythm": bool(require_acoustic_rhythm),
        "minimum_coverage": occurrence_prior_min_coverage,
        "weight_scale": occurrence_prior_weight_scale,
        "allow_fortis_acoustic_rhythm": bool(allow_fortis_acoustic_rhythm),
        "rhythm_tracking": occurrence_rhythm_tracking,
        "applied": occurrence_priors_applied,
    }
    result = FrameScoreResult(
        emission_scores=dict(provider_result.emission_scores),
        transition_start_scores=dict(provider_result.transition_start_scores),
        warnings=tuple(dict.fromkeys([*warnings, *provider_result.warnings])),
        diagnostics=dict(provider_result.diagnostics),
        provenance={
            **dict(provider_result.provenance),
            "configured_scorer_backend": project.alignment.scorer_backend,
            "configured_scorer_model_sha256": project.alignment.scorer_model_sha256,
            **(
                {"fallback_reason": configured_fallback_reason}
                if configured_fallback_reason is not None
                else {}
            ),
        },
    )
    _validate_frame_score_result(result, states=states, evidence=evidence)
    return _FrameScoreCompilation(
        result=result,
        occurrence_priors_applied=occurrence_priors_applied,
        occurrence_diagnostics=occurrence_diagnostics,
    )


def _validate_frame_score_result(
    result: FrameScoreResult,
    *,
    states: Sequence[CompiledState],
    evidence: AcousticEvidence,
) -> None:
    expected_ids = {state.instance_id for state in states}
    actual_ids = set(result.emission_scores)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise AlignmentError(
            f"frame score provider state contract mismatch: missing={missing}, unexpected={unexpected}"
        )
    expected_frames = len(evidence.frame_times_ms)
    for field, score_map in (
        ("emission_scores", result.emission_scores),
        ("transition_start_scores", result.transition_start_scores),
    ):
        unexpected = sorted(set(score_map) - expected_ids)
        if unexpected:
            raise AlignmentError(f"frame score provider {field} has unknown states: {unexpected}")
        for state_id, values in score_map.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (expected_frames,) or not np.all(np.isfinite(values)):
                raise AlignmentError(f"frame score provider {field} is invalid for {state_id}")
            if field == "emission_scores" and (np.any(values < 0.0) or np.any(values > 1.0)):
                raise AlignmentError(f"frame score provider emission_scores must be clipped for {state_id}")


def _compile_frame_scores_legacy_reference(
    states: Sequence[CompiledState],
    project: ProjectSpec,
    evidence: AcousticEvidence,
    profile: AlignmentProfile,
    *,
    ctc_prior: CTCPrior | None = None,
    ctc_prior_weight: float = 0.0,
    recording: RecordingSpec | None = None,
    use_occurrence_priors: bool = False,
    require_acoustic_rhythm: bool = False,
    occurrence_prior_min_coverage: float = 0.60,
    occurrence_prior_weight_scale: float = 1.0,
    allow_fortis_acoustic_rhythm: bool = False,
    occurrence_rhythm_tracking: str = "uniform_cells",
) -> tuple[dict[str, np.ndarray], tuple[str, ...], bool, Mapping[str, object]]:
    result: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    occurrence_priors, acoustic_rhythm_used, occurrence_diagnostics = (
        _occurrence_priors(
            states,
            evidence,
            recording,
            minimum_coverage=occurrence_prior_min_coverage,
            allow_fortis_acoustic_rhythm=allow_fortis_acoustic_rhythm,
            rhythm_tracking=occurrence_rhythm_tracking,
        )
        if use_occurrence_priors
        else ({}, False, {"requested": False, "reason": "not_requested"})
    )
    if require_acoustic_rhythm and not acoustic_rhythm_used:
        occurrence_priors = {}
        if use_occurrence_priors:
            warnings.append("occurrence_prior_acoustic_gate_not_met")
    occurrence_priors_applied = bool(occurrence_priors)
    if acoustic_rhythm_used:
        warnings.append("acoustic_rhythm_occurrence_prior")
    breath_end_ms = _leading_breath_end_ms(evidence) if occurrence_priors_applied else 0.0
    for state in states:
        weights = profile.role_weights.get(state.acoustic_role)
        if weights is None:
            weights = profile.role_weights["other"]
            warnings.append(f"unknown_acoustic_role:{state.acoustic_role}")
        score = np.zeros(len(evidence.frame_times_ms), dtype=np.float64)
        for track, weight in weights.items():
            values = evidence.tracks.get(track)
            if values is None:
                raise AlignmentError(f"profile이 요구하는 acoustic track이 없습니다: {track}")
            score += float(weight) * values
        phone = project.inventory[state.phone_id]
        feature_score = _phone_feature_score(phone.features, evidence.tracks)
        if feature_score is not None:
            score = 0.90 * score + 0.10 * feature_score
        if ctc_prior is not None:
            if state.phone_index >= len(ctc_prior.phone_posteriors):
                raise AlignmentError("CTC prior phone posterior 수가 compiled sequence와 다릅니다")
            score = (1.0 - ctc_prior_weight) * score + ctc_prior_weight * ctc_prior.phone_posteriors[state.phone_index]
        occurrence_prior = occurrence_priors.get(state.instance_id)
        if occurrence_prior is not None:
            sustained = state.acoustic_role in {
                "vowel_stable",
                "vowel_nucleus",
                "periodic",
                "irregular_periodic",
                "silence",
            }
            if acoustic_rhythm_used:
                prior_weight = 0.08 if sustained else 0.20
            else:
                prior_weight = 0.05 if sustained else 0.14
            prior_weight = min(0.45, prior_weight * occurrence_prior_weight_scale)
            score = (1.0 - prior_weight) * score + prior_weight * occurrence_prior
        if breath_end_ms > 0.0 and state.acoustic_role not in {"breath", "silence", "sil"}:
            early = evidence.frame_times_ms < breath_end_ms
            if np.any(early):
                score = np.where(early, score * 0.25, score)
        result[state.instance_id] = np.clip(score, 0.0, 1.0)
    occurrence_diagnostics = {
        **dict(occurrence_diagnostics),
        "requested": bool(use_occurrence_priors),
        "require_acoustic_rhythm": bool(require_acoustic_rhythm),
        "minimum_coverage": occurrence_prior_min_coverage,
        "weight_scale": occurrence_prior_weight_scale,
        "allow_fortis_acoustic_rhythm": bool(allow_fortis_acoustic_rhythm),
        "rhythm_tracking": occurrence_rhythm_tracking,
        "applied": occurrence_priors_applied,
    }
    return result, tuple(dict.fromkeys(warnings)), occurrence_priors_applied, occurrence_diagnostics


def _repeated_phone_boundary_competition_scores(
    states: Sequence[CompiledState],
    recording: RecordingSpec,
    evidence: AcousticEvidence,
    *,
    strength: float,
) -> tuple[dict[str, np.ndarray], Mapping[str, object]]:
    """Give repeated-phone boundaries distinct, reliability-gated local peaks.

    The HSMM already preserves order, but repeated phones can still consume the
    same broad acoustic event in adjacent occurrences.  Each eligible boundary
    receives a separate duration-derived cell and only gets a bonus when its
    neural boundary peak clears a local confidence gate.  Weak evidence is not
    converted into a hard duration constraint.
    """
    if strength <= 0.0:
        return {}, {"requested": False, "applied": False, "reason": "disabled"}
    signal = evidence.tracks.get("neural_boundary_likelihood")
    if signal is None:
        raise AlignmentError("neural boundary competition requires neural_boundary_likelihood")
    times = np.asarray(evidence.frame_times_ms, dtype=np.float64)
    values = np.asarray(signal, dtype=np.float64)
    if len(times) < 3 or len(values) != len(times) or not np.all(np.isfinite(values)):
        raise AlignmentError("neural boundary competition received invalid frame evidence")
    phone_count = len(recording.phones)
    if phone_count < 3:
        return {}, {"requested": True, "applied": False, "reason": "fewer_than_three_phones"}
    first_state_by_phone: dict[int, CompiledState] = {}
    phone_modes = np.zeros(phone_count, dtype=np.float64)
    for state in states:
        first_state_by_phone.setdefault(int(state.phone_index), state)
        mode = state.mode_ms if state.mode_ms is not None else state.min_ms + 0.5 * (state.max_ms - state.min_ms)
        phone_modes[int(state.phone_index)] += max(5.0, float(mode))
    if len(first_state_by_phone) != phone_count or np.any(phone_modes <= 0.0):
        return {}, {"requested": True, "applied": False, "reason": "incomplete_phone_state_topology"}
    active_start = float(evidence.active_start_ms)
    active_end = max(active_start + 50.0, float(evidence.active_end_ms))
    expected = active_start + np.cumsum(phone_modes)[:-1] / float(np.sum(phone_modes)) * (active_end - active_start)
    cell_edges = np.asarray(
        [active_start, *(0.5 * (expected[:-1] + expected[1:])), active_end], dtype=np.float64
    )
    phone_tokens = [str(phone).casefold() for phone in recording.phones]
    token_counts = {phone: phone_tokens.count(phone) for phone in set(phone_tokens)}
    clipped = np.clip(values, 0.0, 1.0)
    median = float(np.median(clipped))
    q75 = float(np.quantile(clipped, 0.75))
    minimum_peak = max(0.16, median + 0.03, q75 * 0.75)
    result: dict[str, np.ndarray] = {}
    targets: list[dict[str, object]] = []
    for boundary_ordinal, expected_ms in enumerate(expected, start=1):
        left_phone = phone_tokens[boundary_ordinal - 1]
        right_phone = phone_tokens[boundary_ordinal]
        repeated = token_counts[left_phone] > 1 or token_counts[right_phone] > 1
        if not repeated:
            continue
        cell_start = float(cell_edges[boundary_ordinal - 1])
        cell_end = float(cell_edges[boundary_ordinal])
        candidates = np.flatnonzero((times >= cell_start) & (times < cell_end))
        target: dict[str, object] = {
            "phone_boundary_index": boundary_ordinal,
            "left_phone": left_phone,
            "right_phone": right_phone,
            "expected_ms": float(expected_ms),
            "cell_start_ms": cell_start,
            "cell_end_ms": cell_end,
        }
        if not len(candidates):
            target["applied"] = False
            target["reason"] = "empty_cell"
            targets.append(target)
            continue
        best = int(candidates[int(np.argmax(clipped[candidates]))])
        peak = float(clipped[best])
        target.update({"peak_ms": float(times[best]), "peak_value": peak})
        if peak < minimum_peak:
            target.update({"applied": False, "reason": "low_peak"})
            targets.append(target)
            continue
        reliability = float(np.clip((peak - minimum_peak) / max(1e-6, 1.0 - minimum_peak), 0.0, 1.0))
        sigma_ms = max(18.0, min(70.0, 0.28 * max(20.0, cell_end - cell_start)))
        score = strength * reliability * np.exp(-0.5 * ((times - float(times[best])) / sigma_ms) ** 2)
        current_state = first_state_by_phone[boundary_ordinal]
        result[current_state.instance_id] = np.asarray(score, dtype=np.float64)
        target.update(
            {
                "applied": True,
                "state_instance_id": current_state.instance_id,
                "reliability": reliability,
                "sigma_ms": sigma_ms,
            }
        )
        targets.append(target)
    return result, {
        "requested": True,
        "applied": bool(result),
        "strength": float(strength),
        "minimum_peak": minimum_peak,
        "signal_median": median,
        "signal_q75": q75,
        "eligible_boundary_count": len(targets),
        "applied_boundary_count": len(result),
        "targets": targets,
    }


def _occurrence_ranker_start_scores(
    states: Sequence[CompiledState], project: ProjectSpec, recording: RecordingSpec,
    model_path: Path | None, strength: float, provider: str | None = None,
) -> tuple[dict[str, np.ndarray], Mapping[str, object]]:
    if model_path is None or strength <= 0.0:
        return {}, {"requested": False, "applied": False, "reason": "disabled"}
    if not 0.0 < float(strength) <= BOUNDARY_COMPETITION_MAX_STRENGTH:
        raise AlignmentError("occurrence_ranker_strength must be between 0 and 0.35")
    from .acoustic_evidence import _frame_analysis, _multiband_pitch_tracks
    from .occurrence_ranker_runtime import infer_occurrence_boundary_scores
    analysis = _frame_analysis(
        project.wav_path(recording), internal_sample_rate=project.audio.internal_sample_rate,
        convert_to_mono=project.audio.convert_to_mono, resampler=project.audio.resampler,
        frame_ms=30.0, hop_ms=5.0,
    )
    selected_provider = str(provider or project.alignment.evidence_provider).strip().casefold()
    if selected_provider not in {"cuda", "cpu", "auto"}:
        raise AlignmentError("occurrence_ranker_provider must be cuda, cpu, or auto")
    scores, _indices, diagnostics = infer_occurrence_boundary_scores(
        analysis, _multiband_pitch_tracks(analysis), recording.phones,
        model_path=model_path, provider=selected_provider,
    )
    first_by_phone: dict[int, CompiledState] = {}
    for state in states:
        first_by_phone.setdefault(state.phone_index, state)
    tokens = [str(phone).casefold() for phone in recording.phones]
    result: dict[str, np.ndarray] = {}
    normalizations: list[dict[str, float]] = []
    for ordinal, values in enumerate(scores, start=1):
        if tokens.count(tokens[ordinal - 1]) <= 1 and tokens.count(tokens[ordinal]) <= 1:
            continue
        bonus, normalization = _normalized_ranker_start_bonus(values, strength=float(strength))
        result[first_by_phone[ordinal].instance_id] = bonus
        normalizations.append(normalization)
    return result, {
        **dict(diagnostics),
        "requested_provider": selected_provider,
        "strength": float(strength),
        "score_normalization": "robust_centered_logit_v1",
        "normalization": normalizations,
        "applied_boundary_count": len(result),
    }


def _normalized_ranker_start_bonus(values: np.ndarray, *, strength: float) -> tuple[np.ndarray, dict[str, float]]:
    """Convert one query's relative logit shape into HSMM-compatible start scores.

    A sigmoid is not suitable here: ranker logits may have a large negative
    calibration offset even when their *relative* peak is informative.  The
    decoder only needs the relative preference over possible start frames, so
    center each query independently and use a robust 5th--95th percentile
    scale.  The symmetric bounded score prevents a globally low logit bias
    from suppressing all ranker influence, while retaining the caller's
    development-only strength cap.
    """
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or not len(raw) or not np.all(np.isfinite(raw)):
        raise AlignmentError("occurrence ranker emitted invalid query logits")
    median = float(np.median(raw))
    p05 = float(np.quantile(raw, 0.05))
    p95 = float(np.quantile(raw, 0.95))
    half_span = 0.5 * (p95 - p05)
    if half_span < 1e-4:
        return np.zeros_like(raw), {
            "median_logit": median,
            "p05_logit": p05,
            "p95_logit": p95,
            "robust_half_span": half_span,
            "bonus_min": 0.0,
            "bonus_max": 0.0,
        }
    centered = np.clip((raw - median) / half_span, -2.0, 2.0)
    bonus = float(strength) * centered
    return bonus.astype(np.float64, copy=False), {
        "median_logit": median,
        "p05_logit": p05,
        "p95_logit": p95,
        "robust_half_span": half_span,
        "bonus_min": float(np.min(bonus)),
        "bonus_max": float(np.max(bonus)),
    }


def _occurrence_priors(
    states: Sequence[CompiledState],
    evidence: AcousticEvidence,
    recording: RecordingSpec | None,
    *,
    minimum_coverage: float = 0.60,
    allow_fortis_acoustic_rhythm: bool = False,
    rhythm_tracking: str = "uniform_cells",
    cvvc_slot_rhythm: bool = False,
    cvvc_slot_grid_refit: bool = False,
    syllable_period_ms: float | None = None,
) -> tuple[dict[str, np.ndarray], bool, Mapping[str, object]]:
    if not states or not len(evidence.frame_times_ms):
        return {}, False, {"reason": "empty_states_or_frames"}
    if syllable_period_ms is not None:
        # The recording tempo replaces the other occurrence priors; a file that
        # does not follow it gets no prior at all (never the uniform fallback).
        return _bpm_grid_occurrence_priors(states, evidence, recording, float(syllable_period_ms))
    if cvvc_slot_rhythm:
        slot_result = _cvvc_slot_occurrence_priors(states, evidence, recording,
                                                   grid_refit=cvvc_slot_grid_refit)
        if slot_result is not None:
            return slot_result
    modes = np.asarray(
        [
            state.mode_ms
            if state.mode_ms is not None
            else state.min_ms + 0.5 * (state.max_ms - state.min_ms)
            for state in states
        ],
        dtype=np.float64,
    )
    modes = np.maximum(modes, 5.0)
    cumulative = np.cumsum(modes)
    normalized_centers = (cumulative - 0.5 * modes) / max(float(cumulative[-1]), 1.0)
    active_start = float(evidence.active_start_ms)
    active_span = max(50.0, float(evidence.active_end_ms) - active_start)
    expected_centers = active_start + normalized_centers * active_span
    acoustic_rhythm_used = False
    rhythm_diagnostics: Mapping[str, object] = {"reason": "topology_not_supported"}
    if _supports_acoustic_rhythm_prior(recording) or (
        allow_fortis_acoustic_rhythm and _is_vcv_recording(recording)
    ):
        expected_centers, acoustic_rhythm_used, rhythm_diagnostics = _rhythm_warped_centers(
            states,
            evidence,
            expected_centers,
            active_start=active_start,
            active_end=active_start + active_span,
            minimum_coverage=minimum_coverage,
            tracking_mode=rhythm_tracking,
        )
    sigma = max(55.0, min(260.0, active_span / max(2.0, float(len(states)) * 0.75)))
    guide_by_phone: dict[int, list[object]] = {}
    if recording is not None:
        for guide in recording.guide_anchors:
            if guide.phone_index is not None and guide.confidence >= 0.55:
                guide_by_phone.setdefault(int(guide.phone_index), []).append(guide)
    priors: dict[str, np.ndarray] = {}
    times = evidence.frame_times_ms
    for state, expected_center, mode in zip(states, expected_centers, modes):
        prior = np.exp(-0.5 * ((times - expected_center) / sigma) ** 2)
        guides = guide_by_phone.get(state.phone_index, ())
        if guides:
            guide = max(guides, key=lambda item: float(item.confidence))
            guide_center = float(guide.time_ms)
            if guide.label in {"phone_boundary", "closure_start", "release", "voicing_onset", "vowel_transition"}:
                guide_center += 0.5 * float(mode)
            guide_sigma = max(30.0, min(95.0, 0.45 * float(mode) + 25.0))
            guide_prior = np.exp(-0.5 * ((times - guide_center) / guide_sigma) ** 2)
            guide_weight = min(0.80, 0.35 + 0.45 * float(guide.confidence))
            prior = (1.0 - guide_weight) * prior + guide_weight * guide_prior
        priors[state.instance_id] = np.clip(prior, 0.0, 1.0)
    return priors, acoustic_rhythm_used, rhythm_diagnostics


def _cvvc_slot_occurrence_priors(states, evidence, recording, *, grid_refit=False):
    """Opt-in hypothesis for regularly recorded Korean CV(C) syllables.

    Extra coda states must not give one syllable a larger rhythm cell. Acoustic
    peaks only locate those cells; they are not reviewed phone boundaries.
    Unsupported topology or weak/irregular cues retain the existing decoder.
    """
    if (recording is None or recording.transcript_source != "original_filename_only"
            or recording.label_source != "none" or recording.guide_anchors
            or recording.metadata.get("format_hint") != "cvvc"
            or recording.metadata.get("language_hint") != "korean"):
        return None
    slots = recording.metadata.get("phone_slots", ())
    if not isinstance(slots, (tuple, list)) or len(slots) < 3:
        return None
    vowels = {"a", "e", "i", "o", "u", "eu", "eo", "ae", "oe", "wi"}
    codas = {"ng", "n", "m", "l"}
    cursor = 0
    for slot in slots:
        if not isinstance(slot, Mapping):
            return None
        start, vowel, end = (slot.get(key) for key in ("start", "vowel", "end"))
        if (any(type(value) is not int for value in (start, vowel, end))
                or start != cursor or vowel != start + 1 or end not in {vowel, vowel + 1}
                or end >= len(recording.phones) or recording.phones[vowel] not in vowels
                or recording.phones[start] in vowels | {"vf", "exh", "q", "sil"}
                or (end > vowel and recording.phones[end] not in codas)):
            return None
        cursor = end + 1
    if cursor != len(recording.phones):
        return None
    required = ("vowel_boundary_likelihood", "transition", "periodicity_rise", "onset_strength")
    if any(key not in evidence.tracks for key in required):
        return None
    times = evidence.frame_times_ms
    period = (evidence.active_end_ms - evidence.active_start_ms) / len(slots)
    if not 120. <= period <= 1200.:
        return None
    signal = sum(weight * evidence.tracks[key] for weight, key in zip((.45, .25, .20, .10), required))
    if signal.shape != times.shape or not np.all(np.isfinite(signal)):
        return None
    step = float(np.median(np.diff(times))) if len(times) > 1 else 5.
    peaks, _ = find_peaks(signal, distance=max(1, round(50. / step)), prominence=.06)
    anchors = []
    strengths = []
    for center in evidence.active_start_ms + np.arange(len(slots)) * period + 40.:
        candidates = peaks[(np.abs(times[peaks] - center) <= .30 * period) & (signal[peaks] >= .35)]
        if not len(candidates):
            return None
        score = signal[candidates] * np.exp(-.5 * ((times[candidates] - center) / (.18 * period)) ** 2)
        best = int(candidates[np.argmax(score)])
        if signal[best] < .35:
            return None
        anchors.append(float(times[best]))
        strengths.append(float(signal[best]))
    gaps = np.diff(anchors)
    if np.any(gaps < .55 * period) or np.any(gaps > 1.45 * period):
        return None
    grid_diagnostics = None
    if grid_refit:
        refit = _slot_anchor_grid_refit(anchors, peaks, signal, times)
        grid_diagnostics = {"applied": False, "reason": "no_peak_near_grid"} if refit is None else refit[2]
        if refit is not None:
            anchors, strengths = refit[0], refit[1]
            grid_diagnostics = {**grid_diagnostics, "applied": True}
    priors = {}
    for index, slot in enumerate(slots):
        local = [s for s in states if slot["start"] <= s.phone_index <= slot["end"]]
        if not local:
            return None
        start = max(evidence.active_start_ms - 70., anchors[index] - 90.)
        end = anchors[index + 1] - 90. if index + 1 < len(slots) else evidence.active_end_ms
        modes = np.array([s.mode_ms or (s.min_ms + s.max_ms) / 2. for s in local])
        centers = start + (np.cumsum(modes) - .5 * modes) / modes.sum() * (end - start)
        for state, center in zip(local, centers):
            priors[state.instance_id] = np.exp(-.5 * ((times - center) / max(55., .20 * period)) ** 2)
    return priors, True, {"reason": "cvvc_slot_rhythm_applied", "profile": "cvvc_slot_rhythm_v1",
        "review_only": True, "promotion_eligible": False, "anchors_are_gold": False,
        "slot_count": len(slots), "period_ms": float(period), "anchors_ms": anchors,
        "peak_strengths": strengths, "internal_gap_policy": "contiguous", "trailing_gap_weighted": True,
        **({"grid_refit": grid_diagnostics} if grid_diagnostics is not None else {})}


_BPM_PAUSE_MARK_RE = re.compile(r"[_'\-]{2,}")
BPM_GRID_SPECIAL_PHONES = frozenset({"exh", "vf", "sil", "q", "pau", "br", "breath"})


def _bpm_grid_occurrence_priors(states, evidence, recording, period_ms):
    """Occurrence priors from the recording tempo (user BPM or the bank estimate).

    List recordings place one syllable per beat (or per half/double beat). The
    grid start is fitted to the file's syllable-onset cues; each grid point is
    then snapped to the nearest cue within 0.2 periods. A file whose cues do not
    follow the grid (unguided or irregular pace, pauses) gets no prior.
    """
    def skipped(reason, **extra):
        return {}, False, {"reason": reason, "profile": "bpm_grid_v1", "period_ms": float(period_ms), **extra}

    if recording is None:
        return skipped("no_recording")
    if _BPM_PAUSE_MARK_RE.search(Path(recording.wav).stem.lstrip("_'-")):
        # A pause written in the name (ga'ga'_'gya, ba'-'bya) breaks the beat grid.
        return skipped("pause_in_file_name")
    raw_slots = recording.metadata.get("phone_slots", ())
    if not isinstance(raw_slots, (list, tuple)):
        return skipped("no_phone_slots")
    slots = []
    for slot in raw_slots:
        if not isinstance(slot, Mapping) or any(type(slot.get(key)) is not int for key in ("start", "vowel", "end")):
            return skipped("invalid_phone_slots")
        if recording.phones[slot["vowel"]] in BPM_GRID_SPECIAL_PHONES:
            if slots and slot is not raw_slots[-1]:
                return skipped("interior_pause_or_fry")
            continue
        slots.append(slot)
    if len(slots) < 2:
        return skipped("single_syllable")
    required = ("vowel_boundary_likelihood", "transition", "periodicity_rise", "onset_strength")
    if any(key not in evidence.tracks for key in required):
        return skipped("missing_onset_tracks")
    times = evidence.frame_times_ms
    signal = sum(weight * evidence.tracks[key] for weight, key in zip((.45, .25, .20, .10), required))
    if signal.shape != times.shape or not np.all(np.isfinite(signal)) or len(times) < 3:
        return skipped("invalid_onset_signal")
    step = float(np.median(np.diff(times)))
    peaks, _ = find_peaks(signal, distance=max(1, round(50. / step)), prominence=.06)
    peaks = peaks[signal[peaks] >= .30]
    if len(peaks) < 2:
        return skipped("no_onset_peaks")
    peak_times = times[peaks]
    active_start, active_end = float(evidence.active_start_ms), float(evidence.active_end_ms)
    count = len(slots)
    required_hits = max(2, int(np.ceil(.75 * count)))
    best = None
    # The requested period first; half/double only when the file does not follow it
    # (one or two syllables per beat).
    for candidate in (period_ms, period_ms / 2.0, period_ms * 2.0):
        if candidate < 150.0 or active_start + (count - 1) * candidate > active_end:
            continue
        candidate_best = None
        for start in np.arange(active_start - .25 * candidate, active_start + .60 * candidate, 5.0):
            grid = start + candidate * np.arange(count)
            distance = np.abs(peak_times[None, :] - grid[:, None])
            near = distance <= .20 * candidate
            hits = int(np.any(near, axis=1).sum())
            weights = np.where(near, signal[peaks][None, :] * np.exp(-.5 * (distance / (.10 * candidate)) ** 2), 0.)
            score = float(weights.max(axis=1).sum())
            if candidate_best is None or (hits, score) > (candidate_best[0], candidate_best[1]):
                candidate_best = (hits, score, float(candidate), float(start))
        if candidate_best is not None and (best is None or candidate_best[0] > best[0]):
            best = candidate_best
        if best is not None and best[0] >= required_hits:
            break
    if best is None:
        return skipped("grid_longer_than_recording")
    hits, score, period, start = best
    if hits < required_hits:
        return skipped("pace_does_not_follow_tempo", grid_hits=hits, slot_count=count)
    anchors = []
    for index in range(count):
        center = start + index * period
        near = np.abs(peak_times - center) <= .20 * period
        if np.any(near):
            local = peaks[near]
            choice = local[np.argmax(signal[local] * np.exp(-.5 * ((times[local] - center) / (.10 * period)) ** 2))]
            anchors.append(float(times[choice]))
        else:
            anchors.append(float(center))
    if np.any(np.diff(anchors) <= .4 * period):
        return skipped("grid_anchors_collapsed")
    priors = {}
    for index, slot in enumerate(slots):
        local = [state for state in states if slot["start"] <= state.phone_index <= slot["end"]]
        if not local:
            continue
        lead = 90. if slot["vowel"] > slot["start"] else 20.
        begin = max(active_start - 70., anchors[index] - lead)
        if index + 1 < count:
            following = slots[index + 1]
            end = anchors[index + 1] - (90. if following["vowel"] > following["start"] else 20.)
        else:
            end = active_end
        if end <= begin:
            return skipped("grid_cell_empty")
        modes = np.array([state.mode_ms or (state.min_ms + state.max_ms) / 2. for state in local], dtype=np.float64)
        modes = np.maximum(modes, 5.)
        centers = begin + (np.cumsum(modes) - .5 * modes) / modes.sum() * (end - begin)
        for state, center in zip(local, centers):
            priors[state.instance_id] = np.exp(-.5 * ((times - center) / max(55., .20 * period)) ** 2)
    return priors, True, {
        "reason": "bpm_grid_applied", "profile": "bpm_grid_v1", "review_only": True, "anchors_are_gold": False,
        "requested_period_ms": float(period_ms), "period_ms": period, "grid_start_ms": start,
        "grid_hits": hits, "slot_count": count, "anchors_ms": anchors,
    }


def _slot_anchor_grid_refit(anchors, peaks, signal, times):
    """Re-pick slot anchors on a robust equal-spacing grid (cvvc_slot_rhythm_v3).

    The first pass searches around active_span/slots, which the trailing tail
    inflates, so one late syllable can lock onto a peak in its neighbour. A
    Theil-Sen line through the first-pass anchors gives the sung grid; each
    anchor is re-picked within 0.2 grid periods. Without a peak there, the
    first pass is kept.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    index = np.arange(len(anchors), dtype=np.float64)
    slopes = [(anchors[j] - anchors[i]) / (j - i) for i in range(len(anchors)) for j in range(i + 1, len(anchors))]
    grid_period = float(np.median(slopes))
    if grid_period <= 0.0:
        return None
    grid_start = float(np.median(anchors - grid_period * index))
    grid = grid_start + grid_period * index
    chosen, strengths = [], []
    for center in grid:
        candidates = peaks[(np.abs(times[peaks] - center) <= 0.20 * grid_period) & (signal[peaks] >= .35)]
        if not len(candidates):
            return None
        score = signal[candidates] * np.exp(-.5 * ((times[candidates] - center) / (.10 * grid_period)) ** 2)
        best = int(candidates[np.argmax(score)])
        chosen.append(float(times[best]))
        strengths.append(float(signal[best]))
    if np.any(np.diff(chosen) <= 0.0):
        return None
    return chosen, strengths, {
        "grid_period_ms": grid_period, "grid_start_ms": grid_start,
        "first_pass_anchors_ms": [float(value) for value in anchors],
        "moved_slots": [i for i, (old, new) in enumerate(zip(anchors, chosen)) if abs(old - new) > 1e-6],
    }


def _supports_acoustic_rhythm_prior(recording: RecordingSpec | None) -> bool:
    if recording is None:
        return False
    # Measured benefit is currently limited to non-fortis VCV sequences.
    # CVVC conventions and Korean fortis VCV are kept on their established
    # priors until manual landmark Gold can justify a broader gate.
    if not _is_vcv_recording(recording):
        return False
    fortis = {"bb", "dd", "gg", "ss", "jj"}
    return not any(str(phone).casefold() in fortis for phone in recording.phones)


def _is_vcv_recording(recording: RecordingSpec | None) -> bool:
    if recording is None:
        return False
    format_hint = str(recording.metadata.get("format_hint", "")).strip().casefold()
    has_vcv_role = any(str(alias.role or "").casefold() == "vcv" for alias in recording.aliases)
    return format_hint == "vcv" or has_vcv_role


def _rhythm_warped_centers(
    states: Sequence[CompiledState],
    evidence: AcousticEvidence,
    expected_centers: np.ndarray,
    *,
    active_start: float,
    active_end: float,
    minimum_coverage: float = 0.60,
    tracking_mode: str = "uniform_cells",
) -> tuple[np.ndarray, bool, Mapping[str, object]]:
    """Warp repeated-state occurrence centers toward independent acoustic peaks.

    Only vowel-transition states are used as anchors. Their disjoint search
    cells preserve monotonic order, while a prominence and coverage gate keeps
    weak/noisy tracks on the established duration-only prior.
    """

    all_anchor_indices = [
        index for index, state in enumerate(states) if state.acoustic_role == "vowel_transition"
    ]
    phone_by_index: dict[int, str] = {}
    for state in states:
        phone_by_index.setdefault(int(state.phone_index), str(state.phone_id).casefold())
    glide_phones = {"w", "y", "j"}
    excluded_glide_anchor_phone_indices = [
        int(states[index].phone_index)
        for index in all_anchor_indices
        if phone_by_index.get(int(states[index].phone_index) - 1) in glide_phones
    ]
    anchor_indices = [
        index
        for index in all_anchor_indices
        if int(states[index].phone_index) not in excluded_glide_anchor_phone_indices
    ]
    if len(anchor_indices) < 3:
        return expected_centers, False, {
            "reason": "fewer_than_three_vowel_transition_anchors",
            "anchor_count": len(anchor_indices),
            "total_anchor_count": len(all_anchor_indices),
            "excluded_glide_anchor_phone_indices": excluded_glide_anchor_phone_indices,
        }
    required = (
        "vowel_boundary_likelihood",
        "transition",
        "periodicity_rise",
        "onset_strength",
    )
    if any(name not in evidence.tracks for name in required):
        return expected_centers, False, {
            "reason": "missing_acoustic_tracks",
            "anchor_count": len(anchor_indices),
            "total_anchor_count": len(all_anchor_indices),
            "excluded_glide_anchor_phone_indices": excluded_glide_anchor_phone_indices,
        }
    signal = (
        0.45 * np.asarray(evidence.tracks["vowel_boundary_likelihood"], dtype=np.float64)
        + 0.25 * np.asarray(evidence.tracks["transition"], dtype=np.float64)
        + 0.20 * np.asarray(evidence.tracks["periodicity_rise"], dtype=np.float64)
        + 0.10 * np.asarray(evidence.tracks["onset_strength"], dtype=np.float64)
    )
    times = np.asarray(evidence.frame_times_ms, dtype=np.float64)
    if len(signal) != len(times) or len(times) < 3:
        return expected_centers, False, {
            "reason": "invalid_acoustic_track_length",
            "anchor_count": len(anchor_indices),
            "total_anchor_count": len(all_anchor_indices),
            "excluded_glide_anchor_phone_indices": excluded_glide_anchor_phone_indices,
        }
    baseline = float(np.median(signal))
    high = float(np.quantile(signal, 0.72))
    minimum_peak = max(0.24, baseline + 0.06, high * 0.82)
    uniform = np.asarray([expected_centers[index] for index in anchor_indices], dtype=np.float64)
    cell_edges = np.empty(len(uniform) + 1, dtype=np.float64)
    cell_edges[0] = max(active_start, uniform[0] - 0.55 * (uniform[1] - uniform[0]))
    cell_edges[-1] = min(active_end, uniform[-1] + 0.55 * (uniform[-1] - uniform[-2]))
    cell_edges[1:-1] = 0.5 * (uniform[:-1] + uniform[1:])
    selected_uniform: list[float] = []
    selected_acoustic: list[float] = []
    selected_anchor_ordinals: list[int] = []
    selected_phone_indices: list[int] = []
    for index, center in enumerate(uniform):
        search_start = float(cell_edges[index])
        search_end = float(cell_edges[index + 1])
        if tracking_mode == "adaptive_interval" and selected_acoustic:
            nominal_step = float(np.median(np.diff(uniform)))
            tracked_step = nominal_step
            if len(selected_acoustic) >= 3:
                ordinal_steps = np.diff(np.asarray(selected_anchor_ordinals, dtype=np.float64))
                acoustic_steps = np.diff(np.asarray(selected_acoustic, dtype=np.float64))
                valid_steps = ordinal_steps > 0.0
                if np.any(valid_steps):
                    tracked_step = float(np.median(acoustic_steps[valid_steps] / ordinal_steps[valid_steps]))
                    tracked_step = float(
                        np.clip(tracked_step, 0.45 * nominal_step, 1.55 * nominal_step)
                    )
            ordinal_gap = index - selected_anchor_ordinals[-1]
            predicted = selected_acoustic[-1] + tracked_step * ordinal_gap
            half_width = max(80.0, 0.38 * tracked_step)
            search_start = max(active_start, predicted - half_width)
            search_end = min(active_end, predicted + half_width)
        mask = (times >= search_start) & (times < search_end)
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            continue
        best = int(candidates[int(np.argmax(signal[candidates]))])
        if float(signal[best]) < minimum_peak:
            continue
        selected_uniform.append(float(center))
        selected_acoustic.append(float(times[best]))
        selected_anchor_ordinals.append(index)
        selected_phone_indices.append(int(states[anchor_indices[index]].phone_index))
    coverage = len(selected_uniform) / max(1, len(uniform))
    if len(selected_uniform) < 3 or coverage < minimum_coverage:
        return expected_centers, False, {
            "reason": "insufficient_peak_coverage",
            "anchor_count": len(anchor_indices),
            "total_anchor_count": len(all_anchor_indices),
            "excluded_glide_anchor_phone_indices": excluded_glide_anchor_phone_indices,
            "selected_anchor_count": len(selected_uniform),
            "coverage": coverage,
            "required_coverage": minimum_coverage,
            "minimum_peak": minimum_peak,
            "signal_median": baseline,
            "signal_q72": high,
            "tracking_mode": tracking_mode,
            "selected_uniform_ms": selected_uniform,
            "selected_acoustic_ms": selected_acoustic,
            "selected_anchor_ordinals": selected_anchor_ordinals,
            "selected_phone_indices": selected_phone_indices,
        }
    source = np.asarray([active_start, *selected_uniform, active_end], dtype=np.float64)
    target = np.asarray([active_start, *selected_acoustic, active_end], dtype=np.float64)
    if np.any(np.diff(target) <= 0.0):
        return expected_centers, False, {
            "reason": "non_monotonic_acoustic_peaks",
            "anchor_count": len(anchor_indices),
            "total_anchor_count": len(all_anchor_indices),
            "excluded_glide_anchor_phone_indices": excluded_glide_anchor_phone_indices,
            "selected_anchor_count": len(selected_uniform),
            "coverage": coverage,
            "required_coverage": minimum_coverage,
            "minimum_peak": minimum_peak,
            "tracking_mode": tracking_mode,
            "selected_uniform_ms": selected_uniform,
            "selected_acoustic_ms": selected_acoustic,
            "selected_anchor_ordinals": selected_anchor_ordinals,
            "selected_phone_indices": selected_phone_indices,
        }
    warped = np.interp(expected_centers, source, target)
    return np.asarray(warped, dtype=np.float64), True, {
        "reason": "acoustic_rhythm_applied",
        "anchor_count": len(anchor_indices),
        "total_anchor_count": len(all_anchor_indices),
        "excluded_glide_anchor_phone_indices": excluded_glide_anchor_phone_indices,
        "selected_anchor_count": len(selected_uniform),
        "coverage": coverage,
        "required_coverage": minimum_coverage,
        "minimum_peak": minimum_peak,
        "signal_median": baseline,
        "signal_q72": high,
        "tracking_mode": tracking_mode,
        "selected_uniform_ms": selected_uniform,
        "selected_acoustic_ms": selected_acoustic,
        "selected_anchor_ordinals": selected_anchor_ordinals,
        "selected_phone_indices": selected_phone_indices,
    }


def _leading_breath_end_ms(evidence: AcousticEvidence) -> float:
    breath = evidence.tracks.get("breath_likelihood")
    voicing = evidence.tracks.get("voicing")
    if breath is None or voicing is None or len(voicing) < 3:
        return 0.0
    hop_ms = float(evidence.metadata["timebase"]["frame_shift_ms"])
    sustained_frames = max(3, int(round(30.0 / max(hop_ms, 1e-6))))
    voiced = np.asarray(voicing) >= 0.42
    run = np.convolve(voiced.astype(np.int16), np.ones(sustained_frames, dtype=np.int16), mode="same")
    indices = np.flatnonzero(run >= sustained_frames)
    if not len(indices):
        return 0.0
    voiced_onset_ms = float(evidence.frame_times_ms[indices[0]])
    breath_end_ms = voiced_onset_ms - 200.0
    if breath_end_ms <= 0.0:
        return 0.0
    leading = evidence.frame_times_ms < breath_end_ms
    if not np.any(leading) or float(np.max(np.asarray(breath)[leading])) < 0.35:
        return 0.0
    return breath_end_ms


def _build_hsmm_diagnostics(
    compiled: Sequence[CompiledState],
    specs: Sequence[HSMMStateSpec],
    decoded_states: Sequence[object],
    frame_scores: Mapping[str, np.ndarray],
    frame_times_ms: np.ndarray,
    *,
    decode_meta: Mapping[str, object] | None = None,
    explicit_gap_topology: bool = False,
    usable_start_ms: float | None = None,
    usable_end_ms: float | None = None,
) -> dict[str, object]:
    state_rows: list[dict[str, object]] = []
    if len(frame_times_ms) > 1:
        frame_shift_ms = float(np.median(np.diff(frame_times_ms)))
    else:
        frame_shift_ms = 5.0
    for state, spec, decoded in zip(compiled, specs, decoded_states):
        score = np.asarray(frame_scores[state.instance_id], dtype=np.float64)
        start_ms = float(getattr(decoded, "start_ms"))
        end_ms = float(getattr(decoded, "end_ms"))
        duration_ms = max(frame_shift_ms, end_ms - start_ms)
        duration_frames = max(1, int(round(duration_ms / max(frame_shift_ms, 1e-6))))
        segment_means = _fixed_duration_segment_means(score, duration_frames)
        candidate_starts = frame_times_ms[: len(segment_means)]
        selected_start = int(np.argmin(np.abs(candidate_starts - start_ms))) if len(candidate_starts) else 0
        selected_mean = float(segment_means[selected_start]) if len(segment_means) else 0.0
        local_mask = np.abs(candidate_starts - start_ms) <= LOCAL_MARGIN_SEARCH_WINDOW_MS
        local_values = segment_means[local_mask]
        if len(local_values):
            ordered = np.sort(local_values)
            best_local = float(ordered[-1])
            second_local = float(ordered[-2]) if len(ordered) > 1 else best_local
        else:
            best_local = selected_mean
            second_local = selected_mean
        if len(segment_means):
            global_index = int(np.argmax(segment_means))
            best_global = float(segment_means[global_index])
            global_start_ms = float(candidate_starts[global_index])
        else:
            best_global = selected_mean
            global_start_ms = start_ms
        local_margin = selected_mean - best_local
        global_margin = selected_mean - best_global
        sigma = max(10.0, float(spec.duration_sigma_ms))
        duration_z = (duration_ms - float(spec.mode_duration_ms)) / sigma
        wrong_occurrence = bool(
            global_margin < WRONG_OCCURRENCE_MARGIN_THRESHOLD
            and abs(global_start_ms - start_ms) > LOCAL_MARGIN_SEARCH_WINDOW_MS
        )
        # A stronger region anywhere in the file is not evidence of a wrong
        # phoneme. Keep the historical metric for gates, but identify whether
        # its candidate actually overlaps another instance of this phone/state.
        same_phone_competitors = []
        if wrong_occurrence:
            for other, other_decoded in zip(compiled, decoded_states):
                if (other.phone_index == state.phone_index or other.phone_id != state.phone_id
                        or other.state_id != state.state_id or other.acoustic_role != state.acoustic_role):
                    continue
                overlap = max(0.0, min(global_start_ms + duration_ms, other_decoded.end_ms)
                    - max(global_start_ms, other_decoded.start_ms))
                if overlap >= 0.5 * duration_ms:
                    same_phone_competitors.append(other.phone_index)
        state_rows.append(
            {
                "instance_id": state.instance_id,
                "phone_index": state.phone_index,
                "phone_id": state.phone_id,
                "occurrence_index": state.occurrence_index,
                "occurrence_count": state.occurrence_count,
                "state_id": state.state_id,
                "acoustic_role": state.acoustic_role,
                "selected_mean_score": selected_mean,
                "selected_vs_best_local_margin": local_margin,
                "selected_vs_second_local_margin": selected_mean - second_local,
                "selected_vs_global_best_margin": global_margin,
                "global_best_start_ms": global_start_ms,
                "wrong_occurrence_risk": wrong_occurrence,
                "global_acoustic_competitor_risk": wrong_occurrence,
                "same_phone_competitor_indices": same_phone_competitors,
                "repeated_phone_ambiguity": bool(same_phone_competitors),
                "occurrence_error_confirmed": False,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
                "duration_z_abs": abs(duration_z),
            }
        )
    local_margins = [float(item["selected_vs_best_local_margin"]) for item in state_rows]
    global_margins = [float(item["selected_vs_global_best_margin"]) for item in state_rows]
    duration_z = [float(item["duration_z_abs"]) for item in state_rows]
    gap_meta = dict(decode_meta or {})
    max_internal_gap_ms = float(gap_meta.get("max_internal_gap_observed_ms", 0.0) or 0.0)
    unmodeled_gap_threshold_ms = 30.0 if explicit_gap_topology else 80.0
    coverage = _lexical_coverage_diagnostics(
        compiled,
        decoded_states,
        usable_start_ms=usable_start_ms,
        usable_end_ms=usable_end_ms,
    )
    return {
        "schema_version": 1,
        "local_margin_search_window_ms": LOCAL_MARGIN_SEARCH_WINDOW_MS,
        "wrong_occurrence_margin_threshold": WRONG_OCCURRENCE_MARGIN_THRESHOLD,
        "min_selected_vs_best_local_margin": min(local_margins) if local_margins else 0.0,
        "min_selected_vs_global_best_margin": min(global_margins) if global_margins else 0.0,
        "wrong_occurrence_risk_state_count": sum(
            1 for item in state_rows if bool(item["wrong_occurrence_risk"])
        ),
        "repeated_phone_ambiguity_state_count": sum(
            bool(item["repeated_phone_ambiguity"]) for item in state_rows
        ),
        "occurrence_risk_semantics": "global_acoustic_competitor_not_confirmed_error",
        "max_duration_z_abs": max(duration_z) if duration_z else 0.0,
        "explicit_gap_topology": bool(explicit_gap_topology),
        "internal_gap_ms": float(gap_meta.get("internal_gap_ms", 0.0) or 0.0),
        "internal_gap_count": int(gap_meta.get("internal_gap_count", 0) or 0),
        "max_internal_gap_observed_ms": max_internal_gap_ms,
        "unmodeled_internal_gap_threshold_ms": unmodeled_gap_threshold_ms,
        "unmodeled_internal_gap": max_internal_gap_ms > unmodeled_gap_threshold_ms,
        "internal_gaps": list(gap_meta.get("internal_gaps", ())),
        "decoder_trailing_gap_ms": float(gap_meta.get("trailing_gap_ms", 0.0) or 0.0),
        **coverage,
        "states": state_rows,
    }


def _lexical_coverage_diagnostics(
    compiled: Sequence[CompiledState],
    decoded_states: Sequence[object],
    *,
    usable_start_ms: float | None,
    usable_end_ms: float | None,
) -> dict[str, object]:
    """Measure rendered lexical coverage independently of decoder metadata."""
    unavailable = {
        "lexical_coverage_available": False,
        "lexical_coverage_ratio": None,
        "lexical_coverage_threshold": MIN_LEXICAL_COVERAGE_RATIO,
        "lexical_start_ms": None,
        "lexical_end_ms": None,
        "lexical_leading_uncovered_ms": None,
        "lexical_trailing_uncovered_ms": None,
        "lexical_coverage_collapse": False,
    }
    lexical = [
        decoded
        for state, decoded in zip(compiled, decoded_states)
        if state.acoustic_role not in EXPLICIT_GAP_ROLES
    ]
    if not lexical or usable_start_ms is None or usable_end_ms is None:
        return unavailable
    usable_start = float(usable_start_ms)
    usable_end = float(usable_end_ms)
    active_span = usable_end - usable_start
    if active_span <= 0.0:
        return unavailable
    lexical_start = min(float(getattr(item, "start_ms")) for item in lexical)
    lexical_end = max(float(getattr(item, "end_ms")) for item in lexical)
    coverage_ratio = (lexical_end - lexical_start) / active_span
    return {
        "lexical_coverage_available": True,
        "lexical_coverage_ratio": float(coverage_ratio),
        "lexical_coverage_threshold": MIN_LEXICAL_COVERAGE_RATIO,
        "lexical_start_ms": lexical_start,
        "lexical_end_ms": lexical_end,
        "lexical_leading_uncovered_ms": max(0.0, lexical_start - usable_start),
        "lexical_trailing_uncovered_ms": max(0.0, usable_end - lexical_end),
        "lexical_coverage_collapse": bool(coverage_ratio < MIN_LEXICAL_COVERAGE_RATIO),
    }


def _fixed_duration_segment_means(score: np.ndarray, duration_frames: int) -> np.ndarray:
    if not len(score):
        return np.zeros(0, dtype=np.float64)
    width = min(max(1, int(duration_frames)), len(score))
    prefix = np.concatenate(([0.0], np.cumsum(score, dtype=np.float64)))
    return (prefix[width:] - prefix[:-width]) / float(width)


def _diagnostic_warnings(diagnostics: Mapping[str, object]) -> tuple[str, ...]:
    warnings: list[str] = []
    if int(diagnostics.get("wrong_occurrence_risk_state_count", 0)) > 0:
        warnings.append("hsmm_wrong_occurrence_risk")
    if float(diagnostics.get("min_selected_vs_best_local_margin", 0.0)) < -0.12:
        warnings.append("hsmm_low_local_margin")
    if float(diagnostics.get("max_duration_z_abs", 0.0)) >= 3.5:
        warnings.append("hsmm_duration_outlier")
    if bool(diagnostics.get("unmodeled_internal_gap", False)):
        warnings.append("hsmm_unmodeled_internal_gap")
    if bool(diagnostics.get("lexical_coverage_collapse", False)):
        warnings.append("hsmm_coverage_collapse")
    return tuple(warnings)


def _guide_agreement_warnings(
    phones: Sequence[PhoneSegment],
    recording: RecordingSpec,
) -> tuple[str, ...]:
    warnings: list[str] = []
    phone_by_index = {phone.phone_index: phone for phone in phones}
    for guide in recording.guide_anchors:
        if guide.phone_index is None or guide.confidence < 0.80:
            continue
        phone = phone_by_index.get(int(guide.phone_index))
        if phone is None:
            continue
        candidate_ms: float | None = None
        if guide.label == "phone_boundary":
            candidate_ms = phone.start_ms
        else:
            for state in phone.state_segments:
                for label, time_ms in _state_landmark_candidates(state):
                    if label == guide.label:
                        candidate_ms = time_ms
                        break
                if candidate_ms is not None:
                    break
        if candidate_ms is not None and abs(candidate_ms - guide.time_ms) > 250.0:
            warnings.append("guide_agreement_failed")
    return tuple(dict.fromkeys(warnings))


def _phone_feature_score(features: Mapping[str, object], tracks: Mapping[str, np.ndarray]) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    if features.get("syllabic") is True or features.get("vowel_like") is True:
        candidates.append(tracks["vowel_like"])
    if features.get("voiced") is True or features.get("periodic") is True:
        candidates.append(tracks["periodicity"])
    if features.get("voiced") is False:
        candidates.append(tracks["aperiodic"])
    if features.get("continuant") is False or features.get("manner") == "stop":
        candidates.append(tracks["consonant_like"])
    if features.get("manner") in {"fricative", "affricate"}:
        candidates.append(tracks["high_band_energy"])
    if not candidates:
        return None
    return np.mean(np.stack(candidates, axis=0), axis=0)


def _manual_occurrence_anchor_start_scores(
    compiled: Sequence[CompiledState],
    frame_times_ms: Sequence[float],
    anchors: Mapping[str, float] | None,
) -> dict[str, np.ndarray]:
    """Create bounded start bonuses from exact-recording review anchors.

    The reviews identify an acoustic occurrence, but are not a complete
    state-duration partition. A bounded bonus preserves the HSMM topology and
    duration constraints instead of forcing an impossible manual path.
    """

    if not anchors:
        return {}
    times = [float(value) for value in frame_times_ms]
    if not times:
        raise AlignmentError("manual occurrence anchors require acoustic frames")
    known = {state.instance_id for state in compiled}
    unknown = sorted(str(state_id) for state_id in anchors if str(state_id) not in known)
    if unknown:
        raise AlignmentError("manual occurrence anchors are absent from compiled states: " + ", ".join(unknown))
    scores: dict[str, np.ndarray] = {}
    for state_id, raw_start_ms in anchors.items():
        start_ms = float(raw_start_ms)
        if not np.isfinite(start_ms) or start_ms < times[0] - 50.0 or start_ms > times[-1] + 100.0:
            raise AlignmentError(f"{state_id}: manual occurrence anchor is outside acoustic frames")
        distance = np.abs(np.asarray(times, dtype=np.float64) - start_ms)
        # This is development-only, exact-recording review evidence. It is
        # stronger than the learned-ranker prior so a short reviewed consonant
        # is not discarded for its weaker local emission. It remains a bounded
        # bonus, not a forced frame/state assignment.
        scores[str(state_id)] = MANUAL_OCCURRENCE_ANCHOR_START_BONUS * np.exp(
            -0.5 * np.square(distance / 40.0)
        )
    return scores


def _state_spec(
    state: CompiledState,
    duration_constraints: Mapping[tuple[str, str], PhoneStateDurationConstraint] | None = None,
) -> HSMMStateSpec:
    constraint = (duration_constraints or {}).get((state.phone_id, state.state_id))
    if constraint is not None and state.acoustic_role not in constraint.acoustic_roles:
        raise AlignmentError(
            f"duration constraint acoustic role mismatch: {state.phone_id}.{state.state_id}/{state.acoustic_role}"
        )
    min_ms = constraint.min_duration_ms if constraint is not None else state.min_ms
    max_ms = constraint.max_duration_ms if constraint is not None else state.max_ms
    mode = (
        constraint.mode_duration_ms
        if constraint is not None
        else state.mode_ms if state.mode_ms is not None else min_ms + 0.5 * (max_ms - min_ms)
    )
    sigma = max(10.0, (max_ms - min_ms) / 2.5)
    over_scale = 0.30 if state.acoustic_role in {"vowel_stable", "vowel_nucleus", "periodic", "silence"} else 1.0
    return HSMMStateSpec(
        state_id=state.instance_id,
        state_type=state.acoustic_role,
        min_duration_ms=min_ms,
        max_duration_ms=max_ms,
        mode_duration_ms=mode,
        duration_sigma_ms=sigma,
        over_duration_penalty_scale=over_scale,
    )


def _segment_confidence(evidence: AcousticEvidence, scores: np.ndarray, start_ms: float, end_ms: float) -> float:
    selected = (evidence.frame_times_ms >= start_ms) & (evidence.frame_times_ms < end_ms)
    if not np.any(selected):
        return 0.0
    mean_score = float(np.mean(scores[selected]))
    baseline = float(np.median(scores))
    return float(np.clip(0.30 + 0.55 * mean_score + 0.25 * max(0.0, mean_score - baseline), 0.0, 1.0))


def _phone_segments(recording: RecordingSpec, states: Sequence[StateSegment]) -> tuple[PhoneSegment, ...]:
    grouped: dict[int, list[StateSegment]] = {index: [] for index in range(len(recording.phones))}
    for state in states:
        grouped[state.phone_index].append(state)
    result: list[PhoneSegment] = []
    for phone_index, phone_id in enumerate(recording.phones):
        phone_states = tuple(grouped[phone_index])
        if not phone_states:
            raise AlignmentError(f"decoded state가 없는 phone occurrence입니다: {phone_index}:{phone_id}")
        result.append(
            PhoneSegment(
                phone_index=phone_index,
                phone_id=phone_id,
                start_ms=phone_states[0].start_ms,
                end_ms=phone_states[-1].end_ms,
                confidence=min(state.confidence for state in phone_states),
                state_segments=phone_states,
            )
        )
    return tuple(result)


def _build_landmarks(phones: Sequence[PhoneSegment], evidence: AcousticEvidence) -> tuple[Landmark, ...]:
    landmarks: list[Landmark] = []
    for phone in phones:
        landmarks.append(
            _refine_landmark(
                Landmark("phone_boundary", phone.start_ms, phone.phone_index, phone.confidence, "hsmm_raw", phone.start_ms),
                evidence,
                phone,
            )
        )
        for state in phone.state_segments:
            for label, time_ms in _state_landmark_candidates(state):
                landmarks.append(
                    _refine_landmark(
                        Landmark(label, time_ms, phone.phone_index, state.confidence, "hsmm_raw", time_ms),
                        evidence,
                        phone,
                    )
                )
    unique: dict[tuple[str, int | None, int], Landmark] = {}
    for landmark in landmarks:
        key = (landmark.label, landmark.phone_index, int(round(landmark.time_ms * 10.0)))
        existing = unique.get(key)
        if existing is None or landmark.confidence > existing.confidence:
            unique[key] = landmark
    return tuple(sorted(unique.values(), key=lambda item: (item.time_ms, item.phone_index if item.phone_index is not None else -1, item.label)))


def _state_landmark_candidates(state: StateSegment) -> tuple[tuple[str, float], ...]:
    role = state.acoustic_role
    if role == "closure":
        return (("closure_start", state.start_ms),)
    if role in {"burst", "release"}:
        return (("release", state.start_ms),)
    if role == "frication":
        return (("frication_start", state.start_ms), ("frication_end", state.end_ms))
    if role in {"voicing_onset", "irregular_voicing_onset"}:
        return (("voicing_onset", state.start_ms),)
    if role in {"vowel_transition", "transition"}:
        return (("vowel_transition", state.start_ms),)
    if role in {"vowel_stable", "vowel_nucleus", "periodic", "irregular_periodic"}:
        return (
            ("vowel_stable_start", state.start_ms),
            ("nucleus", 0.5 * (state.start_ms + state.end_ms)),
            ("vowel_stable_end", state.end_ms),
        )
    if role == "breath":
        return (("breath_start", state.start_ms), ("breath_end", state.end_ms))
    if role == "silence":
        return (("silence_start", state.start_ms), ("silence_end", state.end_ms))
    return ()


_REFINER_TRACKS = {
    "phone_boundary": "boundary",
    "closure_start": "boundary",
    "release": "flux",
    "frication_start": "high_band_energy",
    "voicing_onset": "periodicity_rise",
    "vowel_transition": "transition",
    "vowel_stable_start": "vowel_like",
    "nucleus": "vowel_like",
}


def _refine_landmark(
    landmark: Landmark,
    evidence: AcousticEvidence,
    phone: PhoneSegment,
) -> Landmark:
    track_name = _REFINER_TRACKS.get(landmark.label)
    if track_name is None:
        return landmark
    times = evidence.frame_times_ms
    track = evidence.tracks[track_name]
    radius = 80.0 if landmark.label != "phone_boundary" else 110.0
    # Partition the occurrence by its raw events. Independent peak searches
    # must not swap closure/release/transition/stable events or merge them onto
    # one strong peak belonging to another state.
    raw_events = sorted({phone.start_ms, phone.end_ms, *(
        time_ms for state in phone.state_segments
        for _label, time_ms in _state_landmark_candidates(state)
    )})
    previous = max((t for t in raw_events if t < landmark.time_ms), default=phone.start_ms)
    following = min((t for t in raw_events if t > landmark.time_ms), default=phone.end_ms)
    lower = max(phone.start_ms, landmark.time_ms - radius, (previous + landmark.time_ms) / 2.0)
    upper = min(phone.end_ms, landmark.time_ms + radius, (following + landmark.time_ms) / 2.0)
    indices = np.flatnonzero(
        (times >= lower) & (times <= upper) & np.isfinite(track)
    )
    if len(indices) < 3:
        return landmark
    peak_indices = indices[track[indices] == np.max(track[indices])]
    best_index = int(peak_indices[np.argmin(np.abs(times[peak_indices] - landmark.time_ms))])
    best_score = float(track[best_index])
    raw_index = int(np.argmin(np.abs(times - landmark.time_ms)))
    raw_score = float(track[raw_index])
    baseline = float(np.median(track[indices]))
    margin = best_score - max(raw_score, baseline)
    refined_time = float(times[best_index])
    if margin < 0.06 or abs(refined_time - landmark.time_ms) > radius:
        return landmark
    confidence = float(np.clip(0.45 + 0.7 * (best_score - baseline), 0.0, 1.0))
    return Landmark(
        label=landmark.label,
        time_ms=refined_time,
        phone_index=landmark.phone_index,
        confidence=min(landmark.confidence, confidence),
        source="hsmm_local_refined",
        raw_time_ms=landmark.time_ms,
        refinement_reason=f"{track_name}_local_peak",
    )


def _apply_guide_anchors(
    landmarks: Sequence[Landmark], recording: RecordingSpec
) -> tuple[Landmark, ...]:
    if not recording.guide_anchors:
        return tuple(landmarks)
    result = list(landmarks)
    for guide in recording.guide_anchors:
        result = [
            item
            for item in result
            if not (item.label == guide.label and item.phone_index == guide.phone_index)
        ]
        result.append(
            Landmark(
                label=guide.label,
                time_ms=guide.time_ms,
                phone_index=guide.phone_index,
                confidence=guide.confidence,
                source=guide.source,
                raw_time_ms=None,
                refinement_reason="guide_anchor_override",
            )
        )
    return tuple(sorted(result, key=lambda item: (item.time_ms, item.phone_index if item.phone_index is not None else -1, item.label)))


def _recording_sha256(recording: RecordingSpec) -> str:
    payload = {
        "wav": recording.wav,
        "phones": list(recording.phones),
        "aliases": [
            {
                "alias": alias.alias,
                "phone_indices": list(alias.phone_indices),
                "template": alias.template,
                "role": alias.role,
                "overrides": dict(alias.overrides),
            }
            for alias in recording.aliases
        ],
        "transcript_source": recording.transcript_source,
        "label_source": recording.label_source,
        "metadata": dict(recording.metadata),
        "guide_anchors": [anchor.__dict__ for anchor in recording.guide_anchors],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _calibrate_timeline(project: ProjectSpec, timeline: LandmarkTimeline) -> LandmarkTimeline:
    if not project.calibration.enabled or not project.calibration.profile:
        return timeline
    profile_path = (project.root / project.calibration.profile).resolve()
    profile = load_confidence_profile(profile_path)
    all_states = [state for phone in timeline.phone_segments for state in phone.state_segments]
    min_state_confidence = min((state.confidence for state in all_states), default=timeline.confidence)
    duration_z_abs = 0.0
    for phone, definition_id in zip(timeline.phone_segments, (phone.phone_id for phone in timeline.phone_segments)):
        definition = project.inventory[definition_id]
        definitions = {state.id: state for state in definition.states}
        for segment in phone.state_segments:
            state_definition = definitions.get(segment.state_id)
            if state_definition is None:
                continue
            mode = state_definition.mode_ms
            if mode is None:
                mode = state_definition.min_ms + 0.5 * (state_definition.max_ms - state_definition.min_ms)
            sigma = max(10.0, (state_definition.max_ms - state_definition.min_ms) / 2.5)
            duration_z_abs = max(duration_z_abs, abs((segment.end_ms - segment.start_ms - mode) / sigma))
    features = runtime_confidence_features(
        raw_confidence=timeline.confidence,
        warning_count=len(timeline.warnings),
        duration_z_abs=duration_z_abs,
        boundary_margin=min_state_confidence - 0.5,
    )
    risks = predict_risks(profile, features)
    return LandmarkTimeline(
        **{
            **timeline.__dict__,
            "provenance": {
                **dict(timeline.provenance),
                "confidence_calibrated": True,
                "confidence_profile": str(profile_path),
                "confidence_profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                "confidence_features": features,
                "risk_probabilities": risks,
            },
        }
    )
