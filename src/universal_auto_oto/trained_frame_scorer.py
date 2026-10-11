"""Hash-bound, opt-in runtime for reviewed sub-phone score artifacts.

The artifact is intentionally a small, inspectable linear scorer.  It is not
an alternate decoder: it produces only the frame emissions and optional
transition-start bonuses consumed by the existing HSMM.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .acoustic_evidence import AcousticEvidence
from .audio import file_sha256
from .domain import CompiledState, RecordingSpec
from .frame_scorer import FrameScoreProvider, FrameScoreProviderError, FrameScoreResult
from .project import ProjectSpec


TRAINED_SUBPHONE_SCORER_SCHEMA_VERSION = 1
TRAINED_SUBPHONE_SCORER_KIND = "trained_subphone_v1"
_MANUAL_LABEL_SOURCES = frozenset({"manual_gold", "manual_confirmed_subphone_transition"})
_SUPPORTED_STATE_CONTEXT_FEATURES = frozenset(
    {
        "phone_occurrence_ordinal",
        "phone_occurrence_count",
        "normalized_phone_position",
        "normalized_frame_position",
        "is_repeated_phone",
        "is_transition_role",
    }
)


class TrainedSubphoneScorerError(FrameScoreProviderError):
    """An artifact is malformed, unbound, or cannot safely run."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TrainedSubphoneScorerArtifact:
    path: Path
    sha256: str
    feature_backend: str
    feature_names: tuple[str, ...]
    state_context_feature_names: tuple[str, ...]
    normalization_mean: np.ndarray
    normalization_scale: np.ndarray
    emission_heads: Mapping[str, np.ndarray]
    transition_start_heads: Mapping[str, np.ndarray]
    minimum_mean_confidence: float
    transition_start_roles: frozenset[str] | None
    transition_start_bonus_scale: float
    training: Mapping[str, object]


@dataclass(frozen=True)
class TrainedSubphoneFrameScoreProvider:
    """Execute one validated artifact without changing HSMM topology."""

    artifact: TrainedSubphoneScorerArtifact

    def score(
        self,
        *,
        states: Sequence[CompiledState],
        project: ProjectSpec,
        recording: RecordingSpec | None,
        evidence: AcousticEvidence,
    ) -> FrameScoreResult:
        del recording
        if evidence.metadata.get("feature_backend") != self.artifact.feature_backend:
            raise TrainedSubphoneScorerError(
                "feature_backend_mismatch",
                "trained sub-phone scorer feature backend does not match acoustic evidence",
            )
        if project.inventory_sha256 not in _inventory_allowlist(self.artifact.training):
            raise TrainedSubphoneScorerError(
                "inventory_hash_mismatch",
                "trained sub-phone scorer inventory hash does not match project",
            )
        acoustic_features = _feature_matrix(evidence, self.artifact.feature_names)
        emissions: dict[str, np.ndarray] = {}
        transitions: dict[str, np.ndarray] = {}
        confidence_values: list[float] = []
        maximum_phone_index = max((state.phone_index for state in states), default=0)
        for state in states:
            context = _state_context_matrix(
                state,
                evidence,
                self.artifact.state_context_feature_names,
                maximum_phone_index=maximum_phone_index,
            )
            features = np.column_stack((acoustic_features, context))
            normalized = (features - self.artifact.normalization_mean) / self.artifact.normalization_scale
            if self.artifact.emission_heads:
                emission = _score_head(self.artifact.emission_heads, state.acoustic_role, normalized)
                emissions[state.instance_id] = emission
                confidence_values.append(float(np.mean(np.abs(emission - 0.5) * 2.0)))
            if self.artifact.transition_start_heads and (
                self.artifact.transition_start_roles is None
                or state.acoustic_role in self.artifact.transition_start_roles
            ):
                transition_score = _score_head(
                    self.artifact.transition_start_heads,
                    state.acoustic_role,
                    normalized,
                )
                transitions[state.instance_id] = transition_score * self.artifact.transition_start_bonus_scale
                if not self.artifact.emission_heads:
                    confidence_values.append(float(np.mean(np.abs(transition_score - 0.5) * 2.0)))
        mean_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
        if mean_confidence < self.artifact.minimum_mean_confidence:
            raise TrainedSubphoneScorerError(
                "low_artifact_confidence",
                "trained sub-phone scorer confidence is below its artifact floor",
            )
        return FrameScoreResult(
            emission_scores=emissions,
            transition_start_scores=transitions,
            diagnostics={
                "emission_state_count": len(emissions),
                "transition_start_score_state_count": len(transitions),
                "mean_artifact_confidence": mean_confidence,
            },
            provenance={
                "provider": TRAINED_SUBPHONE_SCORER_KIND,
                "fallback": False,
                "artifact_path": str(self.artifact.path),
                "artifact_sha256": self.artifact.sha256,
                "artifact_schema_version": TRAINED_SUBPHONE_SCORER_SCHEMA_VERSION,
                "feature_backend": self.artifact.feature_backend,
                "feature_schema_sha256": _feature_schema_sha256(
                    self.artifact.feature_names,
                    self.artifact.state_context_feature_names,
                ),
                "fit_split": self.artifact.training["fit_split"],
                "validation_labels_used": self.artifact.training["validation_labels_used"],
                "locked_test_labels_used": self.artifact.training["locked_test_labels_used"],
                "transition_start_enabled": bool(self.artifact.transition_start_heads),
                "transition_start_bonus_scale": self.artifact.transition_start_bonus_scale,
                "emission_mode": "learned" if self.artifact.emission_heads else "heuristic_passthrough",
            },
        )


def load_trained_subphone_scorer(
    path: Path,
    *,
    project: ProjectSpec,
    expected_sha256: str,
) -> TrainedSubphoneFrameScoreProvider:
    """Load a project-local, hash-bound scorer artifact or raise with a code."""

    artifact_path = path.resolve()
    try:
        artifact_path.relative_to(project.root.resolve())
    except ValueError as exc:
        raise TrainedSubphoneScorerError(
            "artifact_outside_project",
            "trained sub-phone scorer artifact must stay inside the project root",
        ) from exc
    if not artifact_path.is_file():
        raise TrainedSubphoneScorerError("artifact_missing", f"trained sub-phone scorer artifact is missing: {artifact_path}")
    actual_sha256 = file_sha256(artifact_path)
    if actual_sha256 != expected_sha256:
        raise TrainedSubphoneScorerError(
            "artifact_hash_mismatch",
            "trained sub-phone scorer SHA-256 does not match project configuration",
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainedSubphoneScorerError("artifact_json_invalid", "trained sub-phone scorer artifact is not valid JSON") from exc
    artifact = _parse_artifact(payload, path=artifact_path, sha256=actual_sha256, project=project)
    return TrainedSubphoneFrameScoreProvider(artifact)


def _parse_artifact(
    payload: object,
    *,
    path: Path,
    sha256: str,
    project: ProjectSpec,
) -> TrainedSubphoneScorerArtifact:
    if not isinstance(payload, Mapping):
        raise TrainedSubphoneScorerError("artifact_root_invalid", "trained sub-phone scorer root must be an object")
    if payload.get("schema_version") != TRAINED_SUBPHONE_SCORER_SCHEMA_VERSION:
        raise TrainedSubphoneScorerError("artifact_schema_invalid", "unsupported trained sub-phone scorer schema")
    if payload.get("model_kind") != TRAINED_SUBPHONE_SCORER_KIND:
        raise TrainedSubphoneScorerError("artifact_kind_invalid", "trained sub-phone scorer model_kind mismatch")
    feature_backend = _text(payload.get("feature_backend"), "feature_backend")
    if feature_backend != project.alignment.feature_backend:
        raise TrainedSubphoneScorerError(
            "feature_backend_mismatch",
            "trained sub-phone scorer feature backend does not match project",
        )
    feature_names = _text_tuple(payload.get("feature_names"), "feature_names")
    state_context_feature_names = _text_tuple(
        payload.get("state_context_feature_names"), "state_context_feature_names"
    )
    unknown_context = sorted(set(state_context_feature_names) - _SUPPORTED_STATE_CONTEXT_FEATURES)
    if unknown_context:
        raise TrainedSubphoneScorerError(
            "state_context_schema_invalid",
            f"trained sub-phone scorer has unknown state context features: {unknown_context}",
        )
    if _text(payload.get("feature_schema_sha256"), "feature_schema_sha256") != _feature_schema_sha256(
        feature_names, state_context_feature_names
    ):
        raise TrainedSubphoneScorerError("feature_schema_hash_mismatch", "trained sub-phone scorer feature schema hash mismatch")
    training = _mapping(payload.get("training"), "training")
    _validate_training(training, project=project)
    normalization = _mapping(payload.get("normalization"), "normalization")
    feature_count = len(feature_names) + len(state_context_feature_names)
    mean = _vector(normalization.get("mean"), feature_count, "normalization.mean")
    scale = _vector(normalization.get("scale"), feature_count, "normalization.scale")
    if np.any(scale <= 0.0):
        raise TrainedSubphoneScorerError("normalization_invalid", "trained sub-phone scorer normalization scales must be positive")
    emission_heads = _heads(
        payload.get("emission_heads", {}),
        feature_count=feature_count,
        name="emission_heads",
        required=False,
    )
    transition_heads = _heads(
        payload.get("transition_start_heads", {}),
        feature_count=feature_count,
        name="transition_start_heads",
        required=False,
    )
    if transition_heads:
        transition = _mapping(training.get("transition_labels"), "training.transition_labels")
        if transition.get("source") != "manual_confirmed_subphone_transition":
            raise TrainedSubphoneScorerError(
                "transition_label_source_invalid",
                "transition-start head requires confirmed manual sub-phone markers",
            )
        if _positive_int(transition.get("marker_count"), "training.transition_labels.marker_count") < 1:
            raise TrainedSubphoneScorerError(
                "transition_label_count_invalid",
                "transition-start head requires at least one confirmed marker",
            )
    if not emission_heads and not transition_heads:
        raise TrainedSubphoneScorerError(
            "artifact_heads_empty", "trained sub-phone scorer requires an emission or transition-start head"
        )
    runtime = _mapping(payload.get("runtime"), "runtime")
    transition_start_roles = _optional_text_set(runtime.get("transition_start_roles"), "runtime.transition_start_roles")
    if transition_start_roles is not None and not transition_heads:
        raise TrainedSubphoneScorerError(
            "transition_role_scope_invalid", "transition_start_roles requires a transition-start head"
        )
    transition_start_bonus_scale = _bounded_bonus_scale(
        runtime.get("transition_start_bonus_scale", 1.0), "runtime.transition_start_bonus_scale"
    )
    minimum_mean_confidence = _probability(
        runtime.get("minimum_mean_confidence", 0.0), "runtime.minimum_mean_confidence"
    )
    return TrainedSubphoneScorerArtifact(
        path=path,
        sha256=sha256,
        feature_backend=feature_backend,
        feature_names=feature_names,
        state_context_feature_names=state_context_feature_names,
        normalization_mean=mean,
        normalization_scale=scale,
        emission_heads=emission_heads,
        transition_start_heads=transition_heads,
        minimum_mean_confidence=minimum_mean_confidence,
        transition_start_roles=transition_start_roles,
        transition_start_bonus_scale=transition_start_bonus_scale,
        training=dict(training),
    )


@dataclass(frozen=True)
class HeuristicEmissionTransitionOverlayProvider:
    """Keep parity emissions while adding bounded reviewed transition evidence.

    A transition-only artifact has no defensible state-span labels for replacing
    the established heuristic emissions.  The overlay therefore leaves those
    emissions byte-for-byte in the heuristic provider's responsibility and
    contributes only optional HSMM start bonuses.
    """

    heuristic_provider: FrameScoreProvider
    trained_provider: TrainedSubphoneFrameScoreProvider

    def score(
        self,
        *,
        states: Sequence[CompiledState],
        project: ProjectSpec,
        recording: RecordingSpec | None,
        evidence: AcousticEvidence,
    ) -> FrameScoreResult:
        heuristic = self.heuristic_provider.score(
            states=states, project=project, recording=recording, evidence=evidence
        )
        trained = self.trained_provider.score(
            states=states, project=project, recording=recording, evidence=evidence
        )
        if trained.emission_scores:
            return trained
        return FrameScoreResult(
            emission_scores=dict(heuristic.emission_scores),
            transition_start_scores=dict(trained.transition_start_scores),
            warnings=tuple(dict.fromkeys((*heuristic.warnings, *trained.warnings))),
            diagnostics={
                **dict(heuristic.diagnostics),
                **dict(trained.diagnostics),
                "emission_mode": "heuristic_passthrough",
            },
            provenance={
                **dict(trained.provenance),
                "emission_provider": dict(heuristic.provenance).get("provider"),
                "emission_mode": "heuristic_passthrough",
                "fallback": False,
            },
        )


def _validate_training(training: Mapping[str, object], *, project: ProjectSpec) -> None:
    if training.get("fit_split") != "train":
        raise TrainedSubphoneScorerError("training_split_invalid", "trained sub-phone scorer must be fitted on train only")
    if training.get("validation_labels_used") is not False or training.get("locked_test_labels_used") is not False:
        raise TrainedSubphoneScorerError(
            "label_leakage",
            "trained sub-phone scorer artifact records validation or locked-test label use",
        )
    if project.inventory_sha256 not in _inventory_allowlist(training):
        raise TrainedSubphoneScorerError("inventory_hash_mismatch", "trained sub-phone scorer inventory hash mismatch")
    _required_sha256(training, "source_table_sha256")
    labels = _text_tuple(training.get("label_sources"), "training.label_sources")
    if "manual_gold" not in labels or any(label not in _MANUAL_LABEL_SOURCES for label in labels):
        raise TrainedSubphoneScorerError(
            "label_source_invalid",
            "trained sub-phone scorer labels must be reviewed manual labels",
        )


def _feature_matrix(evidence: AcousticEvidence, feature_names: Sequence[str]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    frame_count = len(evidence.frame_times_ms)
    for name in feature_names:
        values = evidence.tracks.get(name)
        if values is None:
            raise TrainedSubphoneScorerError("feature_missing", f"trained sub-phone scorer requires missing track: {name}")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (frame_count,) or not np.all(np.isfinite(array)):
            raise TrainedSubphoneScorerError("feature_invalid", f"trained sub-phone scorer track is invalid: {name}")
        arrays.append(array)
    return np.column_stack(arrays)


def _state_context_matrix(
    state: CompiledState,
    evidence: AcousticEvidence,
    names: Sequence[str],
    *,
    maximum_phone_index: int,
) -> np.ndarray:
    frame_count = len(evidence.frame_times_ms)
    if not names:
        return np.empty((frame_count, 0), dtype=np.float64)
    duration = max(float(evidence.duration_ms), 1.0)
    values: list[np.ndarray] = []
    for name in names:
        if name == "phone_occurrence_ordinal":
            denominator = max(1, state.occurrence_count - 1)
            value = float(state.occurrence_index) / denominator
        elif name == "phone_occurrence_count":
            value = min(float(state.occurrence_count), 8.0) / 8.0
        elif name == "normalized_phone_position":
            value = float(state.phone_index) / max(1, maximum_phone_index)
        elif name == "normalized_frame_position":
            values.append(np.clip(np.asarray(evidence.frame_times_ms, dtype=np.float64) / duration, 0.0, 1.0))
            continue
        elif name == "is_repeated_phone":
            value = float(state.occurrence_count > 1)
        elif name == "is_transition_role":
            value = float("transition" in state.acoustic_role)
        else:  # guarded at artifact load; retain a runtime fail-closed check.
            raise TrainedSubphoneScorerError("state_context_schema_invalid", f"unknown state context feature: {name}")
        values.append(np.full(frame_count, value, dtype=np.float64))
    return np.column_stack(values)


def _score_head(heads: Mapping[str, np.ndarray], role: str, normalized: np.ndarray) -> np.ndarray:
    coefficients = heads.get(role, heads["other"])
    logits = coefficients[0] + normalized @ coefficients[1:]
    return _sigmoid(logits)


def _heads(
    value: object,
    *,
    feature_count: int,
    name: str,
    required: bool = True,
) -> Mapping[str, np.ndarray]:
    heads = _mapping(value, name)
    if not heads and not required:
        return {}
    if "other" not in heads:
        raise TrainedSubphoneScorerError("head_fallback_missing", f"{name}.other is required")
    result: dict[str, np.ndarray] = {}
    for role, raw in heads.items():
        if not isinstance(role, str) or not role.strip():
            raise TrainedSubphoneScorerError("head_role_invalid", f"{name} role is invalid")
        coefficients = _vector(raw, feature_count + 1, f"{name}.{role}")
        if np.any(np.abs(coefficients) > 100.0):
            raise TrainedSubphoneScorerError("head_coefficient_invalid", f"{name}.{role} coefficients are outside safety bounds")
        result[role] = coefficients
    return result


def _feature_schema_sha256(
    feature_names: Sequence[str],
    state_context_feature_names: Sequence[str] = (),
) -> str:
    text = json.dumps(
        {
            "frame_features": list(feature_names),
            "state_context_features": list(state_context_feature_names),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float64, copy=False)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrainedSubphoneScorerError("artifact_mapping_invalid", f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainedSubphoneScorerError("artifact_text_invalid", f"{name} must be a non-empty string")
    return value.strip()


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TrainedSubphoneScorerError("artifact_list_invalid", f"{name} must be a non-empty list")
    items = tuple(_text(item, name) for item in value)
    if len(set(items)) != len(items):
        raise TrainedSubphoneScorerError("artifact_list_duplicate", f"{name} has duplicate values")
    return items


def _optional_text_set(value: object, name: str) -> frozenset[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TrainedSubphoneScorerError("artifact_list_invalid", f"{name} must be a non-empty list when present")
    values = frozenset(_text(item, name) for item in value)
    if len(values) != len(value):
        raise TrainedSubphoneScorerError("artifact_list_duplicate", f"{name} has duplicate values")
    return values


def _vector(value: object, length: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise TrainedSubphoneScorerError("artifact_vector_invalid", f"{name} has an invalid length")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TrainedSubphoneScorerError("artifact_vector_invalid", f"{name} must be numeric") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise TrainedSubphoneScorerError("artifact_vector_invalid", f"{name} must be finite")
    return result


def _required_sha256(values: Mapping[str, object], name: str) -> str:
    value = _text(values.get(name), name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.casefold()):
        raise TrainedSubphoneScorerError("artifact_hash_invalid", f"{name} must be a SHA-256 hex digest")
    return value.casefold()


def _inventory_allowlist(training: Mapping[str, object]) -> frozenset[str]:
    """Read a strict single hash (v1) or an explicit multi-project allowlist."""

    raw = training.get("inventory_sha256_allowlist")
    if raw is None:
        return frozenset({_required_sha256(training, "inventory_sha256")})
    if not isinstance(raw, list) or not raw:
        raise TrainedSubphoneScorerError("artifact_list_invalid", "training.inventory_sha256_allowlist must be non-empty")
    values = frozenset(_required_sha256({"value": item}, "value") for item in raw)
    if len(values) != len(raw):
        raise TrainedSubphoneScorerError("artifact_list_duplicate", "training.inventory_sha256_allowlist has duplicate values")
    return values


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrainedSubphoneScorerError("artifact_probability_invalid", f"{name} must be finite")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise TrainedSubphoneScorerError("artifact_probability_invalid", f"{name} must be between 0 and 1")
    return result


def _bounded_bonus_scale(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrainedSubphoneScorerError("artifact_bonus_invalid", f"{name} must be finite")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise TrainedSubphoneScorerError("artifact_bonus_invalid", f"{name} must be between 0 and 1")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainedSubphoneScorerError("artifact_count_invalid", f"{name} must be a positive integer")
    return value


__all__ = [
    "TRAINED_SUBPHONE_SCORER_KIND",
    "TRAINED_SUBPHONE_SCORER_SCHEMA_VERSION",
    "TrainedSubphoneFrameScoreProvider",
    "HeuristicEmissionTransitionOverlayProvider",
    "TrainedSubphoneScorerError",
    "load_trained_subphone_scorer",
]
