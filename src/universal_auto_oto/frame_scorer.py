"""Frame-emission score providers for the known-sequence HSMM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import numpy as np

from .acoustic_evidence import AcousticEvidence
from .ctc_prior import CTCPrior
from .domain import CompiledState, RecordingSpec
from .project import ProjectSpec


class FrameScoreProviderError(ValueError):
    """Raised when a score provider cannot honour the HSMM score contract."""


@dataclass(frozen=True)
class FrameScoreResult:
    """Provider output consumed by the existing HSMM decoder.

    Emission values are bounded probabilities aligned to
    evidence.frame_times_ms. Transition starts are optional bonuses for the
    existing decoder; they cannot alter state topology or set hard durations.
    """

    emission_scores: Mapping[str, np.ndarray]
    transition_start_scores: Mapping[str, np.ndarray] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)


class FrameScoreProvider(Protocol):
    """Stable internal contract for score-generation backends."""

    def score(
        self,
        *,
        states: Sequence[CompiledState],
        project: ProjectSpec,
        recording: RecordingSpec | None,
        evidence: AcousticEvidence,
    ) -> FrameScoreResult: ...


@dataclass(frozen=True)
class HeuristicFrameScoreProvider:
    """Parity implementation of the legacy YAML-weighted track scorer."""

    profile_id: str
    profile_sha256: str
    role_weights: Mapping[str, Mapping[str, float]]
    ctc_prior: CTCPrior | None = None
    ctc_prior_weight: float = 0.0
    occurrence_priors: Mapping[str, np.ndarray] = field(default_factory=dict)
    acoustic_rhythm_used: bool = False
    occurrence_prior_weight_scale: float = 1.0
    breath_end_ms: float = 0.0

    def score(
        self,
        *,
        states: Sequence[CompiledState],
        project: ProjectSpec,
        recording: RecordingSpec | None,
        evidence: AcousticEvidence,
    ) -> FrameScoreResult:
        del recording
        result: dict[str, np.ndarray] = {}
        warnings: list[str] = []
        expected_frames = len(evidence.frame_times_ms)
        for state in states:
            weights = self.role_weights.get(state.acoustic_role)
            if weights is None:
                weights = self.role_weights.get("other")
                if weights is None:
                    raise FrameScoreProviderError("heuristic profile does not define role_weights.other")
                warnings.append(f"unknown_acoustic_role:{state.acoustic_role}")
            score = np.zeros(expected_frames, dtype=np.float64)
            for track, weight in weights.items():
                values = evidence.tracks.get(track)
                if values is None:
                    raise FrameScoreProviderError(f"profile requires missing acoustic track: {track}")
                values = np.asarray(values, dtype=np.float64)
                if len(values) != expected_frames or not np.all(np.isfinite(values)):
                    raise FrameScoreProviderError(f"invalid acoustic track: {track}")
                score += float(weight) * values
            phone = project.inventory[state.phone_id]
            feature_score = _phone_feature_score(phone.features, evidence.tracks)
            if feature_score is not None:
                score = 0.90 * score + 0.10 * feature_score
            if self.ctc_prior is not None:
                if state.phone_index >= len(self.ctc_prior.phone_posteriors):
                    raise FrameScoreProviderError("CTC prior phone posterior does not match compiled sequence")
                score = (
                    (1.0 - self.ctc_prior_weight) * score
                    + self.ctc_prior_weight * self.ctc_prior.phone_posteriors[state.phone_index]
                )
            occurrence_prior = self.occurrence_priors.get(state.instance_id)
            if occurrence_prior is not None:
                occurrence_prior = np.asarray(occurrence_prior, dtype=np.float64)
                if len(occurrence_prior) != expected_frames or not np.all(np.isfinite(occurrence_prior)):
                    raise FrameScoreProviderError(f"invalid occurrence prior: {state.instance_id}")
                sustained = state.acoustic_role in {
                    "vowel_stable",
                    "vowel_nucleus",
                    "periodic",
                    "irregular_periodic",
                    "silence",
                }
                if self.acoustic_rhythm_used:
                    prior_weight = 0.08 if sustained else 0.20
                else:
                    prior_weight = 0.05 if sustained else 0.14
                prior_weight = min(0.45, prior_weight * self.occurrence_prior_weight_scale)
                score = (1.0 - prior_weight) * score + prior_weight * occurrence_prior
            if self.breath_end_ms > 0.0 and state.acoustic_role not in {"breath", "silence", "sil"}:
                early = evidence.frame_times_ms < self.breath_end_ms
                if np.any(early):
                    score = np.where(early, score * 0.25, score)
            result[state.instance_id] = np.clip(score, 0.0, 1.0)
        return FrameScoreResult(
            emission_scores=result,
            warnings=tuple(dict.fromkeys(warnings)),
            diagnostics={
                "emission_state_count": len(result),
                "transition_start_score_state_count": 0,
                "ctc_blend_enabled": self.ctc_prior is not None,
                "occurrence_priors_applied": bool(self.occurrence_priors),
            },
            provenance={
                "provider": "heuristic_frame_score_v1",
                "profile_id": self.profile_id,
                "profile_sha256": self.profile_sha256,
                "fallback": True,
                "fallback_reason": "no_trained_frame_score_provider_configured",
            },
        )


def _phone_feature_score(features: Mapping[str, object], tracks: Mapping[str, np.ndarray]) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    if features.get("syllabic") or features.get("vowel_like"):
        value = tracks.get("vowel_like")
        if value is not None:
            candidates.append(value)
    if features.get("voiced") or features.get("periodic"):
        value = tracks.get("periodicity")
        if value is not None:
            candidates.append(value)
    if features.get("voiced") is False:
        value = tracks.get("aperiodic")
        if value is not None:
            candidates.append(value)
    if features.get("continuant") is False or features.get("manner") == "stop":
        value = tracks.get("consonant_like")
        if value is not None:
            candidates.append(value)
    if features.get("manner") in {"fricative", "affricate"}:
        value = tracks.get("high_band_energy")
        if value is not None:
            candidates.append(value)
    if not candidates:
        return None
    return np.mean(np.stack(candidates), axis=0)
