from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from .acoustic_evidence import AcousticEvidence


EXTERNAL_EVIDENCE_SCHEMA_VERSION = 1
FUSABLE_TRACKS = frozenset(
    {"boundary", "transition", "vowel_like", "consonant_like", "periodicity", "aperiodic"}
)


class ExternalEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class ExternalEvidence:
    frame_times_ms: np.ndarray
    scores: Mapping[str, np.ndarray]
    confidence: np.ndarray
    provenance: Mapping[str, object]


def load_external_evidence(
    path: Path,
    *,
    expected_wav_sha256: str,
) -> ExternalEvidence:
    resolved = path.resolve()
    try:
        with np.load(resolved, allow_pickle=False) as artifact:
            required = {
                "schema_version",
                "wav_sha256",
                "frame_times_ms",
                "track_names",
                "scores",
                "confidence",
                "metadata_json",
            }
            missing = sorted(required - set(artifact.files))
            if missing:
                raise ExternalEvidenceError(f"external evidence fields missing: {', '.join(missing)}")
            schema_version = int(_scalar(artifact["schema_version"], "schema_version"))
            wav_sha256 = str(_scalar(artifact["wav_sha256"], "wav_sha256"))
            frame_times = np.asarray(artifact["frame_times_ms"], dtype=np.float64)
            track_names = tuple(str(item) for item in np.asarray(artifact["track_names"]).tolist())
            scores = np.asarray(artifact["scores"], dtype=np.float64)
            confidence = np.asarray(artifact["confidence"], dtype=np.float64)
            metadata_text = str(_scalar(artifact["metadata_json"], "metadata_json"))
    except (OSError, ValueError) as error:
        if isinstance(error, ExternalEvidenceError):
            raise
        raise ExternalEvidenceError(f"cannot load external evidence NPZ: {resolved}: {error}") from error
    if schema_version != EXTERNAL_EVIDENCE_SCHEMA_VERSION:
        raise ExternalEvidenceError(f"unsupported external evidence schema: {schema_version}")
    if wav_sha256 != expected_wav_sha256:
        raise ExternalEvidenceError("external evidence WAV SHA-256 mismatch")
    if frame_times.ndim != 1 or len(frame_times) < 2 or not np.all(np.isfinite(frame_times)):
        raise ExternalEvidenceError("external evidence frame_times_ms must be finite 1-D data")
    if np.any(np.diff(frame_times) <= 0.0):
        raise ExternalEvidenceError("external evidence frame_times_ms must be strictly increasing")
    if not track_names or len(set(track_names)) != len(track_names):
        raise ExternalEvidenceError("external evidence track_names must be non-empty and unique")
    unsupported = sorted(set(track_names) - FUSABLE_TRACKS)
    if unsupported:
        raise ExternalEvidenceError(f"unsupported external evidence tracks: {unsupported}")
    if scores.shape != (len(frame_times), len(track_names)) or not np.all(np.isfinite(scores)):
        raise ExternalEvidenceError("external evidence scores shape/values are invalid")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ExternalEvidenceError("external evidence scores must be probabilities in [0, 1]")
    if confidence.shape != frame_times.shape or not np.all(np.isfinite(confidence)):
        raise ExternalEvidenceError("external evidence confidence shape/values are invalid")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ExternalEvidenceError("external evidence confidence must be in [0, 1]")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise ExternalEvidenceError(f"external evidence metadata_json is invalid: {error}") from error
    if not isinstance(metadata, Mapping):
        raise ExternalEvidenceError("external evidence metadata_json must be an object")
    provider = str(metadata.get("provider", "")).strip()
    model_sha256 = str(metadata.get("model_sha256", "")).strip().casefold()
    if not provider or len(model_sha256) != 64 or any(char not in "0123456789abcdef" for char in model_sha256):
        raise ExternalEvidenceError("external evidence metadata requires provider and a valid model_sha256")
    return ExternalEvidence(
        frame_times_ms=frame_times,
        scores={name: scores[:, index] for index, name in enumerate(track_names)},
        confidence=confidence,
        provenance={
            "schema_version": EXTERNAL_EVIDENCE_SCHEMA_VERSION,
            "artifact": str(resolved),
            "artifact_sha256": _sha256_file(resolved),
            "provider": provider,
            "model_sha256": model_sha256,
            "encoder_kind": str(metadata.get("encoder_kind", "external_acoustic_evidence")),
            "source_frame_count": len(frame_times),
            "tracks": list(track_names),
            "wav_sha256_verified": True,
        },
    )


def fuse_external_evidence(
    evidence: AcousticEvidence,
    external: ExternalEvidence,
    *,
    maximum_weight: float = 0.25,
) -> AcousticEvidence:
    if not 0.0 < maximum_weight <= 0.5:
        raise ExternalEvidenceError("external evidence maximum_weight must be greater than 0 and at most 0.5")
    target_times = evidence.frame_times_ms
    confidence = np.interp(
        target_times,
        external.frame_times_ms,
        external.confidence,
        left=0.0,
        right=0.0,
    )
    effective_weight = np.clip(maximum_weight * confidence, 0.0, maximum_weight)
    tracks = dict(evidence.tracks)
    for name, source_values in external.scores.items():
        if name not in tracks:
            raise ExternalEvidenceError(f"runtime evidence track is missing: {name}")
        interpolated = np.interp(
            target_times,
            external.frame_times_ms,
            source_values,
            left=0.0,
            right=0.0,
        )
        tracks[name] = np.clip(
            (1.0 - effective_weight) * tracks[name] + effective_weight * interpolated,
            0.0,
            1.0,
        )
    mean_confidence = float(np.mean(confidence)) if len(confidence) else 0.0
    warnings = list(evidence.warnings)
    if mean_confidence < 0.20:
        warnings.append("low_external_evidence_confidence")
    provenance = {
        **dict(external.provenance),
        "maximum_weight": maximum_weight,
        "mean_interpolated_confidence": mean_confidence,
        "mean_effective_weight": float(np.mean(effective_weight)) if len(effective_weight) else 0.0,
        "confidence_gated": True,
        "heuristic_residual_preserved": True,
    }
    return replace(
        evidence,
        tracks=tracks,
        warnings=tuple(dict.fromkeys(warnings)),
        metadata={**dict(evidence.metadata), "external_evidence": provenance},
    )


def inspect_external_evidence(path: Path, wav_path: Path) -> dict[str, object]:
    wav_sha256 = _sha256_file(wav_path.resolve())
    external = load_external_evidence(path, expected_wav_sha256=wav_sha256)
    return {
        "valid": True,
        "artifact": str(path.resolve()),
        "wav": str(wav_path.resolve()),
        "wav_sha256": wav_sha256,
        "frame_count": len(external.frame_times_ms),
        "start_ms": float(external.frame_times_ms[0]),
        "end_ms": float(external.frame_times_ms[-1]),
        "mean_confidence": float(np.mean(external.confidence)),
        "tracks": sorted(external.scores),
        "provenance": dict(external.provenance),
    }


def _scalar(value: np.ndarray, name: str) -> object:
    array = np.asarray(value)
    if array.size != 1:
        raise ExternalEvidenceError(f"external evidence {name} must be scalar")
    return array.reshape(-1)[0].item()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
