from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp


CTC_PRIOR_SCHEMA_VERSION = 1


class CTCPriorError(ValueError):
    pass


@dataclass(frozen=True)
class CTCPrior:
    phone_posteriors: tuple[np.ndarray, ...]
    confidence: float
    provenance: Mapping[str, object]


def load_ctc_prior(
    path: Path,
    *,
    expected_wav_sha256: str,
    target_tokens: Sequence[str],
    target_frame_times_ms: np.ndarray,
) -> CTCPrior:
    resolved = path.resolve()
    try:
        with np.load(resolved, allow_pickle=False) as artifact:
            required = {"schema_version", "wav_sha256", "frame_times_ms", "logits", "tokens", "blank_token", "metadata_json"}
            missing = sorted(required - set(artifact.files))
            if missing:
                raise CTCPriorError(f"CTC prior 필드가 없습니다: {', '.join(missing)}")
            schema_version = int(_scalar(artifact["schema_version"], "schema_version"))
            wav_sha256 = str(_scalar(artifact["wav_sha256"], "wav_sha256"))
            frame_times = np.asarray(artifact["frame_times_ms"], dtype=np.float64)
            logits = np.asarray(artifact["logits"], dtype=np.float64)
            tokens = tuple(str(item) for item in np.asarray(artifact["tokens"]).tolist())
            blank_token = str(_scalar(artifact["blank_token"], "blank_token"))
            metadata_text = str(_scalar(artifact["metadata_json"], "metadata_json"))
    except (OSError, ValueError) as error:
        if isinstance(error, CTCPriorError):
            raise
        raise CTCPriorError(f"CTC prior NPZ를 읽을 수 없습니다: {resolved}: {error}") from error
    if schema_version != CTC_PRIOR_SCHEMA_VERSION:
        raise CTCPriorError(f"지원하지 않는 CTC prior schema_version입니다: {schema_version}")
    if wav_sha256 != expected_wav_sha256:
        raise CTCPriorError("CTC prior WAV hash가 현재 녹음과 다릅니다")
    if len(wav_sha256) != 64 or any(character not in "0123456789abcdef" for character in wav_sha256.casefold()):
        raise CTCPriorError("CTC prior wav_sha256가 유효하지 않습니다")
    if frame_times.ndim != 1 or len(frame_times) < 2 or not np.all(np.isfinite(frame_times)):
        raise CTCPriorError("CTC prior frame_times_ms는 finite 1차원 배열이어야 합니다")
    if np.any(np.diff(frame_times) <= 0.0):
        raise CTCPriorError("CTC prior frame_times_ms는 엄격히 증가해야 합니다")
    if logits.ndim != 2 or logits.shape != (len(frame_times), len(tokens)) or not np.all(np.isfinite(logits)):
        raise CTCPriorError("CTC prior logits shape/값이 유효하지 않습니다")
    if not tokens or len(set(tokens)) != len(tokens) or any(not token for token in tokens):
        raise CTCPriorError("CTC prior tokens는 비어 있지 않은 고유 문자열이어야 합니다")
    if blank_token not in tokens:
        raise CTCPriorError("CTC prior blank_token이 tokens에 없습니다")
    requested = tuple(str(token) for token in target_tokens)
    if not requested:
        raise CTCPriorError("CTC target token이 비어 있습니다")
    missing_tokens = sorted(set(requested) - set(tokens))
    if missing_tokens:
        raise CTCPriorError(f"CTC vocabulary에 target token이 없습니다: {', '.join(missing_tokens)}")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise CTCPriorError(f"CTC prior metadata_json이 유효하지 않습니다: {error}") from error
    if not isinstance(metadata, Mapping):
        raise CTCPriorError("CTC prior metadata_json root는 mapping이어야 합니다")
    if not str(metadata.get("provider", "")).strip() or not str(metadata.get("model_sha256", "")).strip():
        raise CTCPriorError("CTC prior metadata에는 provider와 model_sha256가 필요합니다")

    token_to_id = {token: index for index, token in enumerate(tokens)}
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    posterior, log_likelihood = known_sequence_ctc_posteriors(
        log_probabilities,
        [token_to_id[token] for token in requested],
        blank_id=token_to_id[blank_token],
    )
    interpolated: list[np.ndarray] = []
    for column in range(posterior.shape[1]):
        values = np.interp(
            target_frame_times_ms,
            frame_times,
            posterior[:, column],
            left=0.0,
            right=0.0,
        )
        interpolated.append(np.clip(values, 0.0, 1.0))
    confidence = float(np.clip(np.mean(np.max(posterior, axis=0)), 0.0, 1.0))
    return CTCPrior(
        phone_posteriors=tuple(interpolated),
        confidence=confidence,
        provenance={
            "schema_version": CTC_PRIOR_SCHEMA_VERSION,
            "artifact": str(resolved),
            "artifact_sha256": _sha256_file(resolved),
            "provider": str(metadata["provider"]),
            "model_sha256": str(metadata["model_sha256"]),
            "encoder_kind": str(metadata.get("encoder_kind", "external_ctc")),
            "token_count": len(requested),
            "vocabulary_size": len(tokens),
            "source_frame_count": len(frame_times),
            "log_likelihood": float(log_likelihood),
            "confidence": confidence,
            "wav_sha256_verified": True,
        },
    )


def known_sequence_ctc_posteriors(
    log_probabilities: np.ndarray,
    target_ids: Sequence[int],
    *,
    blank_id: int,
) -> tuple[np.ndarray, float]:
    log_probabilities = np.asarray(log_probabilities, dtype=np.float64)
    if log_probabilities.ndim != 2 or not np.all(np.isfinite(log_probabilities)):
        raise CTCPriorError("CTC log probability는 finite [frames, vocab] 배열이어야 합니다")
    frame_count, vocabulary_size = log_probabilities.shape
    targets = tuple(int(value) for value in target_ids)
    if not targets or frame_count < len(targets):
        raise CTCPriorError("CTC frame 수가 target sequence를 정렬하기에 부족합니다")
    if blank_id < 0 or blank_id >= vocabulary_size or any(value < 0 or value >= vocabulary_size for value in targets):
        raise CTCPriorError("CTC token id가 vocabulary 범위를 벗어났습니다")
    if blank_id in targets:
        raise CTCPriorError("CTC target sequence에 blank token을 넣을 수 없습니다")
    extended = [blank_id]
    for target in targets:
        extended.extend((target, blank_id))
    state_count = len(extended)
    alpha = np.full((frame_count, state_count), -np.inf, dtype=np.float64)
    alpha[0, 0] = log_probabilities[0, blank_id]
    alpha[0, 1] = log_probabilities[0, extended[1]]
    for frame in range(1, frame_count):
        for state in range(state_count):
            sources = [alpha[frame - 1, state]]
            if state > 0:
                sources.append(alpha[frame - 1, state - 1])
            if state > 1 and extended[state] != blank_id and extended[state] != extended[state - 2]:
                sources.append(alpha[frame - 1, state - 2])
            alpha[frame, state] = log_probabilities[frame, extended[state]] + logsumexp(sources)
    log_likelihood = float(logsumexp((alpha[-1, -1], alpha[-1, -2])))
    if not np.isfinite(log_likelihood):
        raise CTCPriorError("CTC known-sequence 경로가 없습니다")
    beta = np.full((frame_count, state_count), -np.inf, dtype=np.float64)
    beta[-1, -1] = 0.0
    beta[-1, -2] = 0.0
    for frame in range(frame_count - 2, -1, -1):
        for state in range(state_count):
            destinations = [beta[frame + 1, state] + log_probabilities[frame + 1, extended[state]]]
            if state + 1 < state_count:
                destinations.append(
                    beta[frame + 1, state + 1] + log_probabilities[frame + 1, extended[state + 1]]
                )
            if (
                state + 2 < state_count
                and extended[state + 2] != blank_id
                and extended[state + 2] != extended[state]
            ):
                destinations.append(
                    beta[frame + 1, state + 2] + log_probabilities[frame + 1, extended[state + 2]]
                )
            beta[frame, state] = logsumexp(destinations)
    state_posterior = np.exp(np.clip(alpha + beta - log_likelihood, -80.0, 0.0))
    token_posterior = np.column_stack(
        [state_posterior[:, 2 * target_index + 1] for target_index in range(len(targets))]
    )
    return np.clip(token_posterior, 0.0, 1.0), log_likelihood


def _scalar(value: np.ndarray, name: str) -> object:
    array = np.asarray(value)
    if array.size != 1:
        raise CTCPriorError(f"CTC prior {name}은 scalar여야 합니다")
    return array.reshape(-1)[0].item()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
