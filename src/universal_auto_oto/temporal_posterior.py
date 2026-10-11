from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .acoustic_evidence import AcousticEvidence, extract_acoustic_evidence
from .audio import file_sha256
from .oto_io import write_lines_atomic


TEMPORAL_POSTERIOR_SCHEMA_VERSION = 1
TEMPORAL_POSTERIOR_MODEL_KIND = "manual_gold_temporal_cnn_posterior_v1"
TEMPORAL_POSTERIOR_HEADS = (
    "boundary",
    "transition",
    "vowel_like",
    "consonant_like",
    "periodicity",
    "aperiodic",
)
TEMPORAL_SPLITS = ("train", "validation", "locked_test")


class TemporalPosteriorError(ValueError):
    pass


@dataclass(frozen=True)
class TemporalPosteriorSequence:
    """One recording worth of frame features and sparse manual targets.

    Every target vector must have one entry per frame. ``NaN`` is the explicit
    unlabelled value and is excluded from both loss and metrics.
    """

    sequence_id: str
    speaker_id: str
    features: np.ndarray
    targets: Mapping[str, np.ndarray]
    source_sha256: str = ""


@dataclass(frozen=True)
class TemporalPosteriorModel:
    feature_names: tuple[str, ...]
    feature_backend: str
    kernel_size: int
    hidden_channels: int
    means: np.ndarray
    scales: np.ndarray
    conv_weights: np.ndarray
    conv_bias: np.ndarray
    head_weights: np.ndarray
    head_bias: np.ndarray
    training_provenance: Mapping[str, object]
    evaluation: Mapping[str, object]

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        matrix = _validate_inference_features(features, len(self.feature_names))
        normalized = (matrix - self.means) / self.scales
        _logits, probabilities, _cache = _forward(
            normalized,
            self.conv_weights,
            self.conv_bias,
            self.head_weights,
            self.head_bias,
        )
        return {
            head: probabilities[:, index].copy()
            for index, head in enumerate(TEMPORAL_POSTERIOR_HEADS)
        }

    def to_artifact(self) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_POSTERIOR_SCHEMA_VERSION,
            "model_kind": TEMPORAL_POSTERIOR_MODEL_KIND,
            "architecture": {
                "type": "conv1d_relu_independent_sigmoid_heads",
                "convolution_layers": 1,
                "kernel_size": self.kernel_size,
                "hidden_channels": self.hidden_channels,
                "padding": "same_zero",
                "heads": list(TEMPORAL_POSTERIOR_HEADS),
            },
            "feature_names": list(self.feature_names),
            "feature_backend": self.feature_backend,
            "normalization": {
                "source_split": "train",
                "means": self.means.tolist(),
                "scales": self.scales.tolist(),
            },
            "parameters": {
                "conv_weights": self.conv_weights.tolist(),
                "conv_bias": self.conv_bias.tolist(),
                "head_weights": self.head_weights.tolist(),
                "head_bias": self.head_bias.tolist(),
            },
            "training_provenance": dict(self.training_provenance),
            "evaluation": dict(self.evaluation),
        }

    def save(self, output_model: Path) -> None:
        artifact = self.to_artifact()
        text = json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        write_lines_atomic(
            output_model.resolve(),
            text.splitlines(),
            encoding="utf-8",
            newline="\n",
            backup_existing=False,
        )


def train_temporal_posterior_model(
    train: Sequence[TemporalPosteriorSequence],
    validation: Sequence[TemporalPosteriorSequence],
    locked_test: Sequence[TemporalPosteriorSequence],
    *,
    feature_names: Sequence[str],
    feature_backend: str,
    output_model: Path | None = None,
    hidden_channels: int = 8,
    kernel_size: int = 5,
    epochs: int = 300,
    learning_rate: float = 0.01,
    l2: float = 1e-4,
    early_stopping_patience: int = 30,
    seed: int = 0,
) -> TemporalPosteriorModel:
    """Fit a small temporal CNN without consulting locked-test labels.

    Normalization statistics and gradients come only from ``train``.
    ``validation`` controls early stopping. ``locked_test`` is evaluated once,
    after the selected validation checkpoint has been restored.
    """

    names = tuple(str(name).strip() for name in feature_names)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise TemporalPosteriorError("feature_names must be non-empty and unique")
    backend = str(feature_backend).strip()
    if not backend:
        raise TemporalPosteriorError("feature_backend must be explicit")
    if hidden_channels < 1:
        raise TemporalPosteriorError("hidden_channels must be at least 1")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise TemporalPosteriorError("kernel_size must be a positive odd integer")
    if epochs < 1:
        raise TemporalPosteriorError("epochs must be at least 1")
    if not 0.0 < learning_rate < 1.0:
        raise TemporalPosteriorError("learning_rate must be between 0 and 1")
    if l2 < 0.0:
        raise TemporalPosteriorError("l2 must be non-negative")
    if early_stopping_patience < 1:
        raise TemporalPosteriorError("early_stopping_patience must be at least 1")

    split_sequences = {
        "train": _validate_split("train", train, len(names), required=True),
        "validation": _validate_split("validation", validation, len(names), required=True),
        "locked_test": _validate_split("locked_test", locked_test, len(names), required=False),
    }
    _reject_split_leakage(split_sequences)

    train_features = np.concatenate(
        [sequence.features for sequence in split_sequences["train"]], axis=0
    )
    means = np.mean(train_features, axis=0, dtype=np.float64)
    scales = np.std(train_features, axis=0, dtype=np.float64)
    scales = np.where(scales < 1e-8, 1.0, scales)
    normalized = {
        split: tuple(
            (
                (sequence.features - means) / scales,
                sequence.targets,
            )
            for sequence in sequences
        )
        for split, sequences in split_sequences.items()
        if split != "locked_test"
    }

    train_support = _target_support(split_sequences["train"])
    active_heads = tuple(head for head in TEMPORAL_POSTERIOR_HEADS if train_support[head]["labelled"] > 0)
    if not active_heads:
        raise TemporalPosteriorError("train split has no finite targets")
    validation_supported = tuple(
        head
        for head in active_heads
        if any(np.isfinite(targets[head]).any() for _features, targets in normalized["validation"])
    )
    if not validation_supported:
        raise TemporalPosteriorError("validation split has no finite target for a trained head")

    rng = np.random.default_rng(seed)
    feature_count = len(names)
    conv_weights = rng.normal(
        0.0,
        np.sqrt(2.0 / (kernel_size * feature_count)),
        size=(hidden_channels, kernel_size, feature_count),
    )
    conv_bias = np.zeros(hidden_channels, dtype=np.float64)
    head_weights = np.zeros((len(TEMPORAL_POSTERIOR_HEADS), hidden_channels), dtype=np.float64)
    for head in active_heads:
        index = TEMPORAL_POSTERIOR_HEADS.index(head)
        head_weights[index] = rng.normal(
            0.0,
            np.sqrt(1.0 / hidden_channels),
            size=hidden_channels,
        )
    head_bias = np.zeros(len(TEMPORAL_POSTERIOR_HEADS), dtype=np.float64)
    parameters = {
        "conv_weights": conv_weights,
        "conv_bias": conv_bias,
        "head_weights": head_weights,
        "head_bias": head_bias,
    }
    moments = {
        name: [np.zeros_like(value), np.zeros_like(value)]
        for name, value in parameters.items()
    }

    best_parameters: dict[str, np.ndarray] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_run = 0
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        _loss, gradients = _training_loss_and_gradients(
            normalized["train"],
            parameters,
            active_heads,
            l2,
        )
        _adam_update(parameters, gradients, moments, epoch, learning_rate)
        epochs_run = epoch
        validation_loss = _data_loss(
            normalized["validation"],
            parameters,
            validation_supported,
        )
        if validation_loss < best_validation_loss - 1e-9:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_parameters = {name: value.copy() for name, value in parameters.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break
    if best_parameters is None:
        raise TemporalPosteriorError("training did not produce a finite validation checkpoint")
    parameters = best_parameters

    # Deliberately normalize/evaluate locked-test only after model selection.
    normalized_locked = tuple(
        ((sequence.features - means) / scales, sequence.targets)
        for sequence in split_sequences["locked_test"]
    )
    evaluation = {
        "train": _evaluate_split(normalized["train"], parameters),
        "validation": _evaluate_split(normalized["validation"], parameters),
        "locked_test": _evaluate_split(normalized_locked, parameters),
    }
    split_hashes = {
        split: _split_sha256(sequences, names, backend)
        for split, sequences in split_sequences.items()
    }
    split_speakers = {
        split: sorted({sequence.speaker_id for sequence in sequences})
        for split, sequences in split_sequences.items()
    }
    split_sequence_ids = {
        split: [sequence.sequence_id for sequence in sequences]
        for split, sequences in split_sequences.items()
    }
    target_support = {
        split: _target_support(sequences)
        for split, sequences in split_sequences.items()
    }
    provenance: dict[str, object] = {
        "seed": int(seed),
        "optimizer": "adam_full_sequence",
        "learning_rate": float(learning_rate),
        "l2": float(l2),
        "epochs_requested": int(epochs),
        "epochs_run": int(epochs_run),
        "early_stopping_patience": int(early_stopping_patience),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "normalization_source": "train_only",
        "gradient_source": "train_only",
        "model_selection_source": "validation_only",
        "locked_test_policy": "final_evaluation_only_after_model_selection",
        "split_leakage_check": "speaker_ids_disjoint",
        "split_speakers": split_speakers,
        "split_sequence_ids": split_sequence_ids,
        "input_sha256": split_hashes,
        "target_support": target_support,
        "trained_heads": list(active_heads),
        "untrained_heads": [head for head in TEMPORAL_POSTERIOR_HEADS if head not in active_heads],
        "production_promotion_claimed": False,
    }
    model = TemporalPosteriorModel(
        feature_names=names,
        feature_backend=backend,
        kernel_size=kernel_size,
        hidden_channels=hidden_channels,
        means=means.copy(),
        scales=scales.copy(),
        conv_weights=parameters["conv_weights"].copy(),
        conv_bias=parameters["conv_bias"].copy(),
        head_weights=parameters["head_weights"].copy(),
        head_bias=parameters["head_bias"].copy(),
        training_provenance=provenance,
        evaluation=evaluation,
    )
    if output_model is not None:
        model.save(output_model)
    return model


def train_temporal_posterior_from_manifest(
    items_csv: Path,
    output_model: Path,
    *,
    feature_backend: str | None = None,
    minimum_train_recordings: int = 20,
    minimum_validation_recordings: int = 5,
    hidden_channels: int = 8,
    kernel_size: int = 5,
    epochs: int = 300,
    learning_rate: float = 0.01,
    l2: float = 1e-4,
    early_stopping_patience: int = 30,
    seed: int = 0,
) -> dict[str, object]:
    """Train the temporal CNN from the existing speaker-locked Gold CSV.

    This adapter deliberately reuses the same feature extraction and sparse
    manual-phone target construction as the pointwise posterior baseline.
    """

    from .posterior_model import (
        POSTERIOR_FEATURE_NAMES,
        _frame_targets,
        _load_manifest_training_items,
    )

    items_path = items_csv.resolve()
    items = _load_manifest_training_items(items_path)
    available_backends = sorted({item.project.alignment.feature_backend for item in items})
    selected_backend = str(feature_backend or "").strip()
    if not selected_backend:
        if len(available_backends) != 1:
            raise TemporalPosteriorError(
                f"feature_backend is required when manifest has multiple backends: {available_backends}"
            )
        selected_backend = available_backends[0]
    selected = tuple(
        item for item in items if item.project.alignment.feature_backend == selected_backend
    )
    split_items = {
        split: tuple(item for item in selected if item.split == split)
        for split in TEMPORAL_SPLITS
    }
    if len(split_items["train"]) < minimum_train_recordings:
        raise TemporalPosteriorError(
            f"manual Gold train recordings are insufficient: "
            f"{len(split_items['train'])} < {minimum_train_recordings}"
        )
    if len(split_items["validation"]) < minimum_validation_recordings:
        raise TemporalPosteriorError(
            f"manual Gold validation recordings are insufficient: "
            f"{len(split_items['validation'])} < {minimum_validation_recordings}"
        )

    sequences: dict[str, tuple[TemporalPosteriorSequence, ...]] = {}
    for split, values in split_items.items():
        compiled: list[TemporalPosteriorSequence] = []
        for item in sorted(
            values,
            key=lambda value: (value.bank_id, value.recording_index, value.annotation.wav_sha256),
        ):
            recording = item.project.recordings[item.recording_index]
            evidence = extract_acoustic_evidence(
                item.project.wav_path(recording),
                internal_sample_rate=item.project.audio.internal_sample_rate,
                convert_to_mono=item.project.audio.convert_to_mono,
                resampler=item.project.audio.resampler,
                feature_backend=selected_backend,
                evidence_model=(item.project.root / item.project.alignment.evidence_model).resolve() if item.project.alignment.evidence_model else None,
                evidence_provider=item.project.alignment.evidence_provider,
            )
            features = np.column_stack(
                [evidence.tracks[name] for name in POSTERIOR_FEATURE_NAMES]
            ).astype(np.float64)
            targets = _frame_targets(
                item.project,
                recording,
                item.annotation,
                evidence.frame_times_ms,
            )
            source_payload = {
                "project_sha256": item.project.project_sha256,
                "inventory_sha256": item.project.inventory_sha256,
                "annotation_sha256": file_sha256(item.annotation_path),
                "wav_sha256": evidence.wav_sha256,
            }
            source_sha256 = hashlib.sha256(
                json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            compiled.append(
                TemporalPosteriorSequence(
                    sequence_id=f"{item.bank_id}:{item.recording_index}:{evidence.wav_sha256[:12]}",
                    speaker_id=item.speaker_id,
                    features=features,
                    targets=targets,
                    source_sha256=source_sha256,
                )
            )
        sequences[split] = tuple(compiled)

    model = train_temporal_posterior_model(
        sequences["train"],
        sequences["validation"],
        sequences["locked_test"],
        feature_names=POSTERIOR_FEATURE_NAMES,
        feature_backend=selected_backend,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    )
    inventory_allowlist = sorted({item.project.inventory_sha256 for item in selected})
    model = replace(
        model,
        training_provenance={
            **dict(model.training_provenance),
            "items_csv": str(items_path),
            "items_csv_sha256": file_sha256(items_path),
            "inventory_sha256_allowlist": inventory_allowlist,
            "available_feature_backends": available_backends,
            "manual_gold_only": True,
        },
    )
    model.save(output_model)
    return {
        "schema_version": 1,
        "model": str(output_model.resolve()),
        "model_sha256": file_sha256(output_model.resolve()),
        "model_kind": TEMPORAL_POSTERIOR_MODEL_KIND,
        "feature_backend": selected_backend,
        "recording_counts": {split: len(values) for split, values in split_items.items()},
        "best_epoch": model.training_provenance["best_epoch"],
        "evaluation": dict(model.evaluation),
        "production_promotion_claimed": False,
    }


def apply_temporal_posterior_model(
    evidence: AcousticEvidence,
    model: TemporalPosteriorModel,
    *,
    model_path: Path,
    posterior_blend: float = 0.75,
) -> AcousticEvidence:
    """Blend a temporal model into acoustic tracks for HSMM decoding."""

    if not 0.0 < posterior_blend <= 1.0:
        raise TemporalPosteriorError("posterior_blend must be in (0, 1]")
    if model.feature_backend != evidence.metadata.get("feature_backend"):
        raise TemporalPosteriorError("temporal posterior feature backend mismatch")
    missing = [name for name in model.feature_names if name not in evidence.tracks]
    if missing:
        raise TemporalPosteriorError(f"acoustic evidence tracks are missing: {missing}")
    features = np.column_stack([evidence.tracks[name] for name in model.feature_names])
    learned = model.predict(features)
    trained_heads = set(model.training_provenance.get("trained_heads", ()))
    tracks = dict(evidence.tracks)
    for head in TEMPORAL_POSTERIOR_HEADS:
        if head not in trained_heads:
            continue
        heuristic = np.asarray(tracks[head], dtype=np.float64)
        tracks[head] = np.clip(
            posterior_blend * learned[head] + (1.0 - posterior_blend) * heuristic,
            0.0,
            1.0,
        )
    return replace(
        evidence,
        tracks=tracks,
        metadata={
            **dict(evidence.metadata),
            "posterior_model": {
                "model_kind": TEMPORAL_POSTERIOR_MODEL_KIND,
                "model_path": str(model_path.resolve()),
                "model_sha256": file_sha256(model_path.resolve()),
                "posterior_blend": posterior_blend,
                "manual_gold_trained": True,
                "temporal_context": True,
                "production_promotion_claimed": False,
            },
        },
    )


def load_temporal_posterior_model(model_path: Path) -> TemporalPosteriorModel:
    path = model_path.resolve()
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemporalPosteriorError(f"cannot read temporal posterior artifact: {path}") from error
    if not isinstance(artifact, dict):
        raise TemporalPosteriorError("temporal posterior artifact root must be an object")
    if artifact.get("schema_version") != TEMPORAL_POSTERIOR_SCHEMA_VERSION:
        raise TemporalPosteriorError("unsupported temporal posterior schema_version")
    if artifact.get("model_kind") != TEMPORAL_POSTERIOR_MODEL_KIND:
        raise TemporalPosteriorError("unsupported temporal posterior model_kind")
    architecture = _required_mapping(artifact.get("architecture"), "architecture")
    if architecture.get("type") != "conv1d_relu_independent_sigmoid_heads":
        raise TemporalPosteriorError("unsupported temporal posterior architecture")
    if tuple(architecture.get("heads", ())) != TEMPORAL_POSTERIOR_HEADS:
        raise TemporalPosteriorError("temporal posterior head order mismatch")
    feature_names = tuple(str(value) for value in artifact.get("feature_names", ()))
    feature_backend = str(artifact.get("feature_backend", "")).strip()
    if not feature_names or not feature_backend:
        raise TemporalPosteriorError("artifact feature contract is incomplete")
    kernel_size = _required_positive_int(architecture.get("kernel_size"), "kernel_size")
    hidden_channels = _required_positive_int(architecture.get("hidden_channels"), "hidden_channels")
    normalization = _required_mapping(artifact.get("normalization"), "normalization")
    if normalization.get("source_split") != "train":
        raise TemporalPosteriorError("artifact normalization must come from train")
    parameters = _required_mapping(artifact.get("parameters"), "parameters")
    model = TemporalPosteriorModel(
        feature_names=feature_names,
        feature_backend=feature_backend,
        kernel_size=kernel_size,
        hidden_channels=hidden_channels,
        means=_finite_array(normalization.get("means"), (len(feature_names),), "normalization.means"),
        scales=_finite_array(normalization.get("scales"), (len(feature_names),), "normalization.scales"),
        conv_weights=_finite_array(
            parameters.get("conv_weights"),
            (hidden_channels, kernel_size, len(feature_names)),
            "parameters.conv_weights",
        ),
        conv_bias=_finite_array(parameters.get("conv_bias"), (hidden_channels,), "parameters.conv_bias"),
        head_weights=_finite_array(
            parameters.get("head_weights"),
            (len(TEMPORAL_POSTERIOR_HEADS), hidden_channels),
            "parameters.head_weights",
        ),
        head_bias=_finite_array(
            parameters.get("head_bias"),
            (len(TEMPORAL_POSTERIOR_HEADS),),
            "parameters.head_bias",
        ),
        training_provenance=_required_mapping(
            artifact.get("training_provenance"), "training_provenance"
        ),
        evaluation=_required_mapping(artifact.get("evaluation"), "evaluation"),
    )
    if np.any(model.scales <= 0.0):
        raise TemporalPosteriorError("artifact normalization scales must be positive")
    return model


def _validate_split(
    split: str,
    sequences: Sequence[TemporalPosteriorSequence],
    feature_count: int,
    *,
    required: bool,
) -> tuple[TemporalPosteriorSequence, ...]:
    if required and not sequences:
        raise TemporalPosteriorError(f"{split} split must not be empty")
    validated: list[TemporalPosteriorSequence] = []
    seen_ids: set[str] = set()
    for raw in sequences:
        sequence_id = str(raw.sequence_id).strip()
        speaker_id = str(raw.speaker_id).strip()
        if not sequence_id:
            raise TemporalPosteriorError(f"{split}: sequence_id must not be empty")
        if sequence_id in seen_ids:
            raise TemporalPosteriorError(f"{split}: duplicate sequence_id: {sequence_id}")
        seen_ids.add(sequence_id)
        if not speaker_id:
            raise TemporalPosteriorError(f"{split}/{sequence_id}: speaker_id must not be empty")
        features = _validate_inference_features(raw.features, feature_count)
        if features.shape[0] < 1:
            raise TemporalPosteriorError(f"{split}/{sequence_id}: features must contain frames")
        targets: dict[str, np.ndarray] = {}
        extra_heads = sorted(set(raw.targets) - set(TEMPORAL_POSTERIOR_HEADS))
        missing_heads = sorted(set(TEMPORAL_POSTERIOR_HEADS) - set(raw.targets))
        if extra_heads or missing_heads:
            raise TemporalPosteriorError(
                f"{split}/{sequence_id}: target heads mismatch; missing={missing_heads}, extra={extra_heads}"
            )
        for head in TEMPORAL_POSTERIOR_HEADS:
            target = np.asarray(raw.targets[head], dtype=np.float64)
            if target.shape != (features.shape[0],):
                raise TemporalPosteriorError(
                    f"{split}/{sequence_id}/{head}: target shape must be ({features.shape[0]},)"
                )
            if np.isinf(target).any():
                raise TemporalPosteriorError(f"{split}/{sequence_id}/{head}: target contains infinity")
            finite = np.isfinite(target)
            if np.any((target[finite] < 0.0) | (target[finite] > 1.0)):
                raise TemporalPosteriorError(
                    f"{split}/{sequence_id}/{head}: finite targets must be in [0, 1]"
                )
            targets[head] = target.copy()
        validated.append(
            TemporalPosteriorSequence(
                sequence_id=sequence_id,
                speaker_id=speaker_id,
                features=features.copy(),
                targets=targets,
                source_sha256=str(raw.source_sha256).strip(),
            )
        )
    return tuple(sorted(validated, key=lambda item: item.sequence_id))


def _reject_split_leakage(
    split_sequences: Mapping[str, Sequence[TemporalPosteriorSequence]],
) -> None:
    speaker_splits: dict[str, set[str]] = {}
    sequence_splits: dict[str, set[str]] = {}
    for split, sequences in split_sequences.items():
        for sequence in sequences:
            speaker_splits.setdefault(sequence.speaker_id, set()).add(split)
            sequence_splits.setdefault(sequence.sequence_id, set()).add(split)
    leaking_speakers = sorted(speaker for speaker, splits in speaker_splits.items() if len(splits) > 1)
    leaking_sequences = sorted(
        sequence_id for sequence_id, splits in sequence_splits.items() if len(splits) > 1
    )
    if leaking_speakers or leaking_sequences:
        raise TemporalPosteriorError(
            "split leakage detected; "
            f"speaker_ids={leaking_speakers}, sequence_ids={leaking_sequences}"
        )


def _validate_inference_features(features: np.ndarray, feature_count: int) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != feature_count:
        raise TemporalPosteriorError(f"features must have shape [frames, {feature_count}]")
    if not np.isfinite(matrix).all():
        raise TemporalPosteriorError("features must be finite")
    return matrix


def _windows(features: np.ndarray, kernel_size: int) -> np.ndarray:
    padding = kernel_size // 2
    padded = np.pad(features, ((padding, padding), (0, 0)), mode="constant")
    frame_count, feature_count = features.shape
    result = np.empty((frame_count, kernel_size, feature_count), dtype=np.float64)
    for offset in range(kernel_size):
        result[:, offset, :] = padded[offset : offset + frame_count, :]
    return result


def _forward(
    normalized_features: np.ndarray,
    conv_weights: np.ndarray,
    conv_bias: np.ndarray,
    head_weights: np.ndarray,
    head_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    windows = _windows(normalized_features, conv_weights.shape[1])
    pre_activation = np.einsum("tkf,hkf->th", windows, conv_weights) + conv_bias
    hidden = np.maximum(pre_activation, 0.0)
    logits = hidden @ head_weights.T + head_bias
    probabilities = _sigmoid(logits)
    return logits, probabilities, (windows, pre_activation, hidden)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _training_loss_and_gradients(
    sequences: Sequence[tuple[np.ndarray, Mapping[str, np.ndarray]]],
    parameters: Mapping[str, np.ndarray],
    active_heads: Sequence[str],
    l2: float,
) -> tuple[float, dict[str, np.ndarray]]:
    gradients = {name: np.zeros_like(value) for name, value in parameters.items()}
    counts = {
        head: sum(int(np.isfinite(targets[head]).sum()) for _features, targets in sequences)
        for head in active_heads
    }
    active_count = len(active_heads)
    data_loss = 0.0
    for features, targets in sequences:
        logits, probabilities, (windows, pre_activation, hidden) = _forward(
            features,
            parameters["conv_weights"],
            parameters["conv_bias"],
            parameters["head_weights"],
            parameters["head_bias"],
        )
        dlogits = np.zeros_like(logits)
        for head in active_heads:
            head_index = TEMPORAL_POSTERIOR_HEADS.index(head)
            target = targets[head]
            mask = np.isfinite(target)
            if not mask.any():
                continue
            selected_logits = logits[mask, head_index]
            selected_targets = target[mask]
            data_loss += float(
                np.sum(
                    np.maximum(selected_logits, 0.0)
                    - selected_logits * selected_targets
                    + np.log1p(np.exp(-np.abs(selected_logits)))
                )
                / counts[head]
                / active_count
            )
            dlogits[mask, head_index] = (
                probabilities[mask, head_index] - selected_targets
            ) / counts[head] / active_count
        gradients["head_weights"] += dlogits.T @ hidden
        gradients["head_bias"] += np.sum(dlogits, axis=0)
        dhidden = dlogits @ parameters["head_weights"]
        dpre_activation = dhidden * (pre_activation > 0.0)
        gradients["conv_weights"] += np.einsum("th,tkf->hkf", dpre_activation, windows)
        gradients["conv_bias"] += np.sum(dpre_activation, axis=0)
    regularized_names = ("conv_weights", "head_weights")
    regularization = 0.0
    for name in regularized_names:
        regularization += 0.5 * l2 * float(np.sum(parameters[name] ** 2))
        gradients[name] += l2 * parameters[name]
    return data_loss + regularization, gradients


def _data_loss(
    sequences: Sequence[tuple[np.ndarray, Mapping[str, np.ndarray]]],
    parameters: Mapping[str, np.ndarray],
    allowed_heads: Sequence[str],
) -> float:
    losses: list[float] = []
    for head in allowed_heads:
        head_index = TEMPORAL_POSTERIOR_HEADS.index(head)
        numerator = 0.0
        denominator = 0
        for features, targets in sequences:
            logits, _probabilities, _cache = _forward(
                features,
                parameters["conv_weights"],
                parameters["conv_bias"],
                parameters["head_weights"],
                parameters["head_bias"],
            )
            target = targets[head]
            mask = np.isfinite(target)
            selected_logits = logits[mask, head_index]
            selected_targets = target[mask]
            numerator += float(
                np.sum(
                    np.maximum(selected_logits, 0.0)
                    - selected_logits * selected_targets
                    + np.log1p(np.exp(-np.abs(selected_logits)))
                )
            )
            denominator += int(mask.sum())
        if denominator:
            losses.append(numerator / denominator)
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def _adam_update(
    parameters: Mapping[str, np.ndarray],
    gradients: Mapping[str, np.ndarray],
    moments: Mapping[str, list[np.ndarray]],
    step: int,
    learning_rate: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    for name, parameter in parameters.items():
        first, second = moments[name]
        first *= beta1
        first += (1.0 - beta1) * gradients[name]
        second *= beta2
        second += (1.0 - beta2) * gradients[name] ** 2
        corrected_first = first / (1.0 - beta1**step)
        corrected_second = second / (1.0 - beta2**step)
        parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)


def _evaluate_split(
    sequences: Sequence[tuple[np.ndarray, Mapping[str, np.ndarray]]],
    parameters: Mapping[str, np.ndarray],
) -> dict[str, object]:
    heads: dict[str, object] = {}
    for head_index, head in enumerate(TEMPORAL_POSTERIOR_HEADS):
        losses: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        target_values: list[np.ndarray] = []
        ignored_nan = 0
        for features, targets in sequences:
            logits, probabilities, _cache = _forward(
                features,
                parameters["conv_weights"],
                parameters["conv_bias"],
                parameters["head_weights"],
                parameters["head_bias"],
            )
            target = targets[head]
            mask = np.isfinite(target)
            ignored_nan += int((~mask).sum())
            if not mask.any():
                continue
            selected_logits = logits[mask, head_index]
            selected_targets = target[mask]
            losses.append(
                np.maximum(selected_logits, 0.0)
                - selected_logits * selected_targets
                + np.log1p(np.exp(-np.abs(selected_logits)))
            )
            predictions.append(probabilities[mask, head_index])
            target_values.append(selected_targets)
        if not target_values:
            heads[head] = {
                "labelled_frames": 0,
                "ignored_nan_frames": ignored_nan,
                "binary_cross_entropy": None,
                "accuracy_at_0_5": None,
            }
            continue
        joined_predictions = np.concatenate(predictions)
        joined_targets = np.concatenate(target_values)
        heads[head] = {
            "labelled_frames": int(joined_targets.size),
            "ignored_nan_frames": ignored_nan,
            "binary_cross_entropy": float(np.mean(np.concatenate(losses))),
            "accuracy_at_0_5": float(
                np.mean((joined_predictions >= 0.5) == (joined_targets >= 0.5))
            ),
        }
    return {
        "sequence_count": len(sequences),
        "frame_count": int(sum(features.shape[0] for features, _targets in sequences)),
        "heads": heads,
    }


def _target_support(
    sequences: Sequence[TemporalPosteriorSequence],
) -> dict[str, dict[str, int]]:
    support: dict[str, dict[str, int]] = {}
    for head in TEMPORAL_POSTERIOR_HEADS:
        labelled = sum(int(np.isfinite(sequence.targets[head]).sum()) for sequence in sequences)
        total = sum(int(sequence.targets[head].size) for sequence in sequences)
        support[head] = {"labelled": labelled, "ignored_nan": total - labelled}
    return support


def _split_sha256(
    sequences: Sequence[TemporalPosteriorSequence],
    feature_names: Sequence[str],
    feature_backend: str,
) -> str:
    digest = hashlib.sha256()
    contract = json.dumps(
        {"feature_names": list(feature_names), "feature_backend": feature_backend},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest.update(contract.encode("utf-8"))
    for sequence in sequences:
        metadata = json.dumps(
            {
                "sequence_id": sequence.sequence_id,
                "speaker_id": sequence.speaker_id,
                "source_sha256": sequence.source_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(metadata.encode("utf-8"))
        _update_array_digest(digest, sequence.features)
        for head in TEMPORAL_POSTERIOR_HEADS:
            _update_array_digest(digest, sequence.targets[head])
    return digest.hexdigest()


def _update_array_digest(digest: "hashlib._Hash", array: np.ndarray) -> None:
    normalized = np.asarray(array, dtype="<f8").copy(order="C")
    if np.isnan(normalized).any():
        normalized[np.isnan(normalized)] = np.nan
    digest.update(json.dumps(list(normalized.shape), separators=(",", ":")).encode("ascii"))
    digest.update(normalized.tobytes(order="C"))


def _required_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TemporalPosteriorError(f"artifact {field} must be an object")
    return value


def _required_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TemporalPosteriorError(f"artifact {field} must be a positive integer")
    return value


def _finite_array(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TemporalPosteriorError(f"artifact {field} is not numeric") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise TemporalPosteriorError(f"artifact {field} must have finite shape {shape}")
    return array
