from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .neural_evidence_dataset import EVENT_TRACKS, NEURAL_LABEL_SCHEMA_VERSION


NEURAL_FEATURE_BACKEND_ID = "onnx_neural_evidence_v1"
NEURAL_MODEL_KIND = "crnn_neural_evidence_v1"
NEURAL_INPUT_FEATURES = tuple(f"logmel_{index:02d}" for index in range(80)) + ("relative_energy", "log_f0", "voicing")
NEURAL_MANIFEST_SCHEMA_VERSION = 1


class NeuralEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class NeuralEvidenceManifest:
    model_path: Path
    manifest_path: Path
    model_sha256: str
    feature_names: tuple[str, ...]
    event_tracks: tuple[str, ...]
    sample_rate: int
    hop_ms: float
    provider: str
    raw: Mapping[str, object]


def manifest_path_for_model(model_path: Path) -> Path:
    return model_path.with_suffix(".manifest.json")


def load_neural_evidence_manifest(model_path: Path, *, provider: str) -> NeuralEvidenceManifest:
    resolved_model = model_path.resolve()
    if not resolved_model.is_file():
        raise NeuralEvidenceError(f"neural evidence model is missing: {resolved_model}")
    manifest_path = manifest_path_for_model(resolved_model)
    if not manifest_path.is_file():
        raise NeuralEvidenceError(f"neural evidence manifest is missing: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise NeuralEvidenceError(f"invalid neural evidence manifest: {manifest_path}") from error
    if not isinstance(raw, Mapping) or raw.get("schema_version") != NEURAL_MANIFEST_SCHEMA_VERSION:
        raise NeuralEvidenceError("unsupported neural evidence manifest schema")
    if raw.get("model_kind") != NEURAL_MODEL_KIND:
        raise NeuralEvidenceError("neural evidence manifest model_kind mismatch")
    model_sha = _sha256(resolved_model)
    if raw.get("model_sha256") != model_sha:
        raise NeuralEvidenceError("neural evidence model SHA-256 does not match its manifest")
    feature_names = _string_tuple(raw.get("feature_names"), "feature_names")
    event_tracks = _string_tuple(raw.get("event_tracks"), "event_tracks")
    if feature_names != NEURAL_INPUT_FEATURES:
        raise NeuralEvidenceError("neural evidence feature schema mismatch")
    if event_tracks != EVENT_TRACKS:
        raise NeuralEvidenceError("neural evidence event-track schema mismatch")
    if raw.get("label_schema_version") != NEURAL_LABEL_SCHEMA_VERSION:
        raise NeuralEvidenceError("neural evidence label schema mismatch")
    sample_rate = _positive_int(raw.get("sample_rate"), "sample_rate")
    hop_ms = _positive_float(raw.get("hop_ms"), "hop_ms")
    selected_provider = str(provider).strip().casefold()
    if selected_provider not in {"cuda", "cpu", "auto"}:
        raise NeuralEvidenceError(f"unsupported neural evidence provider: {provider}")
    return NeuralEvidenceManifest(
        model_path=resolved_model,
        manifest_path=manifest_path,
        model_sha256=model_sha,
        feature_names=feature_names,
        event_tracks=event_tracks,
        sample_rate=sample_rate,
        hop_ms=hop_ms,
        provider=selected_provider,
        raw=dict(raw),
    )


def neural_input_features(analysis: object, primitives: Mapping[str, np.ndarray]) -> np.ndarray:
    """Create the exact per-frame CRNN input schema from existing frame analysis."""
    magnitude = np.asarray(getattr(analysis, "magnitude"), dtype=np.float64)
    frequencies = np.asarray(getattr(analysis, "frequencies"), dtype=np.float64)
    sample_rate = int(getattr(analysis, "analysis_sample_rate"))
    mel = _log_mel(magnitude, frequencies, sample_rate, bands=80)
    relative_energy = _zscore(np.asarray(primitives["energy"], dtype=np.float64))
    f0 = np.asarray(primitives["f0_hz"], dtype=np.float64)
    log_f0_values = np.zeros_like(f0)
    voiced = f0 > 0.0
    log_f0_values[voiced] = np.log(f0[voiced])
    log_f0 = _zscore(log_f0_values)
    voicing = np.asarray(primitives["voicing"], dtype=np.float64)
    result = np.column_stack((mel, relative_energy, log_f0, voicing)).astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape[1] != len(NEURAL_INPUT_FEATURES) or not np.all(np.isfinite(result)):
        raise NeuralEvidenceError("neural evidence input feature construction failed")
    return result


def infer_neural_tracks(
    analysis: object,
    primitives: Mapping[str, np.ndarray],
    *,
    model_path: Path | None,
    provider: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if model_path is None:
        raise NeuralEvidenceError("onnx_neural_evidence_v1 requires alignment.evidence_model")
    manifest = load_neural_evidence_manifest(model_path, provider=provider)
    actual_sample_rate = int(getattr(analysis, "analysis_sample_rate"))
    actual_hop = float(getattr(analysis, "hop_ms"))
    if manifest.sample_rate != actual_sample_rate or not math.isclose(manifest.hop_ms, actual_hop, abs_tol=1e-6):
        raise NeuralEvidenceError(
            "neural evidence model timebase does not match project audio configuration "
            f"(model={manifest.sample_rate}Hz/{manifest.hop_ms}ms, audio={actual_sample_rate}Hz/{actual_hop}ms)"
        )
    features = neural_input_features(analysis, primitives)
    logits, runtime_details = _run_onnx(manifest, features)
    if logits.shape != (len(features), len(EVENT_TRACKS)):
        raise NeuralEvidenceError(f"neural evidence output shape mismatch: {logits.shape}")
    probabilities = _sigmoid(logits)
    tracks = {f"neural_{name}": probabilities[:, index].astype(np.float64, copy=False) for index, name in enumerate(EVENT_TRACKS)}
    entropy = -np.mean(
        probabilities * np.log(np.clip(probabilities, 1e-6, 1.0))
        + (1.0 - probabilities) * np.log(np.clip(1.0 - probabilities, 1e-6, 1.0)),
        axis=1,
    )
    tracks["neural_evidence_confidence"] = np.clip(1.0 - entropy / math.log(2.0), 0.0, 1.0).astype(np.float64)
    return tracks, {
        "neural_model_kind": NEURAL_MODEL_KIND,
        "neural_model_path": str(manifest.model_path),
        "neural_model_sha256": manifest.model_sha256,
        "neural_manifest_path": str(manifest.manifest_path),
        "neural_manifest_sha256": _sha256(manifest.manifest_path),
        "neural_label_schema_version": NEURAL_LABEL_SCHEMA_VERSION,
        "neural_feature_schema": list(NEURAL_INPUT_FEATURES),
        "neural_event_tracks": list(EVENT_TRACKS),
        **runtime_details,
    }


def write_neural_evidence_manifest(
    model_path: Path,
    *,
    training_manifest_sha256: str,
    training_summary: Mapping[str, object],
    sample_rate: int = 44100,
    hop_ms: float = 5.0,
) -> Path:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise NeuralEvidenceError(f"cannot write manifest for missing model: {model_path}")
    target = manifest_path_for_model(model_path)
    payload = {
        "schema_version": NEURAL_MANIFEST_SCHEMA_VERSION,
        "model_kind": NEURAL_MODEL_KIND,
        "model_sha256": _sha256(model_path),
        "label_schema_version": NEURAL_LABEL_SCHEMA_VERSION,
        "feature_names": list(NEURAL_INPUT_FEATURES),
        "event_tracks": list(EVENT_TRACKS),
        "sample_rate": int(sample_rate),
        "hop_ms": float(hop_ms),
        "training_manifest_sha256": str(training_manifest_sha256),
        "training_summary": dict(training_summary),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _run_onnx(manifest: NeuralEvidenceManifest, features: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise NeuralEvidenceError(
            "onnx_neural_evidence_v1 requires onnxruntime-gpu in the selected Auto OTO environment"
        ) from error
    available = tuple(ort.get_available_providers())
    if manifest.provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise NeuralEvidenceError("CUDAExecutionProvider is unavailable; install a CUDA-enabled onnxruntime-gpu build")
        providers = ["CUDAExecutionProvider"]
    elif manifest.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    session = _onnx_session(str(manifest.model_path), manifest.model_sha256, tuple(providers))
    actual_providers = tuple(session.get_providers())
    execution_provider = actual_providers[0] if actual_providers else "unknown"
    if manifest.provider == "cuda" and execution_provider != "CUDAExecutionProvider":
        raise NeuralEvidenceError(
            "CUDA neural evidence was requested but ONNX Runtime fell back to "
            f"{execution_provider}; install/load the matching CUDA and cuDNN runtime"
        )
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise NeuralEvidenceError("neural evidence ONNX model must expose exactly one input")
    output = session.run(None, {inputs[0].name: features[None, :, :]})
    if not output:
        raise NeuralEvidenceError("neural evidence ONNX model produced no outputs")
    values = np.asarray(output[0], dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    return values, {
        "neural_runtime": "onnxruntime",
        "neural_runtime_version": getattr(ort, "__version__", "unknown"),
        "neural_execution_provider": execution_provider,
        "neural_available_providers": list(available),
    }


@lru_cache(maxsize=4)
def _onnx_session(model_path: str, model_sha256: str, providers: tuple[str, ...]) -> object:
    """Cache only hash-bound sessions so a replaced model cannot reuse stale weights."""
    del model_sha256
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise NeuralEvidenceError("onnxruntime is unavailable") from error
    if "CUDAExecutionProvider" in providers:
        _preload_pytorch_cuda_libraries()
    return ort.InferenceSession(model_path, providers=list(providers))


def _preload_pytorch_cuda_libraries() -> None:
    """Load PyTorch's bundled CUDA/cuDNN DLLs before ONNX Runtime on Windows.

    The reused Auto OTO environment intentionally supplies CUDA through
    PyTorch.  A CLI that imports only ONNX Runtime otherwise misses cudnn64_9
    even though the same environment can train and run the model.
    """
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return


def _log_mel(magnitude: np.ndarray, frequencies: np.ndarray, sample_rate: int, *, bands: int) -> np.ndarray:
    power = np.square(np.asarray(magnitude, dtype=np.float64))
    filters = _mel_filterbank(frequencies, sample_rate, bands)
    mel = np.maximum(power @ filters.T, 1e-10)
    return _zscore_matrix(np.log(mel)).astype(np.float32)


def _mel_filterbank(frequencies: np.ndarray, sample_rate: int, bands: int) -> np.ndarray:
    maximum = float(sample_rate) / 2.0
    mel_min = 0.0
    mel_max = 2595.0 * np.log10(1.0 + maximum / 700.0)
    mel_points = np.linspace(mel_min, mel_max, bands + 2)
    hz = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    filters = np.zeros((bands, len(frequencies)), dtype=np.float64)
    for index in range(bands):
        left, center, right = hz[index : index + 3]
        filters[index] = np.maximum(0.0, np.minimum((frequencies - left) / max(center - left, 1e-9), (right - frequencies) / max(right - center, 1e-9)))
    return filters


def _zscore(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    scale = float(np.median(np.abs(values - median))) * 1.4826
    return np.clip((values - median) / max(scale, 1e-4), -6.0, 6.0)


def _zscore_matrix(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    scale = np.median(np.abs(values - median), axis=0, keepdims=True) * 1.4826
    return np.clip((values - median) / np.maximum(scale, 1e-4), -6.0, 6.0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not all(isinstance(item, str) for item in value):
        raise NeuralEvidenceError(f"neural evidence manifest {name} is invalid")
    return tuple(value)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise NeuralEvidenceError(f"neural evidence manifest {name} must be a positive integer")
    return int(value)


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (float, int)) or float(value) <= 0.0:
        raise NeuralEvidenceError(f"neural evidence manifest {name} must be positive")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "NEURAL_FEATURE_BACKEND_ID",
    "NEURAL_INPUT_FEATURES",
    "NEURAL_MANIFEST_SCHEMA_VERSION",
    "NEURAL_MODEL_KIND",
    "NeuralEvidenceError",
    "infer_neural_tracks",
    "load_neural_evidence_manifest",
    "manifest_path_for_model",
    "neural_input_features",
    "write_neural_evidence_manifest",
]
