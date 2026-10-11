from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .neural_evidence import NEURAL_INPUT_FEATURES, _preload_pytorch_cuda_libraries, neural_input_features
from .occurrence_ranker import build_occurrence_boundary_queries, monotonic_boundary_argmax, query_feature_matrix


class OccurrenceRankerRuntimeError(RuntimeError):
    pass


def infer_occurrence_boundary_scores(
    analysis: object,
    primitives: Mapping[str, np.ndarray],
    phones: Sequence[str],
    *,
    model_path: Path,
    provider: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Run the explicit occurrence ranker and jointly decode its frame scores."""
    model = model_path.resolve()
    if not model.is_file():
        raise OccurrenceRankerRuntimeError(f"occurrence ranker model is missing: {model}")
    features = neural_input_features(analysis, primitives)
    queries = build_occurrence_boundary_queries(phones)
    matrix = query_feature_matrix(queries)
    if not len(queries):
        return matrix, np.zeros(0, dtype=np.int64), {"requested": True, "applied": False, "reason": "fewer_than_two_phones"}
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise OccurrenceRankerRuntimeError("occurrence ranker requires onnxruntime-gpu") from error
    selected = str(provider).strip().casefold()
    if selected not in {"cuda", "cpu", "auto"}:
        raise OccurrenceRankerRuntimeError(f"unsupported occurrence ranker provider: {provider}")
    if selected == "cuda":
        _preload_pytorch_cuda_libraries()
        providers = ["CUDAExecutionProvider"]
    elif selected == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        _preload_pytorch_cuda_libraries()
        providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model), providers=providers)
    actual = session.get_providers()[0]
    if selected == "cuda" and actual != "CUDAExecutionProvider":
        raise OccurrenceRankerRuntimeError("CUDA occurrence ranker requested but ONNX Runtime fell back to CPU")
    outputs = session.run(None, {"features": features[None].astype(np.float32), "queries": matrix[None]})
    scores = np.asarray(outputs[0], dtype=np.float64)
    if scores.shape != (1, len(queries), len(features)) or not np.all(np.isfinite(scores)):
        raise OccurrenceRankerRuntimeError(f"occurrence ranker output shape mismatch: {scores.shape}")
    indices = monotonic_boundary_argmax(scores[0])
    return scores[0], indices, {
        "requested": True,
        "applied": True,
        "model_path": str(model),
        "model_sha256": _sha256(model),
        "execution_provider": actual,
        "query_count": len(queries),
        "query_feature_count": int(matrix.shape[1]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["OccurrenceRankerRuntimeError", "infer_occurrence_boundary_scores"]
