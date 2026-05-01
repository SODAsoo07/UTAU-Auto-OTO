from __future__ import annotations

import os
from typing import Literal

import numpy as np

from core.cvn_types import CvnPosterior, FrameFeatures

# C/V binary classifier. Non C/V frames should be handled as ignore in training.
CVN_LABELS: tuple[str, str] = ("C", "V")


def _safe_clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _normalize_rows(probs: np.ndarray) -> np.ndarray:
    sums = np.sum(probs, axis=1, keepdims=True)
    sums = np.where(sums <= 1e-8, 1.0, sums)
    return (probs / sums).astype(np.float32)


def _moving_average_rows(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    if values.shape[0] <= 1:
        return values.astype(np.float32)

    win = int(max(1, window))
    left = int(win // 2)
    right = int(win - 1 - left)
    padded = np.pad(values, ((left, right), (0, 0)), mode="edge")
    out = np.zeros_like(values, dtype=np.float32)
    kernel = np.ones((win,), dtype=np.float32) / float(win)
    for col in range(values.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out


def _enforce_min_run_length(labels: np.ndarray, min_run_frames: int) -> np.ndarray:
    if min_run_frames <= 1 or labels.size <= 1:
        return labels

    out = labels.copy()
    n = int(out.size)
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_run_frames:
            prev_label = out[i - 1] if i > 0 else None
            next_label = out[j] if j < n else None
            if prev_label is not None and next_label is not None:
                new_label = prev_label if prev_label == next_label else prev_label
            elif prev_label is not None:
                new_label = prev_label
            elif next_label is not None:
                new_label = next_label
            else:
                new_label = out[i]
            out[i:j] = new_label
        i = j
    return out


def _rule_based_probs(features: FrameFeatures) -> np.ndarray:
    rms = np.asarray(features.rms, dtype=np.float32)
    zcr = np.asarray(features.zcr, dtype=np.float32)
    flatness = _safe_clip01(np.asarray(features.spectral_flatness, dtype=np.float32))
    centroid = np.asarray(features.spectral_centroid, dtype=np.float32)
    voiced = np.asarray(features.voiced_mask, dtype=np.float32)
    f0 = np.asarray(features.f0_hz, dtype=np.float32)

    nyquist = max(float(features.sample_rate) * 0.5, 1.0)
    centroid_norm = _safe_clip01(centroid / float(nyquist))
    zcr_norm = _safe_clip01(zcr)
    f0_norm = _safe_clip01(f0 / 500.0)

    rms_p50 = float(np.percentile(rms, 50.0)) if rms.size else 0.0
    energy_scale = _safe_clip01(rms / max(rms_p50 + 1e-6, 1e-4))
    energy_scale = np.maximum(energy_scale, 0.20)

    p_v_raw = (
        0.58 * voiced
        + 0.17 * (1.0 - flatness)
        + 0.15 * (1.0 - centroid_norm)
        + 0.10 * f0_norm
    )
    p_c_raw = (
        0.40 * centroid_norm
        + 0.24 * zcr_norm
        + 0.24 * flatness
        + 0.12 * (1.0 - voiced)
    )
    p_v = _safe_clip01(p_v_raw * energy_scale)
    p_c = _safe_clip01(p_c_raw * energy_scale)
    probs = np.stack([p_c, p_v], axis=1)
    return _normalize_rows(probs)


def _feature_matrix_for_linear(features: FrameFeatures) -> np.ndarray:
    rms = np.asarray(features.rms, dtype=np.float32)
    zcr = np.asarray(features.zcr, dtype=np.float32)
    nyquist = max(float(features.sample_rate) * 0.5, 1.0)
    centroid = _safe_clip01(np.asarray(features.spectral_centroid, dtype=np.float32) / float(nyquist))
    flatness = _safe_clip01(np.asarray(features.spectral_flatness, dtype=np.float32))
    f0 = _safe_clip01(np.asarray(features.f0_hz, dtype=np.float32) / 500.0)
    voiced = np.asarray(features.voiced_mask, dtype=np.float32)
    return np.column_stack([rms, zcr, centroid, flatness, f0, voiced]).astype(np.float32)


def _resolve_c_threshold(raw: object, default: float = 0.5) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(np.clip(value, 0.01, 0.99))


def _resolve_env_c_threshold(default: float) -> float:
    raw = str(os.environ.get("UTOA_CVN_C_THRESHOLD", "") or "").strip()
    if raw:
        return _resolve_c_threshold(raw, default=default)
    return float(default)


def _linear_model_infer(features: FrameFeatures, model_path: str) -> tuple[np.ndarray, float] | None:
    if not model_path:
        return None
    try:
        data = np.load(model_path, allow_pickle=False)
    except Exception:
        return None

    if "weights" not in data or "bias" not in data:
        return None
    w = np.asarray(data["weights"], dtype=np.float32)
    b = np.asarray(data["bias"], dtype=np.float32)
    x = _feature_matrix_for_linear(features)

    if w.ndim != 2 or b.ndim != 1:
        return None
    if w.shape[0] != x.shape[1] or w.shape[1] != len(CVN_LABELS) or b.shape[0] != len(CVN_LABELS):
        return None

    logits = (x @ w) + b
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expv = np.exp(logits)
    probs = _normalize_rows(expv)
    c_threshold = 0.5
    if "c_threshold" in data:
        c_threshold = _resolve_c_threshold(data["c_threshold"], default=0.5)
    return probs, float(c_threshold)


def _labels_from_probs(probs: np.ndarray, *, c_threshold: float = 0.5) -> np.ndarray:
    th = _resolve_c_threshold(c_threshold, default=0.5)
    idx = np.where(np.asarray(probs[:, 0], dtype=np.float32) >= float(th), 0, 1).astype(np.int64)
    return np.asarray([CVN_LABELS[int(i)] for i in idx], dtype="<U1")


def predict_cvn(
    features: FrameFeatures,
    *,
    model_path: str = "",
    backend: Literal["rule", "logreg", "gbdt"] = "rule",
    smooth_window: int = 5,
    min_run_frames: int = 3,
    c_threshold: float | None = None,
) -> CvnPosterior:
    infer_c_threshold = 0.5
    if backend in {"logreg", "gbdt"}:
        infer = _linear_model_infer(features, model_path=model_path)
        if infer is None:
            probs = _rule_based_probs(features)
            infer_c_threshold = 0.5
        else:
            probs, infer_c_threshold = infer
    else:
        probs = _rule_based_probs(features)
        infer_c_threshold = 0.5

    probs = _moving_average_rows(probs, int(max(1, smooth_window)))
    probs = _normalize_rows(probs)

    if c_threshold is None:
        eff_c_threshold = _resolve_env_c_threshold(infer_c_threshold)
    else:
        eff_c_threshold = _resolve_c_threshold(c_threshold, default=infer_c_threshold)
    labels = _labels_from_probs(probs, c_threshold=eff_c_threshold)
    labels = _enforce_min_run_length(labels, int(max(1, min_run_frames)))

    idx_map = {label: i for i, label in enumerate(CVN_LABELS)}
    one_hot = np.zeros_like(probs, dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, idx_map[str(label)]] = 1.0
    blended = (0.65 * probs) + (0.35 * one_hot)
    blended = _normalize_rows(blended)

    return CvnPosterior(
        times_ms=np.asarray(features.times_ms, dtype=np.float32),
        probs=blended.astype(np.float32),
        labels=labels.astype("<U1"),
        label_order=CVN_LABELS,
    )


__all__ = ["CVN_LABELS", "predict_cvn"]
