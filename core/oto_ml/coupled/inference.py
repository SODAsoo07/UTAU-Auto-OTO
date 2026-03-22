"""
Coupled mel+OTO inference.

모델 로드 및 추론(델타 예측) 로직을 포함합니다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from core.oto_ml.coupled.model import (
    ANCHOR_TARGET_NAMES,
    CATEGORICAL_FEATURES,
    COUPLED_MODEL_FILE,
    DELTA_TARGET_NAMES,
    FEATURE_NAMES,
    PATCH_FEATURES,
    TARGET_NAMES,
    _build_model,
    _build_model_rawmel,
    _import_torch,
    _require_numpy,
    _resolve_device,
    _row_categorical_index_vector,
    _row_feature_vector,
    _row_patch_vector,
    np,
)
from core.oto_ml.features.schema import get_feature_schema
from core.oto_ml_reliability import compute_mel_reliability_score


def load_coupled_bundle(
    model_dir: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    device: str = "auto",
):
    _require_numpy()
    torch, nn, _F = _import_torch()
    model_path = os.path.join(model_dir, COUPLED_MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    payload = torch.load(model_path, map_location="cpu")
    feature_schema = schema or get_feature_schema()
    feature_names = list(payload.get("feature_names") or feature_schema.get("feature_names") or FEATURE_NAMES)
    categorical_features = list(payload.get("categorical_features") or CATEGORICAL_FEATURES)
    categorical_bucket_sizes = list(payload.get("categorical_bucket_sizes") or [])
    patch_features = list(payload.get("patch_features") or PATCH_FEATURES)
    aux_dim = int(payload.get("aux_dim", 0) or 0)

    rawmel_enabled = bool(payload.get("rawmel_enabled", False))
    head_mode = str(payload.get("head_mode", "") or "single").strip().lower()
    if rawmel_enabled:
        mel_bins = int(payload.get("mel_bins", 80) or 80)
        onset_frames = int(payload.get("onset_frames", 1) or 1)
        tail_frames = int(payload.get("tail_frames", 1) or 1)
        model = _build_model_rawmel(
            torch,
            nn,
            in_dim=int(payload.get("in_dim", len(feature_names))),
            patch_dim=int(payload.get("patch_dim", len(patch_features))),
            mel_bins=mel_bins,
            onset_frames=onset_frames,
            tail_frames=tail_frames,
            hidden_dim=int(payload.get("hidden_dim", 160)),
            aux_dim=aux_dim,
            categorical_bucket_sizes=categorical_bucket_sizes,
            head_mode=head_mode,
        )
    else:
        model = _build_model(
            torch,
            nn,
            in_dim=int(payload.get("in_dim", len(feature_names))),
            patch_dim=int(payload.get("patch_dim", len(patch_features))),
            hidden_dim=int(payload.get("hidden_dim", 160)),
            aux_dim=aux_dim,
            categorical_bucket_sizes=categorical_bucket_sizes,
            head_mode=head_mode,
        )
    model.load_state_dict(payload["state_dict"])
    run_device = _resolve_device(torch, device)
    model = model.to(run_device)
    model.eval()
    return {
        "model": model,
        "meta": meta or {},
        "schema": feature_schema,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "categorical_bucket_sizes": categorical_bucket_sizes,
        "patch_features": patch_features,
        "head_mode": head_mode,
        "anchor_targets": list(payload.get("anchor_targets") or ANCHOR_TARGET_NAMES),
        "delta_targets": list(payload.get("delta_targets") or DELTA_TARGET_NAMES),
        "rawmel_enabled": rawmel_enabled,
        "mel_patch_spec": payload.get("mel_patch_spec") or {},
        "device": run_device,
    }


def predict_coupled_deltas(
    payload,
    feature_row: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], float]:
    _require_numpy()
    torch, _nn, _F = _import_torch()
    feature_names = list(payload.get("feature_names") or (schema or {}).get("feature_names") or FEATURE_NAMES)
    categorical_features = list(payload.get("categorical_features") or CATEGORICAL_FEATURES)
    categorical_bucket_sizes = list(payload.get("categorical_bucket_sizes") or [])
    patch_features = list(payload.get("patch_features") or PATCH_FEATURES)
    model = payload["model"]
    device = str(payload.get("device", "cpu"))

    x_vec = _row_feature_vector(feature_row, feature_names, categorical_features)
    cat_vec = _row_categorical_index_vector(feature_row, categorical_features, categorical_bucket_sizes)
    patch_vec = np.asarray(
        [float(feature_row.get(name, 0.0) or 0.0) if name in feature_row else 0.0 for name in patch_features],
        dtype=np.float32,
    )
    x_t = torch.tensor(x_vec.reshape(1, -1), dtype=torch.float32, device=device)
    cat_t = torch.tensor(cat_vec.reshape(1, -1), dtype=torch.long, device=device) if len(cat_vec) else None
    p_t = torch.tensor(patch_vec.reshape(1, -1), dtype=torch.float32, device=device)
    rawmel_enabled = bool(payload.get("rawmel_enabled", False))
    if rawmel_enabled:
        onset_patch = feature_row.get("mel_onset_patch")
        tail_patch = feature_row.get("mel_tail_patch")
        if onset_patch is None or tail_patch is None:
            raise RuntimeError("raw mel patches are required for coupled v2 inference")
        onset_np = np.asarray(onset_patch, dtype=np.float32)
        tail_np = np.asarray(tail_patch, dtype=np.float32)
        onset_t = torch.tensor(onset_np.reshape(1, 1, onset_np.shape[0], onset_np.shape[1]), dtype=torch.float32, device=device)
        tail_t = torch.tensor(tail_np.reshape(1, 1, tail_np.shape[0], tail_np.shape[1]), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(x_t, p_t, onset_t, tail_t, cat_t)
    else:
        with torch.no_grad():
            out = model(x_t, p_t, cat_t)

    head_mode = str(payload.get("head_mode", "") or "single").strip().lower()
    if head_mode == "split":
        if isinstance(out, tuple) and len(out) == 4:
            anchor_t, delta_t, conf_t, _aux_t = out
        else:
            anchor_t, delta_t, conf_t = out
        anchor_np = anchor_t.detach().cpu().numpy().reshape(-1)
        delta_np = delta_t.detach().cpu().numpy().reshape(-1)
        conf = float(conf_t.detach().cpu().numpy().reshape(-1)[0])
        scale_gamma = float(os.environ.get("UTOA_ML_ANCHOR_MEL_GAMMA", 1.0) or 1.0)
        if scale_gamma <= 0.0:
            scale_gamma = 1.0
        anchor_scale = float(compute_mel_reliability_score(feature_row)) ** float(scale_gamma)
        anchor_scale = max(0.0, min(1.0, anchor_scale))
        anchor_np = anchor_np * anchor_scale
        anchor_targets = list(payload.get("anchor_targets") or ANCHOR_TARGET_NAMES)
        delta_targets = list(payload.get("delta_targets") or DELTA_TARGET_NAMES)
        out_vals = {name: 0.0 for name in TARGET_NAMES}
        for idx, name in enumerate(anchor_targets):
            out_vals[name] = float(anchor_np[idx]) if idx < len(anchor_np) else 0.0
        for idx, name in enumerate(delta_targets):
            out_vals[name] = float(delta_np[idx]) if idx < len(delta_np) else 0.0
        return out_vals, conf
    if isinstance(out, tuple) and len(out) == 3:
        deltas_t, conf_t, _aux_t = out
    else:
        deltas_t, conf_t = out
    deltas_np = deltas_t.detach().cpu().numpy().reshape(-1)
    conf = float(conf_t.detach().cpu().numpy().reshape(-1)[0])
    out_vals = {target: float(deltas_np[i]) for i, target in enumerate(TARGET_NAMES)}
    return out_vals, conf


def predict_coupled_deltas_batch(
    payload,
    feature_rows: List[Dict[str, Any]],
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Dict[str, float], float]]:
    _require_numpy()
    if not feature_rows:
        return []

    torch, _nn, _F = _import_torch()
    feature_names = list(payload.get("feature_names") or (schema or {}).get("feature_names") or FEATURE_NAMES)
    categorical_features = list(payload.get("categorical_features") or CATEGORICAL_FEATURES)
    categorical_bucket_sizes = list(payload.get("categorical_bucket_sizes") or [])
    patch_features = list(payload.get("patch_features") or PATCH_FEATURES)
    model = payload["model"]
    device = str(payload.get("device", "cpu"))

    x_rows = []
    cat_rows = []
    patch_rows = []
    onset_rows = []
    tail_rows = []
    rawmel_enabled = bool(payload.get("rawmel_enabled", False))

    for feature_row in feature_rows:
        x_rows.append(_row_feature_vector(feature_row, feature_names, categorical_features))
        cat_rows.append(_row_categorical_index_vector(feature_row, categorical_features, categorical_bucket_sizes))
        patch_rows.append(
            np.asarray(
                [float(feature_row.get(name, 0.0) or 0.0) if name in feature_row else 0.0 for name in patch_features],
                dtype=np.float32,
            )
        )
        if rawmel_enabled:
            onset_patch = feature_row.get("mel_onset_patch")
            tail_patch = feature_row.get("mel_tail_patch")
            if onset_patch is None or tail_patch is None:
                raise RuntimeError("raw mel patches are required for coupled v2 batch inference")
            onset_rows.append(np.asarray(onset_patch, dtype=np.float32))
            tail_rows.append(np.asarray(tail_patch, dtype=np.float32))

    x_np = np.asarray(x_rows, dtype=np.float32)
    cat_np = np.asarray(cat_rows, dtype=np.int64) if cat_rows else np.zeros((len(feature_rows), 0), dtype=np.int64)
    patch_np = np.asarray(patch_rows, dtype=np.float32)

    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    cat_t = torch.tensor(cat_np, dtype=torch.long, device=device) if cat_np.shape[1] > 0 else None
    p_t = torch.tensor(patch_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        if rawmel_enabled:
            onset_np = np.asarray(onset_rows, dtype=np.float32)
            tail_np = np.asarray(tail_rows, dtype=np.float32)
            onset_t = torch.tensor(onset_np[:, None, :, :], dtype=torch.float32, device=device)
            tail_t = torch.tensor(tail_np[:, None, :, :], dtype=torch.float32, device=device)
            out = model(x_t, p_t, onset_t, tail_t, cat_t)
        else:
            out = model(x_t, p_t, cat_t)

    head_mode = str(payload.get("head_mode", "") or "single").strip().lower()
    results: List[Tuple[Dict[str, float], float]] = []

    if head_mode == "split":
        if isinstance(out, tuple) and len(out) == 4:
            anchor_t, delta_t, conf_t, _aux_t = out
        else:
            anchor_t, delta_t, conf_t = out
        anchor_np = anchor_t.detach().cpu().numpy()
        delta_np = delta_t.detach().cpu().numpy()
        conf_np = conf_t.detach().cpu().numpy().reshape(-1)
        scale_gamma = float(os.environ.get("UTOA_ML_ANCHOR_MEL_GAMMA", 1.0) or 1.0)
        if scale_gamma <= 0.0:
            scale_gamma = 1.0
        anchor_targets = list(payload.get("anchor_targets") or ANCHOR_TARGET_NAMES)
        delta_targets = list(payload.get("delta_targets") or DELTA_TARGET_NAMES)
        for row_idx, feature_row in enumerate(feature_rows):
            anchor_scale = float(compute_mel_reliability_score(feature_row)) ** float(scale_gamma)
            anchor_scale = max(0.0, min(1.0, anchor_scale))
            a_row = anchor_np[row_idx] * anchor_scale
            d_row = delta_np[row_idx]
            out_vals = {name: 0.0 for name in TARGET_NAMES}
            for idx, name in enumerate(anchor_targets):
                out_vals[name] = float(a_row[idx]) if idx < len(a_row) else 0.0
            for idx, name in enumerate(delta_targets):
                out_vals[name] = float(d_row[idx]) if idx < len(d_row) else 0.0
            results.append((out_vals, float(conf_np[row_idx] if row_idx < len(conf_np) else 0.0)))
        return results

    if isinstance(out, tuple) and len(out) == 3:
        deltas_t, conf_t, _aux_t = out
    else:
        deltas_t, conf_t = out
    deltas_np = deltas_t.detach().cpu().numpy()
    conf_np = conf_t.detach().cpu().numpy().reshape(-1)
    for row_idx in range(len(feature_rows)):
        row_vals = deltas_np[row_idx]
        out_vals = {target: float(row_vals[i]) for i, target in enumerate(TARGET_NAMES)}
        results.append((out_vals, float(conf_np[row_idx] if row_idx < len(conf_np) else 0.0)))
    return results
