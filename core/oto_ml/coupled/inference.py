"""
Coupled mel+OTO inference.

모델 로드 및 추론(델타 예측) 로직을 포함합니다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from core.oto_ml.coupled.model import (
    CATEGORICAL_FEATURES,
    COUPLED_MODEL_FILE,
    FEATURE_NAMES,
    PATCH_FEATURES,
    TARGET_NAMES,
    _build_model,
    _build_model_rawmel,
    _import_torch,
    _require_numpy,
    _resolve_device,
    _row_feature_vector,
    _row_patch_vector,
    np,
)
from core.oto_ml.features.schema import get_feature_schema


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
    patch_features = list(payload.get("patch_features") or PATCH_FEATURES)
    aux_dim = int(payload.get("aux_dim", 0) or 0)

    rawmel_enabled = bool(payload.get("rawmel_enabled", False))
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
        )
    else:
        model = _build_model(
            torch,
            nn,
            in_dim=int(payload.get("in_dim", len(feature_names))),
            patch_dim=int(payload.get("patch_dim", len(patch_features))),
            hidden_dim=int(payload.get("hidden_dim", 160)),
            aux_dim=aux_dim,
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
        "patch_features": patch_features,
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
    patch_features = list(payload.get("patch_features") or PATCH_FEATURES)
    model = payload["model"]
    device = str(payload.get("device", "cpu"))

    x_vec = _row_feature_vector(feature_row, feature_names, categorical_features)
    patch_vec = np.asarray(
        [float(feature_row.get(name, 0.0) or 0.0) if name in feature_row else 0.0 for name in patch_features],
        dtype=np.float32,
    )
    x_t = torch.tensor(x_vec.reshape(1, -1), dtype=torch.float32, device=device)
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
            out = model(x_t, p_t, onset_t, tail_t)
    else:
        with torch.no_grad():
            out = model(x_t, p_t)
        if isinstance(out, tuple) and len(out) == 3:
            deltas_t, conf_t, _aux_t = out
        else:
            deltas_t, conf_t = out
    deltas_np = deltas_t.detach().cpu().numpy().reshape(-1)
    conf = float(conf_t.detach().cpu().numpy().reshape(-1)[0])
    out = {target: float(deltas_np[i]) for i, target in enumerate(TARGET_NAMES)}
    return out, conf
