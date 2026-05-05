"""
Coupled mel+OTO inference.
"""

from __future__ import annotations

import json
import logging
import os
import ctypes
from typing import Any, Dict, List, Optional, Tuple

from core.oto_ml.coupled.model import (
    ANCHOR_TARGET_NAMES,
    CATEGORICAL_FEATURES,
    COUPLED_MODEL_FILE,
    COUPLED_MODEL_ONNX_FILE,
    COUPLED_MODEL_ONNX_META_FILE,
    DELTA_TARGET_NAMES,
    FEATURE_NAMES,
    PATCH_FEATURES,
    TARGET_NAMES,
    _build_model,
    _build_model_rawmel,
    _import_onnxruntime,
    _import_torch,
    _require_numpy,
    _resolve_device,
    _row_categorical_index_vector,
    _row_feature_vector,
    np,
)
from core.oto_ml.features.schema import get_feature_schema
from core.oto_ml_reliability import compute_mel_reliability_score

logger = logging.getLogger(__name__)


def _onnx_enabled() -> bool:
    raw = str(os.environ.get("UTOA_ML_COUPLED_ONNX_ENABLE", "1") or "").strip().lower()
    return raw not in {"0", "false", "off", "no", "n"}


def _load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f) or {})
    except Exception:
        return {}


def _select_onnx_providers(ort, device: str = "auto") -> Tuple[List[str], str]:
    req = str(device or "auto").strip().lower()
    available = set(ort.get_available_providers() or [])
    if req in {"auto", "cuda"} and "CUDAExecutionProvider" in available and _onnx_cuda_runtime_ready():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
    if req == "cuda":
        logger.warning("[OTO-ML] CUDA requested but runtime dependencies are unavailable. Falling back to CPU.")
    if req == "dml" and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"], "dml"
    return ["CPUExecutionProvider"], "cpu"


def _onnx_cuda_runtime_ready() -> bool:
    if str(os.name).strip().lower() != "nt":
        return True
    required = ["cudnn64_9.dll"]
    for dll_name in required:
        try:
            ctypes.WinDLL(dll_name)
        except Exception:
            logger.info("[OTO-ML] ONNX CUDA dependency missing: %s", dll_name)
            return False
    return True


def _resolve_onnx_io(session, sidecar: Dict[str, Any]) -> Dict[str, Any]:
    sidecar_inputs = dict(sidecar.get("input_names") or {})
    sidecar_outputs = dict(sidecar.get("output_names") or {})
    inputs = list(session.get_inputs() or [])
    outputs = list(session.get_outputs() or [])
    input_names = [str(v.name) for v in inputs]
    output_names = [str(v.name) for v in outputs]

    rawmel_enabled = bool(sidecar.get("rawmel_enabled", False))
    use_cat_input = bool(sidecar.get("use_cat_input", False))
    if not sidecar:
        rawmel_enabled = len(input_names) >= 4
        use_cat_input = len(input_names) >= (5 if rawmel_enabled else 3)

    if rawmel_enabled:
        x_name = sidecar_inputs.get("x") or (input_names[0] if len(input_names) >= 1 else "")
        patch_name = sidecar_inputs.get("patch") or (input_names[1] if len(input_names) >= 2 else "")
        onset_name = sidecar_inputs.get("onset") or (input_names[2] if len(input_names) >= 3 else "")
        tail_name = sidecar_inputs.get("tail") or (input_names[3] if len(input_names) >= 4 else "")
        cat_name = sidecar_inputs.get("cat") or (input_names[4] if len(input_names) >= 5 else "")
    else:
        x_name = sidecar_inputs.get("x") or (input_names[0] if len(input_names) >= 1 else "")
        patch_name = sidecar_inputs.get("patch") or (input_names[1] if len(input_names) >= 2 else "")
        onset_name = ""
        tail_name = ""
        cat_name = sidecar_inputs.get("cat") or (input_names[2] if len(input_names) >= 3 else "")

    head_mode = str(sidecar.get("head_mode", "single") or "single").strip().lower()
    if head_mode == "split":
        anchor_name = sidecar_outputs.get("anchor") or (output_names[0] if len(output_names) >= 1 else "")
        delta_name = sidecar_outputs.get("delta") or (output_names[1] if len(output_names) >= 2 else "")
        conf_name = sidecar_outputs.get("confidence") or (output_names[2] if len(output_names) >= 3 else "")
        deltas_name = ""
    else:
        deltas_name = sidecar_outputs.get("deltas") or (output_names[0] if len(output_names) >= 1 else "")
        conf_name = sidecar_outputs.get("confidence") or (output_names[1] if len(output_names) >= 2 else "")
        anchor_name = ""
        delta_name = ""

    return {
        "rawmel_enabled": bool(rawmel_enabled),
        "use_cat_input": bool(use_cat_input and cat_name),
        "head_mode": head_mode if head_mode in {"single", "split"} else "single",
        "input_names": {
            "x": x_name,
            "patch": patch_name,
            "onset": onset_name,
            "tail": tail_name,
            "cat": cat_name,
        },
        "output_names": {
            "deltas": deltas_name,
            "anchor": anchor_name,
            "delta": delta_name,
            "confidence": conf_name,
        },
    }


def _compose_split_row(
    payload: Dict[str, Any],
    feature_row: Dict[str, Any],
    anchor_row: "np.ndarray",
    delta_row: "np.ndarray",
) -> Dict[str, float]:
    scale_gamma = float(os.environ.get("UTOA_ML_ANCHOR_MEL_GAMMA", 1.0) or 1.0)
    if scale_gamma <= 0.0:
        scale_gamma = 1.0
    anchor_scale = float(compute_mel_reliability_score(feature_row)) ** float(scale_gamma)
    anchor_scale = max(0.0, min(1.0, anchor_scale))
    scaled_anchor = anchor_row * anchor_scale
    anchor_targets = list(payload.get("anchor_targets") or ANCHOR_TARGET_NAMES)
    delta_targets = list(payload.get("delta_targets") or DELTA_TARGET_NAMES)
    out_vals = {name: 0.0 for name in TARGET_NAMES}
    for idx, name in enumerate(anchor_targets):
        out_vals[name] = float(scaled_anchor[idx]) if idx < len(scaled_anchor) else 0.0
    for idx, name in enumerate(delta_targets):
        out_vals[name] = float(delta_row[idx]) if idx < len(delta_row) else 0.0
    return out_vals


def _load_coupled_bundle_onnx(
    model_dir: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    device: str = "auto",
):
    _require_numpy()
    ort = _import_onnxruntime()
    onnx_path = os.path.join(model_dir, COUPLED_MODEL_ONNX_FILE)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(onnx_path)
    sidecar = _load_json(os.path.join(model_dir, COUPLED_MODEL_ONNX_META_FILE))
    providers, run_device = _select_onnx_providers(ort, device=device)
    session = ort.InferenceSession(onnx_path, providers=providers)
    io_meta = _resolve_onnx_io(session, sidecar)
    feature_schema = schema or get_feature_schema()
    feature_names = list(sidecar.get("feature_names") or feature_schema.get("feature_names") or FEATURE_NAMES)
    categorical_features = list(sidecar.get("categorical_features") or CATEGORICAL_FEATURES)
    categorical_bucket_sizes = list(sidecar.get("categorical_bucket_sizes") or [])
    patch_features = list(sidecar.get("patch_features") or PATCH_FEATURES)
    rawmel_enabled = bool(io_meta.get("rawmel_enabled", False))
    head_mode = str(io_meta.get("head_mode", sidecar.get("head_mode", "single")) or "single").strip().lower()
    return {
        "runtime": "onnx",
        "onnx_session": session,
        "onnx_path": onnx_path,
        "onnx_input_names": dict(io_meta.get("input_names") or {}),
        "onnx_output_names": dict(io_meta.get("output_names") or {}),
        "use_cat_input": bool(io_meta.get("use_cat_input", False)),
        "meta": meta or {},
        "schema": feature_schema,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "categorical_bucket_sizes": categorical_bucket_sizes,
        "patch_features": patch_features,
        "head_mode": head_mode,
        "alias_branch_mode": str(sidecar.get("alias_branch_mode", "shared") or "shared"),
        "alias_branch_experts": int(sidecar.get("alias_branch_experts", 4) or 4),
        "alias_type_cat_index": int(sidecar.get("alias_type_cat_index", -1) or -1),
        "alias_type_bucket_size": int(sidecar.get("alias_type_bucket_size", 0) or 0),
        "alias_fallback_ids": [int(v) for v in (sidecar.get("alias_fallback_ids", []) or [])],
        "anchor_targets": list(sidecar.get("anchor_targets") or ANCHOR_TARGET_NAMES),
        "delta_targets": list(sidecar.get("delta_targets") or DELTA_TARGET_NAMES),
        "rawmel_enabled": rawmel_enabled,
        "mel_patch_spec": sidecar.get("mel_patch_spec") or {},
        "mel_bins": int(sidecar.get("mel_bins", 80) or 80),
        "onset_frames": int(sidecar.get("onset_frames", 8) or 8),
        "tail_frames": int(sidecar.get("tail_frames", 8) or 8),
        "device": run_device,
    }


def _load_coupled_bundle_torch(
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
            alias_branch_mode=str(payload.get("alias_branch_mode", "shared") or "shared"),
            alias_branch_experts=int(payload.get("alias_branch_experts", 4) or 4),
            alias_type_cat_index=int(payload.get("alias_type_cat_index", -1) or -1),
            alias_type_bucket_size=int(payload.get("alias_type_bucket_size", 0) or 0),
            alias_fallback_ids=list(payload.get("alias_fallback_ids", []) or []),
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
            alias_branch_mode=str(payload.get("alias_branch_mode", "shared") or "shared"),
            alias_branch_experts=int(payload.get("alias_branch_experts", 4) or 4),
            alias_type_cat_index=int(payload.get("alias_type_cat_index", -1) or -1),
            alias_type_bucket_size=int(payload.get("alias_type_bucket_size", 0) or 0),
            alias_fallback_ids=list(payload.get("alias_fallback_ids", []) or []),
        )
    model.load_state_dict(payload["state_dict"])
    run_device = _resolve_device(torch, device)
    model = model.to(run_device)
    model.eval()
    return {
        "runtime": "torch",
        "model": model,
        "meta": meta or {},
        "schema": feature_schema,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "categorical_bucket_sizes": categorical_bucket_sizes,
        "patch_features": patch_features,
        "head_mode": head_mode,
        "alias_branch_mode": str(payload.get("alias_branch_mode", "shared") or "shared"),
        "alias_branch_experts": int(payload.get("alias_branch_experts", 4) or 4),
        "alias_type_cat_index": int(payload.get("alias_type_cat_index", -1) or -1),
        "alias_type_bucket_size": int(payload.get("alias_type_bucket_size", 0) or 0),
        "alias_fallback_ids": [int(v) for v in (payload.get("alias_fallback_ids", []) or [])],
        "anchor_targets": list(payload.get("anchor_targets") or ANCHOR_TARGET_NAMES),
        "delta_targets": list(payload.get("delta_targets") or DELTA_TARGET_NAMES),
        "rawmel_enabled": rawmel_enabled,
        "mel_patch_spec": payload.get("mel_patch_spec") or {},
        "mel_bins": int(payload.get("mel_bins", 80) or 80),
        "onset_frames": int(payload.get("onset_frames", 8) or 8),
        "tail_frames": int(payload.get("tail_frames", 8) or 8),
        "device": run_device,
    }


def load_coupled_bundle(
    model_dir: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    device: str = "auto",
):
    if _onnx_enabled():
        onnx_path = os.path.join(model_dir, COUPLED_MODEL_ONNX_FILE)
        if os.path.exists(onnx_path):
            try:
                return _load_coupled_bundle_onnx(model_dir, meta=meta, schema=schema, device=device)
            except Exception as exc:
                logger.warning("Failed to load coupled ONNX bundle (%s): %s. Falling back to torch.", model_dir, exc)
    return _load_coupled_bundle_torch(model_dir, meta=meta, schema=schema, device=device)


def _predict_coupled_deltas_torch(
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
        out_vals = _compose_split_row(payload, feature_row, anchor_np, delta_np)
        return out_vals, conf
    if isinstance(out, tuple) and len(out) == 3:
        deltas_t, conf_t, _aux_t = out
    else:
        deltas_t, conf_t = out
    deltas_np = deltas_t.detach().cpu().numpy().reshape(-1)
    conf = float(conf_t.detach().cpu().numpy().reshape(-1)[0])
    out_vals = {target: float(deltas_np[i]) for i, target in enumerate(TARGET_NAMES)}
    return out_vals, conf


def _predict_coupled_deltas_onnx(
    payload,
    feature_row: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], float]:
    _require_numpy()
    feature_names = list(payload.get("feature_names") or (schema or {}).get("feature_names") or FEATURE_NAMES)
    categorical_features = list(payload.get("categorical_features") or CATEGORICAL_FEATURES)
    categorical_bucket_sizes = list(payload.get("categorical_bucket_sizes") or [])
    patch_features = list(payload.get("patch_features") or PATCH_FEATURES)
    session = payload["onnx_session"]
    input_names = dict(payload.get("onnx_input_names") or {})
    output_names = dict(payload.get("onnx_output_names") or {})

    x_np = _row_feature_vector(feature_row, feature_names, categorical_features).reshape(1, -1).astype(np.float32)
    cat_np = _row_categorical_index_vector(feature_row, categorical_features, categorical_bucket_sizes).reshape(1, -1).astype(np.int64)
    patch_np = np.asarray(
        [float(feature_row.get(name, 0.0) or 0.0) if name in feature_row else 0.0 for name in patch_features],
        dtype=np.float32,
    ).reshape(1, -1)

    feed: Dict[str, Any] = {
        str(input_names.get("x", "x")): x_np,
        str(input_names.get("patch", "patch")): patch_np,
    }
    if bool(payload.get("use_cat_input", False)):
        feed[str(input_names.get("cat", "cat_idx"))] = cat_np

    if bool(payload.get("rawmel_enabled", False)):
        onset_patch = feature_row.get("mel_onset_patch")
        tail_patch = feature_row.get("mel_tail_patch")
        if onset_patch is None or tail_patch is None:
            raise RuntimeError("raw mel patches are required for coupled v2 ONNX inference")
        onset_np = np.asarray(onset_patch, dtype=np.float32)
        tail_np = np.asarray(tail_patch, dtype=np.float32)
        feed[str(input_names.get("onset", "onset"))] = onset_np.reshape(1, 1, onset_np.shape[0], onset_np.shape[1])
        feed[str(input_names.get("tail", "tail"))] = tail_np.reshape(1, 1, tail_np.shape[0], tail_np.shape[1])

    raw_out = session.run(None, feed)
    head_mode = str(payload.get("head_mode", "") or "single").strip().lower()
    if head_mode == "split":
        anchor_np = np.asarray(raw_out[0], dtype=np.float32).reshape(1, -1)[0]
        delta_np = np.asarray(raw_out[1], dtype=np.float32).reshape(1, -1)[0]
        conf = float(np.asarray(raw_out[2], dtype=np.float32).reshape(-1)[0]) if len(raw_out) >= 3 else 0.0
        out_vals = _compose_split_row(payload, feature_row, anchor_np, delta_np)
        return out_vals, conf
    deltas_np = np.asarray(raw_out[0], dtype=np.float32).reshape(1, -1)[0]
    conf = float(np.asarray(raw_out[1], dtype=np.float32).reshape(-1)[0]) if len(raw_out) >= 2 else 0.0
    out_vals = {target: float(deltas_np[i]) for i, target in enumerate(TARGET_NAMES)}
    return out_vals, conf


def predict_coupled_deltas(
    payload,
    feature_row: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], float]:
    runtime = str(payload.get("runtime", "torch")).strip().lower()
    if runtime == "onnx":
        return _predict_coupled_deltas_onnx(payload, feature_row, meta=meta, schema=schema)
    return _predict_coupled_deltas_torch(payload, feature_row, meta=meta, schema=schema)


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

    runtime = str(payload.get("runtime", "torch")).strip().lower()
    if runtime == "onnx":
        feature_names = list(payload.get("feature_names") or (schema or {}).get("feature_names") or FEATURE_NAMES)
        categorical_features = list(payload.get("categorical_features") or CATEGORICAL_FEATURES)
        categorical_bucket_sizes = list(payload.get("categorical_bucket_sizes") or [])
        patch_features = list(payload.get("patch_features") or PATCH_FEATURES)
        session = payload["onnx_session"]
        input_names = dict(payload.get("onnx_input_names") or {})

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
                    raise RuntimeError("raw mel patches are required for coupled v2 batch ONNX inference")
                onset_rows.append(np.asarray(onset_patch, dtype=np.float32))
                tail_rows.append(np.asarray(tail_patch, dtype=np.float32))

        x_np = np.asarray(x_rows, dtype=np.float32)
        cat_np = np.asarray(cat_rows, dtype=np.int64) if cat_rows else np.zeros((len(feature_rows), 0), dtype=np.int64)
        patch_np = np.asarray(patch_rows, dtype=np.float32)

        feed: Dict[str, Any] = {
            str(input_names.get("x", "x")): x_np,
            str(input_names.get("patch", "patch")): patch_np,
        }
        if bool(payload.get("use_cat_input", False)):
            feed[str(input_names.get("cat", "cat_idx"))] = cat_np
        if rawmel_enabled:
            onset_np = np.asarray(onset_rows, dtype=np.float32)
            tail_np = np.asarray(tail_rows, dtype=np.float32)
            feed[str(input_names.get("onset", "onset"))] = onset_np[:, None, :, :]
            feed[str(input_names.get("tail", "tail"))] = tail_np[:, None, :, :]

        raw_out = session.run(None, feed)
        head_mode = str(payload.get("head_mode", "") or "single").strip().lower()
        results: List[Tuple[Dict[str, float], float]] = []
        if head_mode == "split":
            anchor_np = np.asarray(raw_out[0], dtype=np.float32)
            delta_np = np.asarray(raw_out[1], dtype=np.float32)
            conf_np = np.asarray(raw_out[2], dtype=np.float32).reshape(-1) if len(raw_out) >= 3 else np.zeros((len(feature_rows),), dtype=np.float32)
            for row_idx, feature_row in enumerate(feature_rows):
                out_vals = _compose_split_row(payload, feature_row, anchor_np[row_idx], delta_np[row_idx])
                results.append((out_vals, float(conf_np[row_idx] if row_idx < len(conf_np) else 0.0)))
            return results

        deltas_np = np.asarray(raw_out[0], dtype=np.float32)
        conf_np = np.asarray(raw_out[1], dtype=np.float32).reshape(-1) if len(raw_out) >= 2 else np.zeros((len(feature_rows),), dtype=np.float32)
        for row_idx in range(len(feature_rows)):
            row_vals = deltas_np[row_idx]
            out_vals = {target: float(row_vals[i]) for i, target in enumerate(TARGET_NAMES)}
            results.append((out_vals, float(conf_np[row_idx] if row_idx < len(conf_np) else 0.0)))
        return results

    # torch path
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
        for row_idx, feature_row in enumerate(feature_rows):
            out_vals = _compose_split_row(payload, feature_row, anchor_np[row_idx], delta_np[row_idx])
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
