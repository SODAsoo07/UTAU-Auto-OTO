"""
Coupled mel+OTO backend (coupled_nn_v1).

This backend predicts all OTO deltas jointly and emits model confidence.
Runtime can fall back to LightGBM/base when torch/model inference is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception as _pd_exc:  # pragma: no cover
    pd = None
    PANDAS_IMPORT_ERROR = _pd_exc
else:
    PANDAS_IMPORT_ERROR = None

try:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupShuffleSplit
except Exception as _sk_exc:  # pragma: no cover
    mean_absolute_error = None
    GroupShuffleSplit = None
    SKLEARN_IMPORT_ERROR = _sk_exc
else:
    SKLEARN_IMPORT_ERROR = None

from core.format_type_utils import normalize_format_type
from core.oto_ml_features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TARGET_NAMES,
    build_training_rows,
    canonicalize_feature_row,
    get_feature_schema,
    write_dataset_csv,
    write_feature_schema,
)

logger = logging.getLogger(__name__)

COUPLED_BACKEND = "coupled_nn_v1"
COUPLED_MODEL_FILE = "coupled_model.pt"
PATCH_FEATURES = [
    "onset_patch_energy_mean",
    "onset_patch_voiced_ratio",
    "onset_patch_unvoiced_ratio",
    "tail_patch_energy_mean",
    "tail_patch_silence_ratio",
    "blank_span_confidence",
]


def _require_numpy():
    if np is None:
        raise RuntimeError("numpy is required for coupled backend")


def _require_training_stack():
    _require_numpy()
    if pd is None:
        raise RuntimeError(f"pandas is required: {PANDAS_IMPORT_ERROR}")
    if mean_absolute_error is None:
        raise RuntimeError(f"scikit-learn is required: {SKLEARN_IMPORT_ERROR}")


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"torch is required for coupled backend: {exc}") from exc


def _stable_hash_to_unit(text: str) -> float:
    raw = str(text or "").strip().encode("utf-8", errors="replace")
    digest = hashlib.sha1(raw).digest()
    value = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return float(value % 1_000_000) / 1_000_000.0


def _row_feature_vector(feature_row: Dict[str, Any], feature_names: List[str], categorical_features: List[str]) -> np.ndarray:
    canon = canonicalize_feature_row(feature_row, feature_names=feature_names)
    out: List[float] = []
    cat_set = set(categorical_features)
    for name in feature_names:
        if name in cat_set:
            out.append(_stable_hash_to_unit(canon.get(name, "")))
        else:
            try:
                out.append(float(canon.get(name, 0.0)))
            except Exception:
                out.append(0.0)
    return np.asarray(out, dtype=np.float32)


def _row_patch_vector(feature_row: Dict[str, Any]) -> np.ndarray:
    vals: List[float] = []
    for key in PATCH_FEATURES:
        try:
            vals.append(float(feature_row.get(key, 0.0) or 0.0))
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


def _build_model(torch, nn, in_dim: int, patch_dim: int, hidden_dim: int = 160):
    class _CoupledModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.patch_net = nn.Sequential(
                nn.Linear(patch_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.joint = nn.Sequential(
                nn.Linear(hidden_dim + 64, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.delta_head = nn.Linear(hidden_dim // 2, len(TARGET_NAMES))
            self.conf_head = nn.Sequential(nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

        def forward(self, x, patch):
            xf = self.feature_net(x)
            xp = self.patch_net(patch)
            z = self.joint(torch.cat([xf, xp], dim=1))
            deltas = self.delta_head(z)
            conf = self.conf_head(z)
            return deltas, conf

    return _CoupledModel()


def _resolve_device(torch, requested: str = "auto") -> str:
    req = str(requested or "auto").strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return req


def build_and_save_coupled_dataset(
    language: str,
    auto_oto_path: str,
    manual_oto_path: str,
    tg_dir: str,
    wav_dir: str,
    out_csv: str,
    custom_phonemes_path: str = "",
    voicebank_id: str = "",
    append: bool = False,
    format_type_override: str = "",
) -> Dict[str, int]:
    rows, stats = build_training_rows(
        language=language,
        auto_oto_path=auto_oto_path,
        manual_oto_path=manual_oto_path,
        tg_dir=tg_dir,
        wav_dir=wav_dir,
        custom_phonemes_path=custom_phonemes_path,
        voicebank_id=voicebank_id,
        format_type_override=format_type_override,
    )
    if rows:
        write_dataset_csv(out_csv, rows, append=append)
    out = dict(stats)
    out["saved_rows"] = int(len(rows))
    out["out_csv"] = os.path.abspath(out_csv)
    return out


def _prepare_training_frame(
    df,
    language: str,
    format_type: str,
    alias_types: Optional[List[str]] = None,
    min_mapping_confidence: float = 0.0,
):
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == lang]
    if "format_type" in df.columns and fmt and fmt != "general":
        df = df[
            df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type(lang, v)) == fmt
        ]
    if alias_types and "alias_type" in df.columns:
        normalized = [str(v).strip().lower() for v in alias_types if str(v).strip()]
        if normalized:
            df = df[df["alias_type"].astype(str).str.lower().isin(normalized)]
    if float(min_mapping_confidence) > 0.0 and "mapping_confidence" in df.columns:
        df = df[pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(0.0) >= float(min_mapping_confidence)]
    return df


def train_coupled_bundle(
    language: str,
    format_type: str,
    dataset_csv: str,
    out_dir: str,
    *,
    group_column: str = "voicebank_id",
    alias_types: Optional[List[str]] = None,
    min_mapping_confidence: float = 0.0,
    device: str = "auto",
    epochs: int = 70,
    batch_size: int = 192,
    learning_rate: float = 1e-3,
    min_confidence: float = 0.55,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.exists(dataset_csv):
        raise FileNotFoundError(dataset_csv)

    torch, nn, F = _import_torch()
    df = pd.read_csv(dataset_csv, low_memory=False)
    df = _prepare_training_frame(
        df,
        language=language,
        format_type=format_type,
        alias_types=alias_types,
        min_mapping_confidence=min_mapping_confidence,
    )
    if len(df) < 16:
        raise RuntimeError("Coupled dataset is too small (need >= 16 rows).")

    schema = get_feature_schema()
    feature_names = list(schema.get("feature_names") or FEATURE_NAMES)
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in feature_names]

    x_rows = []
    p_rows = []
    for _, row in df.iterrows():
        as_dict = row.to_dict()
        x_rows.append(_row_feature_vector(as_dict, feature_names, categorical_features))
        p_rows.append(_row_patch_vector(as_dict))
    X = np.asarray(x_rows, dtype=np.float32)
    P = np.asarray(p_rows, dtype=np.float32)
    Y = np.stack(
        [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in TARGET_NAMES],
        axis=1,
    )
    base = np.stack(
        [
            pd.to_numeric(df["base_offset"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_cons"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_cutoff_abs"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_pre"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["base_ovl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    boundary = np.stack(
        [
            pd.to_numeric(df["mel_offset_candidate_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(df["mel_cutoff_candidate_ms"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    if "sample_weight" in df.columns:
        W = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    else:
        W = np.ones((len(df),), dtype=np.float32)

    if group_column in df.columns and df[group_column].nunique() >= 2 and GroupShuffleSplit is not None:
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(split.split(df, groups=df[group_column]))
    else:
        split_at = max(1, int(len(df) * 0.8))
        train_idx = np.arange(split_at)
        valid_idx = np.arange(split_at, len(df))
        if len(valid_idx) <= 0:
            valid_idx = np.arange(max(0, len(df) - 1), len(df))

    train_idx = np.asarray(train_idx, dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)

    model = _build_model(torch, nn, in_dim=int(X.shape[1]), patch_dim=int(P.shape[1]))
    run_device = _resolve_device(torch, requested=device)
    model = model.to(run_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=run_device)
    P_train = torch.tensor(P[train_idx], dtype=torch.float32, device=run_device)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32, device=run_device)
    B_train = torch.tensor(base[train_idx], dtype=torch.float32, device=run_device)
    M_train = torch.tensor(boundary[train_idx], dtype=torch.float32, device=run_device)
    W_train = torch.tensor(W[train_idx], dtype=torch.float32, device=run_device)

    X_valid = torch.tensor(X[valid_idx], dtype=torch.float32, device=run_device)
    P_valid = torch.tensor(P[valid_idx], dtype=torch.float32, device=run_device)
    Y_valid = torch.tensor(Y[valid_idx], dtype=torch.float32, device=run_device)
    B_valid = torch.tensor(base[valid_idx], dtype=torch.float32, device=run_device)
    M_valid = torch.tensor(boundary[valid_idx], dtype=torch.float32, device=run_device)
    W_valid = torch.tensor(W[valid_idx], dtype=torch.float32, device=run_device)

    target_weights = torch.tensor([1.00, 0.90, 0.95, 1.00, 0.65], dtype=torch.float32, device=run_device).view(1, -1)
    cons_margin = 10.0
    cut_margin = 10.0

    best_state = None
    best_val = float("inf")
    wait = 0
    patience = 10

    train_n = int(X_train.shape[0])
    batch_n = max(1, int(batch_size))

    for _epoch in range(max(1, int(epochs))):
        model.train()
        perm = torch.randperm(train_n, device=run_device)
        for start in range(0, train_n, batch_n):
            batch_idx = perm[start:start + batch_n]
            xb = X_train[batch_idx]
            pb = P_train[batch_idx]
            yb = Y_train[batch_idx]
            bb = B_train[batch_idx]
            mb = M_train[batch_idx]
            wb = W_train[batch_idx]

            pred, conf = model(xb, pb)
            base_loss = F.smooth_l1_loss(pred, yb, reduction="none")
            base_loss = torch.mean(base_loss * target_weights, dim=1)
            base_loss = torch.mean(base_loss * wb)

            offset = bb[:, 0] + pred[:, 0]
            consonant = bb[:, 1] + pred[:, 1]
            cutoff_abs = bb[:, 2] + pred[:, 2]
            pre = bb[:, 3] + pred[:, 3]
            ovl = bb[:, 4] + pred[:, 4]
            penalty = (
                torch.relu(ovl - pre)
                + torch.relu((pre + cons_margin) - consonant)
                + torch.relu((consonant + cut_margin) - cutoff_abs)
                + torch.relu(-offset)
            )
            penalty_loss = torch.mean(penalty * wb)

            align_loss = F.smooth_l1_loss(offset, mb[:, 0], reduction="none") + F.smooth_l1_loss(
                cutoff_abs, mb[:, 1], reduction="none"
            )
            align_loss = torch.mean(align_loss * wb)

            err = torch.mean(torch.abs(pred - yb), dim=1)
            conf_target = torch.exp(-torch.clamp(err / 80.0, min=0.0, max=8.0))
            conf_loss = F.binary_cross_entropy(conf.squeeze(1), conf_target.detach(), reduction="none")
            conf_loss = torch.mean(conf_loss * wb)

            total_loss = base_loss + (0.25 * penalty_loss) + (0.12 * align_loss) + (0.05 * conf_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val, conf_val = model(X_valid, P_valid)
            val_base = F.smooth_l1_loss(pred_val, Y_valid, reduction="none")
            val_base = torch.mean(val_base * target_weights, dim=1)
            val_base = torch.mean(val_base * W_valid)
            offset_v = B_valid[:, 0] + pred_val[:, 0]
            consonant_v = B_valid[:, 1] + pred_val[:, 1]
            cutoff_abs_v = B_valid[:, 2] + pred_val[:, 2]
            pre_v = B_valid[:, 3] + pred_val[:, 3]
            ovl_v = B_valid[:, 4] + pred_val[:, 4]
            val_penalty = (
                torch.relu(ovl_v - pre_v)
                + torch.relu((pre_v + cons_margin) - consonant_v)
                + torch.relu((consonant_v + cut_margin) - cutoff_abs_v)
                + torch.relu(-offset_v)
            )
            val_penalty = torch.mean(val_penalty * W_valid)
            val_align = F.smooth_l1_loss(offset_v, M_valid[:, 0], reduction="none") + F.smooth_l1_loss(
                cutoff_abs_v, M_valid[:, 1], reduction="none"
            )
            val_align = torch.mean(val_align * W_valid)
            err_v = torch.mean(torch.abs(pred_val - Y_valid), dim=1)
            conf_target_v = torch.exp(-torch.clamp(err_v / 80.0, min=0.0, max=8.0))
            val_conf = F.binary_cross_entropy(conf_val.squeeze(1), conf_target_v.detach(), reduction="none")
            val_conf = torch.mean(val_conf * W_valid)
            val_total = float((val_base + (0.25 * val_penalty) + (0.12 * val_align) + (0.05 * val_conf)).item())

        if val_total < best_val:
            best_val = val_total
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_valid, conf_valid = model(X_valid, P_valid)
    pred_valid_np = pred_valid.detach().cpu().numpy()
    conf_valid_np = conf_valid.detach().cpu().numpy().reshape(-1)
    truth_valid_np = Y[valid_idx]

    metrics = {}
    for col_i, target in enumerate(TARGET_NAMES):
        truth = truth_valid_np[:, col_i]
        pred = pred_valid_np[:, col_i]
        metrics[target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }

    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "categorical_features": categorical_features,
            "target_names": list(TARGET_NAMES),
            "patch_features": list(PATCH_FEATURES),
            "in_dim": int(X.shape[1]),
            "patch_dim": int(P.shape[1]),
            "hidden_dim": 160,
        },
        os.path.join(out_dir, COUPLED_MODEL_FILE),
    )
    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))

    meta = {
        "backend": COUPLED_BACKEND,
        "language": str(language or "").strip().lower(),
        "format_type": normalize_format_type(language, format_type) or "general",
        "model_version": "v1",
        "feature_version": schema.get("feature_version", ""),
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "targets": list(TARGET_NAMES),
        "mel_patch_spec": list(PATCH_FEATURES),
        "min_confidence": float(min_confidence),
        "fallback_order": [COUPLED_BACKEND, "lightgbm", "base"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "holdout_metrics": metrics,
        "holdout_confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
        "device_used": str(run_device),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_min": float(np.min(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_max": float(np.max(conf_valid_np)) if len(conf_valid_np) else 0.0,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return meta


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

    model = _build_model(
        torch,
        nn,
        in_dim=int(payload.get("in_dim", len(feature_names))),
        patch_dim=int(payload.get("patch_dim", len(patch_features))),
        hidden_dim=int(payload.get("hidden_dim", 160)),
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
    with torch.no_grad():
        deltas_t, conf_t = model(x_t, p_t)
    deltas_np = deltas_t.detach().cpu().numpy().reshape(-1)
    conf = float(conf_t.detach().cpu().numpy().reshape(-1)[0])
    out = {target: float(deltas_np[i]) for i, target in enumerate(TARGET_NAMES)}
    return out, conf


def evaluate_coupled_bundle(
    model_dir: str,
    dataset_csv: str,
    *,
    language: str = "",
    format_type: str = "",
    device: str = "auto",
) -> Dict[str, Any]:
    _require_training_stack()
    payload_meta = {}
    meta_path = os.path.join(model_dir, "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            payload_meta = json.load(f)

    bundle = load_coupled_bundle(model_dir, meta=payload_meta, schema=get_feature_schema(), device=device)
    df = pd.read_csv(dataset_csv, low_memory=False)
    lang = str(language or payload_meta.get("language", "")).strip().lower()
    fmt = normalize_format_type(lang, format_type or payload_meta.get("format_type", "")) or "general"
    df = _prepare_training_frame(df, language=lang, format_type=fmt)
    if len(df) == 0:
        return {"rows": 0, "targets": {}, "confidence_mean": 0.0}

    preds = []
    confs = []
    for _, row in df.iterrows():
        d, c = predict_coupled_deltas(bundle, row.to_dict())
        preds.append(d)
        confs.append(float(c))

    summary = {"rows": int(len(df)), "targets": {}, "confidence_mean": float(np.mean(confs)) if confs else 0.0}
    for target in TARGET_NAMES:
        truth = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        pred = np.asarray([float(p.get(target, 0.0)) for p in preds], dtype=np.float64)
        summary["targets"][target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    return summary


__all__ = [
    "COUPLED_BACKEND",
    "COUPLED_MODEL_FILE",
    "PATCH_FEATURES",
    "build_and_save_coupled_dataset",
    "evaluate_coupled_bundle",
    "load_coupled_bundle",
    "predict_coupled_deltas",
    "train_coupled_bundle",
]
