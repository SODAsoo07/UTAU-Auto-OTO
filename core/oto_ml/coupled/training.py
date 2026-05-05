"""
Coupled mel+OTO training.

학습 루프, 데이터 전처리, 데이터셋 빌드, 평가 로직을 포함합니다.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
import wave
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
    try:  # pragma: no cover
        from pandas.errors import ParserError
    except Exception:  # pragma: no cover
        ParserError = Exception
except Exception:  # pragma: no cover
    pd = None
    ParserError = Exception

try:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupShuffleSplit
except Exception:  # pragma: no cover
    mean_absolute_error = None
    GroupShuffleSplit = None

from core.format_type_utils import normalize_format_type
from core.oto_ml_policy import alias_family_to_alias_types, normalize_alias_family
from core.oto_ml.coupled.model import (
    ANCHOR_TARGET_NAMES,
    CATEGORICAL_FEATURES,
    COUPLED_BACKEND,
    COUPLED_BACKEND_RAWMEL,
    COUPLED_MODEL_FILE,
    COUPLED_MODEL_ONNX_FILE,
    COUPLED_MODEL_ONNX_META_FILE,
    DELTA_TARGET_NAMES,
    FEATURE_NAMES,
    PATCH_FEATURES,
    TARGET_NAMES,
    _build_model,
    _build_model_rawmel,
    _default_categorical_bucket_sizes,
    _env_int,
    _import_torch,
    _require_training_stack,
    _resolve_device,
    _row_categorical_index_vector,
    _row_feature_vector,
    _row_patch_vector,
)
from core.oto_ml.features.schema import (
    AUX_TARGET_NAMES,
    get_feature_schema,
    write_dataset_csv,
    write_feature_schema,
)
from core.oto_ml.features.mel_patches import (
    MelPatchCacheIndex,
    make_mel_patch_key,
    patch_spec_hash,
)
from core.oto_ml.pairing.vc_cv_pairing import _batch_pair_positions, _build_vc_cv_pair_map

logger = logging.getLogger(__name__)

_PITCH_NOTE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Ga-g])([#b]?)(-?[0-8])(?![A-Za-z0-9])")
_NOTE_PITCH_CLASS = {
    "c": 0,
    "d": 2,
    "e": 4,
    "f": 5,
    "g": 7,
    "a": 9,
    "b": 11,
}
_WAV_DIR_INDEX_CACHE: Dict[str, Dict[Tuple[str, str, str], List[str]]] = {}
_WAV_PITCH_HZ_CACHE: Dict[str, float] = {}


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _export_coupled_onnx(
    torch,
    model,
    out_dir: str,
    *,
    feature_names: List[str],
    categorical_features: List[str],
    categorical_bucket_sizes: List[int],
    patch_features: List[str],
    head_mode: str,
    alias_branch_mode: str,
    alias_branch_experts: int,
    alias_type_cat_index: int,
    alias_type_bucket_size: int,
    alias_fallback_ids: Optional[List[int]],
    anchor_targets: List[str],
    delta_targets: List[str],
    rawmel_enabled: bool = False,
    mel_bins: int = 80,
    onset_frames: int = 8,
    tail_frames: int = 8,
    mel_patch_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not _env_flag("UTOA_ML_EXPORT_ONNX", True):
        return {"enabled": False, "status": "disabled"}

    onnx_path = os.path.join(out_dir, COUPLED_MODEL_ONNX_FILE)
    sidecar_path = os.path.join(out_dir, COUPLED_MODEL_ONNX_META_FILE)
    use_cat_input = bool(categorical_bucket_sizes)
    head_mode_norm = str(head_mode or "single").strip().lower()

    class _ExportWrapper(torch.nn.Module):
        def __init__(self, base_model, is_rawmel: bool, use_cat: bool, mode: str):
            super().__init__()
            self.base_model = base_model
            self.is_rawmel = bool(is_rawmel)
            self.use_cat = bool(use_cat)
            self.mode = str(mode or "single").strip().lower()

        def _encode_rawmel_export(self, encoder, mel_tensor):
            # ONNX export compatibility:
            # AdaptiveAvgPool2d((4,1)) fails for some frame lengths (e.g. 49),
            # so we use bilinear resize to the same target shape during export.
            x = encoder.net[0](mel_tensor)
            x = encoder.net[1](x)
            x = encoder.net[2](x)
            x = encoder.net[3](x)
            x = encoder.net[4](x)
            x = encoder.net[5](x)
            x = torch.nn.functional.interpolate(x, size=(4, 1), mode="bilinear", align_corners=False)
            x = encoder.proj[0](x)
            x = encoder.proj[1](x)
            x = encoder.proj[2](x)
            x = encoder.proj[3](x)
            x = encoder.proj[4](x)
            return x

        def forward(self, *inputs):
            idx = 0
            x = inputs[idx]
            idx += 1
            patch = inputs[idx]
            idx += 1
            if self.is_rawmel:
                onset = inputs[idx]
                idx += 1
                tail = inputs[idx]
                idx += 1
                cat = inputs[idx] if self.use_cat else None
                # Build the rawmel forward path explicitly to avoid
                # exporter limitation in AdaptiveAvgPool2d for non-factor sizes.
                xf = self.base_model.feature_net(x)
                xp = self.base_model.patch_net(patch)
                xo = self._encode_rawmel_export(self.base_model.onset_encoder, onset)
                xt = self._encode_rawmel_export(self.base_model.tail_encoder, tail)
                pieces = [xf, xp, xo, xt]
                cat_repr = self.base_model._cat_repr(x, cat)
                if cat_repr is not None:
                    pieces.append(cat_repr)
                z = self.base_model.joint(torch.cat(pieces, dim=1))
                conf = self.base_model.conf_head(z)
                if self.mode == "split":
                    anchor, delta = self.base_model._predict_heads(z, cat)
                    out = (anchor, delta, conf)
                else:
                    deltas = self.base_model._predict_heads(z, cat)
                    out = (deltas, conf)
            else:
                cat = inputs[idx] if self.use_cat else None
                out = self.base_model(x, patch, cat)
            if not isinstance(out, tuple):
                raise RuntimeError("Coupled model export failed: unexpected non-tuple output.")
            if self.mode == "split":
                if len(out) >= 3:
                    return out[0], out[1], out[2]
                raise RuntimeError("Coupled split head export failed: missing outputs.")
            if len(out) >= 2:
                return out[0], out[1]
            raise RuntimeError("Coupled single head export failed: missing outputs.")

    try:
        wrapper = _ExportWrapper(model, rawmel_enabled, use_cat_input, head_mode_norm).to("cpu")
        wrapper.eval()

        dummy_x = torch.zeros((1, max(1, len(feature_names))), dtype=torch.float32)
        dummy_patch = torch.zeros((1, max(1, len(patch_features))), dtype=torch.float32)
        input_tensors = [dummy_x, dummy_patch]
        input_names = ["x", "patch"]
        dynamic_axes = {
            "x": {0: "batch"},
            "patch": {0: "batch"},
        }

        if rawmel_enabled:
            dummy_onset = torch.zeros((1, 1, int(onset_frames), int(mel_bins)), dtype=torch.float32)
            dummy_tail = torch.zeros((1, 1, int(tail_frames), int(mel_bins)), dtype=torch.float32)
            input_tensors.extend([dummy_onset, dummy_tail])
            input_names.extend(["onset", "tail"])
            dynamic_axes["onset"] = {0: "batch"}
            dynamic_axes["tail"] = {0: "batch"}

        if use_cat_input:
            dummy_cat = torch.zeros((1, len(categorical_bucket_sizes)), dtype=torch.long)
            input_tensors.append(dummy_cat)
            input_names.append("cat_idx")
            dynamic_axes["cat_idx"] = {0: "batch"}

        if head_mode_norm == "split":
            output_names = ["anchor", "delta", "confidence"]
            dynamic_axes.update(
                {
                    "anchor": {0: "batch"},
                    "delta": {0: "batch"},
                    "confidence": {0: "batch"},
                }
            )
        else:
            output_names = ["deltas", "confidence"]
            dynamic_axes.update(
                {
                    "deltas": {0: "batch"},
                    "confidence": {0: "batch"},
                }
            )

        torch.onnx.export(
            wrapper,
            tuple(input_tensors),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=int(os.environ.get("UTOA_ML_ONNX_OPSET", "17") or 17),
            do_constant_folding=True,
        )

        sidecar = {
            "feature_names": list(feature_names),
            "categorical_features": list(categorical_features),
            "categorical_bucket_sizes": [int(v) for v in categorical_bucket_sizes],
            "patch_features": list(patch_features),
            "head_mode": head_mode_norm if head_mode_norm in {"single", "split"} else "single",
            "alias_branch_mode": str(alias_branch_mode or "shared").strip().lower(),
            "alias_branch_experts": int(alias_branch_experts),
            "alias_type_cat_index": int(alias_type_cat_index),
            "alias_type_bucket_size": int(alias_type_bucket_size),
            "alias_fallback_ids": [int(v) for v in (alias_fallback_ids or [])],
            "anchor_targets": list(anchor_targets),
            "delta_targets": list(delta_targets),
            "rawmel_enabled": bool(rawmel_enabled),
            "mel_bins": int(mel_bins),
            "onset_frames": int(onset_frames),
            "tail_frames": int(tail_frames),
            "mel_patch_spec": dict(mel_patch_spec or {}),
            "use_cat_input": bool(use_cat_input),
            "input_names": {
                "x": "x",
                "patch": "patch",
                "onset": "onset" if rawmel_enabled else "",
                "tail": "tail" if rawmel_enabled else "",
                "cat": "cat_idx" if use_cat_input else "",
            },
            "output_names": {
                "anchor": "anchor" if head_mode_norm == "split" else "",
                "delta": "delta" if head_mode_norm == "split" else "",
                "deltas": "deltas" if head_mode_norm != "split" else "",
                "confidence": "confidence",
            },
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
        return {
            "enabled": True,
            "status": "ok",
            "onnx_path": onnx_path,
            "sidecar_path": sidecar_path,
            "input_names": list(input_names),
            "output_names": list(output_names),
        }
    except Exception as exc:
        logger.warning("Failed to export coupled ONNX bundle: %s", exc)
        return {"enabled": True, "status": "failed", "error": str(exc)}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return str(default)
    text = str(raw).strip()
    return text or str(default)


def _resolve_head_mode(raw_value: str, default: str = "split") -> str:
    mode = str(raw_value or default).strip().lower()
    if mode in {"single", "split"}:
        return mode
    return str(default).strip().lower() or "split"


def _resolve_alias_branch_mode(raw_value: str, default: str = "shared") -> str:
    mode = str(raw_value or default).strip().lower()
    if mode in {"shared", "shared_heads", "moe"}:
        return mode
    return str(default or "shared").strip().lower() or "shared"


def _resolve_alias_branch_settings(
    *,
    mode_env: str,
    experts_env: str,
    min_rows_env: str,
    categorical_features: List[str],
    categorical_bucket_sizes: List[int],
    cat_matrix: "np.ndarray",
    train_idx: "np.ndarray",
) -> Dict[str, Any]:
    requested_mode = _resolve_alias_branch_mode(_env_str(mode_env, "shared"), default="shared")
    experts = max(2, _env_int(experts_env, 4))
    min_rows = max(1, _env_int(min_rows_env, 80))
    alias_type_cat_index = -1
    alias_type_bucket_size = 0
    active_alias_ids = 0
    strong_alias_ids = 0
    fallback_ids: List[int] = []
    reason = "ok"
    applied_mode = requested_mode

    if "alias_type" in categorical_features:
        alias_type_cat_index = int(categorical_features.index("alias_type"))
        if 0 <= alias_type_cat_index < len(categorical_bucket_sizes):
            alias_type_bucket_size = max(2, int(categorical_bucket_sizes[alias_type_cat_index] or 0))

    if requested_mode != "shared":
        if alias_type_cat_index < 0 or alias_type_bucket_size <= 1:
            applied_mode = "shared"
            reason = "alias_type_unavailable"
        elif cat_matrix is None or len(cat_matrix) <= 0 or len(train_idx) <= 0:
            applied_mode = "shared"
            reason = "empty_cat_matrix"
        else:
            try:
                alias_vals = np.asarray(cat_matrix[train_idx, alias_type_cat_index], dtype=np.int64).reshape(-1)
            except Exception:
                alias_vals = np.asarray([], dtype=np.int64)
            if alias_vals.size <= 0:
                applied_mode = "shared"
                reason = "empty_alias_values"
            else:
                counts: Dict[int, int] = {}
                for v in alias_vals.tolist():
                    key = int(v)
                    counts[key] = int(counts.get(key, 0)) + 1
                active_alias_ids = int(sum(1 for _k, c in counts.items() if c > 0))
                strong_alias = [int(k) for k, c in counts.items() if int(c) >= int(min_rows)]
                strong_alias_ids = int(len(strong_alias))
                if requested_mode == "shared_heads":
                    fallback_ids = [int(k) for k, c in counts.items() if int(c) < int(min_rows)]
                    if strong_alias_ids < 2:
                        applied_mode = "shared"
                        reason = "insufficient_strong_alias_types"
                elif requested_mode == "moe":
                    if active_alias_ids < 2:
                        applied_mode = "shared"
                        reason = "insufficient_alias_types"

    return {
        "requested_mode": str(requested_mode),
        "applied_mode": str(applied_mode),
        "experts": int(experts),
        "min_rows": int(min_rows),
        "alias_type_cat_index": int(alias_type_cat_index),
        "alias_type_bucket_size": int(alias_type_bucket_size),
        "active_alias_ids": int(active_alias_ids),
        "strong_alias_ids": int(strong_alias_ids),
        "fallback_ids": [int(v) for v in fallback_ids],
        "reason": str(reason),
    }


def _resolve_min_mapping_confidence(lang: str, fmt: str, min_mapping_confidence: float) -> float:
    try:
        if float(min_mapping_confidence) > 0.0:
            return float(min_mapping_confidence)
    except Exception:
        pass
    env_val = os.environ.get("UTOA_ML_TRAIN_MIN_MAPPING_CONF")
    if env_val is not None:
        try:
            return max(0.0, float(env_val))
        except Exception:
            return 0.0
    if str(lang).strip().lower() == "korean":
        if str(fmt).strip().lower() in {"cvvc", "cvc"}:
            return 0.55
        return 0.50
    return 0.0


def _resolve_min_quality_score(lang: str, fmt: str) -> float:
    env_val = os.environ.get("UTOA_ML_TRAIN_MIN_QUALITY_SCORE")
    if env_val is not None:
        try:
            return max(0.0, float(env_val))
        except Exception:
            return 0.0
    if str(lang).strip().lower() == "korean":
        if str(fmt).strip().lower() in {"cvvc", "cvc"}:
            return 55.0
        return 45.0
    return 0.0


def _is_kr_cvc(language: str, format_type: str) -> bool:
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower()
    return lang == "korean" and fmt == "cvc"


def _name_env_token(name: str) -> str:
    token = str(name or "").strip().upper()
    token = token.replace("DELTA_", "")
    token = token.replace("AUX_", "")
    token = token.replace("_REL", "")
    return token


def _huber_loss(torch, pred, truth, delta: float):
    delta_v = max(1e-3, float(delta))
    err = torch.abs(pred - truth)
    quadratic = torch.clamp(err, max=delta_v)
    linear = err - quadratic
    return (0.5 * quadratic.pow(2) / delta_v) + linear


def _loss_matrix(torch, pred, truth, loss_kinds: List[str], huber_deltas: List[float]):
    cols = []
    for idx in range(int(pred.shape[1])):
        kind = str(loss_kinds[idx] if idx < len(loss_kinds) else "huber").strip().lower()
        if kind in {"l1", "mae", "abs"}:
            col = torch.abs(pred[:, idx] - truth[:, idx])
        else:
            col = _huber_loss(torch, pred[:, idx], truth[:, idx], huber_deltas[idx] if idx < len(huber_deltas) else 16.0)
        cols.append(col.unsqueeze(1))
    return torch.cat(cols, dim=1) if cols else pred[:, :0]


def _resolve_loss_config(env_prefix: str, names: List[str], default_kinds: List[str], default_deltas: List[float]):
    loss_kinds: List[str] = []
    huber_deltas: List[float] = []
    for idx, name in enumerate(names):
        token = _name_env_token(name)
        default_kind = str(default_kinds[idx] if idx < len(default_kinds) else "huber").strip().lower()
        kind = _env_str(f"{env_prefix}LOSS_{token}", default_kind).strip().lower()
        if kind not in {"huber", "smooth_l1", "l1", "mae", "abs"}:
            kind = default_kind
        delta = _env_float(
            f"{env_prefix}HUBER_{token}",
            float(default_deltas[idx] if idx < len(default_deltas) else 16.0),
        )
        loss_kinds.append(kind)
        huber_deltas.append(max(1e-3, float(delta)))
    return loss_kinds, huber_deltas


def _target_indices(target_names: List[str], subset_names: List[str]) -> List[int]:
    indices = []
    for name in subset_names:
        if name not in target_names:
            raise RuntimeError(f"Missing target name: {name}")
        indices.append(target_names.index(name))
    return indices


def _combine_predictions(torch, anchor_pred, delta_pred, anchor_idx: List[int], delta_idx: List[int], total_dim: int):
    combined = torch.zeros((int(anchor_pred.shape[0]), int(total_dim)), dtype=anchor_pred.dtype, device=anchor_pred.device)
    combined[:, torch.tensor(anchor_idx, device=anchor_pred.device)] = anchor_pred
    combined[:, torch.tensor(delta_idx, device=anchor_pred.device)] = delta_pred
    return combined


def _compute_static_hard_example_boost(
    df,
    alias_type_arr,
    mapping_conf_np,
    blank_conf_np,
    jump_blocked_np,
    aux_mask,
    *,
    strength: float,
):
    strength_v = max(0.0, float(strength))
    boost = np.ones((len(df),), dtype=np.float32)
    if strength_v <= 0.0:
        return boost
    low_conf = np.clip(1.0 - mapping_conf_np, 0.0, 1.0)
    boost *= 1.0 + (0.60 * strength_v * low_conf)
    cv_mask = np.isin(alias_type_arr, ["cv", "cv_head"])
    bridge_mask = np.isin(alias_type_arr, ["vc", "vv", "vcv"])
    boost *= np.where(cv_mask & (jump_blocked_np > 0.5), 1.0 + (0.20 * strength_v), 1.0)
    boost *= np.where(bridge_mask, 1.0 + (0.20 * strength_v), 1.0)
    if aux_mask is not None and int(aux_mask.shape[1]) >= 3:
        next_onset_mask = np.asarray(aux_mask[:, 2] > 0.5, dtype=bool)
        boost *= np.where(next_onset_mask, 1.0 + (0.20 * strength_v), 1.0)
    if "coda_type" in df.columns:
        coda = df["coda_type"].astype(str).str.strip().str.lower().to_numpy()
        closed_mask = ~np.isin(coda, ["", "none", "open", "vowel"])
        boost *= np.where(closed_mask, 1.0 + (0.15 * strength_v), 1.0)
    if "is_diphthong" in df.columns:
        diph_mask = pd.to_numeric(df["is_diphthong"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        boost *= np.where(diph_mask, 1.0 + (0.12 * strength_v), 1.0)
    if "used_alias_occurrence_mapping" in df.columns:
        occurrence_mask = (
            pd.to_numeric(df["used_alias_occurrence_mapping"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        )
        boost *= np.where(occurrence_mask, 1.0 + (0.10 * strength_v), 1.0)
    if "used_nuclei_fallback" in df.columns:
        nuclei_mask = pd.to_numeric(df["used_nuclei_fallback"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        boost *= np.where(nuclei_mask, 1.0 + (0.08 * strength_v), 1.0)
    if "used_alias_based_syllables" in df.columns:
        alias_based_mask = (
            pd.to_numeric(df["used_alias_based_syllables"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        )
        boost *= np.where(alias_based_mask, 1.0 + (0.06 * strength_v), 1.0)
    if "mapping_reason_code" in df.columns:
        reason = df["mapping_reason_code"].astype(str).str.strip().str.lower().to_numpy()
        risky_reason_mask = np.isin(
            reason,
            [
                "order_locked_length_mismatch",
                "order_locked_glide_mismatch",
                "order_locked_low_phone_quality",
                "alias_based_recover",
                "alias_based_empty_words",
            ],
        )
        recover_reason_mask = np.isin(
            reason,
            [
                "alias_based_cvvc",
                "words_low_phone_quality",
                "alias_phone_minimal",
            ],
        )
        boost *= np.where(risky_reason_mask, 1.0 + (0.14 * strength_v), 1.0)
        boost *= np.where(recover_reason_mask, 1.0 + (0.09 * strength_v), 1.0)
    if "train_quality_score" in df.columns:
        quality_np = pd.to_numeric(df["train_quality_score"], errors="coerce").fillna(100.0).to_numpy(dtype=np.float32)
        boost *= np.where(quality_np < 70.0, 1.0 + (0.06 * strength_v), 1.0)
    if "train_keep_default" in df.columns:
        keep_default_np = pd.to_numeric(df["train_keep_default"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        # Keep low-quality labels from dominating while still keeping them in training.
        boost *= np.where(keep_default_np <= 0.5, 1.0 - (0.10 * strength_v), 1.0)
    if "blank_risk_score" in df.columns:
        blank_risk_np = pd.to_numeric(df["blank_risk_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        severe_blank_mask = cv_mask & (blank_risk_np >= 0.72)
        boost *= np.where(severe_blank_mask, 1.0 - (0.08 * strength_v), 1.0)
    return np.clip(boost.astype(np.float32), 0.70, 3.00)


def _apply_blank_risk_weight(df, weights: "np.ndarray") -> "np.ndarray":
    if df is None or weights is None or len(weights) == 0:
        return weights
    if "blank_risk_score" not in df.columns or "alias_type" not in df.columns:
        return weights
    weight = max(0.0, min(0.90, _env_float("UTOA_ML_BLANK_RISK_WEIGHT", 0.45)))
    if weight <= 0.0:
        return weights
    blank_score = pd.to_numeric(df["blank_risk_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    alias_type_arr = df["alias_type"].astype(str).str.strip().str.lower().to_numpy()
    cv_mask = np.isin(alias_type_arr, ["cv", "cv_head"])
    factor = np.clip(1.0 - (weight * blank_score), 0.25, 1.0)
    return (weights * np.where(cv_mask, factor, 1.0)).astype(np.float32)


def _blank_attach_alias_set() -> set:
    raw = str(os.environ.get("UTOA_ML_BLANK_ATTACH_ALIAS_TYPES", "") or "").strip().lower()
    if raw:
        toks = [t.strip().lower() for t in raw.split(",") if t.strip()]
        if toks:
            return set(toks)
    return {"cv", "cv_head", "vv", "v"}


def _safe_numeric_np(df, col: str, default: float = 0.0) -> "np.ndarray":
    if df is None or col not in df.columns:
        return np.full((0 if df is None else len(df),), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default)).to_numpy(dtype=np.float64)


def _note_token_to_hz(note: str, accidental: str, octave: str) -> Optional[float]:
    key = str(note or "").strip().lower()
    if key not in _NOTE_PITCH_CLASS:
        return None
    semitone = int(_NOTE_PITCH_CLASS[key])
    acc = str(accidental or "").strip()
    if acc == "#":
        semitone += 1
    elif acc == "b":
        semitone -= 1
    semitone %= 12
    try:
        octave_i = int(str(octave or "").strip())
    except Exception:
        return None
    midi = (octave_i + 1) * 12 + semitone
    return float(440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


def _extract_note_hz_from_text(text: str) -> Optional[float]:
    token = str(text or "").strip()
    if not token:
        return None
    for m in _PITCH_NOTE_TOKEN_RE.finditer(token):
        hz = _note_token_to_hz(m.group(1), m.group(2), m.group(3))
        if hz is not None and np.isfinite(hz) and hz > 0.0:
            return float(hz)
    return None


def _workspace_root_from_dataset_csv(dataset_csv: str) -> str:
    path = os.path.abspath(str(dataset_csv or "").strip())
    if not path:
        return ""
    base = os.path.dirname(os.path.dirname(path))
    if os.path.basename(base).strip().lower() == "datasets":
        return os.path.dirname(base)
    return base


def _load_wav_dir_index(dataset_csv: str) -> Dict[Tuple[str, str, str], List[str]]:
    workspace_root = _workspace_root_from_dataset_csv(dataset_csv)
    if not workspace_root:
        return {}
    if workspace_root in _WAV_DIR_INDEX_CACHE:
        return _WAV_DIR_INDEX_CACHE[workspace_root]
    index: Dict[Tuple[str, str, str], List[str]] = {}
    manifest_csv = os.path.join(workspace_root, "_manifest", "training_candidates.csv")
    if not os.path.isfile(manifest_csv):
        _WAV_DIR_INDEX_CACHE[workspace_root] = index
        return index
    try:
        with open(manifest_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lang = str(row.get("language", "") or "").strip().lower()
                fmt_raw = str(row.get("format_type", "") or "").strip().lower()
                fmt = normalize_format_type(lang, fmt_raw) or fmt_raw
                voicebank_id = str(row.get("voicebank_id", "") or "").strip()
                wav_dir = os.path.abspath(str(row.get("wav_dir", "") or "").strip())
                if not (lang and fmt and voicebank_id and wav_dir):
                    continue
                key = (lang, fmt, voicebank_id)
                bucket = index.setdefault(key, [])
                if wav_dir not in bucket:
                    bucket.append(wav_dir)
    except Exception:
        index = {}
    _WAV_DIR_INDEX_CACHE[workspace_root] = index
    return index


def _resolve_wav_path_from_row(
    row: Dict[str, object],
    *,
    language: str,
    format_type: str,
    wav_dir_index: Dict[Tuple[str, str, str], List[str]],
) -> str:
    wav_name = str(row.get("wav", "") or "").strip()
    voicebank_id = str(row.get("voicebank_id", "") or "").strip()
    if not (wav_name and voicebank_id):
        return ""
    fmt = normalize_format_type(language, format_type) or str(format_type or "").strip().lower()
    candidates = list(wav_dir_index.get((str(language).strip().lower(), fmt, voicebank_id), []))
    for wav_dir in candidates:
        cand = os.path.abspath(os.path.join(wav_dir, wav_name))
        if os.path.isfile(cand):
            return cand
    return ""


def _read_wav_mono_float64(wav_path: str) -> Tuple[Optional["np.ndarray"], int]:
    if not wav_path or not os.path.isfile(wav_path):
        return None, 0
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = int(wf.getframerate())
            n_channels = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            n_frames = int(wf.getnframes())
            raw = wf.readframes(n_frames)
    except Exception:
        return None, 0
    if not raw or sr <= 0:
        return None, 0
    if sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        return None, 0
    if n_channels > 1:
        frame_count = int(len(arr) // n_channels)
        if frame_count <= 0:
            return None, 0
        arr = arr[: frame_count * n_channels].reshape(frame_count, n_channels).mean(axis=1)
    return arr.astype(np.float64), sr


def _estimate_pitch_hz_from_wav(wav_path: str) -> float:
    cache_key = os.path.abspath(str(wav_path or "").strip())
    if not cache_key:
        return float("nan")
    if cache_key in _WAV_PITCH_HZ_CACHE:
        return float(_WAV_PITCH_HZ_CACHE[cache_key])
    audio, sr = _read_wav_mono_float64(cache_key)
    if audio is None or sr <= 0 or len(audio) < int(sr * 0.08):
        _WAV_PITCH_HZ_CACHE[cache_key] = float("nan")
        return float("nan")

    min_hz = max(40.0, float(_env_float("UTOA_ML_BLANK_ATTACH_WAV_PITCH_MIN_HZ", 70.0)))
    max_hz = max(min_hz + 30.0, float(_env_float("UTOA_ML_BLANK_ATTACH_WAV_PITCH_MAX_HZ", 1100.0)))
    min_lag = max(2, int(sr / max_hz))
    max_lag = min(int(sr / min_hz), int(sr * 0.06))
    if max_lag <= min_lag + 1:
        _WAV_PITCH_HZ_CACHE[cache_key] = float("nan")
        return float("nan")

    frame_len = max(int(sr * 0.05), max_lag + 2)
    hop = max(1, int(sr * 0.01))
    if len(audio) < frame_len:
        _WAV_PITCH_HZ_CACHE[cache_key] = float("nan")
        return float("nan")

    max_frames = max(4, int(_env_int("UTOA_ML_BLANK_ATTACH_WAV_PITCH_TOP_FRAMES", 10)))
    rms_list: List[Tuple[float, int]] = []
    for start in range(0, len(audio) - frame_len + 1, hop):
        fr = audio[start : start + frame_len]
        rms = float(np.sqrt(np.mean(fr * fr) + 1e-12))
        rms_list.append((rms, int(start)))
    if not rms_list:
        _WAV_PITCH_HZ_CACHE[cache_key] = float("nan")
        return float("nan")
    rms_list.sort(key=lambda x: x[0], reverse=True)

    min_clarity = float(_env_float("UTOA_ML_BLANK_ATTACH_WAV_PITCH_MIN_CLARITY", 0.22))
    max_zcr = float(_env_float("UTOA_ML_BLANK_ATTACH_WAV_PITCH_MAX_ZCR", 0.26))
    f0_candidates: List[float] = []
    for _rms, start in rms_list[:max_frames]:
        fr = audio[start : start + frame_len].astype(np.float64)
        fr = fr - float(np.mean(fr))
        if np.max(np.abs(fr)) <= 1e-6:
            continue
        zcr = float(np.mean((fr[:-1] * fr[1:]) < 0.0)) if len(fr) >= 2 else 1.0
        if zcr > max_zcr:
            continue
        ac = np.correlate(fr, fr, mode="full")
        ac = ac[len(fr) - 1 :]
        if len(ac) <= max_lag:
            continue
        ac0 = float(ac[0])
        if ac0 <= 1e-9:
            continue
        seg = ac[min_lag : max_lag + 1]
        rel_idx = int(np.argmax(seg))
        lag = int(min_lag + rel_idx)
        peak = float(seg[rel_idx])
        clarity = float(peak / (ac0 + 1e-9))
        if clarity < min_clarity:
            continue
        f0 = float(sr) / float(max(1, lag))
        if min_hz <= f0 <= max_hz and np.isfinite(f0):
            f0_candidates.append(float(f0))
    if len(f0_candidates) < 2:
        hz = float("nan")
    else:
        hz = float(np.median(np.asarray(f0_candidates, dtype=np.float64)))
    _WAV_PITCH_HZ_CACHE[cache_key] = float(hz)
    return float(hz)


def _estimate_pitch_hz_np(
    df,
    *,
    language: str = "",
    format_type: str = "",
    dataset_csv: str = "",
) -> "np.ndarray":
    n = 0 if df is None else len(df)
    out = np.full((n,), np.nan, dtype=np.float64)
    if df is None or n <= 0:
        return out
    for col in ("f0_note_hint_hz", "note_hint_hz", "note_hz"):
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(val) & (val > 0.0)
            out[ok] = val[ok]
    text_cols = [col for col in ("wav", "wav_norm", "voicebank_id", "alias") if col in df.columns]
    if text_cols:
        unresolved = np.where(~np.isfinite(out))[0]
        for idx in unresolved.tolist():
            hz = None
            for col in text_cols:
                hz = _extract_note_hz_from_text(df.iloc[idx][col])
                if hz is not None:
                    break
            if hz is not None:
                out[idx] = float(hz)

    use_wav_pitch = _env_flag("UTOA_ML_BLANK_ATTACH_WAV_PITCH_ENABLE", True)
    if use_wav_pitch and np.any(~np.isfinite(out)) and dataset_csv:
        wav_dir_index = _load_wav_dir_index(dataset_csv)
        unresolved = np.where(~np.isfinite(out))[0]
        for idx in unresolved.tolist():
            row = df.iloc[idx].to_dict()
            wav_path = _resolve_wav_path_from_row(
                row,
                language=str(language or "").strip().lower(),
                format_type=str(format_type or "").strip().lower(),
                wav_dir_index=wav_dir_index,
            )
            if not wav_path:
                continue
            wav_hint_hz = _extract_note_hz_from_text(wav_path)
            if wav_hint_hz is not None:
                out[idx] = float(wav_hint_hz)
                continue
            wav_hz = _estimate_pitch_hz_from_wav(wav_path)
            if np.isfinite(wav_hz) and float(wav_hz) > 0.0:
                out[idx] = float(wav_hz)
    return out


def _blank_attach_high_pitch_mask(
    df,
    *,
    language: str = "",
    format_type: str = "",
    dataset_csv: str = "",
) -> "np.ndarray":
    n = 0 if df is None else len(df)
    mask = np.zeros((n,), dtype=bool)
    if df is None or n <= 0:
        return mask
    high_zone_tokens = {"high", "upper", "upper_mid", "upper-mid", "hi"}
    if "f0_pitch_zone" in df.columns:
        zone = df["f0_pitch_zone"].astype(str).str.strip().str.lower().to_numpy()
        mask |= np.isin(zone, list(high_zone_tokens))
    pitch_hz = _estimate_pitch_hz_np(
        df,
        language=language,
        format_type=format_type,
        dataset_csv=dataset_csv,
    )
    hz_th = float(_env_float("UTOA_ML_BLANK_ATTACH_HIGH_PITCH_HZ", 500.0))
    mask |= np.isfinite(pitch_hz) & (pitch_hz >= hz_th)
    if "f0_max_hz" in df.columns:
        f0_max_hz = pd.to_numeric(df["f0_max_hz"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        hard_high = float(_env_float("UTOA_ML_BLANK_ATTACH_HIGH_F0MAX_HZ", 860.0))
        mask |= (f0_max_hz >= hard_high)
    return mask


def _blank_like_score_np(df) -> "np.ndarray":
    if df is None:
        return np.zeros((0,), dtype=np.float64)
    n = len(df)
    cols = [
        "blank_risk_score",
        "blank_span_confidence",
        "syllable_blank_confidence",
        "mel_silence_sparse_ratio",
        "mel_silence_sparse_conf",
    ]
    arrs = []
    for col in cols:
        if col in df.columns:
            arrs.append(_safe_numeric_np(df, col, 0.0))
    if not arrs:
        return np.zeros((n,), dtype=np.float64)
    stacked = np.vstack(arrs)
    return np.clip(np.max(stacked, axis=0), 0.0, 1.0)


def _blank_attach_scope_masks(df, *, language: str = "", format_type: str = ""):
    if df is None:
        z = np.zeros((0,), dtype=bool)
        return z, z, z
    n = len(df)
    alias_types = _blank_attach_alias_set()
    alias_arr = (
        df["alias_type"].astype(str).str.strip().str.lower().to_numpy()
        if "alias_type" in df.columns
        else np.full((n,), "", dtype=object)
    )
    alias_mask = np.isin(alias_arr, list(alias_types))

    fmt_fallback = normalize_format_type(language, format_type) or str(format_type or "").strip().lower()
    fmt_series = (
        df["format_type"].astype(str).str.strip().str.lower().map(
            lambda v: normalize_format_type(language, v) or str(v or "").strip().lower()
        ).to_numpy()
        if "format_type" in df.columns
        else np.full((n,), fmt_fallback, dtype=object)
    )
    cvvc_only = _env_flag("UTOA_ML_BLANK_ATTACH_SCOPE_CVVC_ONLY", True)
    fmt_mask = np.ones((n,), dtype=bool) if not cvvc_only else (fmt_series == "cvvc")

    head_row = np.zeros((n,), dtype=bool)
    if "is_head_row" in df.columns:
        head_row |= (_safe_numeric_np(df, "is_head_row", 0.0) >= 0.5)
    if "mora_position" in df.columns:
        head_row |= (df["mora_position"].astype(str).str.strip().str.lower() == "head").to_numpy()
    head_row |= np.isin(alias_arr, ["cv_head"])
    vv_mask = np.isin(alias_arr, ["vv", "v"])

    scope_mask = alias_mask & fmt_mask
    head_scope = scope_mask & (head_row | vv_mask)
    vv_scope = scope_mask & vv_mask
    return scope_mask, head_scope, vv_scope


def _derive_blank_attach_label(df, *, language: str = "", format_type: str = ""):
    if df is None or len(df) == 0:
        return np.zeros((0,), dtype=np.int32), {"rows": 0, "positive_rows": 0, "positive_rate": 0.0}
    scope_mask, head_scope, _vv_scope = _blank_attach_scope_masks(df, language=language, format_type=format_type)
    blank_like = _blank_like_score_np(df)
    blank_th = float(_env_float("UTOA_ML_BLANK_ATTACH_RISK_TH", 0.55))
    margin_ms = float(_env_float("UTOA_ML_BLANK_ATTACH_MARGIN_MS", 14.0))
    expected_anchor = (
        _safe_numeric_np(df, "expected_anchor_ms", 0.0)
        if "expected_anchor_ms" in df.columns
        else _safe_numeric_np(df, "curr_vowel_start_ms", 0.0)
    )
    manual_offset = (
        _safe_numeric_np(df, "manual_offset", np.nan)
        if "manual_offset" in df.columns
        else (_safe_numeric_np(df, "base_offset", 0.0) + _safe_numeric_np(df, "delta_offset", 0.0))
    )
    finite = np.isfinite(expected_anchor) & np.isfinite(manual_offset)
    lead_gap = expected_anchor - manual_offset
    attach = finite & scope_mask & head_scope & (blank_like >= blank_th) & (lead_gap >= margin_ms)
    label = attach.astype(np.int32)
    rows = int(np.sum(scope_mask & head_scope))
    pos = int(np.sum(attach))
    rate = (float(pos) / float(max(1, rows))) if rows > 0 else 0.0
    return label, {"rows": rows, "positive_rows": pos, "positive_rate": float(rate)}


def _apply_blank_attach_focus_weight(df, weights: "np.ndarray", *, language: str = "", format_type: str = "") -> "np.ndarray":
    if df is None or weights is None or len(weights) == 0:
        return weights
    focus_weight = float(_env_float("UTOA_ML_BLANK_ATTACH_FOCUS_WEIGHT", 1.15))
    if focus_weight <= 1.0:
        return weights
    scope_mask, head_scope, _vv_scope = _blank_attach_scope_masks(df, language=language, format_type=format_type)
    blank_like = _blank_like_score_np(df)
    focus_factor = 1.0 + ((focus_weight - 1.0) * np.clip(blank_like, 0.0, 1.0))
    out = weights.astype(np.float32).copy()
    out *= np.where(head_scope, focus_factor.astype(np.float32), 1.0)
    if "blank_attach_label" in df.columns:
        pos = (_safe_numeric_np(df, "blank_attach_label", 0.0) > 0.5) & head_scope
        pos_weight = max(1.0, float(_env_float("UTOA_ML_BLANK_ATTACH_POS_WEIGHT", 1.30)))
        out *= np.where(pos, np.float32(pos_weight), np.float32(1.0))
    return out.astype(np.float32)


def _compute_blank_attach_kpi(
    df,
    *,
    pred_delta_offset,
    truth_delta_offset=None,
    language: str = "",
    format_type: str = "",
    row_mask=None,
):
    if df is None or len(df) == 0:
        return {
            "enabled": False,
            "scope_rows": 0,
            "head_rows": 0,
            "vv_rows": 0,
            "pred_attach_rate": 0.0,
            "head_pred_attach_rate": 0.0,
            "vv_pred_attach_rate": 0.0,
            "head_gold_attach_rate": 0.0,
            "gold_attach_rows": 0,
            "pred_minus_gold_rate": 0.0,
            "primary_kpi_name": "head_pred_attach_rate",
            "primary_kpi_value": 0.0,
            "has_gold_labels": False,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    n = len(df)
    pred_delta = np.asarray(pred_delta_offset, dtype=np.float64).reshape(-1)
    if pred_delta.shape[0] != n:
        pred_delta = np.resize(pred_delta, n)
    truth_delta = None
    if truth_delta_offset is not None:
        truth_delta = np.asarray(truth_delta_offset, dtype=np.float64).reshape(-1)
        if truth_delta.shape[0] != n:
            truth_delta = np.resize(truth_delta, n)

    base_offset = _safe_numeric_np(df, "base_offset", 0.0)
    expected_anchor = (
        _safe_numeric_np(df, "expected_anchor_ms", 0.0)
        if "expected_anchor_ms" in df.columns
        else _safe_numeric_np(df, "curr_vowel_start_ms", 0.0)
    )
    pred_offset_abs = base_offset + pred_delta
    if truth_delta is not None:
        truth_offset_abs = base_offset + truth_delta
    elif "manual_offset" in df.columns:
        truth_offset_abs = _safe_numeric_np(df, "manual_offset", np.nan)
    else:
        truth_offset_abs = np.full((n,), np.nan, dtype=np.float64)

    blank_like = _blank_like_score_np(df)
    blank_th = float(_env_float("UTOA_ML_BLANK_ATTACH_RISK_TH", 0.55))
    margin_ms = float(_env_float("UTOA_ML_BLANK_ATTACH_MARGIN_MS", 14.0))
    scope_mask, head_scope, vv_scope = _blank_attach_scope_masks(df, language=language, format_type=format_type)
    if row_mask is not None:
        row_mask_np = np.asarray(row_mask, dtype=bool).reshape(-1)
        if row_mask_np.shape[0] != n:
            row_mask_np = np.resize(row_mask_np, n)
        scope_mask = scope_mask & row_mask_np
        head_scope = head_scope & row_mask_np
        vv_scope = vv_scope & row_mask_np
    finite_pred = np.isfinite(expected_anchor) & np.isfinite(pred_offset_abs)
    pred_attach = finite_pred & head_scope & (blank_like >= blank_th) & ((expected_anchor - pred_offset_abs) >= margin_ms)

    has_gold = False
    if "blank_attach_label" in df.columns:
        gold_attach = (_safe_numeric_np(df, "blank_attach_label", 0.0) > 0.5) & head_scope
        has_gold = True
    else:
        finite_truth = np.isfinite(expected_anchor) & np.isfinite(truth_offset_abs)
        gold_attach = finite_truth & head_scope & (blank_like >= blank_th) & ((expected_anchor - truth_offset_abs) >= margin_ms)
        has_gold = bool(np.any(finite_truth & head_scope))

    def _rate(mask, flag):
        rows = int(np.sum(mask))
        if rows <= 0:
            return 0.0, rows, 0
        pos = int(np.sum(mask & flag))
        return float(pos) / float(rows), rows, pos

    pred_rate, scope_rows, pred_rows = _rate(head_scope, pred_attach)
    overall_rate, overall_rows, overall_pred_rows = _rate(scope_mask, pred_attach)
    vv_rate, vv_rows, vv_pred_rows = _rate(vv_scope, pred_attach)
    gold_rate, gold_rows, gold_pos = _rate(head_scope, gold_attach)

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    tp = fp = fn = 0
    if has_gold and gold_rows > 0:
        tp = int(np.sum(head_scope & pred_attach & gold_attach))
        fp = int(np.sum(head_scope & pred_attach & (~gold_attach)))
        fn = int(np.sum(head_scope & (~pred_attach) & gold_attach))
        precision = float(tp) / float(max(1, tp + fp))
        recall = float(tp) / float(max(1, tp + fn))
        if (precision + recall) > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "enabled": bool(scope_rows > 0),
        "scope_rows": int(overall_rows),
        "head_rows": int(scope_rows),
        "vv_rows": int(vv_rows),
        "pred_attach_rows": int(pred_rows),
        "pred_attach_rate": float(overall_rate),
        "head_pred_attach_rate": float(pred_rate),
        "vv_pred_attach_rate": float(vv_rate),
        "head_gold_attach_rate": float(gold_rate),
        "gold_attach_rows": int(gold_pos),
        "pred_minus_gold_rate": float(pred_rate - gold_rate),
        "primary_kpi_name": "head_pred_attach_rate",
        "primary_kpi_value": float(pred_rate),
        "risk_threshold": float(blank_th),
        "attach_margin_ms": float(margin_ms),
        "has_gold_labels": bool(has_gold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _compute_blank_attach_kpi_pitch_zones(
    df,
    *,
    pred_delta_offset,
    truth_delta_offset=None,
    language: str = "",
    format_type: str = "",
    dataset_csv: str = "",
) -> Dict[str, Dict[str, object]]:
    if df is None or len(df) == 0:
        return {"high": {"enabled": False, "zone": "high", "scope_rows": 0, "head_rows": 0, "vv_rows": 0}}
    high_mask = _blank_attach_high_pitch_mask(
        df,
        language=language,
        format_type=format_type,
        dataset_csv=dataset_csv,
    )
    high_kpi = _compute_blank_attach_kpi(
        df,
        pred_delta_offset=pred_delta_offset,
        truth_delta_offset=truth_delta_offset,
        language=language,
        format_type=format_type,
        row_mask=high_mask,
    )
    high_kpi["zone"] = "high"
    high_kpi["pitch_hz_threshold"] = float(_env_float("UTOA_ML_BLANK_ATTACH_HIGH_PITCH_HZ", 500.0))
    return {"high": high_kpi}


def _resolve_sampling_group_values(df, train_idx, preferred_column: str = ""):
    candidates = [preferred_column, "voicebank_id", "wav_norm"]
    for name in candidates:
        if name and name in df.columns:
            return df.iloc[train_idx][name].astype(str).fillna("").to_numpy(), str(name)
    fallback = np.asarray([str(v) for v in train_idx.tolist()], dtype=object)
    return fallback, "__index__"


def _sample_group_balanced_indices(group_values, sample_weights, sample_count: int, rng):
    count = max(1, int(sample_count))
    groups = np.asarray(group_values)
    if groups.size <= 1:
        return rng.permutation(count).astype(np.int64)
    group_to_positions: Dict[str, List[int]] = {}
    for pos, group in enumerate(groups.tolist()):
        key = str(group or f"group_{pos}")
        group_to_positions.setdefault(key, []).append(int(pos))
    if len(group_to_positions) <= 1:
        return rng.permutation(count).astype(np.int64)
    keys = list(group_to_positions.keys())
    group_choices = rng.integers(0, len(keys), size=count, endpoint=False)
    out = np.empty((count,), dtype=np.int64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    for key_idx, key in enumerate(keys):
        take_positions = np.where(group_choices == key_idx)[0]
        if take_positions.size <= 0:
            continue
        members = np.asarray(group_to_positions[key], dtype=np.int64)
        probs = weights[members] if weights.size == groups.size else np.ones((len(members),), dtype=np.float64)
        if (not np.all(np.isfinite(probs))) or float(np.sum(probs)) <= 0.0:
            probs = None
        else:
            probs = probs / float(np.sum(probs))
        out[take_positions] = rng.choice(members, size=int(take_positions.size), replace=True, p=probs)
    rng.shuffle(out)
    return out


def _pair_weight_for_epoch(base_weight: float, epoch_idx: int, warmup_epochs: int) -> float:
    base = max(0.0, float(base_weight))
    warmup = max(0, int(warmup_epochs))
    if base <= 0.0 or warmup <= 0:
        return base
    progress = min(1.0, float(epoch_idx + 1) / float(warmup))
    return base * progress


def _estimate_rawmel_patch_cache_mb(sample_count: int, onset_frames: int, tail_frames: int, mel_bins: int) -> float:
    rows = max(0, int(sample_count))
    frames = max(0, int(onset_frames)) + max(0, int(tail_frames))
    bins = max(1, int(mel_bins))
    total_bytes = rows * frames * bins * 4
    return float(total_bytes) / float(1024 * 1024)


def _resolve_rawmel_prefetch_mode(
    rawmel_prefetch: str,
    *,
    run_device,
    train_count: int,
    valid_count: int,
    onset_frames: int,
    tail_frames: int,
    mel_bins: int,
) -> Dict[str, object]:
    requested = str(rawmel_prefetch or "auto").strip().lower()
    if requested not in {"none", "train", "gpu", "auto"}:
        requested = "auto"
    device_type = str(getattr(run_device, "type", run_device) or "").strip().lower()
    est_patch_mb = _estimate_rawmel_patch_cache_mb(
        int(train_count) + int(valid_count),
        onset_frames=int(onset_frames),
        tail_frames=int(tail_frames),
        mel_bins=int(mel_bins),
    )
    if requested == "auto":
        if device_type != "cuda":
            return {"mode": "none", "reason": "cpu_device", "estimated_patch_mb": float(est_patch_mb)}
        gpu_limit_mb = max(128, _env_int("UTOA_ML_RAWMEL_GPU_CACHE_MAX_MB", 1536))
        if est_patch_mb <= float(gpu_limit_mb):
            return {
                "mode": "gpu",
                "reason": f"auto_gpu_le_{gpu_limit_mb}mb",
                "estimated_patch_mb": float(est_patch_mb),
            }
        return {"mode": "train", "reason": "auto_train_cache", "estimated_patch_mb": float(est_patch_mb)}
    if requested == "gpu" and device_type != "cuda":
        return {"mode": "none", "reason": "gpu_prefetch_requires_cuda", "estimated_patch_mb": float(est_patch_mb)}
    return {"mode": requested, "reason": "explicit", "estimated_patch_mb": float(est_patch_mb)}


def _boundary_consistency_row_loss(torch, aux_pred, aux_mask, offset_abs, cutoff_abs, *, next_onset_margin: float = 6.0):
    if aux_pred is None or aux_mask is None:
        return None
    mask = aux_mask.float()
    vowel_start = aux_pred[:, 0]
    vowel_end = aux_pred[:, 1]
    next_onset = aux_pred[:, 2]
    row_loss = torch.zeros_like(vowel_start)
    row_loss = row_loss + (mask[:, 0] * mask[:, 1] * torch.relu(vowel_start - vowel_end))
    row_loss = row_loss + (mask[:, 1] * mask[:, 2] * torch.relu(vowel_end - next_onset))
    predicted_next_onset_abs = offset_abs + next_onset
    row_loss = row_loss + (mask[:, 2] * torch.relu((cutoff_abs + float(next_onset_margin)) - predicted_next_onset_abs))
    return row_loss


def _confidence_target_from_errors(torch, delta_row_err, align_row_err, penalty_row, boundary_row_err=None):
    score = (delta_row_err / 80.0) + (0.35 * (align_row_err / 120.0)) + (0.20 * (penalty_row / 40.0))
    if boundary_row_err is not None:
        score = score + (0.35 * (boundary_row_err / 140.0))
    return torch.exp(-torch.clamp(score, min=0.0, max=8.0))


def _read_dataset_csv(path: str):
    df = pd.read_csv(path, low_memory=False)
    return df


def _read_dataset_csv_resilient(path: str):
    try:
        return _read_dataset_csv(path)
    except Exception as exc:
        if isinstance(exc, ParserError):
            msg = f"[TRAIN] CSV parse failed, retrying with python engine (bad lines may be skipped): {exc}"
            print(msg)
            logger.warning(msg)
            try:
                return pd.read_csv(path, engine="python", on_bad_lines="warn")
            except Exception:
                msg2 = "[TRAIN] CSV parse still failing; retrying with relaxed quoting (skipping bad lines)."
                print(msg2)
                logger.warning(msg2)
                return pd.read_csv(
                    path,
                    engine="python",
                    on_bad_lines="skip",
                    quoting=csv.QUOTE_NONE,
                    escapechar="\\",
                )
        raise


def _infer_mapping_reason_codes(df):
    if df is None or "mapping_reason_code" not in df.columns:
        return df, 0, 0
    reasons = df["mapping_reason_code"].astype(str).str.strip().str.lower()
    unknown_mask = reasons.isin(["", "unknown", "none", "nan", "null"])
    unknown_before = int(unknown_mask.sum())
    if unknown_before <= 0:
        return df, 0, 0

    inferred = np.asarray(["unspecified"] * len(df), dtype=object)
    if "used_alias_occurrence_mapping" in df.columns:
        occ = pd.to_numeric(df["used_alias_occurrence_mapping"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        inferred = np.where(occ, "alias_occurrence", inferred)
    if "used_alias_based_syllables" in df.columns:
        alias_based = pd.to_numeric(df["used_alias_based_syllables"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        inferred = np.where(alias_based, "alias_based_syllables", inferred)
    if "used_nuclei_fallback" in df.columns:
        nuclei = pd.to_numeric(df["used_nuclei_fallback"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        inferred = np.where(nuclei, "nuclei_fallback", inferred)
    if "jump_blocked_flag" in df.columns:
        jump_blocked = pd.to_numeric(df["jump_blocked_flag"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        inferred = np.where(jump_blocked, "jump_blocked", inferred)
    if "blank_risk_flag" in df.columns:
        blank_flag = pd.to_numeric(df["blank_risk_flag"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        inferred = np.where(blank_flag, "blank_risk", inferred)
    if "words_vs_alias_score_margin" in df.columns:
        margin = pd.to_numeric(df["words_vs_alias_score_margin"], errors="coerce").fillna(0.0).to_numpy()
        inferred = np.where(margin <= -0.05, "words_alias_margin_low", inferred)

    out = df.copy()
    reason_arr = reasons.to_numpy(dtype=object)
    reason_arr[unknown_mask.to_numpy()] = inferred[unknown_mask.to_numpy()]
    out["mapping_reason_code"] = reason_arr

    after_reasons = out["mapping_reason_code"].astype(str).str.strip().str.lower()
    unknown_after = int(after_reasons.isin(["", "unknown", "none", "nan", "null"]).sum())
    return out, unknown_before, unknown_after


def _prepare_training_frame(
    df,
    language: str,
    format_type: str,
    alias_types: Optional[List[str]] = None,
    alias_family: str = "",
    min_mapping_confidence: float = 0.0,
):
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    family = normalize_alias_family(alias_family)
    if family and not alias_types:
        alias_types = alias_family_to_alias_types(family)
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
    min_map_conf = _resolve_min_mapping_confidence(lang, fmt, min_mapping_confidence)
    if float(min_map_conf) > 0.0 and "mapping_confidence" in df.columns:
        df = df[pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(0.0) >= float(min_map_conf)]
    min_quality_score = _resolve_min_quality_score(lang, fmt)
    if float(min_quality_score) > 0.0 and "train_quality_score" in df.columns:
        df = df[pd.to_numeric(df["train_quality_score"], errors="coerce").fillna(0.0) >= float(min_quality_score)]
    if _env_int("UTOA_ML_TRAIN_KEEP_DEFAULT_ONLY", 0) > 0 and "train_keep_default" in df.columns:
        df = df[pd.to_numeric(df["train_keep_default"], errors="coerce").fillna(0.0) >= 1.0]
    if _env_int("UTOA_ML_INFER_REASON_CODE", 1) > 0:
        df, unknown_before, unknown_after = _infer_mapping_reason_codes(df)
        if unknown_before > 0:
            print(
                f"[TRAIN] mapping_reason_code inferred: unknown {unknown_before} -> {unknown_after}",
                flush=True,
            )
    if _env_int("UTOA_ML_ENABLE_BLANK_ATTACH_LABEL", 1) > 0:
        if "blank_attach_label" not in df.columns:
            labels, label_meta = _derive_blank_attach_label(df, language=lang, format_type=fmt)
            if len(labels) == len(df):
                out = df.copy()
                out["blank_attach_label"] = labels.astype(np.int32)
                df = out
                if int(label_meta.get("rows", 0)) > 0:
                    print(
                        "[TRAIN] blank_attach_label derived: "
                        f"rows={int(label_meta.get('rows', 0))}, "
                        f"positive={int(label_meta.get('positive_rows', 0))}, "
                        f"rate={float(label_meta.get('positive_rate', 0.0)):.4f}",
                        flush=True,
                    )
    return df


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
    auto_oto_policy: str = "",
) -> Dict[str, int]:
    from core.oto_ml_features import build_training_rows

    rows, stats = build_training_rows(
        language=language,
        auto_oto_path=auto_oto_path,
        manual_oto_path=manual_oto_path,
        tg_dir=tg_dir,
        wav_dir=wav_dir,
        custom_phonemes_path=custom_phonemes_path,
        voicebank_id=voicebank_id,
        format_type_override=format_type_override,
        auto_oto_policy=auto_oto_policy,
    )
    if rows:
        write_dataset_csv(out_csv, rows, append=append)
    out = dict(stats)
    out["saved_rows"] = int(len(rows))
    out["out_csv"] = os.path.abspath(out_csv)
    return out


def train_coupled_bundle(
    language: str,
    format_type: str,
    dataset_csv: str,
    out_dir: str,
    *,
    group_column: str = "voicebank_id",
    alias_types: Optional[List[str]] = None,
    alias_family: str = "",
    min_mapping_confidence: float = 0.0,
    device: str = "auto",
    epochs: int = 70,
    batch_size: int = 192,
    learning_rate: float = 1e-3,
    min_confidence: float = 0.55,
    progress_every: int = 0,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.exists(dataset_csv):
        raise FileNotFoundError(dataset_csv)

    torch, nn, F = _import_torch()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    df = _read_dataset_csv_resilient(dataset_csv)
    df = _prepare_training_frame(
        df,
        language=language,
        format_type=format_type,
        alias_types=alias_types,
        alias_family=alias_family,
        min_mapping_confidence=min_mapping_confidence,
    )
    df = df.reset_index(drop=True)
    if len(df) < 16:
        raise RuntimeError("Coupled dataset is too small (need >= 16 rows).")

    schema = get_feature_schema()
    feature_names = list(schema.get("feature_names") or FEATURE_NAMES)
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in feature_names]
    categorical_bucket_sizes = _default_categorical_bucket_sizes(categorical_features)
    prefetch_info = {"mode": "none", "estimated_patch_mb": 0.0, "reason": "not_rawmel"}
    prefetch_mode = "none"

    x_rows = []
    c_rows = []
    p_rows = []
    for _, row in df.iterrows():
        as_dict = row.to_dict()
        x_rows.append(_row_feature_vector(as_dict, feature_names, categorical_features))
        c_rows.append(_row_categorical_index_vector(as_dict, categorical_features, categorical_bucket_sizes))
        p_rows.append(_row_patch_vector(as_dict))
    X = np.asarray(x_rows, dtype=np.float32)
    C = np.asarray(c_rows, dtype=np.int64) if c_rows else np.zeros((len(df), 0), dtype=np.int64)
    P = np.asarray(p_rows, dtype=np.float32)
    Y = np.stack(
        [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in TARGET_NAMES],
        axis=1,
    )
    use_aux = all(name in df.columns for name in AUX_TARGET_NAMES)
    if use_aux:
        A = np.stack(
            [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in AUX_TARGET_NAMES],
            axis=1,
        )
        vowel_start_valid = pd.to_numeric(df["curr_vowel_start_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        vowel_end_valid = pd.to_numeric(df["curr_vowel_end_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        next_onset_valid = (
            pd.to_numeric(df["base_cutoff_to_next_anchor_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        )
        aux_mask = np.stack(
            [vowel_start_valid.astype(np.float32), vowel_end_valid.astype(np.float32), next_onset_valid.astype(np.float32)],
            axis=1,
        )
    else:
        A = None
        aux_mask = None
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
    W = _apply_blank_risk_weight(df, W)
    W = _apply_blank_attach_focus_weight(
        df,
        W,
        language=str(language or "").strip().lower(),
        format_type=normalize_format_type(language, format_type) or str(format_type or "").strip().lower(),
    )
    if "alias_type" in df.columns:
        alias_type_arr = df["alias_type"].astype(str).str.lower().to_numpy()
    else:
        alias_type_arr = np.full((len(df),), "", dtype=object)
    blank_conf_np = (
        pd.to_numeric(df["blank_span_confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "blank_span_confidence" in df.columns
        else np.zeros((len(df),), dtype=np.float32)
    )
    jump_blocked_np = (
        pd.to_numeric(df["jump_blocked_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "jump_blocked_flag" in df.columns
        else np.zeros((len(df),), dtype=np.float32)
    )
    mapping_conf_np = (
        pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        if "mapping_confidence" in df.columns
        else np.ones((len(df),), dtype=np.float32)
    )
    hard_example_strength = max(0.0, _env_float("UTOA_ML_COUPLED_HARD_MINING_STRENGTH", 0.45))
    hard_boost = _compute_static_hard_example_boost(
        df,
        alias_type_arr,
        mapping_conf_np,
        blank_conf_np,
        jump_blocked_np,
        aux_mask,
        strength=hard_example_strength,
    )
    W = np.clip((W * np.sqrt(hard_boost)).astype(np.float32), 0.20, 3.00)
    sampling_weights = np.clip((W * hard_boost).astype(np.float32), 0.20, 4.00)

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

    pair_map = _build_vc_cv_pair_map(df)
    if pair_map:
        train_mask = np.zeros((len(df),), dtype=bool)
        valid_mask = np.zeros((len(df),), dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        pair_train_count = int(sum(1 for a, b in pair_map.items() if train_mask[a] and train_mask[b]))
        pair_valid_count = int(sum(1 for a, b in pair_map.items() if valid_mask[a] and valid_mask[b]))
        pair_total_count = int(len(pair_map))
        global_to_train = {int(g): i for i, g in enumerate(train_idx.tolist())}
        global_to_valid = {int(g): i for i, g in enumerate(valid_idx.tolist())}
        pair_map_train = {
            int(global_to_train[a]): int(global_to_train[b])
            for a, b in pair_map.items()
            if a in global_to_train and b in global_to_train
        }
        pair_map_valid = {
            int(global_to_valid[a]): int(global_to_valid[b])
            for a, b in pair_map.items()
            if a in global_to_valid and b in global_to_valid
        }
        val_src_pos = list(pair_map_valid.keys())
        val_dst_pos = [pair_map_valid[k] for k in val_src_pos]
    else:
        pair_train_count = 0
        pair_valid_count = 0
        pair_total_count = 0
        pair_map_train = {}
        pair_map_valid = {}
        val_src_pos = []
        val_dst_pos = []

    is_kr_cvc = _is_kr_cvc(language, format_type)
    pair_weight_default = 0.20 if is_kr_cvc else 0.12
    pair_weight_base = float(os.environ.get("UTOA_ML_VC_CV_PAIR_WEIGHT", pair_weight_default) or pair_weight_default)
    if pair_weight_base < 0.0:
        pair_weight_base = 0.0
    pair_weight_base = pair_weight_base * 0.5
    pair_warmup_default = 8 if is_kr_cvc else 6
    pair_warmup_epochs = max(0, _env_int("UTOA_ML_COUPLED_PAIR_WARMUP_EPOCHS", pair_warmup_default))

    aux_dim = len(AUX_TARGET_NAMES) if use_aux else 0
    head_mode = _resolve_head_mode(_env_str("UTOA_ML_COUPLED_HEAD_MODE", "split"), default="split")
    alias_branch = _resolve_alias_branch_settings(
        mode_env="UTOA_ML_COUPLED_ALIAS_BRANCH_MODE",
        experts_env="UTOA_ML_COUPLED_ALIAS_BRANCH_EXPERTS",
        min_rows_env="UTOA_ML_COUPLED_ALIAS_BRANCH_MIN_ROWS",
        categorical_features=categorical_features,
        categorical_bucket_sizes=categorical_bucket_sizes,
        cat_matrix=C,
        train_idx=train_idx,
    )
    model = _build_model(
        torch,
        nn,
        in_dim=int(X.shape[1]),
        patch_dim=int(P.shape[1]),
        aux_dim=aux_dim,
        categorical_bucket_sizes=categorical_bucket_sizes,
        head_mode=head_mode,
        alias_branch_mode=str(alias_branch.get("applied_mode", "shared")),
        alias_branch_experts=int(alias_branch.get("experts", 4)),
        alias_type_cat_index=int(alias_branch.get("alias_type_cat_index", -1)),
        alias_type_bucket_size=int(alias_branch.get("alias_type_bucket_size", 0)),
        alias_fallback_ids=list(alias_branch.get("fallback_ids", []) or []),
    )
    run_device = _resolve_device(torch, requested=device)
    if isinstance(run_device, str):
        run_device = torch.device(run_device)
    model = model.to(run_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=run_device)
    C_train = torch.tensor(C[train_idx], dtype=torch.long, device=run_device)
    P_train = torch.tensor(P[train_idx], dtype=torch.float32, device=run_device)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32, device=run_device)
    B_train = torch.tensor(base[train_idx], dtype=torch.float32, device=run_device)
    M_train = torch.tensor(boundary[train_idx], dtype=torch.float32, device=run_device)
    W_train = torch.tensor(W[train_idx], dtype=torch.float32, device=run_device)

    X_valid = torch.tensor(X[valid_idx], dtype=torch.float32, device=run_device)
    C_valid = torch.tensor(C[valid_idx], dtype=torch.long, device=run_device)
    P_valid = torch.tensor(P[valid_idx], dtype=torch.float32, device=run_device)
    Y_valid = torch.tensor(Y[valid_idx], dtype=torch.float32, device=run_device)
    B_valid = torch.tensor(base[valid_idx], dtype=torch.float32, device=run_device)
    M_valid = torch.tensor(boundary[valid_idx], dtype=torch.float32, device=run_device)
    W_valid = torch.tensor(W[valid_idx], dtype=torch.float32, device=run_device)
    if use_aux:
        A_train = torch.tensor(A[train_idx], dtype=torch.float32, device=run_device)
        AM_train = torch.tensor(aux_mask[train_idx], dtype=torch.float32, device=run_device)
        A_valid = torch.tensor(A[valid_idx], dtype=torch.float32, device=run_device)
        AM_valid = torch.tensor(aux_mask[valid_idx], dtype=torch.float32, device=run_device)
    else:
        A_train = None
        AM_train = None
        A_valid = None
        AM_valid = None

    train_sampling_weights = sampling_weights[train_idx]
    sampling_group_values, sampling_group_column = _resolve_sampling_group_values(df, train_idx, preferred_column="voicebank_id")
    sampler_mode = _env_str("UTOA_ML_COUPLED_SAMPLER", "group_balanced").strip().lower()
    if sampler_mode not in {"group_balanced", "shuffle"}:
        sampler_mode = "group_balanced"

    if is_kr_cvc:
        # KR CVC: strengthen consonant containment and overlap stability.
        target_weight_values = [1.00, 1.15, 0.95, 1.00, 1.05]
    else:
        target_weight_values = [1.00, 1.10, 0.95, 1.00, 0.90]
    target_weights = torch.tensor(target_weight_values, dtype=torch.float32, device=run_device).view(1, -1)
    target_loss_kinds, target_huber_deltas = _resolve_loss_config(
        "UTOA_ML_COUPLED_",
        TARGET_NAMES,
        ["huber", "l1", "huber", "l1", "l1"],
        [28.0, 18.0, 34.0, 14.0, 12.0],
    )
    anchor_indices = _target_indices(TARGET_NAMES, ANCHOR_TARGET_NAMES)
    delta_indices = _target_indices(TARGET_NAMES, DELTA_TARGET_NAMES)
    anchor_weight_values = [target_weight_values[i] for i in anchor_indices]
    delta_weight_values = [target_weight_values[i] for i in delta_indices]
    anchor_weights = torch.tensor(anchor_weight_values, dtype=torch.float32, device=run_device).view(1, -1)
    delta_weights = torch.tensor(delta_weight_values, dtype=torch.float32, device=run_device).view(1, -1)
    anchor_loss_kinds = [target_loss_kinds[i] for i in anchor_indices]
    anchor_huber_deltas = [target_huber_deltas[i] for i in anchor_indices]
    delta_loss_kinds = [target_loss_kinds[i] for i in delta_indices]
    delta_huber_deltas = [target_huber_deltas[i] for i in delta_indices]
    anchor_loss_weight = _env_float("UTOA_ML_COUPLED_ANCHOR_WEIGHT", 1.0)
    delta_loss_weight = _env_float("UTOA_ML_COUPLED_DELTA_WEIGHT", 1.0)
    aux_target_weight_values = [1.0, 1.0, 1.35]
    aux_target_weights = torch.tensor(aux_target_weight_values, dtype=torch.float32, device=run_device).view(1, -1)
    aux_loss_kinds, aux_huber_deltas = _resolve_loss_config(
        "UTOA_ML_COUPLED_",
        AUX_TARGET_NAMES,
        ["huber", "huber", "huber"],
        [18.0, 18.0, 24.0],
    )
    cons_margin = max(0.0, _env_float("UTOA_ML_COUPLED_CONS_MARGIN", 10.0))
    cut_margin = max(0.0, _env_float("UTOA_ML_COUPLED_CUT_MARGIN", 10.0))
    penalty_loss_weight = max(0.0, _env_float("UTOA_ML_COUPLED_CONSTRAINT_WEIGHT", 0.25))
    align_loss_weight = max(0.0, _env_float("UTOA_ML_COUPLED_ALIGN_WEIGHT", 0.12))
    conf_loss_weight = max(0.0, _env_float("UTOA_ML_COUPLED_CONF_WEIGHT", 0.05))
    boundary_aux_default = 0.18 if is_kr_cvc else 0.14
    boundary_consistency_default = 0.10 if is_kr_cvc else 0.06
    boundary_aux_weight = _env_float("UTOA_ML_COUPLED_BOUNDARY_AUX_WEIGHT", boundary_aux_default)
    boundary_consistency_weight = _env_float(
        "UTOA_ML_COUPLED_BOUNDARY_CONSISTENCY_WEIGHT",
        boundary_consistency_default,
    )

    best_state = None
    best_val = float("inf")
    wait = 0
    patience = max(3, _env_int("UTOA_ML_COUPLED_PATIENCE", 10))

    train_n = int(X_train.shape[0])
    batch_n = max(1, int(batch_size))
    epochs_n = max(1, int(epochs))
    progress_every = int(progress_every)
    total_batches = max(1, int((train_n + batch_n - 1) / batch_n))
    if progress_every > 0:
        print(
            f"[TRAIN] rows={train_n} batches={total_batches} epochs={epochs_n} device={run_device} "
            f"sampler={sampler_mode}:{sampling_group_column}"
        )

    for epoch in range(epochs_n):
        model.train()
        epoch_pair_weight = _pair_weight_for_epoch(pair_weight_base, epoch, pair_warmup_epochs)
        rng = np.random.default_rng(42 + epoch)
        if sampler_mode == "group_balanced":
            perm_np = _sample_group_balanced_indices(sampling_group_values, train_sampling_weights, train_n, rng)
        else:
            perm_np = rng.permutation(train_n).astype(np.int64)
        perm = torch.tensor(perm_np, dtype=torch.long, device=run_device)
        for batch_i, start in enumerate(range(0, train_n, batch_n), start=1):
            batch_idx = perm[start:start + batch_n]
            xb = X_train[batch_idx]
            cb = C_train[batch_idx]
            pb = P_train[batch_idx]
            yb = Y_train[batch_idx]
            bb = B_train[batch_idx]
            mb = M_train[batch_idx]
            wb = W_train[batch_idx]

            out = model(xb, pb, cb)
            if use_aux:
                if isinstance(out, tuple) and len(out) == 4:
                    anchor_pred, delta_pred, conf, aux_pred = out
                else:
                    pred, conf, aux_pred = out
                    anchor_pred = pred[:, anchor_indices]
                    delta_pred = pred[:, delta_indices]
            else:
                if isinstance(out, tuple) and len(out) == 3:
                    anchor_pred, delta_pred, conf = out
                else:
                    pred, conf = out
                    anchor_pred = pred[:, anchor_indices]
                    delta_pred = pred[:, delta_indices]
                aux_pred = None
            pred = _combine_predictions(torch, anchor_pred, delta_pred, anchor_indices, delta_indices, len(TARGET_NAMES))
            yb_anchor = yb[:, anchor_indices]
            yb_delta = yb[:, delta_indices]
            anchor_matrix = _loss_matrix(torch, anchor_pred, yb_anchor, anchor_loss_kinds, anchor_huber_deltas)
            anchor_row = torch.mean(anchor_matrix * anchor_weights, dim=1)
            anchor_loss = torch.mean(anchor_row * wb)
            delta_matrix = _loss_matrix(torch, delta_pred, yb_delta, delta_loss_kinds, delta_huber_deltas)
            delta_row = torch.mean(delta_matrix * delta_weights, dim=1)
            delta_loss = torch.mean(delta_row * wb)
            base_loss = (anchor_loss * float(anchor_loss_weight)) + (delta_loss * float(delta_loss_weight))

            offset = bb[:, 0] + pred[:, 0]
            consonant = bb[:, 1] + pred[:, 1]
            cutoff_abs = bb[:, 2] + pred[:, 2]
            pre = bb[:, 3] + pred[:, 3]
            ovl = bb[:, 4] + pred[:, 4]
            penalty_row = (
                torch.relu(ovl - pre)
                + torch.relu((pre + cons_margin) - consonant)
                + torch.relu((consonant + cut_margin) - cutoff_abs)
                + torch.relu(-offset)
            )
            penalty_loss = torch.mean(penalty_row * wb)

            align_row = _huber_loss(torch, offset, mb[:, 0], 24.0) + _huber_loss(torch, cutoff_abs, mb[:, 1], 24.0)
            align_loss = torch.mean(align_row * wb)

            aux_loss = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            boundary_consistency_loss = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            boundary_row_err = None
            if use_aux and aux_pred is not None and A_train is not None and AM_train is not None:
                ab = A_train[batch_idx]
                am = AM_train[batch_idx]
                aux_matrix = _loss_matrix(torch, aux_pred, ab, aux_loss_kinds, aux_huber_deltas)
                weighted_mask = am * aux_target_weights
                mask_sum = torch.clamp(torch.sum(weighted_mask, dim=1), min=1.0)
                boundary_row_err = torch.sum(aux_matrix * weighted_mask, dim=1) / mask_sum
                aux_loss = torch.mean(boundary_row_err * wb)
                boundary_consistency_row = _boundary_consistency_row_loss(
                    torch,
                    aux_pred,
                    am,
                    offset,
                    cutoff_abs,
                )
                if boundary_consistency_row is not None:
                    boundary_consistency_loss = torch.mean(boundary_consistency_row * wb)
                    boundary_row_err = boundary_row_err + boundary_consistency_row

            delta_row_err = torch.mean(torch.abs(pred - yb) * target_weights, dim=1)
            conf_target = _confidence_target_from_errors(
                torch,
                delta_row_err.detach(),
                align_row.detach(),
                penalty_row.detach(),
                boundary_row_err.detach() if boundary_row_err is not None else None,
            )
            conf_loss_row = F.binary_cross_entropy(conf.squeeze(1), conf_target, reduction="none")
            conf_loss = torch.mean(conf_loss_row * wb)

            pair_loss = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            if epoch_pair_weight > 0.0 and pair_map_train:
                batch_indices = [int(i) for i in batch_idx.detach().cpu().tolist()]
                src_pos, dst_pos = _batch_pair_positions(batch_indices, pair_map_train)
                if src_pos:
                    src_t = torch.tensor(src_pos, device=run_device, dtype=torch.long)
                    dst_t = torch.tensor(dst_pos, device=run_device, dtype=torch.long)
                    pred_offset = bb[:, 0] + pred[:, 0]
                    true_offset = bb[:, 0] + yb[:, 0]
                    pred_gap = pred_offset[dst_t] - pred_offset[src_t]
                    true_gap = true_offset[dst_t] - true_offset[src_t]
                    pair_err = _huber_loss(torch, pred_gap, true_gap, 20.0)
                    pair_w = 0.5 * (wb[src_t] + wb[dst_t])
                    pair_loss = torch.mean(pair_err * pair_w)

            total_loss = (
                base_loss
                + (float(penalty_loss_weight) * penalty_loss)
                + (float(align_loss_weight) * align_loss)
                + (float(conf_loss_weight) * conf_loss)
                + (float(boundary_aux_weight) * aux_loss)
                + (float(boundary_consistency_weight) * boundary_consistency_loss)
                + (epoch_pair_weight * pair_loss)
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if progress_every > 0 and (batch_i % progress_every == 0 or batch_i == total_batches):
                loss_val = float(total_loss.detach().cpu().item())
                print(
                    f"[TRAIN] epoch={epoch + 1}/{epochs_n} batch={batch_i}/{total_batches} "
                    f"loss={loss_val:.4f} pair_w={epoch_pair_weight:.4f}"
                )

        model.eval()
        with torch.no_grad():
            out_val = model(X_valid, P_valid, C_valid)
            if use_aux:
                if isinstance(out_val, tuple) and len(out_val) == 4:
                    anchor_val, delta_val, conf_val, aux_val = out_val
                else:
                    pred_val, conf_val, aux_val = out_val
                    anchor_val = pred_val[:, anchor_indices]
                    delta_val = pred_val[:, delta_indices]
            else:
                if isinstance(out_val, tuple) and len(out_val) == 3:
                    anchor_val, delta_val, conf_val = out_val
                else:
                    pred_val, conf_val = out_val
                    anchor_val = pred_val[:, anchor_indices]
                    delta_val = pred_val[:, delta_indices]
                aux_val = None
            pred_val = _combine_predictions(torch, anchor_val, delta_val, anchor_indices, delta_indices, len(TARGET_NAMES))
            val_anchor_matrix = _loss_matrix(torch, anchor_val, Y_valid[:, anchor_indices], anchor_loss_kinds, anchor_huber_deltas)
            val_anchor_row = torch.mean(val_anchor_matrix * anchor_weights, dim=1)
            val_anchor_loss = torch.mean(val_anchor_row * W_valid)
            val_delta_matrix = _loss_matrix(torch, delta_val, Y_valid[:, delta_indices], delta_loss_kinds, delta_huber_deltas)
            val_delta_row = torch.mean(val_delta_matrix * delta_weights, dim=1)
            val_delta_loss = torch.mean(val_delta_row * W_valid)
            val_base = (val_anchor_loss * float(anchor_loss_weight)) + (val_delta_loss * float(delta_loss_weight))
            offset_v = B_valid[:, 0] + pred_val[:, 0]
            consonant_v = B_valid[:, 1] + pred_val[:, 1]
            cutoff_abs_v = B_valid[:, 2] + pred_val[:, 2]
            pre_v = B_valid[:, 3] + pred_val[:, 3]
            ovl_v = B_valid[:, 4] + pred_val[:, 4]
            val_penalty_row = (
                torch.relu(ovl_v - pre_v)
                + torch.relu((pre_v + cons_margin) - consonant_v)
                + torch.relu((consonant_v + cut_margin) - cutoff_abs_v)
                + torch.relu(-offset_v)
            )
            val_penalty = torch.mean(val_penalty_row * W_valid)
            val_align_row = _huber_loss(torch, offset_v, M_valid[:, 0], 24.0) + _huber_loss(
                torch, cutoff_abs_v, M_valid[:, 1], 24.0
            )
            val_align = torch.mean(val_align_row * W_valid)
            val_boundary_row_err = None
            val_aux = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            val_boundary_consistency = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            if use_aux and aux_val is not None and A_valid is not None and AM_valid is not None:
                val_aux_matrix = _loss_matrix(torch, aux_val, A_valid, aux_loss_kinds, aux_huber_deltas)
                val_weighted_mask = AM_valid * aux_target_weights
                val_mask_sum = torch.clamp(torch.sum(val_weighted_mask, dim=1), min=1.0)
                val_boundary_row_err = torch.sum(val_aux_matrix * val_weighted_mask, dim=1) / val_mask_sum
                val_aux = torch.mean(val_boundary_row_err * W_valid)
                val_boundary_consistency_row = _boundary_consistency_row_loss(
                    torch,
                    aux_val,
                    AM_valid,
                    offset_v,
                    cutoff_abs_v,
                )
                if val_boundary_consistency_row is not None:
                    val_boundary_consistency = torch.mean(val_boundary_consistency_row * W_valid)
                    val_boundary_row_err = val_boundary_row_err + val_boundary_consistency_row
            delta_row_err_v = torch.mean(torch.abs(pred_val - Y_valid) * target_weights, dim=1)
            conf_target_v = _confidence_target_from_errors(
                torch,
                delta_row_err_v.detach(),
                val_align_row.detach(),
                val_penalty_row.detach(),
                val_boundary_row_err.detach() if val_boundary_row_err is not None else None,
            )
            val_conf = torch.mean(
                F.binary_cross_entropy(conf_val.squeeze(1), conf_target_v, reduction="none") * W_valid
            )
            val_pair = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            if epoch_pair_weight > 0.0 and val_src_pos:
                src_t = torch.tensor(val_src_pos, device=run_device, dtype=torch.long)
                dst_t = torch.tensor(val_dst_pos, device=run_device, dtype=torch.long)
                pred_offset_v = B_valid[:, 0] + pred_val[:, 0]
                true_offset_v = B_valid[:, 0] + Y_valid[:, 0]
                pred_gap_v = pred_offset_v[dst_t] - pred_offset_v[src_t]
                true_gap_v = true_offset_v[dst_t] - true_offset_v[src_t]
                pair_err_v = _huber_loss(torch, pred_gap_v, true_gap_v, 20.0)
                pair_w_v = 0.5 * (W_valid[src_t] + W_valid[dst_t])
                val_pair = torch.mean(pair_err_v * pair_w_v)
            val_total = float(
                (
                    val_base
                    + (float(penalty_loss_weight) * val_penalty)
                    + (float(align_loss_weight) * val_align)
                    + (float(conf_loss_weight) * val_conf)
                    + (float(boundary_aux_weight) * val_aux)
                    + (float(boundary_consistency_weight) * val_boundary_consistency)
                    + (epoch_pair_weight * val_pair)
                ).item()
            )

        new_best = False
        if val_total < best_val:
            best_val = val_total
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            new_best = True
        else:
            wait += 1
            if wait >= patience:
                break
        if progress_every > 0:
            status = "best" if new_best else "wait"
            print(
                f"[TRAIN] epoch={epoch + 1}/{epochs_n} val_loss={val_total:.4f} best={best_val:.4f} "
                f"patience={wait}/{patience} status={status} pair_w={epoch_pair_weight:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_valid = model(X_valid, P_valid, C_valid)
        if use_aux:
            if isinstance(out_valid, tuple) and len(out_valid) == 4:
                anchor_valid, delta_valid, conf_valid, aux_valid = out_valid
            else:
                pred_valid, conf_valid, aux_valid = out_valid
                anchor_valid = pred_valid[:, anchor_indices]
                delta_valid = pred_valid[:, delta_indices]
        else:
            if isinstance(out_valid, tuple) and len(out_valid) == 3:
                anchor_valid, delta_valid, conf_valid = out_valid
            else:
                pred_valid, conf_valid = out_valid
                anchor_valid = pred_valid[:, anchor_indices]
                delta_valid = pred_valid[:, delta_indices]
            aux_valid = None
        pred_valid = _combine_predictions(torch, anchor_valid, delta_valid, anchor_indices, delta_indices, len(TARGET_NAMES))
    pred_valid_np = pred_valid.detach().cpu().numpy()
    conf_valid_np = conf_valid.detach().cpu().numpy().reshape(-1)
    truth_valid_np = Y[valid_idx]
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    blank_attach_kpi = _compute_blank_attach_kpi(
        df_valid,
        pred_delta_offset=pred_valid_np[:, 0],
        truth_delta_offset=truth_valid_np[:, 0],
        language=str(language or "").strip().lower(),
        format_type=normalize_format_type(language, format_type) or str(format_type or "").strip().lower(),
    )
    blank_attach_kpi_pitch_zones = _compute_blank_attach_kpi_pitch_zones(
        df_valid,
        pred_delta_offset=pred_valid_np[:, 0],
        truth_delta_offset=truth_valid_np[:, 0],
        language=str(language or "").strip().lower(),
        format_type=normalize_format_type(language, format_type) or str(format_type or "").strip().lower(),
        dataset_csv=str(dataset_csv or ""),
    )

    metrics = {}
    for col_i, target in enumerate(TARGET_NAMES):
        truth = truth_valid_np[:, col_i]
        pred = pred_valid_np[:, col_i]
        metrics[target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    aux_metrics = {}
    if use_aux and aux_valid is not None and A is not None and aux_mask is not None:
        aux_pred_np = aux_valid.detach().cpu().numpy()
        aux_truth_np = A[valid_idx]
        aux_mask_np = aux_mask[valid_idx]
        for col_i, target in enumerate(AUX_TARGET_NAMES):
            mask = aux_mask_np[:, col_i] > 0.5
            if np.any(mask):
                aux_metrics[target] = {
                    "rows": int(np.sum(mask)),
                    "model_mae": float(mean_absolute_error(aux_truth_np[mask, col_i], aux_pred_np[mask, col_i])),
                }
            else:
                aux_metrics[target] = {"rows": 0, "model_mae": 0.0}

    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "categorical_features": categorical_features,
            "categorical_bucket_sizes": [int(v) for v in categorical_bucket_sizes],
            "target_names": list(TARGET_NAMES),
            "anchor_targets": list(ANCHOR_TARGET_NAMES),
            "delta_targets": list(DELTA_TARGET_NAMES),
            "patch_features": list(PATCH_FEATURES),
            "in_dim": int(X.shape[1]),
            "patch_dim": int(P.shape[1]),
            "hidden_dim": 160,
            "head_mode": str(head_mode),
            "alias_branch_mode": str(alias_branch.get("applied_mode", "shared")),
            "alias_branch_requested_mode": str(alias_branch.get("requested_mode", "shared")),
            "alias_branch_experts": int(alias_branch.get("experts", 4)),
            "alias_branch_min_rows": int(alias_branch.get("min_rows", 80)),
            "alias_type_cat_index": int(alias_branch.get("alias_type_cat_index", -1)),
            "alias_type_bucket_size": int(alias_branch.get("alias_type_bucket_size", 0)),
            "alias_fallback_ids": [int(v) for v in (alias_branch.get("fallback_ids", []) or [])],
            "anchor_dim": int(len(ANCHOR_TARGET_NAMES)),
            "delta_dim": int(len(DELTA_TARGET_NAMES)),
            "aux_dim": int(aux_dim),
            "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        },
        os.path.join(out_dir, COUPLED_MODEL_FILE),
    )
    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))
    onnx_export = _export_coupled_onnx(
        torch,
        model,
        out_dir,
        feature_names=feature_names,
        categorical_features=categorical_features,
        categorical_bucket_sizes=[int(v) for v in categorical_bucket_sizes],
        patch_features=list(PATCH_FEATURES),
        head_mode=str(head_mode),
        alias_branch_mode=str(alias_branch.get("applied_mode", "shared")),
        alias_branch_experts=int(alias_branch.get("experts", 4)),
        alias_type_cat_index=int(alias_branch.get("alias_type_cat_index", -1)),
        alias_type_bucket_size=int(alias_branch.get("alias_type_bucket_size", 0)),
        alias_fallback_ids=[int(v) for v in (alias_branch.get("fallback_ids", []) or [])],
        anchor_targets=list(ANCHOR_TARGET_NAMES),
        delta_targets=list(DELTA_TARGET_NAMES),
        rawmel_enabled=False,
        mel_patch_spec={},
    )

    meta = {
        "backend": COUPLED_BACKEND,
        "language": str(language or "").strip().lower(),
        "format_type": normalize_format_type(language, format_type) or "general",
        "model_version": "v2",
        "feature_version": schema.get("feature_version", ""),
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "phoneme_aware_conditioning": {
            "enabled": bool(categorical_features),
            "categorical_bucket_sizes": [int(v) for v in categorical_bucket_sizes],
        },
        "targets": list(TARGET_NAMES),
        "anchor_targets": list(ANCHOR_TARGET_NAMES),
        "delta_targets": list(DELTA_TARGET_NAMES),
        "head_mode": str(head_mode),
        "alias_branch": {
            "requested_mode": str(alias_branch.get("requested_mode", "shared")),
            "applied_mode": str(alias_branch.get("applied_mode", "shared")),
            "experts": int(alias_branch.get("experts", 4)),
            "min_rows": int(alias_branch.get("min_rows", 80)),
            "alias_type_cat_index": int(alias_branch.get("alias_type_cat_index", -1)),
            "alias_type_bucket_size": int(alias_branch.get("alias_type_bucket_size", 0)),
            "active_alias_ids": int(alias_branch.get("active_alias_ids", 0)),
            "strong_alias_ids": int(alias_branch.get("strong_alias_ids", 0)),
            "fallback_ids": [int(v) for v in (alias_branch.get("fallback_ids", []) or [])],
            "reason": str(alias_branch.get("reason", "ok")),
        },
        "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        "mel_patch_spec": list(PATCH_FEATURES),
        "onnx_export": onnx_export,
        "min_confidence": float(min_confidence),
        "vc_cv_pair_weight": float(pair_weight_base),
        "vc_cv_pair_warmup_epochs": int(pair_warmup_epochs),
        "vc_cv_pair_max_gap": int(os.environ.get("UTOA_ML_VC_CV_MAX_GAP", 5) or 5),
        "vc_cv_pairs_total": int(pair_total_count),
        "vc_cv_pairs_train": int(pair_train_count),
        "vc_cv_pairs_valid": int(pair_valid_count),
        "hard_example_mining": {
            "strength": float(hard_example_strength),
            "mean_boost": float(np.mean(hard_boost)) if len(hard_boost) else 1.0,
        },
        "rawmel_prefetch": {
            "mode": str(prefetch_mode),
            "estimated_patch_mb": float(prefetch_info.get("estimated_patch_mb", 0.0) or 0.0),
            "reason": str(prefetch_info.get("reason", "") or ""),
        },
        "sampler": {
            "mode": str(sampler_mode),
            "group_column": str(sampling_group_column),
        },
        "target_loss": {
            "kinds": list(target_loss_kinds),
            "huber_deltas": [float(v) for v in target_huber_deltas],
            "weights": [float(v) for v in target_weight_values],
        },
        "boundary_aux": {
            "weight": float(boundary_aux_weight),
            "consistency_weight": float(boundary_consistency_weight),
            "target_weights": [float(v) for v in aux_target_weight_values],
            "loss_kinds": list(aux_loss_kinds),
            "huber_deltas": [float(v) for v in aux_huber_deltas],
        },
        "constraint_loss": {
            "penalty_weight": float(penalty_loss_weight),
            "align_weight": float(align_loss_weight),
            "conf_weight": float(conf_loss_weight),
            "cons_margin": float(cons_margin),
            "cut_margin": float(cut_margin),
        },
        "fallback_order": [COUPLED_BACKEND, "lightgbm", "base"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "holdout_metrics": metrics,
        "aux_holdout_metrics": aux_metrics,
        "blank_attach_kpi": blank_attach_kpi,
        "blank_attach_kpi_pitch_zones": blank_attach_kpi_pitch_zones,
        "holdout_confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
        "device_used": str(run_device),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "aux_metrics": aux_metrics,
                "confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_min": float(np.min(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_max": float(np.max(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "blank_attach_kpi": blank_attach_kpi,
                "blank_attach_kpi_pitch_zones": blank_attach_kpi_pitch_zones,
                "kpi_primary": {
                    "name": str(blank_attach_kpi.get("primary_kpi_name", "head_pred_attach_rate")),
                    "value": float(blank_attach_kpi.get("primary_kpi_value", 0.0)),
                },
                "vc_cv_pair_weight": float(pair_weight_base),
                "vc_cv_pairs_total": int(pair_total_count),
                "vc_cv_pairs_valid": int(pair_valid_count),
                "rawmel_prefetch": {
                    "mode": str(prefetch_mode),
                    "estimated_patch_mb": float(prefetch_info.get("estimated_patch_mb", 0.0) or 0.0),
                    "reason": str(prefetch_info.get("reason", "") or ""),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return meta


def _frames_from_spec(window_ms: List[float], hop_ms: float) -> int:
    if not window_ms or len(window_ms) != 2:
        return 1
    span = float(window_ms[1]) - float(window_ms[0])
    return max(1, int(round(span / float(hop_ms))) + 1)


def train_coupled_bundle_rawmel(
    language: str,
    format_type: str,
    dataset_csv: str,
    out_dir: str,
    *,
    rawmel_cache_dir: str,
    group_column: str = "voicebank_id",
    alias_types: Optional[List[str]] = None,
    alias_family: str = "",
    min_mapping_confidence: float = 0.0,
    device: str = "auto",
    epochs: int = 70,
    batch_size: int = 192,
    learning_rate: float = 1e-3,
    min_confidence: float = 0.55,
    progress_every: int = 0,
    rawmel_prefetch: str = "auto",
    rawmel_max_shard_cache: int = 2,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.exists(dataset_csv):
        raise FileNotFoundError(dataset_csv)
    if not rawmel_cache_dir or not os.path.isdir(rawmel_cache_dir):
        raise FileNotFoundError(rawmel_cache_dir)

    torch, nn, F = _import_torch()
    df = _read_dataset_csv_resilient(dataset_csv)
    df = _prepare_training_frame(
        df,
        language=language,
        format_type=format_type,
        alias_types=alias_types,
        alias_family=alias_family,
        min_mapping_confidence=min_mapping_confidence,
    )
    df = df.reset_index(drop=True)
    if len(df) < 16:
        raise RuntimeError("Coupled dataset is too small (need >= 16 rows).")

    required_key_cols = [
        "language",
        "format_type",
        "voicebank_id",
        "wav_norm",
        "alias_norm",
        "occurrence_index",
        "row_index_in_wav",
    ]
    if "mel_patch_key" not in df.columns:
        missing = [c for c in required_key_cols if c not in df.columns]
        if missing:
            raise RuntimeError(
                "Dataset is missing mel_patch_key and required columns: "
                + ", ".join(missing)
                + " (rebuild dataset CSV)."
            )
        print("[TRAIN] mel_patch_key missing; computing from columns.")
        def _build_patch_key(row):
            row_lang = str(row.get("language", "") or "").strip().lower()
            row_fmt = normalize_format_type(row_lang, row.get("format_type", "")) or str(row.get("format_type", "") or "").strip().lower()
            return make_mel_patch_key(
                language=row_lang,
                format_type=row_fmt,
                voicebank_id=str(row.get("voicebank_id", "") or ""),
                wav_norm=str(row.get("wav_norm", "") or ""),
                alias_norm=str(row.get("alias_norm", "") or ""),
                occurrence_index=int(float(row.get("occurrence_index", 0) or 0)),
                row_index_in_wav=int(float(row.get("row_index_in_wav", 0) or 0)),
            )
        df["mel_patch_key"] = df.apply(_build_patch_key, axis=1)
    else:
        # Fill empty keys if present
        key_series = df["mel_patch_key"].astype(str)
        empty_mask = key_series.str.strip() == ""
        if empty_mask.any():
            missing = [c for c in required_key_cols if c not in df.columns]
            if missing:
                raise RuntimeError(
                    "Dataset has empty mel_patch_key but missing required columns: "
                    + ", ".join(missing)
                    + " (rebuild dataset CSV)."
                )
            print(f"[TRAIN] mel_patch_key empty rows={int(empty_mask.sum())}; computing from columns.")
            def _build_patch_key(row):
                row_lang = str(row.get("language", "") or "").strip().lower()
                row_fmt = normalize_format_type(row_lang, row.get("format_type", "")) or str(row.get("format_type", "") or "").strip().lower()
                return make_mel_patch_key(
                    language=row_lang,
                    format_type=row_fmt,
                    voicebank_id=str(row.get("voicebank_id", "") or ""),
                    wav_norm=str(row.get("wav_norm", "") or ""),
                    alias_norm=str(row.get("alias_norm", "") or ""),
                    occurrence_index=int(float(row.get("occurrence_index", 0) or 0)),
                    row_index_in_wav=int(float(row.get("row_index_in_wav", 0) or 0)),
                )
            df.loc[empty_mask, "mel_patch_key"] = df[empty_mask].apply(_build_patch_key, axis=1)

    cache_index = MelPatchCacheIndex.load(rawmel_cache_dir, max_shard_cache=int(rawmel_max_shard_cache))
    patch_spec = cache_index.patch_spec or {}
    hop_ms = float(patch_spec.get("frame_hop_ms", 5.0))
    mel_bins = int(patch_spec.get("mel_bins", 80))
    onset_frames = _frames_from_spec(patch_spec.get("onset_window_ms", [0.0, 0.0]), hop_ms)
    tail_frames = _frames_from_spec(patch_spec.get("tail_window_ms", [0.0, 0.0]), hop_ms)
    patch_hash = patch_spec_hash(patch_spec)

    keys = df["mel_patch_key"].astype(str).tolist()
    missing_sample_n = max(1, int(_env_int("UTOA_ML_RAWMEL_MISSING_KEYS_SAMPLE", 20) or 20))
    missing_count = 0
    missing_sample: List[str] = []
    key_exists_mask: List[bool] = []
    for key in keys:
        has_key = cache_index.has_key(key)
        key_exists_mask.append(bool(has_key))
        if has_key:
            continue
        missing_count += 1
        if len(missing_sample) < missing_sample_n:
            missing_sample.append(str(key))
    if missing_count > 0:
        os.makedirs(out_dir, exist_ok=True)
        missing_report = os.path.join(out_dir, "rawmel_missing_keys.sample.json")
        with open(missing_report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cache_dir": rawmel_cache_dir,
                    "missing_count": int(missing_count),
                    "sample_limit": int(missing_sample_n),
                    "sample_keys": missing_sample,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        missing_policy = str(os.environ.get("UTOA_ML_RAWMEL_MISSING_KEYS_POLICY", "error") or "error").strip().lower()
        max_drop_ratio = float(_env_float("UTOA_ML_RAWMEL_MISSING_KEYS_MAX_RATIO", 0.01) or 0.01)
        total_keys = max(1, len(keys))
        missing_ratio = float(missing_count) / float(total_keys)
        if missing_policy in {"drop", "warn"} and missing_ratio <= max_drop_ratio:
            keep_mask_np = np.asarray(key_exists_mask, dtype=bool)
            dropped = int((~keep_mask_np).sum())
            df = df.loc[keep_mask_np].reset_index(drop=True)
            print(
                (
                    f"[TRAIN] rawmel missing keys drop enabled: dropped={dropped}, "
                    f"kept={len(df)}, missing_ratio={missing_ratio:.6f}, "
                    f"max_ratio={max_drop_ratio:.6f}, report={missing_report}"
                ),
                flush=True,
            )
            if len(df) == 0:
                raise RuntimeError(
                    "Raw mel cache missing-key drop removed all rows. "
                    f"sample_report={missing_report}"
                )
        else:
            raise RuntimeError(
                f"Raw mel cache missing keys (count={int(missing_count)}, ratio={missing_ratio:.6f}). "
                f"sample_report={missing_report}. "
                "Set UTOA_ML_RAWMEL_MISSING_KEYS_POLICY=drop and "
                "UTOA_ML_RAWMEL_MISSING_KEYS_MAX_RATIO (default 0.01) to allow small drops."
            )

    schema = get_feature_schema()
    feature_names = list(schema.get("feature_names") or FEATURE_NAMES)
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in feature_names]
    categorical_bucket_sizes = _default_categorical_bucket_sizes(categorical_features)

    x_rows = []
    c_rows = []
    p_rows = []
    for _, row in df.iterrows():
        as_dict = row.to_dict()
        x_rows.append(_row_feature_vector(as_dict, feature_names, categorical_features))
        c_rows.append(_row_categorical_index_vector(as_dict, categorical_features, categorical_bucket_sizes))
        p_rows.append(_row_patch_vector(as_dict))
    X = np.asarray(x_rows, dtype=np.float32)
    C = np.asarray(c_rows, dtype=np.int64) if c_rows else np.zeros((len(df), 0), dtype=np.int64)
    P = np.asarray(p_rows, dtype=np.float32)
    Y = np.stack(
        [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in TARGET_NAMES],
        axis=1,
    )
    use_aux = all(name in df.columns for name in AUX_TARGET_NAMES)
    if use_aux:
        A = np.stack(
            [pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for target in AUX_TARGET_NAMES],
            axis=1,
        )
        vowel_start_valid = pd.to_numeric(df["curr_vowel_start_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        vowel_end_valid = pd.to_numeric(df["curr_vowel_end_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        next_onset_valid = (
            pd.to_numeric(df["base_cutoff_to_next_anchor_ms"], errors="coerce").fillna(0.0).to_numpy() > 0.0
        )
        aux_mask = np.stack(
            [vowel_start_valid.astype(np.float32), vowel_end_valid.astype(np.float32), next_onset_valid.astype(np.float32)],
            axis=1,
        )
    else:
        A = None
        aux_mask = None
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
    if "alias_type" in df.columns:
        alias_type_arr = df["alias_type"].astype(str).str.lower().to_numpy()
        cv_family_mask_np = np.isin(alias_type_arr, ["cv", "cv_head"]).astype(np.float32)
    else:
        cv_family_mask_np = np.zeros((len(df),), dtype=np.float32)
    blank_conf_np = (
        pd.to_numeric(df["blank_span_confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "blank_span_confidence" in df.columns
        else np.zeros((len(df),), dtype=np.float32)
    )
    jump_blocked_np = (
        pd.to_numeric(df["jump_blocked_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "jump_blocked_flag" in df.columns
        else np.zeros((len(df),), dtype=np.float32)
    )
    mapping_conf_np = (
        pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        if "mapping_confidence" in df.columns
        else np.ones((len(df),), dtype=np.float32)
    )
    is_kr_cvc = _is_kr_cvc(language, format_type)
    risk_blank_th = _env_float("UTOA_ML_RAWMEL_RISK_BLANK_TH", 0.55)
    risk_map_conf_th = _env_float("UTOA_ML_RAWMEL_RISK_MAP_CONF_TH", 0.64)
    risk_boost_blank_default = 1.12 if is_kr_cvc else 0.90
    risk_boost_blank = _env_float("UTOA_ML_RAWMEL_RISK_BOOST_BLANK", risk_boost_blank_default)
    risk_boost_blank = max(0.50, min(1.50, float(risk_boost_blank)))
    risk_boost_jump = max(1.0, _env_float("UTOA_ML_RAWMEL_RISK_BOOST_JUMP", 1.15))
    risk_boost_low_conf = max(1.0, _env_float("UTOA_ML_RAWMEL_RISK_BOOST_LOW_CONF", 1.08))
    risk_boost = np.ones((len(df),), dtype=np.float32)
    risk_boost *= np.where((cv_family_mask_np > 0.5) & (blank_conf_np >= float(risk_blank_th)), risk_boost_blank, 1.0)
    risk_boost *= np.where((cv_family_mask_np > 0.5) & (jump_blocked_np > 0.5), risk_boost_jump, 1.0)
    risk_boost *= np.where((cv_family_mask_np > 0.5) & (mapping_conf_np < float(risk_map_conf_th)), risk_boost_low_conf, 1.0)
    hard_example_strength = max(0.0, _env_float("UTOA_ML_RAWMEL_HARD_MINING_STRENGTH", 0.55))
    hard_boost = _compute_static_hard_example_boost(
        df,
        alias_type_arr if "alias_type" in df.columns else np.full((len(df),), "", dtype=object),
        mapping_conf_np,
        blank_conf_np,
        jump_blocked_np,
        aux_mask,
        strength=hard_example_strength,
    )
    W = np.clip((W * risk_boost * np.sqrt(hard_boost)).astype(np.float32), 0.20, 3.00)
    W = _apply_blank_attach_focus_weight(
        df,
        W,
        language=str(language or "").strip().lower(),
        format_type=normalize_format_type(language, format_type) or str(format_type or "").strip().lower(),
    )
    sampling_weights = np.clip((W * hard_boost).astype(np.float32), 0.20, 4.00)

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

    pair_map = _build_vc_cv_pair_map(df)
    pair_weight_default = 0.20 if is_kr_cvc else 0.12
    pair_weight_base = float(os.environ.get("UTOA_ML_VC_CV_PAIR_WEIGHT", pair_weight_default) or pair_weight_default)
    if pair_weight_base < 0.0:
        pair_weight_base = 0.0
    pair_weight_base = pair_weight_base * 0.5
    pair_warmup_default = 10 if is_kr_cvc else 8
    pair_warmup_epochs = max(0, _env_int("UTOA_ML_RAWMEL_PAIR_WARMUP_EPOCHS", pair_warmup_default))
    if pair_map:
        train_mask = np.zeros((len(df),), dtype=bool)
        valid_mask = np.zeros((len(df),), dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        pair_train_count = int(sum(1 for a, b in pair_map.items() if train_mask[a] and train_mask[b]))
        pair_valid_count = int(sum(1 for a, b in pair_map.items() if valid_mask[a] and valid_mask[b]))
        pair_total_count = int(len(pair_map))
        global_to_train = {int(g): i for i, g in enumerate(train_idx.tolist())}
        global_to_valid = {int(g): i for i, g in enumerate(valid_idx.tolist())}
        pair_map_train = {
            int(global_to_train[a]): int(global_to_train[b])
            for a, b in pair_map.items()
            if a in global_to_train and b in global_to_train
        }
        pair_map_valid = {
            int(global_to_valid[a]): int(global_to_valid[b])
            for a, b in pair_map.items()
            if a in global_to_valid and b in global_to_valid
        }
        val_src_pos = list(pair_map_valid.keys())
        val_dst_pos = [pair_map_valid[k] for k in val_src_pos]
    else:
        pair_train_count = 0
        pair_valid_count = 0
        pair_total_count = 0
        pair_map_train = {}
        pair_map_valid = {}
        val_src_pos = []
        val_dst_pos = []

    aux_dim = len(AUX_TARGET_NAMES) if use_aux else 0
    head_mode = _resolve_head_mode(_env_str("UTOA_ML_RAWMEL_HEAD_MODE", "split"), default="split")
    alias_branch = _resolve_alias_branch_settings(
        mode_env="UTOA_ML_RAWMEL_ALIAS_BRANCH_MODE",
        experts_env="UTOA_ML_RAWMEL_ALIAS_BRANCH_EXPERTS",
        min_rows_env="UTOA_ML_RAWMEL_ALIAS_BRANCH_MIN_ROWS",
        categorical_features=categorical_features,
        categorical_bucket_sizes=categorical_bucket_sizes,
        cat_matrix=C,
        train_idx=train_idx,
    )
    model = _build_model_rawmel(
        torch,
        nn,
        in_dim=int(X.shape[1]),
        patch_dim=int(P.shape[1]),
        mel_bins=int(mel_bins),
        onset_frames=int(onset_frames),
        tail_frames=int(tail_frames),
        aux_dim=aux_dim,
        categorical_bucket_sizes=categorical_bucket_sizes,
        head_mode=head_mode,
        alias_branch_mode=str(alias_branch.get("applied_mode", "shared")),
        alias_branch_experts=int(alias_branch.get("experts", 4)),
        alias_type_cat_index=int(alias_branch.get("alias_type_cat_index", -1)),
        alias_type_bucket_size=int(alias_branch.get("alias_type_bucket_size", 0)),
        alias_fallback_ids=list(alias_branch.get("fallback_ids", []) or []),
    )
    run_device = _resolve_device(torch, requested=device)
    if isinstance(run_device, str):
        run_device = torch.device(run_device)
    model = model.to(run_device)
    rawmel_weight_decay = max(0.0, _env_float("UTOA_ML_RAWMEL_WEIGHT_DECAY", 1e-4))
    optimizer_name = str(os.environ.get("UTOA_ML_RAWMEL_OPTIMIZER", "adamw") or "adamw").strip().lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        optimizer_name = "adam"
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(rawmel_weight_decay))
        optimizer_name = "adamw"
    scheduler_patience = max(1, int(os.environ.get("UTOA_ML_RAWMEL_LR_PATIENCE", 3) or 3))
    scheduler_factor = max(0.20, min(0.90, _env_float("UTOA_ML_RAWMEL_LR_FACTOR", 0.60)))
    scheduler_min_lr = max(1e-6, _env_float("UTOA_ML_RAWMEL_MIN_LR", 2e-5))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_factor),
        patience=int(scheduler_patience),
        min_lr=float(scheduler_min_lr),
    )
    grad_clip = max(0.0, _env_float("UTOA_ML_RAWMEL_GRAD_CLIP", 1.2))

    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=run_device)
    C_train = torch.tensor(C[train_idx], dtype=torch.long, device=run_device)
    P_train = torch.tensor(P[train_idx], dtype=torch.float32, device=run_device)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32, device=run_device)
    B_train = torch.tensor(base[train_idx], dtype=torch.float32, device=run_device)
    M_train = torch.tensor(boundary[train_idx], dtype=torch.float32, device=run_device)
    W_train = torch.tensor(W[train_idx], dtype=torch.float32, device=run_device)

    X_valid = torch.tensor(X[valid_idx], dtype=torch.float32, device=run_device)
    C_valid = torch.tensor(C[valid_idx], dtype=torch.long, device=run_device)
    P_valid = torch.tensor(P[valid_idx], dtype=torch.float32, device=run_device)
    Y_valid = torch.tensor(Y[valid_idx], dtype=torch.float32, device=run_device)
    B_valid = torch.tensor(base[valid_idx], dtype=torch.float32, device=run_device)
    M_valid = torch.tensor(boundary[valid_idx], dtype=torch.float32, device=run_device)
    W_valid = torch.tensor(W[valid_idx], dtype=torch.float32, device=run_device)
    if use_aux:
        A_train = torch.tensor(A[train_idx], dtype=torch.float32, device=run_device)
        AM_train = torch.tensor(aux_mask[train_idx], dtype=torch.float32, device=run_device)
        A_valid = torch.tensor(A[valid_idx], dtype=torch.float32, device=run_device)
        AM_valid = torch.tensor(aux_mask[valid_idx], dtype=torch.float32, device=run_device)
    else:
        A_train = None
        AM_train = None
        A_valid = None
        AM_valid = None

    train_sampling_weights = sampling_weights[train_idx]
    sampling_group_values, sampling_group_column = _resolve_sampling_group_values(df, train_idx, preferred_column="voicebank_id")
    sampler_mode = _env_str("UTOA_ML_RAWMEL_SAMPLER", "group_balanced").strip().lower()
    if sampler_mode not in {"group_balanced", "shuffle"}:
        sampler_mode = "group_balanced"

    keys_train = [keys[i] for i in train_idx.tolist()]
    keys_valid = [keys[i] for i in valid_idx.tolist()]

    def _np_to_device(np_arr):
        arr = np.ascontiguousarray(np_arr, dtype=np.float32)
        t = torch.from_numpy(arr)
        if run_device.type == "cuda":
            t = t.pin_memory().to(run_device, non_blocking=True)
        else:
            t = t.to(run_device)
        return t

    prefetch_info = _resolve_rawmel_prefetch_mode(
        rawmel_prefetch,
        run_device=run_device,
        train_count=len(keys_train),
        valid_count=len(keys_valid),
        onset_frames=int(onset_frames),
        tail_frames=int(tail_frames),
        mel_bins=int(mel_bins),
    )
    prefetch_mode = str(prefetch_info.get("mode", "none") or "none").strip().lower()
    onset_train_cache = None
    tail_train_cache = None
    onset_valid_cache = None
    tail_valid_cache = None
    onset_train_cache_gpu = None
    tail_train_cache_gpu = None
    onset_valid_cache_gpu = None
    tail_valid_cache_gpu = None
    if prefetch_mode == "gpu":
        print(
            f"[TRAIN] rawmel prefetch: gpu estimated_patch_mb={float(prefetch_info.get('estimated_patch_mb', 0.0)):.1f} "
            f"reason={prefetch_info.get('reason', 'explicit')}"
        )
        try:
            onset_train_np, tail_train_np = cache_index.get_batch(keys_train)
            onset_valid_np, tail_valid_np = cache_index.get_batch(keys_valid)
            onset_train_cache_gpu = _np_to_device(onset_train_np).unsqueeze(1)
            tail_train_cache_gpu = _np_to_device(tail_train_np).unsqueeze(1)
            onset_valid_cache_gpu = _np_to_device(onset_valid_np).unsqueeze(1)
            tail_valid_cache_gpu = _np_to_device(tail_valid_np).unsqueeze(1)
        except RuntimeError as exc:
            if run_device.type != "cuda":
                raise
            print(f"[TRAIN] rawmel gpu prefetch failed; fallback to train cache: {exc}")
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            prefetch_mode = "train"
    if prefetch_mode == "train":
        print(
            f"[TRAIN] rawmel prefetch: train estimated_patch_mb={float(prefetch_info.get('estimated_patch_mb', 0.0)):.1f} "
            f"reason={prefetch_info.get('reason', 'explicit')}"
        )
        onset_train_cache, tail_train_cache = cache_index.get_batch(keys_train)
        onset_valid_cache, tail_valid_cache = cache_index.get_batch(keys_valid)

    target_default_offset = 1.12
    target_default_cons = 1.15 if is_kr_cvc else 1.05
    target_default_cutoff = 1.02
    target_default_pre = 1.08
    target_default_ovl = 1.05 if is_kr_cvc else 0.90
    target_weight_values = [
        _env_float("UTOA_ML_RAWMEL_TARGET_W_OFFSET", target_default_offset),
        _env_float("UTOA_ML_RAWMEL_TARGET_W_CONS", target_default_cons),
        _env_float("UTOA_ML_RAWMEL_TARGET_W_CUTOFF", target_default_cutoff),
        _env_float("UTOA_ML_RAWMEL_TARGET_W_PRE", target_default_pre),
        _env_float("UTOA_ML_RAWMEL_TARGET_W_OVL", target_default_ovl),
    ]
    target_weights = torch.tensor(target_weight_values, dtype=torch.float32, device=run_device).view(1, -1)
    target_loss_kinds, target_huber_deltas = _resolve_loss_config(
        "UTOA_ML_RAWMEL_",
        TARGET_NAMES,
        ["huber", "l1", "huber", "l1", "l1"],
        [30.0, 18.0, 38.0, 14.0, 12.0],
    )
    anchor_indices = _target_indices(TARGET_NAMES, ANCHOR_TARGET_NAMES)
    delta_indices = _target_indices(TARGET_NAMES, DELTA_TARGET_NAMES)
    anchor_weight_values = [target_weight_values[i] for i in anchor_indices]
    delta_weight_values = [target_weight_values[i] for i in delta_indices]
    anchor_weights = torch.tensor(anchor_weight_values, dtype=torch.float32, device=run_device).view(1, -1)
    delta_weights = torch.tensor(delta_weight_values, dtype=torch.float32, device=run_device).view(1, -1)
    anchor_loss_kinds = [target_loss_kinds[i] for i in anchor_indices]
    anchor_huber_deltas = [target_huber_deltas[i] for i in anchor_indices]
    delta_loss_kinds = [target_loss_kinds[i] for i in delta_indices]
    delta_huber_deltas = [target_huber_deltas[i] for i in delta_indices]
    anchor_loss_weight = _env_float("UTOA_ML_RAWMEL_ANCHOR_WEIGHT", 1.0)
    delta_loss_weight = _env_float("UTOA_ML_RAWMEL_DELTA_WEIGHT", 1.0)
    aux_target_weight_values = [1.0, 1.0, 1.45]
    aux_target_weights = torch.tensor(aux_target_weight_values, dtype=torch.float32, device=run_device).view(1, -1)
    aux_loss_kinds, aux_huber_deltas = _resolve_loss_config(
        "UTOA_ML_RAWMEL_",
        AUX_TARGET_NAMES,
        ["huber", "huber", "huber"],
        [18.0, 18.0, 26.0],
    )
    cons_margin = max(0.0, _env_float("UTOA_ML_RAWMEL_CONS_MARGIN", 10.0))
    cut_margin = max(0.0, _env_float("UTOA_ML_RAWMEL_CUT_MARGIN", 10.0))
    penalty_loss_weight = max(0.0, _env_float("UTOA_ML_RAWMEL_CONSTRAINT_WEIGHT", 0.25))
    align_loss_weight = max(0.0, _env_float("UTOA_ML_RAWMEL_ALIGN_WEIGHT", 0.12))
    conf_loss_weight = max(0.0, _env_float("UTOA_ML_RAWMEL_CONF_WEIGHT", 0.05))
    boundary_aux_default = 0.24 if is_kr_cvc else 0.18
    boundary_consistency_default = 0.12 if is_kr_cvc else 0.08
    boundary_aux_weight = _env_float("UTOA_ML_RAWMEL_BOUNDARY_AUX_WEIGHT", boundary_aux_default)
    boundary_consistency_weight = _env_float(
        "UTOA_ML_RAWMEL_BOUNDARY_CONSISTENCY_WEIGHT",
        boundary_consistency_default,
    )

    best_state = None
    best_val = float("inf")
    wait = 0
    patience = max(3, int(os.environ.get("UTOA_ML_RAWMEL_PATIENCE", 12) or 12))

    train_n = int(X_train.shape[0])
    batch_n = max(1, int(batch_size))
    epochs_n = max(1, int(epochs))
    progress_every = int(progress_every)
    total_batches = max(1, int((train_n + batch_n - 1) / batch_n))
    print(
        f"[TRAIN] start rows={train_n} valid={int(X_valid.shape[0])} batches={total_batches} epochs={epochs_n} "
        f"device={run_device} rawmel={mel_bins}x(onset={onset_frames},tail={tail_frames}) "
        f"prefetch={prefetch_mode} sampler={sampler_mode}:{sampling_group_column}"
    )

    for epoch in range(epochs_n):
        epoch_started_at = time.perf_counter()
        model.train()
        epoch_pair_weight = _pair_weight_for_epoch(pair_weight_base, epoch, pair_warmup_epochs)
        rng = np.random.default_rng(42 + epoch)
        if sampler_mode == "group_balanced":
            perm_np = _sample_group_balanced_indices(sampling_group_values, train_sampling_weights, train_n, rng)
        else:
            perm_np = rng.permutation(train_n).astype(np.int64)
        for batch_i, start in enumerate(range(0, train_n, batch_n), start=1):
            batch_idx_np = np.asarray(perm_np[start:start + batch_n], dtype=np.int64)
            idx_list = [int(i) for i in batch_idx_np.tolist()]
            batch_idx = torch.as_tensor(batch_idx_np, dtype=torch.long, device=run_device)
            xb = X_train[batch_idx]
            cb = C_train[batch_idx]
            pb = P_train[batch_idx]
            yb = Y_train[batch_idx]
            bb = B_train[batch_idx]
            mb = M_train[batch_idx]
            wb = W_train[batch_idx]
            if prefetch_mode == "gpu" and onset_train_cache_gpu is not None and tail_train_cache_gpu is not None:
                onset_t = onset_train_cache_gpu[batch_idx]
                tail_t = tail_train_cache_gpu[batch_idx]
            elif prefetch_mode == "train" and onset_train_cache is not None and tail_train_cache is not None:
                onset_np = onset_train_cache[batch_idx_np]
                tail_np = tail_train_cache[batch_idx_np]
                onset_t = _np_to_device(onset_np).unsqueeze(1)
                tail_t = _np_to_device(tail_np).unsqueeze(1)
            else:
                batch_keys = [keys_train[i] for i in idx_list]
                onset_np, tail_np = cache_index.get_batch(batch_keys)
                onset_t = _np_to_device(onset_np).unsqueeze(1)
                tail_t = _np_to_device(tail_np).unsqueeze(1)

            out = model(xb, pb, onset_t, tail_t, cb)
            if use_aux:
                if isinstance(out, tuple) and len(out) == 4:
                    anchor_pred, delta_pred, conf, aux_pred = out
                else:
                    pred, conf, aux_pred = out
                    anchor_pred = pred[:, anchor_indices]
                    delta_pred = pred[:, delta_indices]
            else:
                if isinstance(out, tuple) and len(out) == 3:
                    anchor_pred, delta_pred, conf = out
                else:
                    pred, conf = out
                    anchor_pred = pred[:, anchor_indices]
                    delta_pred = pred[:, delta_indices]
                aux_pred = None
            pred = _combine_predictions(torch, anchor_pred, delta_pred, anchor_indices, delta_indices, len(TARGET_NAMES))
            yb_anchor = yb[:, anchor_indices]
            yb_delta = yb[:, delta_indices]
            anchor_matrix = _loss_matrix(torch, anchor_pred, yb_anchor, anchor_loss_kinds, anchor_huber_deltas)
            anchor_row = torch.mean(anchor_matrix * anchor_weights, dim=1)
            anchor_loss = torch.mean(anchor_row * wb)
            delta_matrix = _loss_matrix(torch, delta_pred, yb_delta, delta_loss_kinds, delta_huber_deltas)
            delta_row = torch.mean(delta_matrix * delta_weights, dim=1)
            delta_loss = torch.mean(delta_row * wb)
            base_loss = (anchor_loss * float(anchor_loss_weight)) + (delta_loss * float(delta_loss_weight))

            offset = bb[:, 0] + pred[:, 0]
            consonant = bb[:, 1] + pred[:, 1]
            cutoff_abs = bb[:, 2] + pred[:, 2]
            pre = bb[:, 3] + pred[:, 3]
            ovl = bb[:, 4] + pred[:, 4]
            penalty_row = (
                torch.relu(ovl - pre)
                + torch.relu((pre + cons_margin) - consonant)
                + torch.relu((consonant + cut_margin) - cutoff_abs)
                + torch.relu(-offset)
            )
            penalty_loss = torch.mean(penalty_row * wb)

            align_row = _huber_loss(torch, offset, mb[:, 0], 24.0) + _huber_loss(torch, cutoff_abs, mb[:, 1], 24.0)
            align_loss = torch.mean(align_row * wb)

            aux_loss = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            boundary_consistency_loss = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            boundary_row_err = None
            if use_aux and aux_pred is not None and A_train is not None and AM_train is not None:
                ab = A_train[batch_idx]
                am = AM_train[batch_idx]
                aux_matrix = _loss_matrix(torch, aux_pred, ab, aux_loss_kinds, aux_huber_deltas)
                weighted_mask = am * aux_target_weights
                mask_sum = torch.clamp(torch.sum(weighted_mask, dim=1), min=1.0)
                boundary_row_err = torch.sum(aux_matrix * weighted_mask, dim=1) / mask_sum
                aux_loss = torch.mean(boundary_row_err * wb)
                boundary_consistency_row = _boundary_consistency_row_loss(
                    torch,
                    aux_pred,
                    am,
                    offset,
                    cutoff_abs,
                )
                if boundary_consistency_row is not None:
                    boundary_consistency_loss = torch.mean(boundary_consistency_row * wb)
                    boundary_row_err = boundary_row_err + boundary_consistency_row

            delta_row_err = torch.mean(torch.abs(pred - yb) * target_weights, dim=1)
            conf_target = _confidence_target_from_errors(
                torch,
                delta_row_err.detach(),
                align_row.detach(),
                penalty_row.detach(),
                boundary_row_err.detach() if boundary_row_err is not None else None,
            )
            conf_loss = torch.mean(
                F.binary_cross_entropy(conf.squeeze(1), conf_target, reduction="none") * wb
            )

            pair_loss = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            if epoch_pair_weight > 0.0 and pair_map_train:
                src_pos, dst_pos = _batch_pair_positions(idx_list, pair_map_train)
                if src_pos:
                    src_t = torch.tensor(src_pos, device=run_device, dtype=torch.long)
                    dst_t = torch.tensor(dst_pos, device=run_device, dtype=torch.long)
                    pred_offset = bb[:, 0] + pred[:, 0]
                    true_offset = bb[:, 0] + yb[:, 0]
                    pred_gap = pred_offset[dst_t] - pred_offset[src_t]
                    true_gap = true_offset[dst_t] - true_offset[src_t]
                    pair_err = _huber_loss(torch, pred_gap, true_gap, 20.0)
                    pair_w = 0.5 * (wb[src_t] + wb[dst_t])
                    pair_loss = torch.mean(pair_err * pair_w)

            total_loss = (
                base_loss
                + (float(penalty_loss_weight) * penalty_loss)
                + (float(align_loss_weight) * align_loss)
                + (float(conf_loss_weight) * conf_loss)
                + (float(boundary_aux_weight) * aux_loss)
                + (float(boundary_consistency_weight) * boundary_consistency_loss)
                + (epoch_pair_weight * pair_loss)
            )
            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            if progress_every > 0 and (batch_i % progress_every == 0 or batch_i == total_batches):
                loss_val = float(total_loss.detach().cpu().item())
                print(
                    f"[TRAIN] epoch={epoch + 1}/{epochs_n} batch={batch_i}/{total_batches} "
                    f"loss={loss_val:.4f} pair_w={epoch_pair_weight:.4f}"
                )

        model.eval()
        with torch.no_grad():
            if prefetch_mode == "gpu" and onset_valid_cache_gpu is not None and tail_valid_cache_gpu is not None:
                onset_valid_t = onset_valid_cache_gpu
                tail_valid_t = tail_valid_cache_gpu
            elif prefetch_mode == "train" and onset_valid_cache is not None and tail_valid_cache is not None:
                onset_valid_np = onset_valid_cache
                tail_valid_np = tail_valid_cache
                onset_valid_t = _np_to_device(onset_valid_np).unsqueeze(1)
                tail_valid_t = _np_to_device(tail_valid_np).unsqueeze(1)
            else:
                onset_valid_np, tail_valid_np = cache_index.get_batch(keys_valid)
                onset_valid_t = _np_to_device(onset_valid_np).unsqueeze(1)
                tail_valid_t = _np_to_device(tail_valid_np).unsqueeze(1)
            out_val = model(X_valid, P_valid, onset_valid_t, tail_valid_t, C_valid)
            if use_aux:
                if isinstance(out_val, tuple) and len(out_val) == 4:
                    anchor_val, delta_val, conf_val, aux_val = out_val
                else:
                    pred_val, conf_val, aux_val = out_val
                    anchor_val = pred_val[:, anchor_indices]
                    delta_val = pred_val[:, delta_indices]
            else:
                if isinstance(out_val, tuple) and len(out_val) == 3:
                    anchor_val, delta_val, conf_val = out_val
                else:
                    pred_val, conf_val = out_val
                    anchor_val = pred_val[:, anchor_indices]
                    delta_val = pred_val[:, delta_indices]
                aux_val = None
            pred_val = _combine_predictions(torch, anchor_val, delta_val, anchor_indices, delta_indices, len(TARGET_NAMES))
            val_anchor_matrix = _loss_matrix(torch, anchor_val, Y_valid[:, anchor_indices], anchor_loss_kinds, anchor_huber_deltas)
            val_anchor_row = torch.mean(val_anchor_matrix * anchor_weights, dim=1)
            val_anchor_loss = torch.mean(val_anchor_row * W_valid)
            val_delta_matrix = _loss_matrix(torch, delta_val, Y_valid[:, delta_indices], delta_loss_kinds, delta_huber_deltas)
            val_delta_row = torch.mean(val_delta_matrix * delta_weights, dim=1)
            val_delta_loss = torch.mean(val_delta_row * W_valid)
            val_base = (val_anchor_loss * float(anchor_loss_weight)) + (val_delta_loss * float(delta_loss_weight))
            offset_v = B_valid[:, 0] + pred_val[:, 0]
            consonant_v = B_valid[:, 1] + pred_val[:, 1]
            cutoff_abs_v = B_valid[:, 2] + pred_val[:, 2]
            pre_v = B_valid[:, 3] + pred_val[:, 3]
            ovl_v = B_valid[:, 4] + pred_val[:, 4]
            val_penalty_row = (
                torch.relu(ovl_v - pre_v)
                + torch.relu((pre_v + cons_margin) - consonant_v)
                + torch.relu((consonant_v + cut_margin) - cutoff_abs_v)
                + torch.relu(-offset_v)
            )
            val_penalty = torch.mean(val_penalty_row * W_valid)
            val_align_row = _huber_loss(torch, offset_v, M_valid[:, 0], 24.0) + _huber_loss(
                torch, cutoff_abs_v, M_valid[:, 1], 24.0
            )
            val_align = torch.mean(val_align_row * W_valid)
            val_aux = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            val_boundary_consistency = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            val_boundary_row_err = None
            if use_aux and aux_val is not None and A_valid is not None and AM_valid is not None:
                val_aux_matrix = _loss_matrix(torch, aux_val, A_valid, aux_loss_kinds, aux_huber_deltas)
                val_weighted_mask = AM_valid * aux_target_weights
                val_mask_sum_v = torch.clamp(torch.sum(val_weighted_mask, dim=1), min=1.0)
                val_boundary_row_err = torch.sum(val_aux_matrix * val_weighted_mask, dim=1) / val_mask_sum_v
                val_aux = torch.mean(val_boundary_row_err * W_valid)
                val_boundary_consistency_row = _boundary_consistency_row_loss(
                    torch,
                    aux_val,
                    AM_valid,
                    offset_v,
                    cutoff_abs_v,
                )
                if val_boundary_consistency_row is not None:
                    val_boundary_consistency = torch.mean(val_boundary_consistency_row * W_valid)
                    val_boundary_row_err = val_boundary_row_err + val_boundary_consistency_row
            delta_row_err_v = torch.mean(torch.abs(pred_val - Y_valid) * target_weights, dim=1)
            conf_target_v = _confidence_target_from_errors(
                torch,
                delta_row_err_v.detach(),
                val_align_row.detach(),
                val_penalty_row.detach(),
                val_boundary_row_err.detach() if val_boundary_row_err is not None else None,
            )
            val_conf = torch.mean(
                F.binary_cross_entropy(conf_val.squeeze(1), conf_target_v, reduction="none") * W_valid
            )
            val_pair = torch.tensor(0.0, dtype=torch.float32, device=run_device)
            if epoch_pair_weight > 0.0 and val_src_pos:
                src_t = torch.tensor(val_src_pos, device=run_device, dtype=torch.long)
                dst_t = torch.tensor(val_dst_pos, device=run_device, dtype=torch.long)
                pred_offset_v = B_valid[:, 0] + pred_val[:, 0]
                true_offset_v = B_valid[:, 0] + Y_valid[:, 0]
                pred_gap_v = pred_offset_v[dst_t] - pred_offset_v[src_t]
                true_gap_v = true_offset_v[dst_t] - true_offset_v[src_t]
                pair_err_v = _huber_loss(torch, pred_gap_v, true_gap_v, 20.0)
                pair_w_v = 0.5 * (W_valid[src_t] + W_valid[dst_t])
                val_pair = torch.mean(pair_err_v * pair_w_v)
            val_total = float(
                (
                    val_base
                    + (float(penalty_loss_weight) * val_penalty)
                    + (float(align_loss_weight) * val_align)
                    + (float(conf_loss_weight) * val_conf)
                    + (float(boundary_aux_weight) * val_aux)
                    + (float(boundary_consistency_weight) * val_boundary_consistency)
                    + (epoch_pair_weight * val_pair)
                ).item()
            )
            lr_scheduler.step(val_total)

        new_best = False
        if val_total < best_val:
            best_val = val_total
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            new_best = True
        else:
            wait += 1
            if wait >= patience:
                break
        status = "best" if new_best else "wait"
        lr_now = float(optimizer.param_groups[0].get("lr", learning_rate))
        epoch_seconds = float(time.perf_counter() - epoch_started_at)
        print(
            f"[TRAIN] epoch={epoch + 1}/{epochs_n} ({int(round(((epoch + 1) / max(1, epochs_n)) * 100.0))}%) "
            f"val_loss={val_total:.4f} best={best_val:.4f} patience={wait}/{patience} status={status} "
            f"lr={lr_now:.6g} pair_w={epoch_pair_weight:.4f} sec={epoch_seconds:.1f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if prefetch_mode == "gpu" and onset_valid_cache_gpu is not None and tail_valid_cache_gpu is not None:
            onset_valid_t = onset_valid_cache_gpu
            tail_valid_t = tail_valid_cache_gpu
        elif prefetch_mode == "train" and onset_valid_cache is not None and tail_valid_cache is not None:
            onset_valid_np = onset_valid_cache
            tail_valid_np = tail_valid_cache
            onset_valid_t = _np_to_device(onset_valid_np).unsqueeze(1)
            tail_valid_t = _np_to_device(tail_valid_np).unsqueeze(1)
        else:
            onset_valid_np, tail_valid_np = cache_index.get_batch(keys_valid)
            onset_valid_t = _np_to_device(onset_valid_np).unsqueeze(1)
            tail_valid_t = _np_to_device(tail_valid_np).unsqueeze(1)
        out_valid = model(X_valid, P_valid, onset_valid_t, tail_valid_t, C_valid)
        if use_aux:
            if isinstance(out_valid, tuple) and len(out_valid) == 4:
                anchor_valid, delta_valid, conf_valid, aux_valid = out_valid
            else:
                pred_valid, conf_valid, aux_valid = out_valid
                anchor_valid = pred_valid[:, anchor_indices]
                delta_valid = pred_valid[:, delta_indices]
        else:
            if isinstance(out_valid, tuple) and len(out_valid) == 3:
                anchor_valid, delta_valid, conf_valid = out_valid
            else:
                pred_valid, conf_valid = out_valid
                anchor_valid = pred_valid[:, anchor_indices]
                delta_valid = pred_valid[:, delta_indices]
            aux_valid = None
        pred_valid = _combine_predictions(torch, anchor_valid, delta_valid, anchor_indices, delta_indices, len(TARGET_NAMES))
    pred_valid_np = pred_valid.detach().cpu().numpy()
    conf_valid_np = conf_valid.detach().cpu().numpy().reshape(-1)
    truth_valid_np = Y[valid_idx]
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    blank_attach_kpi = _compute_blank_attach_kpi(
        df_valid,
        pred_delta_offset=pred_valid_np[:, 0],
        truth_delta_offset=truth_valid_np[:, 0],
        language=str(language or "").strip().lower(),
        format_type=normalize_format_type(language, format_type) or str(format_type or "").strip().lower(),
    )
    blank_attach_kpi_pitch_zones = _compute_blank_attach_kpi_pitch_zones(
        df_valid,
        pred_delta_offset=pred_valid_np[:, 0],
        truth_delta_offset=truth_valid_np[:, 0],
        language=str(language or "").strip().lower(),
        format_type=normalize_format_type(language, format_type) or str(format_type or "").strip().lower(),
        dataset_csv=str(dataset_csv or ""),
    )

    metrics = {}
    for col_i, target in enumerate(TARGET_NAMES):
        truth = truth_valid_np[:, col_i]
        pred = pred_valid_np[:, col_i]
        metrics[target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    aux_metrics = {}
    if use_aux and aux_valid is not None and A is not None and aux_mask is not None:
        aux_pred_np = aux_valid.detach().cpu().numpy()
        aux_truth_np = A[valid_idx]
        aux_mask_np = aux_mask[valid_idx]
        for col_i, target in enumerate(AUX_TARGET_NAMES):
            mask = aux_mask_np[:, col_i] > 0.5
            if np.any(mask):
                aux_metrics[target] = {
                    "rows": int(np.sum(mask)),
                    "model_mae": float(mean_absolute_error(aux_truth_np[mask, col_i], aux_pred_np[mask, col_i])),
                }
            else:
                aux_metrics[target] = {"rows": 0, "model_mae": 0.0}

    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "categorical_features": categorical_features,
            "categorical_bucket_sizes": [int(v) for v in categorical_bucket_sizes],
            "target_names": list(TARGET_NAMES),
            "anchor_targets": list(ANCHOR_TARGET_NAMES),
            "delta_targets": list(DELTA_TARGET_NAMES),
            "patch_features": list(PATCH_FEATURES),
            "in_dim": int(X.shape[1]),
            "patch_dim": int(P.shape[1]),
            "hidden_dim": 160,
            "head_mode": str(head_mode),
            "alias_branch_mode": str(alias_branch.get("applied_mode", "shared")),
            "alias_branch_requested_mode": str(alias_branch.get("requested_mode", "shared")),
            "alias_branch_experts": int(alias_branch.get("experts", 4)),
            "alias_branch_min_rows": int(alias_branch.get("min_rows", 80)),
            "alias_type_cat_index": int(alias_branch.get("alias_type_cat_index", -1)),
            "alias_type_bucket_size": int(alias_branch.get("alias_type_bucket_size", 0)),
            "alias_fallback_ids": [int(v) for v in (alias_branch.get("fallback_ids", []) or [])],
            "anchor_dim": int(len(ANCHOR_TARGET_NAMES)),
            "delta_dim": int(len(DELTA_TARGET_NAMES)),
            "aux_dim": int(aux_dim),
            "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
            "rawmel_enabled": True,
            "mel_bins": int(mel_bins),
            "onset_frames": int(onset_frames),
            "tail_frames": int(tail_frames),
            "mel_patch_spec": patch_spec,
            "mel_patch_spec_hash": patch_hash,
        },
        os.path.join(out_dir, COUPLED_MODEL_FILE),
    )
    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))
    onnx_export = _export_coupled_onnx(
        torch,
        model,
        out_dir,
        feature_names=feature_names,
        categorical_features=categorical_features,
        categorical_bucket_sizes=[int(v) for v in categorical_bucket_sizes],
        patch_features=list(PATCH_FEATURES),
        head_mode=str(head_mode),
        alias_branch_mode=str(alias_branch.get("applied_mode", "shared")),
        alias_branch_experts=int(alias_branch.get("experts", 4)),
        alias_type_cat_index=int(alias_branch.get("alias_type_cat_index", -1)),
        alias_type_bucket_size=int(alias_branch.get("alias_type_bucket_size", 0)),
        alias_fallback_ids=[int(v) for v in (alias_branch.get("fallback_ids", []) or [])],
        anchor_targets=list(ANCHOR_TARGET_NAMES),
        delta_targets=list(DELTA_TARGET_NAMES),
        rawmel_enabled=True,
        mel_bins=int(mel_bins),
        onset_frames=int(onset_frames),
        tail_frames=int(tail_frames),
        mel_patch_spec=dict(patch_spec),
    )

    meta = {
        "backend": COUPLED_BACKEND_RAWMEL,
        "language": str(language or "").strip().lower(),
        "format_type": normalize_format_type(language, format_type) or "general",
        "model_version": "v2",
        "feature_version": schema.get("feature_version", ""),
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "phoneme_aware_conditioning": {
            "enabled": bool(categorical_features),
            "categorical_bucket_sizes": [int(v) for v in categorical_bucket_sizes],
        },
        "targets": list(TARGET_NAMES),
        "anchor_targets": list(ANCHOR_TARGET_NAMES),
        "delta_targets": list(DELTA_TARGET_NAMES),
        "head_mode": str(head_mode),
        "alias_branch": {
            "requested_mode": str(alias_branch.get("requested_mode", "shared")),
            "applied_mode": str(alias_branch.get("applied_mode", "shared")),
            "experts": int(alias_branch.get("experts", 4)),
            "min_rows": int(alias_branch.get("min_rows", 80)),
            "alias_type_cat_index": int(alias_branch.get("alias_type_cat_index", -1)),
            "alias_type_bucket_size": int(alias_branch.get("alias_type_bucket_size", 0)),
            "active_alias_ids": int(alias_branch.get("active_alias_ids", 0)),
            "strong_alias_ids": int(alias_branch.get("strong_alias_ids", 0)),
            "fallback_ids": [int(v) for v in (alias_branch.get("fallback_ids", []) or [])],
            "reason": str(alias_branch.get("reason", "ok")),
        },
        "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        "mel_patch_spec": dict(patch_spec),
        "mel_patch_spec_hash": patch_hash,
        "mel_bins": int(mel_bins),
        "onset_frames": int(onset_frames),
        "tail_frames": int(tail_frames),
        "onnx_export": onnx_export,
        "min_confidence": float(min_confidence),
        "vc_cv_pair_weight": float(pair_weight_base),
        "vc_cv_pair_warmup_epochs": int(pair_warmup_epochs),
        "optimizer": str(optimizer_name),
        "learning_rate": float(learning_rate),
        "weight_decay": float(rawmel_weight_decay),
        "target_weights": [float(v) for v in target_weight_values],
        "target_loss": {
            "kinds": list(target_loss_kinds),
            "huber_deltas": [float(v) for v in target_huber_deltas],
        },
        "boundary_aux": {
            "weight": float(boundary_aux_weight),
            "consistency_weight": float(boundary_consistency_weight),
            "target_weights": [float(v) for v in aux_target_weight_values],
            "loss_kinds": list(aux_loss_kinds),
            "huber_deltas": [float(v) for v in aux_huber_deltas],
        },
        "constraint_loss": {
            "penalty_weight": float(penalty_loss_weight),
            "align_weight": float(align_loss_weight),
            "conf_weight": float(conf_loss_weight),
            "cons_margin": float(cons_margin),
            "cut_margin": float(cut_margin),
        },
        "grad_clip": float(grad_clip),
        "lr_scheduler": {
            "type": "ReduceLROnPlateau",
            "factor": float(scheduler_factor),
            "patience": int(scheduler_patience),
            "min_lr": float(scheduler_min_lr),
        },
        "risk_weighting": {
            "blank_threshold": float(risk_blank_th),
            "mapping_conf_threshold": float(risk_map_conf_th),
            "boost_blank": float(risk_boost_blank),
            "boost_jump": float(risk_boost_jump),
            "boost_low_conf": float(risk_boost_low_conf),
        },
        "hard_example_mining": {
            "strength": float(hard_example_strength),
            "mean_boost": float(np.mean(hard_boost)) if len(hard_boost) else 1.0,
        },
        "sampler": {
            "mode": str(sampler_mode),
            "group_column": str(sampling_group_column),
        },
        "vc_cv_pair_max_gap": int(os.environ.get("UTOA_ML_VC_CV_MAX_GAP", 5) or 5),
        "vc_cv_pairs_total": int(pair_total_count),
        "vc_cv_pairs_train": int(pair_train_count),
        "vc_cv_pairs_valid": int(pair_valid_count),
        "fallback_order": [COUPLED_BACKEND_RAWMEL, COUPLED_BACKEND, "lightgbm", "base"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "holdout_metrics": metrics,
        "aux_holdout_metrics": aux_metrics,
        "blank_attach_kpi": blank_attach_kpi,
        "blank_attach_kpi_pitch_zones": blank_attach_kpi_pitch_zones,
        "holdout_confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
        "device_used": str(run_device),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "aux_metrics": aux_metrics,
                "confidence_mean": float(np.mean(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_min": float(np.min(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "confidence_max": float(np.max(conf_valid_np)) if len(conf_valid_np) else 0.0,
                "blank_attach_kpi": blank_attach_kpi,
                "blank_attach_kpi_pitch_zones": blank_attach_kpi_pitch_zones,
                "kpi_primary": {
                    "name": str(blank_attach_kpi.get("primary_kpi_name", "head_pred_attach_rate")),
                    "value": float(blank_attach_kpi.get("primary_kpi_value", 0.0)),
                },
                "vc_cv_pair_weight": float(pair_weight_base),
                "vc_cv_pairs_total": int(pair_total_count),
                "vc_cv_pairs_valid": int(pair_valid_count),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return meta


def evaluate_coupled_bundle(
    model_dir: str,
    dataset_csv: str,
    *,
    language: str = "",
    format_type: str = "",
    device: str = "auto",
    rawmel_cache_dir: str = "",
) -> Dict[str, Any]:
    _require_training_stack()
    from core.oto_ml.coupled.inference import load_coupled_bundle, predict_coupled_deltas

    payload_meta = {}
    meta_path = os.path.join(model_dir, "model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            payload_meta = json.load(f)

    bundle = load_coupled_bundle(model_dir, meta=payload_meta, schema=get_feature_schema(), device=device)
    df = _read_dataset_csv_resilient(dataset_csv)
    lang = str(language or payload_meta.get("language", "")).strip().lower()
    fmt = normalize_format_type(lang, format_type or payload_meta.get("format_type", "")) or "general"
    df = _prepare_training_frame(df, language=lang, format_type=fmt)
    if len(df) == 0:
        return {"rows": 0, "targets": {}, "confidence_mean": 0.0}

    preds = []
    confs = []
    rawmel_enabled = bool(bundle.get("rawmel_enabled", False))
    cache_index = None
    if rawmel_enabled:
        if not rawmel_cache_dir:
            raise RuntimeError("rawmel_cache_dir is required for coupled_nn_v2_rawmel evaluation")
        cache_index = MelPatchCacheIndex.load(rawmel_cache_dir)
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if rawmel_enabled and cache_index is not None:
            key = str(row_dict.get("mel_patch_key", "") or "").strip()
            if not key:
                raise RuntimeError("missing mel_patch_key in dataset")
            onset_patch, tail_patch = cache_index.get(key)
            row_dict["mel_onset_patch"] = onset_patch
            row_dict["mel_tail_patch"] = tail_patch
        d, c = predict_coupled_deltas(bundle, row_dict)
        preds.append(d)
        confs.append(float(c))

    summary = {"rows": int(len(df)), "targets": {}, "confidence_mean": float(np.mean(confs)) if confs else 0.0}
    pred_delta_offset_np = np.asarray([float(p.get("delta_offset", 0.0)) for p in preds], dtype=np.float64)
    truth_delta_offset_np = (
        pd.to_numeric(df["delta_offset"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "delta_offset" in df.columns
        else None
    )
    blank_attach_kpi = _compute_blank_attach_kpi(
        df.reset_index(drop=True),
        pred_delta_offset=pred_delta_offset_np,
        truth_delta_offset=truth_delta_offset_np,
        language=lang,
        format_type=fmt,
    )
    blank_attach_kpi_pitch_zones = _compute_blank_attach_kpi_pitch_zones(
        df.reset_index(drop=True),
        pred_delta_offset=pred_delta_offset_np,
        truth_delta_offset=truth_delta_offset_np,
        language=lang,
        format_type=fmt,
        dataset_csv=str(dataset_csv or ""),
    )
    summary["blank_attach_kpi"] = blank_attach_kpi
    summary["blank_attach_kpi_pitch_zones"] = blank_attach_kpi_pitch_zones
    summary["kpi_primary"] = {
        "name": str(blank_attach_kpi.get("primary_kpi_name", "head_pred_attach_rate")),
        "value": float(blank_attach_kpi.get("primary_kpi_value", 0.0)),
    }
    for target in TARGET_NAMES:
        truth = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        pred = np.asarray([float(p.get(target, 0.0)) for p in preds], dtype=np.float64)
        summary["targets"][target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    return summary
