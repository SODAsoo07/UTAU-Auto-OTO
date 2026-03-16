"""
Coupled mel+OTO training.

학습 루프, 데이터 전처리, 데이터셋 빌드, 평가 로직을 포함합니다.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    model = _build_model(
        torch,
        nn,
        in_dim=int(X.shape[1]),
        patch_dim=int(P.shape[1]),
        aux_dim=aux_dim,
        categorical_bucket_sizes=categorical_bucket_sizes,
        head_mode=head_mode,
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
            "anchor_dim": int(len(ANCHOR_TARGET_NAMES)),
            "delta_dim": int(len(DELTA_TARGET_NAMES)),
            "aux_dim": int(aux_dim),
            "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        },
        os.path.join(out_dir, COUPLED_MODEL_FILE),
    )
    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))

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
        "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        "mel_patch_spec": list(PATCH_FEATURES),
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
    missing_keys = [k for k in keys if not cache_index.has_key(k)]
    if missing_keys:
        raise RuntimeError(f"Raw mel cache missing keys (count={len(missing_keys)}).")

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
        "aux_targets": list(AUX_TARGET_NAMES) if use_aux else [],
        "mel_patch_spec": dict(patch_spec),
        "mel_patch_spec_hash": patch_hash,
        "mel_bins": int(mel_bins),
        "onset_frames": int(onset_frames),
        "tail_frames": int(tail_frames),
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
    for target in TARGET_NAMES:
        truth = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        pred = np.asarray([float(p.get(target, 0.0)) for p in preds], dtype=np.float64)
        summary["targets"][target] = {
            "baseline_mae": float(mean_absolute_error(truth, np.zeros_like(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    return summary
