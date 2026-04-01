from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import numpy as np
except Exception:
    np = None

from core.oto_ml_features import CATEGORICAL_FEATURES, FEATURE_NAMES, get_feature_schema

TARGET_NAMES = ["delta_offset", "delta_cons", "delta_cutoff", "delta_pre", "delta_ovl"]
DEFAULT_DELTA_CUTOFF_CLIP = 1200.0
ALIAS_GROUP_CHOICES = ("all", "cv", "vc", "vv", "other")
VC_STOP_LIKE_CODA_TYPES = {"stop", "affricate", "plosive"}
CV_STOP_LIKE_ONSET_CLASSES = {"stop", "plosive", "affricate", "sibilant"}
CV_SONORANT_ONSET_CLASSES = {"sonorant", "nasal", "liquid"}
CVC_FAMILY_FORMATS = {"cvc", "cvvc", "vccv"}
KNOWN_CUTOFF_MODES = ("negative_relative", "positive_relative", "positive_absolute", "positive_from_wav_end")
MAX_EXPM1_ARG = 20.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_alias_group(alias_type: object) -> str:
    alias = str(alias_type or "").strip().lower()
    if alias in {"cv", "cv_head", "mono"}:
        return "cv"
    if alias == "vc":
        return "vc"
    if alias in {"vv", "vcv"}:
        return "vv"
    return "other"


def _to_bool(text: object, default: bool = False) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _is_cvc_family_format(format_type: object) -> bool:
    fmt = str(format_type or "").strip().lower()
    return fmt in CVC_FAMILY_FORMATS


def _normalize_cutoff_mode(mode: object) -> str:
    m = str(mode or "").strip().lower()
    if m in KNOWN_CUTOFF_MODES:
        return m
    return ""


def _resolve_cutoff_mode_label(row: Dict[str, object]) -> str:
    mode = _normalize_cutoff_mode(row.get("manual_cutoff_mode", ""))
    if mode:
        return mode
    manual_cutoff = _safe_float(row.get("manual_cutoff", 0.0), 0.0)
    if manual_cutoff < 0.0:
        return "negative_relative"
    if manual_cutoff > 0.0:
        return "positive_relative"
    return "negative_relative"


def _cutoff_prior_key(row: Dict[str, object]) -> str:
    alias_bucket = _normalize_alias_group(row.get("alias_type", ""))
    fmt = str(row.get("format_type", "") or "").strip().lower() or "unknown"
    return f"{alias_bucket}|{fmt}"


def _hash_bucket(cat_name: str, cat_value: str, bins: int) -> Tuple[int, float]:
    n_bins = max(1, int(bins))
    token = (str(cat_name or "").strip().lower() + "=" + str(cat_value or "").strip().lower()).encode("utf-8", errors="replace")
    digest = hashlib.sha1(token).digest()
    idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % n_bins
    sign = 1.0 if (digest[4] & 1) == 0 else -1.0
    return idx, sign


def _build_design_feature_names(
    *,
    cat_hash_bins: int,
    enable_vc_stop_binary: bool,
    enable_cv_subgroup_binary: bool,
) -> List[str]:
    out: List[str] = []
    for name in FEATURE_NAMES:
        if name not in CATEGORICAL_FEATURES:
            out.append(name)
    n_bins = max(0, int(cat_hash_bins))
    if n_bins > 0:
        for cat in CATEGORICAL_FEATURES:
            out.append(f"hash__{cat}")
    if enable_vc_stop_binary:
        out.append("vc_stop_like_flag")
        out.append("vc_non_stop_flag")
    if enable_cv_subgroup_binary:
        out.append("cv_head_flag")
        out.append("cv_onset_stop_like_flag")
        out.append("cv_onset_sonorant_flag")
        out.append("cv_cvc_family_flag")
        out.append("cv_head_x_cvc_flag")
    return out


def _build_design_row(
    row: Dict[str, object],
    *,
    cat_hash_bins: int,
    enable_vc_stop_binary: bool,
    enable_cv_subgroup_binary: bool,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name in CATEGORICAL_FEATURES:
            continue
        out[name] = _safe_float(row.get(name, 0.0), 0.0)

    n_bins = max(0, int(cat_hash_bins))
    if n_bins > 0:
        for cat in CATEGORICAL_FEATURES:
            val = str(row.get(cat, "") or "").strip().lower()
            if not val:
                continue
            idx, sign = _hash_bucket(cat, val, n_bins)
            if n_bins <= 1:
                out[f"hash__{cat}"] = float(sign)
            else:
                out[f"hash__{cat}"] = float(sign) * (float(idx) / float(n_bins - 1))

    if enable_vc_stop_binary:
        alias_group = _normalize_alias_group(row.get("alias_type", ""))
        coda_type = str(row.get("coda_type", "") or "").strip().lower()
        is_stop = alias_group == "vc" and coda_type in VC_STOP_LIKE_CODA_TYPES
        out["vc_stop_like_flag"] = 1.0 if is_stop else 0.0
        out["vc_non_stop_flag"] = 1.0 if (alias_group == "vc" and not is_stop) else 0.0

    if enable_cv_subgroup_binary:
        alias_type = str(row.get("alias_type", "") or "").strip().lower()
        alias_group = _normalize_alias_group(alias_type)
        onset_class = str(row.get("onset_class", "") or "").strip().lower()
        format_type = str(row.get("format_type", "") or "").strip().lower()

        is_cv = alias_group == "cv"
        is_cv_head = is_cv and alias_type == "cv_head"
        is_cv_stop_like = is_cv and onset_class in CV_STOP_LIKE_ONSET_CLASSES
        is_cv_sonorant = is_cv and onset_class in CV_SONORANT_ONSET_CLASSES
        is_cv_cvc_family = is_cv and _is_cvc_family_format(format_type)

        out["cv_head_flag"] = 1.0 if is_cv_head else 0.0
        out["cv_onset_stop_like_flag"] = 1.0 if is_cv_stop_like else 0.0
        out["cv_onset_sonorant_flag"] = 1.0 if is_cv_sonorant else 0.0
        out["cv_cvc_family_flag"] = 1.0 if is_cv_cvc_family else 0.0
        out["cv_head_x_cvc_flag"] = 1.0 if (is_cv_head and is_cv_cvc_family) else 0.0
    return out


def _load_rows(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _build_matrix(rows: List[Dict[str, object]], feature_names: List[str]) -> Tuple[object, List[float]]:
    X_rows: List[List[float]] = []
    w_rows: List[float] = []
    for row in rows:
        feats = [_safe_float(row.get(name, 0.0), 0.0) for name in feature_names]
        X_rows.append(feats)
        w_rows.append(max(0.05, _safe_float(row.get("sample_weight", 1.0), 1.0)))
    if np is None:
        return X_rows, w_rows
    return np.asarray(X_rows, dtype=np.float32), w_rows


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(float(x) for x in values)
    qq = max(0.0, min(1.0, float(q)))
    idx = int((len(v) - 1) * qq)
    return float(v[idx])


def _signed_log1p(value: float) -> float:
    v = float(value)
    if v >= 0.0:
        return float(math.log1p(v))
    return float(-math.log1p(abs(v)))


def _signed_expm1(value: float) -> float:
    v = max(-MAX_EXPM1_ARG, min(MAX_EXPM1_ARG, float(value)))
    if v >= 0.0:
        return float(math.expm1(v))
    return float(-math.expm1(abs(v)))


def _clip_abs(value: float, clip_abs: float) -> float:
    lim = max(0.0, float(clip_abs))
    return max(-lim, min(lim, float(value)))


def _build_target_transform_spec(
    *,
    delta_cutoff_transform: str,
    delta_cutoff_clip: float,
) -> Dict[str, Dict[str, object]]:
    mode = str(delta_cutoff_transform or "none").strip().lower()
    if mode != "log_clip":
        return {}
    return {
        "delta_cutoff": {
            "type": "signed_log1p_clip",
            "clip_abs": max(1.0, float(delta_cutoff_clip)),
        }
    }


def _encode_target(target: str, value: float, transforms: Dict[str, Dict[str, object]]) -> float:
    spec = dict((transforms or {}).get(target) or {})
    t_type = str(spec.get("type", "") or "").strip().lower()
    if t_type == "signed_log1p_clip":
        clip_abs = max(1.0, _safe_float(spec.get("clip_abs", DEFAULT_DELTA_CUTOFF_CLIP), DEFAULT_DELTA_CUTOFF_CLIP))
        return _signed_log1p(_clip_abs(float(value), clip_abs))
    return float(value)


def _decode_target(target: str, value: float, transforms: Dict[str, Dict[str, object]]) -> float:
    spec = dict((transforms or {}).get(target) or {})
    t_type = str(spec.get("type", "") or "").strip().lower()
    if t_type == "signed_log1p_clip":
        clip_abs = max(1.0, _safe_float(spec.get("clip_abs", DEFAULT_DELTA_CUTOFF_CLIP), DEFAULT_DELTA_CUTOFF_CLIP))
        return _clip_abs(_signed_expm1(float(value)), clip_abs)
    return float(value)


def _build_target_weights(
    rows: List[Dict[str, object]],
    base_weights: List[float],
    target: str,
    cutoff_loss_weight: float,
    vv_cutoff_loss_weight: float,
    vv_positive_cutoff_loss_weight: float,
) -> List[float]:
    out = [max(1e-6, float(w)) for w in base_weights]
    if target != "delta_cutoff":
        return out
    boost = max(1.0, float(cutoff_loss_weight))
    vv_boost = max(1.0, float(vv_cutoff_loss_weight))
    vv_pos_boost = max(1.0, float(vv_positive_cutoff_loss_weight))

    mags = [abs(_safe_float(r.get("delta_cutoff", 0.0), 0.0)) for r in rows]
    scale = max(1e-6, _percentile(mags, 0.90))
    for i, mag in enumerate(mags):
        factor = 1.0
        if boost > 1.0:
            norm = max(0.0, min(1.0, float(mag) / scale))
            factor *= 1.0 + (boost - 1.0) * norm

        if _normalize_alias_group(rows[i].get("alias_type", "")) == "vv":
            factor *= vv_boost
            if _resolve_cutoff_mode_label(rows[i]) != "negative_relative":
                factor *= vv_pos_boost

        out[i] = max(1e-6, out[i] * factor)
    return out


def _build_cutoff_mode_adjustment(
    rows: List[Dict[str, object]],
    preds: List[Dict[str, float]],
    *,
    min_key_count: int,
    positive_gate_threshold: float,
    positive_only: bool,
) -> Dict[str, object]:
    if not rows or not preds:
        return {"enabled": False}

    mode_sum: Dict[str, float] = {m: 0.0 for m in KNOWN_CUTOFF_MODES}
    mode_count: Dict[str, int] = {m: 0 for m in KNOWN_CUTOFF_MODES}
    key_mode_count: Dict[str, Dict[str, int]] = {}

    total = min(len(rows), len(preds))
    for i in range(total):
        row = rows[i]
        pred = preds[i]
        mode = _resolve_cutoff_mode_label(row)
        if mode not in KNOWN_CUTOFF_MODES:
            continue
        true_d = _safe_float(row.get("delta_cutoff", 0.0), 0.0)
        pred_d = _safe_float(pred.get("delta_cutoff", 0.0), 0.0)
        residual = true_d - pred_d
        mode_sum[mode] += float(residual)
        mode_count[mode] = int(mode_count[mode]) + 1

        key = _cutoff_prior_key(row)
        bucket = key_mode_count.setdefault(key, {m: 0 for m in KNOWN_CUTOFF_MODES})
        bucket[mode] = int(bucket.get(mode, 0)) + 1

    total_modes = sum(mode_count.values())
    if total_modes <= 0:
        return {"enabled": False}

    mode_bias: Dict[str, float] = {}
    for mode in KNOWN_CUTOFF_MODES:
        n = int(mode_count.get(mode, 0))
        if n <= 0:
            continue
        raw_bias = float(mode_sum.get(mode, 0.0) / float(n))
        shrink = float(n) / float(n + max(8, int(min_key_count)))
        mode_bias[mode] = max(-DEFAULT_DELTA_CUTOFF_CLIP, min(DEFAULT_DELTA_CUTOFF_CLIP, raw_bias * shrink))

    global_priors = {m: float(mode_count.get(m, 0)) / float(total_modes) for m in KNOWN_CUTOFF_MODES}
    priors_by_alias_format: Dict[str, Dict[str, float]] = {}
    min_n = max(1, int(min_key_count))
    for key, counts in key_mode_count.items():
        key_total = int(sum(int(v) for v in counts.values()))
        if key_total < min_n:
            continue
        priors_by_alias_format[key] = {
            m: float(int(counts.get(m, 0)) / float(key_total))
            for m in KNOWN_CUTOFF_MODES
        }

    return {
        "enabled": True,
        "version": "v1",
        "mode_bias": mode_bias,
        "global_priors": global_priors,
        "priors_by_alias_format": priors_by_alias_format,
        "mode_counts": mode_count,
        "min_key_count": int(min_n),
        "positive_gate_threshold": max(0.0, min(1.0, float(positive_gate_threshold))),
        "positive_only": bool(positive_only),
    }


def _fit_weighted_linear(X, y: List[float], w: List[float], ridge: float = 1e-3) -> Dict[str, object]:
    y_arr = [float(v) for v in y]
    if len(y_arr) == 0:
        return {"bias": 0.0, "weights": {}}

    # numpy path
    if np is not None and hasattr(X, "shape"):
        y_np = np.asarray(y_arr, dtype=np.float32)
        w_np = np.asarray(w, dtype=np.float32)
        X_aug = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
        sw = np.sqrt(np.maximum(w_np, 1e-6))[:, None]
        Xw = X_aug * sw
        yw = y_np * sw[:, 0]
        eye = np.eye(Xw.shape[1], dtype=np.float32)
        eye[0, 0] = 0.0
        try:
            beta = np.linalg.solve(Xw.T @ Xw + ridge * eye, Xw.T @ yw)
        except Exception:
            beta = np.zeros((Xw.shape[1],), dtype=np.float32)
            beta[0] = float(np.average(y_np, weights=w_np)) if float(np.sum(w_np)) > 0.0 else float(np.mean(y_np))
        bias = float(beta[0])
        weights = beta[1:].tolist()
        return {"bias": bias, "weights": weights}

    # fallback: bias-only
    s_w = sum(float(v) for v in w)
    if s_w <= 0:
        return {"bias": float(sum(y_arr) / max(1, len(y_arr))), "weights": [0.0 for _ in range(len(X[0]) if X else 0)]}
    bias = sum(float(vy) * float(vw) for vy, vw in zip(y_arr, w)) / s_w
    return {"bias": float(bias), "weights": [0.0 for _ in range(len(X[0]) if X else 0)]}


def _predict_linear(model: Dict[str, object], feature_vec: List[float]) -> float:
    bias = _safe_float(model.get("bias", 0.0), 0.0)
    weights = model.get("weights") or []
    out = bias
    for i, fv in enumerate(feature_vec):
        if i < len(weights):
            out += _safe_float(weights[i], 0.0) * float(fv)
    return float(out)


def _mae(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(abs(v) for v in values) / float(len(values)))


def _constraint_violation_rate(rows: List[Dict[str, object]], preds: List[Dict[str, float]]) -> float:
    if not rows or not preds:
        return 0.0
    bad = 0
    total = min(len(rows), len(preds))
    for i in range(total):
        row = rows[i]
        pred = preds[i]
        off = _safe_float(row.get("base_offset", 0.0), 0.0) + pred.get("delta_offset", 0.0)
        cons = _safe_float(row.get("base_cons", 0.0), 0.0) + pred.get("delta_cons", 0.0)
        pre = _safe_float(row.get("base_pre", 0.0), 0.0) + pred.get("delta_pre", 0.0)
        cut_abs = _safe_float(row.get("base_cutoff_abs", 0.0), 0.0) + pred.get("delta_cutoff", 0.0)
        cutoff = -(float(cut_abs) - float(off))
        if cons < 0.0 or pre < 0.0 or pre > cons or cutoff >= 0.0:
            bad += 1
    return float(bad / max(1, total))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train simple E2E linear model (Phase A scaffold).")
    parser.add_argument("--dataset", required=True, help="Dataset CSV built by build_e2e_dataset.py")
    parser.add_argument("--out-dir", required=True, help="Model output directory")
    parser.add_argument("--language", required=True, choices=["korean", "japanese"])
    parser.add_argument("--format-type", default="cv")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--cutoff-loss-weight",
        type=float,
        default=1.0,
        help="Extra emphasis for delta_cutoff target (>=1.0).",
    )
    parser.add_argument(
        "--delta-cutoff-transform",
        choices=["none", "log_clip"],
        default="none",
        help="Optional target transform for delta_cutoff.",
    )
    parser.add_argument(
        "--delta-cutoff-clip",
        type=float,
        default=DEFAULT_DELTA_CUTOFF_CLIP,
        help="Absolute clip limit used with --delta-cutoff-transform=log_clip.",
    )
    parser.add_argument("--cv-only", default="1", help="1/0. Keep cv/cv_head rows only")
    parser.add_argument(
        "--alias-group",
        choices=list(ALIAS_GROUP_CHOICES),
        default="all",
        help="Optional alias-group subset. Overrides cv-only when not 'all'.",
    )
    parser.add_argument(
        "--cat-hash-bins",
        type=int,
        default=16,
        help="Hashed one-hot bins per categorical feature (0 disables categorical encoding).",
    )
    parser.add_argument(
        "--vc-stop-binary",
        default="1",
        help="1/0. Add vc_stop_like_flag and vc_non_stop_flag derived features.",
    )
    parser.add_argument(
        "--cv-subgroup-binary",
        default="1",
        help="1/0. Add cv subgroup flags (head/onset/cvc-family interactions).",
    )
    parser.add_argument(
        "--vv-cutoff-loss-weight",
        type=float,
        default=1.0,
        help="Extra multiplier for vv rows on delta_cutoff loss (>=1.0).",
    )
    parser.add_argument(
        "--vv-positive-cutoff-loss-weight",
        type=float,
        default=1.0,
        help="Extra multiplier for vv rows where cutoff mode is positive_* (>=1.0).",
    )
    parser.add_argument(
        "--cutoff-mode-adjust",
        default="1",
        help="1/0. Enable mode-aware delta_cutoff post correction metadata.",
    )
    parser.add_argument(
        "--cutoff-mode-min-key-count",
        type=int,
        default=32,
        help="Minimum rows for alias|format prior key used in cutoff mode adjustment.",
    )
    parser.add_argument(
        "--cutoff-mode-positive-gate",
        type=float,
        default=0.35,
        help="Enable cutoff mode adjustment only when positive-mode prior is above this threshold.",
    )
    parser.add_argument(
        "--cutoff-mode-positive-only",
        default="1",
        help="1/0. Use positive-mode-only expectation for cutoff mode adjustment.",
    )
    args = parser.parse_args()

    rows = _load_rows(os.path.abspath(args.dataset))
    alias_group = str(args.alias_group or "all").strip().lower()
    if alias_group != "all":
        rows = [r for r in rows if _normalize_alias_group(r.get("alias_type", "")) == alias_group]
    elif str(args.cv_only).strip() not in {"0", "false", "False"}:
        rows = [r for r in rows if str(r.get("alias_type", "") or "").strip().lower() in {"cv", "cv_head"}]

    if not rows:
        raise RuntimeError("No rows available for training.")

    cat_hash_bins = max(0, int(args.cat_hash_bins))
    enable_vc_stop_binary = _to_bool(args.vc_stop_binary, True)
    enable_cv_subgroup_binary = _to_bool(args.cv_subgroup_binary, True)
    feature_names = _build_design_feature_names(
        cat_hash_bins=cat_hash_bins,
        enable_vc_stop_binary=enable_vc_stop_binary,
        enable_cv_subgroup_binary=enable_cv_subgroup_binary,
    )
    design_rows = [
        _build_design_row(
            r,
            cat_hash_bins=cat_hash_bins,
            enable_vc_stop_binary=enable_vc_stop_binary,
            enable_cv_subgroup_binary=enable_cv_subgroup_binary,
        )
        for r in rows
    ]
    X, weights = _build_matrix(design_rows, feature_names)
    target_transforms = _build_target_transform_spec(
        delta_cutoff_transform=str(args.delta_cutoff_transform or "none"),
        delta_cutoff_clip=float(args.delta_cutoff_clip),
    )

    target_models: Dict[str, Dict[str, object]] = {}
    pred_rows: List[Dict[str, float]] = []
    err_acc: Dict[str, List[float]] = {k: [] for k in TARGET_NAMES}

    # fit one linear head per target
    for target in TARGET_NAMES:
        y_raw = [_safe_float(r.get(target, 0.0), 0.0) for r in rows]
        y = [_encode_target(target, v, target_transforms) for v in y_raw]
        target_weights = _build_target_weights(
            rows,
            weights,
            target,
            cutoff_loss_weight=float(args.cutoff_loss_weight),
            vv_cutoff_loss_weight=float(args.vv_cutoff_loss_weight),
            vv_positive_cutoff_loss_weight=float(args.vv_positive_cutoff_loss_weight),
        )
        fitted = _fit_weighted_linear(X, y, target_weights, ridge=float(args.ridge))
        target_models[target] = {
            "bias": float(fitted.get("bias", 0.0)),
            "weights": {name: float(fitted.get("weights", [0.0] * len(feature_names))[i]) for i, name in enumerate(feature_names)},
        }

    # evaluate train mae quickly
    for row, design_row in zip(rows, design_rows):
        vec = [_safe_float(design_row.get(name, 0.0), 0.0) for name in feature_names]
        pred_item: Dict[str, float] = {}
        for target in TARGET_NAMES:
            model = target_models[target]
            # compact prediction using dict weights
            val = float(model.get("bias", 0.0))
            weights_dict = model.get("weights") or {}
            for i, name in enumerate(feature_names):
                val += _safe_float(weights_dict.get(name, 0.0), 0.0) * vec[i]
            pred_decoded = _decode_target(target, float(val), target_transforms)
            pred_item[target] = float(pred_decoded)
            err_acc[target].append(float(pred_decoded) - _safe_float(row.get(target, 0.0), 0.0))
        pred_rows.append(pred_item)

    enable_cutoff_mode_adjust = _to_bool(args.cutoff_mode_adjust, True)
    cutoff_mode_adjustment = {"enabled": False}
    if enable_cutoff_mode_adjust:
        cutoff_mode_positive_only = _to_bool(args.cutoff_mode_positive_only, True)
        cutoff_mode_adjustment = _build_cutoff_mode_adjustment(
            rows,
            pred_rows,
            min_key_count=max(1, int(args.cutoff_mode_min_key_count)),
            positive_gate_threshold=float(args.cutoff_mode_positive_gate),
            positive_only=cutoff_mode_positive_only,
        )
    else:
        cutoff_mode_positive_only = _to_bool(args.cutoff_mode_positive_only, True)

    # confidence head: calibrated from training quality/mapping confidence
    confidence_head = {
        "bias": 0.20,
        "weights": {
            "mapping_confidence": 0.60,
            "train_quality_score": 0.30,
            "blank_span_confidence": -0.25,
        },
    }

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_payload = {
        "targets": target_models,
        "confidence": confidence_head,
        "target_transforms": target_transforms,
        "cutoff_mode_adjustment": cutoff_mode_adjustment,
    }
    with open(os.path.join(out_dir, "e2e_linear.json"), "w", encoding="utf-8") as f:
        json.dump(model_payload, f, ensure_ascii=False, indent=2)

    schema_path = os.path.join(out_dir, "feature_schema.json")
    schema = dict(get_feature_schema())
    schema["feature_names"] = list(feature_names)
    schema["base_feature_names"] = list(FEATURE_NAMES)
    schema["categorical_hash_bins"] = int(cat_hash_bins)
    schema["categorical_hash_features"] = list(CATEGORICAL_FEATURES)
    schema["categorical_encoding"] = "hash_signed_onehot_v1"
    schema["enable_vc_stop_binary"] = bool(enable_vc_stop_binary)
    schema["enable_cv_subgroup_binary"] = bool(enable_cv_subgroup_binary)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    metrics = {f"valid_mae_{k}": _mae(err_acc[k]) for k in TARGET_NAMES}
    metrics["constraint_violation_rate"] = _constraint_violation_rate(rows, pred_rows)

    model_meta = {
        "backend": "e2e_linear_v1",
        "created_at": dt.datetime.now().isoformat(),
        "language": str(args.language),
        "format_type": str(args.format_type or "cv"),
        "output_mode": "delta",
        "feature_version": "v12",
        "cutoff_loss_weight": max(1.0, float(args.cutoff_loss_weight)),
        "delta_cutoff_transform": str(args.delta_cutoff_transform or "none"),
        "delta_cutoff_clip": max(1.0, float(args.delta_cutoff_clip)),
        "vv_cutoff_loss_weight": max(1.0, float(args.vv_cutoff_loss_weight)),
        "vv_positive_cutoff_loss_weight": max(1.0, float(args.vv_positive_cutoff_loss_weight)),
        "cutoff_mode_adjust": bool(enable_cutoff_mode_adjust),
        "cutoff_mode_min_key_count": max(1, int(args.cutoff_mode_min_key_count)),
        "cutoff_mode_positive_gate": max(0.0, min(1.0, float(args.cutoff_mode_positive_gate))),
        "cutoff_mode_positive_only": bool(cutoff_mode_positive_only),
        "alias_group": alias_group,
        "cat_hash_bins": int(cat_hash_bins),
        "vc_stop_binary": bool(enable_vc_stop_binary),
        "cv_subgroup_binary": bool(enable_cv_subgroup_binary),
        **metrics,
        "rows": int(len(rows)),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)

    print(f"[E2E-Train] rows={len(rows)} out_dir={out_dir}")
    print(f"[E2E-Train] mae={json.dumps(metrics, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
