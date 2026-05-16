from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

try:
    import lightgbm as lgb
except Exception as _lgb_exc:  # pragma: no cover
    lgb = None
    LIGHTGBM_IMPORT_ERROR = _lgb_exc
else:
    LIGHTGBM_IMPORT_ERROR = None

try:
    import pandas as pd
except Exception as _pd_exc:  # pragma: no cover
    pd = None
    PANDAS_IMPORT_ERROR = _pd_exc
else:
    PANDAS_IMPORT_ERROR = None

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params, oto_row_to_absolute_anchors
from core.coarse_crnn.oto_param_builder import (
    apply_pre_cons_gap_guard,
    apply_right_boundary_guard,
    apply_span_length_guard,
    stretch_risk_flags,
)


RESIDUAL_FEATURE_NAMES: tuple[str, ...] = (
    "language",
    "format_type",
    "role",
    "alias_type",
    "transition_type",
    "slot_index",
    "slot_count",
    "line_index",
    "duration_ms",
    "selected_time_ms",
    "selected_rel",
    "left_anchor_ms",
    "right_anchor_ms",
    "anchor_span_ms",
    "center_from_left_ms",
    "center_to_right_ms",
    "confidence",
    "fallback_used",
    "offset_pred",
    "preutterance_pred",
    "overlap_pred",
    "consonant_pred",
    "cutoff_pred",
    "pre_abs_pred",
    "consonant_abs_pred",
    "cutoff_abs_pred",
)

RESIDUAL_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "language",
    "format_type",
    "role",
    "alias_type",
    "transition_type",
)

RESIDUAL_TARGET_NAMES: tuple[str, ...] = (
    "delta_offset",
    "delta_preutterance",
    "delta_overlap",
    "delta_cutoff",
)

DEFAULT_DELTA_CLIPS: dict[str, tuple[float, float]] = {
    "delta_offset": (-60.0, 60.0),
    "delta_preutterance": (-120.0, 120.0),
    "delta_overlap": (-120.0, 120.0),
    "delta_cutoff": (-220.0, 220.0),
}


@dataclass(frozen=True)
class BoundaryResidualBundle:
    model_dir: str
    boosters: dict[str, Any]
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    delta_clips: dict[str, tuple[float, float]]
    active_targets: tuple[str, ...]
    stage1_role_bias: dict[str, dict[str, float]]
    meta: dict[str, Any]


def resolve_boundary_residual_model_dir(*, boundary_model_path: str) -> str:
    env_path = str(os.environ.get("UTOA_BOUNDARY_RESIDUAL_MODEL_DIR", "") or "").strip()
    if env_path:
        return os.path.abspath(env_path)
    base = os.path.dirname(os.path.abspath(str(boundary_model_path or "")))
    if not base:
        return ""
    return os.path.join(base, "boundary_residual_v1")


def residual_enabled_by_env(default: bool = True) -> bool:
    raw = str(os.environ.get("UTOA_BOUNDARY_RESIDUAL_ENABLE", "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def load_boundary_residual_bundle(model_dir: str) -> BoundaryResidualBundle:
    if lgb is None:
        raise RuntimeError(f"lightgbm is required: {LIGHTGBM_IMPORT_ERROR}")
    if pd is None:
        raise RuntimeError(f"pandas is required: {PANDAS_IMPORT_ERROR}")
    root = os.path.abspath(str(model_dir or ""))
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(root or model_dir)
    meta = _read_json(os.path.join(root, "model_meta.json")) or {}
    schema = _read_json(os.path.join(root, "feature_schema.json")) or {}
    feature_names = tuple(schema.get("feature_names") or RESIDUAL_FEATURE_NAMES)
    categorical_features = tuple(schema.get("categorical_features") or RESIDUAL_CATEGORICAL_FEATURES)
    delta_clips = _resolve_delta_clips(meta.get("delta_clips"))
    stage1_role_bias = _read_stage1_role_bias(os.path.join(root, "stage1_role_bias.json"))
    loaded_targets: list[str] = []
    boosters: dict[str, Any] = {}
    for target in RESIDUAL_TARGET_NAMES:
        model_path = os.path.join(root, f"model_{target}.txt")
        if not os.path.isfile(model_path):
            continue
        boosters[target] = lgb.Booster(model_file=model_path)
        loaded_targets.append(target)
    if not boosters and not stage1_role_bias:
        raise FileNotFoundError(f"no residual models found in {root}")
    active_targets = _resolve_active_targets(meta, available_targets=tuple(loaded_targets))
    return BoundaryResidualBundle(
        model_dir=root,
        boosters=boosters,
        feature_names=feature_names,
        categorical_features=categorical_features,
        delta_clips=delta_clips,
        active_targets=active_targets,
        stage1_role_bias=stage1_role_bias,
        meta=meta,
    )


def build_residual_feature_row(
    *,
    row,
    anchor_timeline_ms: list[float],
    params_pred: dict[str, float] | None = None,
) -> dict[str, Any]:
    params = dict(params_pred or absolute_anchors_to_oto_params(row.anchors, duration_ms=float(row.spec.duration_ms)))
    slot_idx = max(0, int(row.spec.slot_index))
    timeline = list(anchor_timeline_ms or [])
    if timeline:
        left_anchor = float(timeline[min(slot_idx, len(timeline) - 1)])
    else:
        left_anchor = 0.0
    if timeline and (slot_idx + 1) < len(timeline):
        right_anchor = float(timeline[slot_idx + 1])
    else:
        right_anchor = float(row.spec.duration_ms)
    duration = max(1.0, float(row.spec.duration_ms))
    span = max(1.0, right_anchor - left_anchor)
    selected = float(row.selected_time_ms)
    alias_type = str((row.spec.meta or {}).get("alias_type", "") or "other").strip().lower() or "other"
    transition_type = str((row.spec.meta or {}).get("transition_type", "") or "other").strip().lower() or "other"
    role = normalize_role(row.spec.role)
    feature = {
        "language": str(row.spec.language or ""),
        "format_type": str(row.spec.format_type or ""),
        "role": str(role or "other"),
        "alias_type": alias_type,
        "transition_type": transition_type,
        "slot_index": float(max(0, int(row.spec.slot_index))),
        "slot_count": float(max(1, int(row.spec.slot_count))),
        "line_index": float(max(0, int(row.spec.line_index))),
        "duration_ms": float(duration),
        "selected_time_ms": float(selected),
        "selected_rel": float(selected / duration),
        "left_anchor_ms": float(left_anchor),
        "right_anchor_ms": float(right_anchor),
        "anchor_span_ms": float(span),
        "center_from_left_ms": float(selected - left_anchor),
        "center_to_right_ms": float(right_anchor - selected),
        "confidence": float(row.anchors.confidence),
        "fallback_used": 1.0 if bool(row.fallback_used) else 0.0,
        "offset_pred": float(params.get("offset", 0.0) or 0.0),
        "preutterance_pred": float(params.get("preutterance", 0.0) or 0.0),
        "overlap_pred": float(params.get("overlap", 0.0) or 0.0),
        "consonant_pred": float(params.get("consonant", 0.0) or 0.0),
        "cutoff_pred": float(params.get("cutoff", 0.0) or 0.0),
        "pre_abs_pred": float(row.anchors.pre_abs),
        "consonant_abs_pred": float(row.anchors.consonant_abs),
        "cutoff_abs_pred": float(row.anchors.cutoff_abs),
    }
    return feature


def predict_boundary_residual_deltas(bundle: BoundaryResidualBundle, feature_row: dict[str, Any]) -> dict[str, float]:
    if pd is None:
        raise RuntimeError(f"pandas is required: {PANDAS_IMPORT_ERROR}")
    frame = _prepare_frame(feature_row, feature_names=bundle.feature_names, categorical_features=bundle.categorical_features)
    out: dict[str, float] = {}
    active = set(bundle.active_targets)
    for target in RESIDUAL_TARGET_NAMES:
        booster = bundle.boosters.get(target)
        if booster is None:
            out[target] = 0.0
            continue
        if target not in active:
            out[target] = 0.0
            continue
        pred = booster.predict(frame)
        val = float(pred[0]) if len(pred) else 0.0
        lo, hi = bundle.delta_clips.get(target, (-9999.0, 9999.0))
        out[target] = max(float(lo), min(float(hi), val))
    return out


def apply_boundary_residual_deltas(
    *,
    params: dict[str, float],
    deltas: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    base = {
        "offset": float(params.get("offset", 0.0) or 0.0),
        "consonant": float(params.get("consonant", 0.0) or 0.0),
        "cutoff": float(params.get("cutoff", 0.0) or 0.0),
        "preutterance": float(params.get("preutterance", 0.0) or 0.0),
        "overlap": float(params.get("overlap", 0.0) or 0.0),
    }
    offset = max(0.0, float(base["offset"]) + float(deltas.get("delta_offset", 0.0)))
    pre = max(0.0, float(base["preutterance"]) + float(deltas.get("delta_preutterance", 0.0)))
    ovl = max(0.0, float(base["overlap"]) + float(deltas.get("delta_overlap", 0.0)))
    if ovl > pre:
        ovl = pre
    cons = max(pre + 1.0, float(base["consonant"]))
    cutoff = float(base["cutoff"]) + float(deltas.get("delta_cutoff", 0.0))
    out = {
        "offset": float(offset),
        "consonant": float(cons),
        "cutoff": float(cutoff),
        "preutterance": float(pre),
        "overlap": float(ovl),
    }
    changed = any(abs(float(out[k]) - float(base[k])) > 1e-4 for k in ("offset", "consonant", "cutoff", "preutterance", "overlap"))
    return out, {"changed": bool(changed)}


def apply_residual_to_decoded_row(
    *,
    row,
    anchor_timeline_ms: list[float],
    params_pred: dict[str, float],
    bundle: BoundaryResidualBundle,
) -> tuple[dict[str, float], dict[str, Any]]:
    feature_row = build_residual_feature_row(
        row=row,
        anchor_timeline_ms=anchor_timeline_ms,
        params_pred=params_pred,
    )
    stage1_deltas = predict_boundary_stage1_role_bias(bundle, feature_row)
    stage1_params, _stage1_audit = apply_boundary_residual_deltas(
        params=params_pred,
        deltas=stage1_deltas,
    )
    feature_row_stage2 = build_residual_feature_row(
        row=row,
        anchor_timeline_ms=anchor_timeline_ms,
        params_pred=stage1_params,
    )
    deltas = predict_boundary_residual_deltas(bundle, feature_row_stage2)
    corrected, audit = apply_boundary_residual_deltas(
        params=stage1_params,
        deltas=deltas,
    )
    slot_idx = max(0, int(row.spec.slot_index))
    has_right_anchor = (slot_idx + 1) < len(anchor_timeline_ms or [])
    right_anchor = float(anchor_timeline_ms[slot_idx + 1]) if has_right_anchor else None
    anchors = oto_row_to_absolute_anchors(
        offset=float(corrected.get("offset", 0.0) or 0.0),
        consonant=float(corrected.get("consonant", 0.0) or 0.0),
        cutoff=float(corrected.get("cutoff", 0.0) or 0.0),
        preutterance=float(corrected.get("preutterance", 0.0) or 0.0),
        overlap=float(corrected.get("overlap", 0.0) or 0.0),
        duration_ms=float(row.spec.duration_ms),
    )
    anchors = apply_right_boundary_guard(
        anchors=anchors,
        role=normalize_role(row.spec.role),
        duration_ms=float(row.spec.duration_ms),
        right_anchor_ms=right_anchor,
    )
    anchors = apply_span_length_guard(
        anchors=anchors,
        role=normalize_role(row.spec.role),
        duration_ms=float(row.spec.duration_ms),
    )
    anchors = apply_pre_cons_gap_guard(
        anchors=anchors,
        role=normalize_role(row.spec.role),
        duration_ms=float(row.spec.duration_ms),
        active_start_ms=0.0,
    )
    final_params = absolute_anchors_to_oto_params(
        anchors,
        duration_ms=float(row.spec.duration_ms),
    )
    audit["delta_preutterance"] = float(deltas.get("delta_preutterance", 0.0))
    audit["delta_overlap"] = float(deltas.get("delta_overlap", 0.0))
    audit["delta_cutoff"] = float(deltas.get("delta_cutoff", 0.0))
    audit["delta_offset"] = float(deltas.get("delta_offset", 0.0))
    audit["stage1_delta_offset"] = float(stage1_deltas.get("delta_offset", 0.0))
    audit["stage1_delta_preutterance"] = float(stage1_deltas.get("delta_preutterance", 0.0))
    audit["stage1_delta_overlap"] = float(stage1_deltas.get("delta_overlap", 0.0))
    audit["stage1_delta_cutoff"] = float(stage1_deltas.get("delta_cutoff", 0.0))
    audit["has_right_anchor"] = bool(has_right_anchor)
    return final_params, audit


def should_apply_residual_for_row(
    *,
    row,
    params_pred: dict[str, float],
    quality_score: float | None = None,
    default_conditional: bool = True,
) -> tuple[bool, str]:
    conditional_on = _env_bool("UTOA_BOUNDARY_RESIDUAL_CONDITIONAL_ENABLE", default_conditional)
    if not conditional_on:
        return True, "conditional_disabled"
    risk_score = 0
    reasons: list[str] = []
    role = normalize_role(getattr(getattr(row, "spec", None), "role", "other"))
    risky_roles = {
        token.strip()
        for token in str(os.environ.get("UTOA_BOUNDARY_RESIDUAL_RISKY_ROLES", "vc,vv,v-cv") or "").split(",")
        if token.strip()
    }
    if role in risky_roles:
        risk_score += 1
        reasons.append(f"role={role}")
    if bool(getattr(row, "fallback_used", False)):
        risk_score += 2
        reasons.append("fallback=1")

    confidence = float(getattr(getattr(row, "anchors", None), "confidence", 0.0) or 0.0)
    conf_max = _env_float("UTOA_BOUNDARY_RESIDUAL_MAX_CONFIDENCE", 0.58)
    if confidence <= conf_max:
        risk_score += 1
        reasons.append(f"conf<={conf_max:.2f}")

    q_val = float(
        quality_score
        if quality_score is not None
        else getattr(row, "quality_score", 0.50)
    )
    quality_max = _env_float("UTOA_BOUNDARY_RESIDUAL_MAX_QUALITY", 0.54)
    if q_val <= quality_max:
        risk_score += 1
        reasons.append(f"quality<={quality_max:.2f}")

    try:
        stretch = stretch_risk_flags(
            params_pred,
            duration_ms=float(getattr(getattr(row, "spec", None), "duration_ms", 0.0) or 0.0),
        )
    except Exception:
        stretch = {}
    if any(bool(v) for v in stretch.values()):
        risk_score += 1
        reasons.append("stretch_risk=1")

    threshold = max(1, int(round(_env_float("UTOA_BOUNDARY_RESIDUAL_RISK_SCORE_MIN", 2.0))))
    apply_now = risk_score >= threshold
    detail = f"risk={risk_score}/{threshold} " + ",".join(reasons or ["none"])
    return bool(apply_now), detail


def predict_boundary_stage1_role_bias(bundle: BoundaryResidualBundle, feature_row: dict[str, Any]) -> dict[str, float]:
    role = normalize_role(feature_row.get("role", "other"))
    out: dict[str, float] = {}
    for target in RESIDUAL_TARGET_NAMES:
        target_map = bundle.stage1_role_bias.get(target, {})
        out[target] = float(target_map.get(role, target_map.get("default", 0.0)))
    return out


def _prepare_frame(feature_row: dict[str, Any], *, feature_names: tuple[str, ...], categorical_features: tuple[str, ...]):
    frame = pd.DataFrame([feature_row], copy=True)
    for name in feature_names:
        if name not in frame.columns:
            frame[name] = "" if name in categorical_features else 0.0
    frame = frame[list(feature_names)].copy()
    for name in feature_names:
        if name in categorical_features:
            frame[name] = frame[name].fillna("").astype("string").astype("category")
        else:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    return frame


def _read_json(path: str) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_delta_clips(payload: Any) -> dict[str, tuple[float, float]]:
    out = dict(DEFAULT_DELTA_CLIPS)
    src = payload if isinstance(payload, dict) else {}
    for key in RESIDUAL_TARGET_NAMES:
        item = src.get(key)
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lo = float(item[0])
            hi = float(item[1])
            if lo > hi:
                lo, hi = hi, lo
            out[key] = (lo, hi)
    return out


def _resolve_active_targets(meta: dict[str, Any], *, available_targets: tuple[str, ...] | None = None) -> tuple[str, ...]:
    allowed = set(available_targets or RESIDUAL_TARGET_NAMES)
    forced_raw = str(os.environ.get("UTOA_BOUNDARY_RESIDUAL_TARGETS", "") or "").strip().lower()
    if forced_raw:
        forced = [token.strip() for token in forced_raw.split(",") if token.strip()]
        forced = [token for token in forced if token in allowed]
        if forced:
            return tuple(forced)
    # Conservative default: keep runtime stable by only enabling shape-correction deltas.
    default_targets = [target for target in ("delta_preutterance", "delta_overlap") if target in allowed]
    if default_targets:
        return tuple(default_targets)

    disabled_raw = str(os.environ.get("UTOA_BOUNDARY_RESIDUAL_DISABLE_TARGETS", "") or "").strip().lower()
    disabled = {token.strip() for token in disabled_raw.split(",") if token.strip()}
    try:
        min_improve = float(os.environ.get("UTOA_BOUNDARY_RESIDUAL_MIN_MAE_IMPROVEMENT", "0.0"))
    except Exception:
        min_improve = 0.0
    metrics = meta.get("metrics") if isinstance(meta, dict) else {}
    active: list[str] = []
    for target in RESIDUAL_TARGET_NAMES:
        if target not in allowed:
            continue
        if target in disabled:
            continue
        if isinstance(metrics, dict) and isinstance(metrics.get(target), dict):
            improve = float(metrics.get(target, {}).get("mae_improvement", 0.0) or 0.0)
            if improve < min_improve:
                continue
        active.append(target)
    if not active:
        active = [target for target in RESIDUAL_TARGET_NAMES if target not in disabled and target in allowed]
    return tuple(active)


def _read_stage1_role_bias(path: str) -> dict[str, dict[str, float]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        return {}
    for target in RESIDUAL_TARGET_NAMES:
        item = targets.get(target)
        if not isinstance(item, dict):
            continue
        by_role = item.get("by_role") if isinstance(item.get("by_role"), dict) else {}
        role_map: dict[str, float] = {}
        for key, value in by_role.items():
            role_map[normalize_role(key)] = float(value)
        role_map["default"] = float(item.get("default", 0.0) or 0.0)
        out[target] = role_map
    return out


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


__all__ = [
    "BoundaryResidualBundle",
    "DEFAULT_DELTA_CLIPS",
    "RESIDUAL_CATEGORICAL_FEATURES",
    "RESIDUAL_FEATURE_NAMES",
    "RESIDUAL_TARGET_NAMES",
    "apply_boundary_residual_deltas",
    "apply_residual_to_decoded_row",
    "build_residual_feature_row",
    "load_boundary_residual_bundle",
    "predict_boundary_residual_deltas",
    "predict_boundary_stage1_role_bias",
    "residual_enabled_by_env",
    "resolve_boundary_residual_model_dir",
    "should_apply_residual_for_row",
]
