from __future__ import annotations

import json
import logging
import math
import os
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from core.format_type_utils import normalize_format_type
from core.oto_ml_features import CATEGORICAL_FEATURES, FEATURE_NAMES, canonicalize_feature_row

logger = logging.getLogger(__name__)


ParamsTuple = Tuple[float, float, float, float, float]
DEFAULT_DELTA_CUTOFF_CLIP = 1200.0
ALIAS_MODEL_BUCKETS = ("cv", "vc", "vv")
VC_STOP_LIKE_CODA_TYPES = {"stop", "affricate", "plosive"}
CV_STOP_LIKE_ONSET_CLASSES = {"stop", "plosive", "affricate", "sibilant"}
CV_SONORANT_ONSET_CLASSES = {"sonorant", "nasal", "liquid"}
CVC_FAMILY_FORMATS = {"cvc", "cvvc", "vccv"}
KNOWN_CUTOFF_MODES = ("negative_relative", "positive_relative", "positive_absolute", "positive_from_wav_end")
MAX_EXPM1_ARG = 20.0


@dataclass
class OtoE2EConfig:
    enabled: bool = False
    mode: str = "hybrid"
    t_low: float = 0.52
    t_high: float = 0.72
    blend_alpha: float = 0.60


@dataclass
class OtoE2EBundle:
    backend: str
    model_dir: str
    meta: Dict[str, Any]
    feature_schema: Dict[str, Any]
    payload: Dict[str, Any]


@dataclass
class OtoE2ERuntimeContext:
    language: str
    format_type: str
    route: str
    config: OtoE2Config
    active: bool
    code: str
    message: str
    bundle: Optional[OtoE2EBundle] = None
    bundles_by_alias: Dict[str, OtoE2EBundle] = field(default_factory=dict)


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"hybrid", "legacy_only", "e2e_only"}:
        return mode
    return "hybrid"


def read_e2e_config_from_env() -> OtoE2EConfig:
    cfg = OtoE2EConfig(
        enabled=_env_flag("UTOA_E2E_ENABLE", False),
        mode=_normalize_mode(os.environ.get("UTOA_E2E_MODE", "hybrid")),
        t_low=_clamp(_env_float("UTOA_E2E_T_LOW", 0.52), 0.0, 1.0),
        t_high=_clamp(_env_float("UTOA_E2E_T_HIGH", 0.72), 0.0, 1.0),
        blend_alpha=_clamp(_env_float("UTOA_E2E_BLEND_ALPHA", 0.60), 0.0, 1.0),
    )
    if cfg.t_high < cfg.t_low:
        cfg.t_high = cfg.t_low
    return cfg


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _norm_path_key(path: str) -> str:
    try:
        return os.path.abspath(str(path or ""))
    except Exception:
        return str(path or "")


def _pick_latest_model_dir(root: str) -> str:
    if not root or not os.path.isdir(root):
        return ""
    if os.path.isfile(os.path.join(root, "model_meta.json")):
        return root
    candidates = []
    try:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if not os.path.isdir(full):
                continue
            if os.path.isfile(os.path.join(full, "model_meta.json")):
                candidates.append(full)
    except Exception:
        return ""
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.basename(p), reverse=True)
    return candidates[0]


def _normalize_alias_bucket(alias_type: object) -> str:
    alias = str(alias_type or "").strip().lower()
    if alias in {"cv", "cv_head", "mono"}:
        return "cv"
    if alias == "vc":
        return "vc"
    if alias in {"vv", "vcv"}:
        return "vv"
    return "other"


def _hash_bucket(cat_name: str, cat_value: str, bins: int) -> Tuple[int, float]:
    n_bins = max(1, int(bins))
    token = (str(cat_name or "").strip().lower() + "=" + str(cat_value or "").strip().lower()).encode("utf-8", errors="replace")
    digest = hashlib.sha1(token).digest()
    idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % n_bins
    sign = 1.0 if (digest[4] & 1) == 0 else -1.0
    return idx, sign


def _is_cvc_family_format(format_type: object) -> bool:
    fmt = str(format_type or "").strip().lower()
    return fmt in CVC_FAMILY_FORMATS


def _build_model_design_row(bundle: OtoE2EBundle, feature_row: Dict[str, Any]) -> Dict[str, Any]:
    schema = dict(bundle.feature_schema or {})
    feature_names = list(schema.get("feature_names") or FEATURE_NAMES)
    base_feature_names = list(schema.get("base_feature_names") or FEATURE_NAMES)
    cat_features = list(schema.get("categorical_hash_features") or CATEGORICAL_FEATURES)
    cat_hash_bins = max(0, int(_safe_float(schema.get("categorical_hash_bins", 0), 0.0)))
    enable_vc_stop_binary = bool(schema.get("enable_vc_stop_binary", False))
    enable_cv_subgroup_binary = bool(schema.get("enable_cv_subgroup_binary", False))

    base_row = canonicalize_feature_row(feature_row, feature_names=base_feature_names)
    out: Dict[str, Any] = {}
    for name in base_feature_names:
        if name in CATEGORICAL_FEATURES:
            continue
        out[name] = _safe_float(base_row.get(name, 0.0), 0.0)

    if cat_hash_bins > 0:
        for cat in cat_features:
            raw_val = feature_row.get(cat, base_row.get(cat, ""))
            cat_val = str(raw_val or "").strip().lower()
            if not cat_val:
                continue
            idx, sign = _hash_bucket(cat, cat_val, cat_hash_bins)
            if cat_hash_bins <= 1:
                out[f"hash__{cat}"] = float(sign)
            else:
                out[f"hash__{cat}"] = float(sign) * (float(idx) / float(cat_hash_bins - 1))

    if enable_vc_stop_binary:
        alias_bucket = _normalize_alias_bucket(feature_row.get("alias_type", ""))
        coda_raw = feature_row.get("coda_type", base_row.get("coda_type", ""))
        coda_type = str(coda_raw or "").strip().lower()
        is_stop = alias_bucket == "vc" and coda_type in VC_STOP_LIKE_CODA_TYPES
        out["vc_stop_like_flag"] = 1.0 if is_stop else 0.0
        out["vc_non_stop_flag"] = 1.0 if (alias_bucket == "vc" and not is_stop) else 0.0

    if enable_cv_subgroup_binary:
        alias_type_raw = feature_row.get("alias_type", base_row.get("alias_type", ""))
        alias_type = str(alias_type_raw or "").strip().lower()
        alias_bucket = _normalize_alias_bucket(alias_type)
        onset_raw = feature_row.get("onset_class", base_row.get("onset_class", ""))
        onset_class = str(onset_raw or "").strip().lower()
        format_raw = feature_row.get("format_type", base_row.get("format_type", ""))
        format_type = str(format_raw or "").strip().lower()

        is_cv = alias_bucket == "cv"
        is_cv_head = is_cv and alias_type == "cv_head"
        is_cv_stop_like = is_cv and onset_class in CV_STOP_LIKE_ONSET_CLASSES
        is_cv_sonorant = is_cv and onset_class in CV_SONORANT_ONSET_CLASSES
        is_cv_cvc_family = is_cv and _is_cvc_family_format(format_type)

        out["cv_head_flag"] = 1.0 if is_cv_head else 0.0
        out["cv_onset_stop_like_flag"] = 1.0 if is_cv_stop_like else 0.0
        out["cv_onset_sonorant_flag"] = 1.0 if is_cv_sonorant else 0.0
        out["cv_cvc_family_flag"] = 1.0 if is_cv_cvc_family else 0.0
        out["cv_head_x_cvc_flag"] = 1.0 if (is_cv_head and is_cv_cvc_family) else 0.0

    return canonicalize_feature_row(out, feature_names=feature_names)


def _pick_alias_model_dirs(root: str) -> Dict[str, str]:
    if not root or not os.path.isdir(root):
        return {}
    out: Dict[str, str] = {}
    for bucket in ALIAS_MODEL_BUCKETS:
        picked = _pick_latest_model_dir(os.path.join(root, bucket))
        if picked:
            out[bucket] = picked
    return out


def resolve_e2e_model_dir(language: str, format_type: str) -> str:
    lang = str(language or "").strip().lower() or "korean"
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower() or "general"

    per_lang_env = "UTOA_JA_E2E_MODEL_DIR" if lang == "japanese" else "UTOA_KR_E2E_MODEL_DIR"
    explicit = str(os.environ.get(per_lang_env, "") or "").strip()
    if explicit:
        return _pick_latest_model_dir(explicit)

    shared = str(os.environ.get("UTOA_E2E_MODEL_DIR", "") or "").strip()
    if shared:
        # shared path can be direct model dir or language root
        direct = _pick_latest_model_dir(shared)
        if direct:
            return direct
        for candidate in (
            os.path.join(shared, lang, "general"),
            os.path.join(shared, lang, fmt),
            os.path.join(shared, lang, "cv"),
        ):
            picked = _pick_latest_model_dir(candidate)
            if picked:
                return picked

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (
        os.path.join(project_root, "assets", "models", "oto_e2e", lang, "general"),
        os.path.join(project_root, "assets", "models", "oto_e2e", lang, fmt),
        os.path.join(project_root, "assets", "models", "oto_e2e", lang, "cv"),
    ):
        picked = _pick_latest_model_dir(candidate)
        if picked:
            return picked
    return ""


def resolve_e2e_alias_model_root(language: str, format_type: str) -> str:
    lang = str(language or "").strip().lower() or "korean"
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower() or "general"

    per_lang_env = "UTOA_JA_E2E_ALIAS_MODEL_DIR" if lang == "japanese" else "UTOA_KR_E2E_ALIAS_MODEL_DIR"
    explicit = str(os.environ.get(per_lang_env, "") or "").strip()
    if explicit:
        return _norm_path_key(explicit)

    shared = str(os.environ.get("UTOA_E2E_ALIAS_MODEL_DIR", "") or "").strip()
    if shared:
        for candidate in (shared, os.path.join(shared, lang), os.path.join(shared, lang, fmt)):
            if candidate and os.path.isdir(candidate):
                return _norm_path_key(candidate)
        return _norm_path_key(shared)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (
        os.path.join(project_root, "assets", "models", "oto_e2e", lang, "by_alias"),
        os.path.join(project_root, "assets", "models", "oto_e2e", lang, fmt, "by_alias"),
    ):
        if os.path.isdir(candidate):
            return _norm_path_key(candidate)
    return ""


def _load_e2e_payload(model_dir: str) -> Optional[Dict[str, Any]]:
    for name in ("e2e_linear.json", "model_linear.json", "weights.json"):
        payload = _load_json(os.path.join(model_dir, name))
        if isinstance(payload, dict):
            return payload
    return None


def load_e2e_bundle(model_dir: str) -> Tuple[Optional[OtoE2EBundle], str, str]:
    if not model_dir:
        return None, "model_unavailable", "e2e model dir is empty"
    model_dir = _norm_path_key(model_dir)
    if not os.path.isdir(model_dir):
        return None, "model_unavailable", f"e2e model dir not found: {model_dir}"

    meta = _load_json(os.path.join(model_dir, "model_meta.json"))
    if not isinstance(meta, dict):
        return None, "model_unavailable", "model_meta.json missing"

    backend = str(meta.get("backend", "") or "").strip().lower()
    if backend and backend not in {"e2e_hybrid_v1", "e2e_linear_v1", "e2e_stub_v1"}:
        return None, "model_unavailable", f"unsupported e2e backend: {backend}"

    schema = _load_json(os.path.join(model_dir, "feature_schema.json"))
    if not isinstance(schema, dict):
        schema = {
            "feature_names": list(FEATURE_NAMES),
            "feature_version": str(meta.get("feature_version", "")),
        }

    payload = _load_e2e_payload(model_dir)
    if not isinstance(payload, dict):
        return None, "model_unavailable", "e2e linear payload missing"

    bundle = OtoE2EBundle(
        backend=backend or "e2e_linear_v1",
        model_dir=model_dir,
        meta=meta,
        feature_schema=schema,
        payload=payload,
    )
    return bundle, "ok", ""


def _predict_linear_head(head: Dict[str, Any], row: Dict[str, Any]) -> float:
    bias = _safe_float(head.get("bias", 0.0), 0.0)
    weights = head.get("weights") or {}
    out = float(bias)
    if isinstance(weights, dict):
        for key, w in weights.items():
            out += _safe_float(w, 0.0) * _safe_float(row.get(str(key), 0.0), 0.0)
    return float(out)


def _decode_target_delta(payload: Dict[str, Any], target: str, value: float) -> float:
    transforms = dict((payload or {}).get("target_transforms") or {})
    spec = dict(transforms.get(str(target)) or {})
    t_type = str(spec.get("type", "") or "").strip().lower()
    if t_type != "signed_log1p_clip":
        return float(value)
    clip_abs = max(1.0, _safe_float(spec.get("clip_abs", DEFAULT_DELTA_CUTOFF_CLIP), DEFAULT_DELTA_CUTOFF_CLIP))
    v = max(-MAX_EXPM1_ARG, min(MAX_EXPM1_ARG, float(value)))
    if v >= 0.0:
        dec = float(math.expm1(v))
    else:
        dec = float(-math.expm1(abs(v)))
    return max(-clip_abs, min(clip_abs, dec))


def _predict_cutoff_mode_bias(payload: Dict[str, Any], feature_row: Dict[str, Any]) -> float:
    adj = dict((payload or {}).get("cutoff_mode_adjustment") or {})
    if not bool(adj.get("enabled", False)):
        return 0.0

    mode_bias_raw = dict(adj.get("mode_bias") or {})
    mode_bias: Dict[str, float] = {}
    for mode in KNOWN_CUTOFF_MODES:
        mode_bias[mode] = _safe_float(mode_bias_raw.get(mode, 0.0), 0.0)

    if not any(abs(v) > 1e-9 for v in mode_bias.values()):
        return 0.0

    alias_bucket = _normalize_alias_bucket(feature_row.get("alias_type", ""))
    fmt = str(feature_row.get("format_type", "") or "").strip().lower() or "unknown"
    key = f"{alias_bucket}|{fmt}"

    priors_by_key = dict(adj.get("priors_by_alias_format") or {})
    priors = dict(priors_by_key.get(key) or {})
    if not priors:
        priors = dict(adj.get("global_priors") or {})
    if not priors:
        return 0.0

    p_neg = max(0.0, _safe_float(priors.get("negative_relative", 0.0), 0.0))
    p_pos = max(0.0, min(1.0, 1.0 - p_neg))
    gate = max(0.0, min(1.0, _safe_float(adj.get("positive_gate_threshold", 0.35), 0.35)))
    if p_pos <= gate:
        return 0.0
    gate_scale = 1.0
    if gate < 1.0:
        gate_scale = max(0.0, min(1.0, (p_pos - gate) / max(1e-6, 1.0 - gate)))

    positive_only = bool(adj.get("positive_only", True))

    prob_sum = 0.0
    weighted = 0.0
    for mode in KNOWN_CUTOFF_MODES:
        if positive_only and mode == "negative_relative":
            continue
        p = max(0.0, _safe_float(priors.get(mode, 0.0), 0.0))
        if p <= 0.0:
            continue
        prob_sum += p
        weighted += p * _safe_float(mode_bias.get(mode, 0.0), 0.0)

    if prob_sum <= 1e-9:
        return 0.0
    expected = float(weighted / prob_sum) * float(gate_scale)
    return max(-DEFAULT_DELTA_CUTOFF_CLIP, min(DEFAULT_DELTA_CUTOFF_CLIP, expected))


def _predict_confidence(payload: Dict[str, Any], row: Dict[str, Any]) -> float:
    mapping_conf = _clamp(_safe_float(row.get("mapping_confidence", 0.0), 0.0), 0.0, 1.0)
    conf_head = payload.get("confidence")
    if isinstance(conf_head, dict):
        raw = _predict_linear_head(conf_head, row)
        # Convert unbounded linear score to probabilistic band, then blend with
        # mapping confidence to avoid hard saturation from uncalibrated heads.
        try:
            model_conf = 1.0 / (1.0 + math.exp(-float(raw)))
        except Exception:
            model_conf = 0.5
        # Keep mapping confidence dominant until per-model calibration is added.
        blended = 0.30 * float(model_conf) + 0.70 * float(mapping_conf)
        return _clamp(blended, 0.0, 1.0)

    return mapping_conf


def _predict_abs_params(bundle: OtoE2EBundle, feature_row: Dict[str, Any]) -> ParamsTuple:
    row = _build_model_design_row(bundle, feature_row)
    payload = dict(bundle.payload or {})
    targets = payload.get("targets") or {}

    output_mode = str((bundle.meta or {}).get("output_mode", "delta") or "delta").strip().lower()
    if output_mode == "absolute":
        off = _predict_linear_head(dict(targets.get("offset") or {}), row)
        cons = _predict_linear_head(dict(targets.get("cons") or {}), row)
        cut_abs = _predict_linear_head(dict(targets.get("cutoff_abs") or {}), row)
        pre = _predict_linear_head(dict(targets.get("pre") or {}), row)
        ovl = _predict_linear_head(dict(targets.get("ovl") or {}), row)
    else:
        d_off = _predict_linear_head(dict(targets.get("delta_offset") or {}), row)
        d_cons = _predict_linear_head(dict(targets.get("delta_cons") or {}), row)
        d_cut = _predict_linear_head(dict(targets.get("delta_cutoff") or {}), row)
        d_pre = _predict_linear_head(dict(targets.get("delta_pre") or {}), row)
        d_ovl = _predict_linear_head(dict(targets.get("delta_ovl") or {}), row)
        d_off = _decode_target_delta(payload, "delta_offset", d_off)
        d_cons = _decode_target_delta(payload, "delta_cons", d_cons)
        d_cut = _decode_target_delta(payload, "delta_cutoff", d_cut)
        d_pre = _decode_target_delta(payload, "delta_pre", d_pre)
        d_ovl = _decode_target_delta(payload, "delta_ovl", d_ovl)
        d_cut += _predict_cutoff_mode_bias(payload, feature_row)

        base_off = _safe_float(feature_row.get("base_offset", 0.0), 0.0)
        base_cons = _safe_float(feature_row.get("base_cons", 0.0), 0.0)
        base_cut_abs = _safe_float(feature_row.get("base_cutoff_abs", 0.0), 0.0)
        base_pre = _safe_float(feature_row.get("base_pre", 0.0), 0.0)
        base_ovl = _safe_float(feature_row.get("base_ovl", 0.0), 0.0)

        off = base_off + d_off
        cons = base_cons + d_cons
        cut_abs = base_cut_abs + d_cut
        pre = base_pre + d_pre
        ovl = base_ovl + d_ovl

    cut_abs = max(float(cut_abs), float(off) + 2.0)
    cutoff = -(cut_abs - float(off))
    return float(off), float(cons), float(cutoff), float(pre), float(ovl)


def build_e2e_runtime_context(language: str, format_type: str, route: str) -> OtoE2ERuntimeContext:
    lang = str(language or "").strip().lower() or "korean"
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower() or "general"
    route_name = str(route or "legacy").strip().lower() or "legacy"
    cfg = read_e2e_config_from_env()

    if not cfg.enabled:
        return OtoE2ERuntimeContext(lang, fmt, route_name, cfg, False, "disabled", "UTOA_E2E_ENABLE=0")
    if cfg.mode == "legacy_only":
        return OtoE2ERuntimeContext(lang, fmt, route_name, cfg, False, "legacy_only", "UTOA_E2E_MODE=legacy_only")
    if route_name != "e2e_hybrid":
        return OtoE2ERuntimeContext(lang, fmt, route_name, cfg, False, "route_off", f"route={route_name}")

    model_dir = resolve_e2e_model_dir(lang, fmt)
    bundle, code, message = load_e2e_bundle(model_dir)

    alias_root = resolve_e2e_alias_model_root(lang, fmt)
    alias_dirs = _pick_alias_model_dirs(alias_root)
    alias_bundles: Dict[str, OtoE2EBundle] = {}
    for bucket, alias_dir in alias_dirs.items():
        alias_bundle, _alias_code, _alias_msg = load_e2e_bundle(alias_dir)
        if alias_bundle is not None:
            alias_bundles[bucket] = alias_bundle

    if bundle is None and not alias_bundles:
        return OtoE2ERuntimeContext(lang, fmt, route_name, cfg, False, code, message)

    return OtoE2ERuntimeContext(
        lang,
        fmt,
        route_name,
        cfg,
        True,
        "ok",
        "",
        bundle=bundle,
        bundles_by_alias=alias_bundles,
    )


def predict_e2e_params(context: OtoE2ERuntimeContext, feature_row: Dict[str, Any]) -> Dict[str, Any]:
    if context is None or not bool(getattr(context, "active", False)):
        return {
            "ok": False,
            "code": str(getattr(context, "code", "disabled") if context is not None else "disabled"),
            "message": str(getattr(context, "message", "") if context is not None else ""),
        }

    alias_bucket = _normalize_alias_bucket(feature_row.get("alias_type", ""))
    bundle = (context.bundles_by_alias or {}).get(alias_bucket)
    if bundle is None:
        bundle = context.bundle
    if bundle is None and context.bundles_by_alias:
        # Fallback to any loaded alias bundle instead of failing hard.
        bundle = next(iter(context.bundles_by_alias.values()), None)
    if bundle is None:
        return {"ok": False, "code": "model_unavailable", "message": "e2e bundle is missing"}

    try:
        params = _predict_abs_params(bundle, feature_row)
        confidence = _predict_confidence(bundle.payload, feature_row)
        return {
            "ok": True,
            "code": "ok",
            "params": params,
            "confidence": float(confidence),
            "model_dir": str(bundle.model_dir),
            "backend": str(bundle.backend),
            "alias_bucket": alias_bucket,
        }
    except Exception as exc:
        logger.warning("E2E prediction failed: %s", exc)
        return {
            "ok": False,
            "code": "predict_failed",
            "message": str(exc),
            "model_dir": str(bundle.model_dir),
            "backend": str(bundle.backend),
            "alias_bucket": alias_bucket,
        }


def predict_e2e_deltas_from_bundle(bundle: OtoE2EBundle, feature_row: Dict[str, Any]) -> Dict[str, Any]:
    params = _predict_abs_params(bundle, feature_row)
    confidence = _predict_confidence(bundle.payload, feature_row)

    base_off = _safe_float(feature_row.get("base_offset", 0.0), 0.0)
    base_cons = _safe_float(feature_row.get("base_cons", 0.0), 0.0)
    base_cut_abs = _safe_float(feature_row.get("base_cutoff_abs", 0.0), 0.0)
    base_pre = _safe_float(feature_row.get("base_pre", 0.0), 0.0)
    base_ovl = _safe_float(feature_row.get("base_ovl", 0.0), 0.0)

    off, cons, cutoff, pre, ovl = [float(x) for x in params]
    cut_abs = float(off) + abs(float(cutoff))
    deltas = {
        "delta_offset": off - base_off,
        "delta_cons": cons - base_cons,
        "delta_cutoff": cut_abs - base_cut_abs,
        "delta_pre": pre - base_pre,
        "delta_ovl": ovl - base_ovl,
    }
    return {
        "deltas": deltas,
        "confidence": float(confidence),
    }


__all__ = [
    "OtoE2EConfig",
    "OtoE2EBundle",
    "OtoE2ERuntimeContext",
    "build_e2e_runtime_context",
    "load_e2e_bundle",
    "predict_e2e_deltas_from_bundle",
    "predict_e2e_params",
    "read_e2e_config_from_env",
    "resolve_e2e_alias_model_root",
    "resolve_e2e_model_dir",
]
