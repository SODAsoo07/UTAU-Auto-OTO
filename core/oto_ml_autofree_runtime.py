from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from core.format_type_utils import normalize_format_type
from core.oto_ml_autofree import (
    AUTOFREE_BACKEND,
    FEATURE_SCHEMA_VERSION,
    TARGET_SCHEMA_VERSION,
    build_rows_for_inference,
    predict_autofree_abs,
)
from core.oto_ml_features import parse_oto_rows
from core.oto_ml_runtime import OtoModelBundle, load_oto_model_bundle
from core.pipeline_status import (
    ML_APPLIED,
    ML_BUNDLE_INVALID,
    ML_DISABLED_ENV,
    ML_INFER_FAILED,
    ML_INPUT_MISSING,
    ML_MODEL_MISSING,
    ML_NO_FEATURES,
    ML_POLICY_OFF,
    ML_ROUTE_UNAVAILABLE,
    OK,
    make_runtime_report,
    normalize_ml_policy,
)


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_short_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _is_pause_like_alias(alias: str, alias_type: str = "") -> bool:
    a = _normalize_short_text(alias)
    t = _normalize_short_text(alias_type)
    if t in {"br", "pause", "rest", "sp", "sil"}:
        return True
    if a in {"", "r", "rr", "sil", "sp", "spn", "pau", "pause", "rest", "_", "-"}:
        return True
    return a.startswith("br")


def _get_validate_func(language: str):
    lang = str(language).strip().lower()
    if lang == "japanese":
        from core.ja_oto_generator import validate_oto_params
    else:
        from core.oto_generator import validate_oto_params
    return validate_oto_params


def _read_bundle_backend(model_dir: str) -> str:
    try:
        import json

        meta_path = os.path.join(model_dir, "model_meta.json")
        if not os.path.isfile(meta_path):
            return ""
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
        return str(meta.get("backend", "") or "").strip().lower()
    except Exception:
        return ""


def _model_root_for_language(language: str) -> str:
    lang = str(language).strip().lower()
    explicit_env = "UTOA_JA_AUTOFREE_ML_DIR" if lang == "japanese" else "UTOA_KR_AUTOFREE_ML_DIR"
    explicit = str(os.environ.get(explicit_env, "") or "").strip()
    if explicit:
        return explicit
    # compatibility fallback
    legacy_env = "UTOA_JA_OTO_ML_DIR" if lang == "japanese" else "UTOA_KR_OTO_ML_DIR"
    legacy = str(os.environ.get(legacy_env, "") or "").strip()
    if legacy:
        return legacy
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "assets", "models", "oto_ml", lang)


def _pick_latest_version_dir(format_root: str) -> Optional[str]:
    if not format_root or not os.path.isdir(format_root):
        return None
    candidates: List[str] = []
    for name in os.listdir(format_root):
        full = os.path.join(format_root, name)
        if not os.path.isdir(full):
            continue
        if os.path.isfile(os.path.join(full, "model_meta.json")):
            candidates.append(full)
    for candidate in sorted(candidates, key=lambda p: os.path.basename(p), reverse=True):
        if _read_bundle_backend(candidate) == AUTOFREE_BACKEND:
            return candidate
    return None


def _resolve_autofree_model_dir(language: str, format_type: str) -> Optional[str]:
    root = _model_root_for_language(language)
    fmt = normalize_format_type(language, format_type) or "general"
    # explicit direct path
    if os.path.isfile(os.path.join(root, "model_meta.json")) and _read_bundle_backend(root) == AUTOFREE_BACKEND:
        return root
    primary = _pick_latest_version_dir(os.path.join(root, fmt))
    if primary:
        return primary
    if fmt != "general":
        return _pick_latest_version_dir(os.path.join(root, "general"))
    return None


def validate_autofree_bundle_contract(bundle: Optional[OtoModelBundle]) -> Tuple[bool, str]:
    if bundle is None:
        return False, "bundle_missing"
    backend = str(bundle.backend or "").strip().lower()
    if backend != AUTOFREE_BACKEND:
        return False, "backend_mismatch"
    meta = dict(bundle.meta or {})
    if not bool(meta.get("incompatible_with_legacy", False)):
        return False, "legacy_compat_flag_missing"
    if str(meta.get("feature_schema_version", "") or "").strip() != FEATURE_SCHEMA_VERSION:
        return False, "feature_schema_version_mismatch"
    if str(meta.get("target_schema_version", "") or "").strip() != TARGET_SCHEMA_VERSION:
        return False, "target_schema_version_mismatch"
    return True, "ok"


def _apply_abs_guard(language: str, row: Dict[str, object], pred: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    validate = _get_validate_func(language)
    offset = _to_float(pred.get("target_offset_ms"), _to_float(row.get("current_offset"), 0.0))
    alias_type = str(row.get("alias_type", "") or "").strip().lower()
    alias = str(row.get("alias", "") or "")

    # Prevent collapse to leading silence when TextGrid onset is far from 0.
    if not _is_pause_like_alias(alias, alias_type):
        onset_ms = _to_float(row.get("onset_ms"), 0.0)
        tail_ms = _to_float(row.get("tail_ms"), 0.0)
        span_ms = max(1.0, tail_ms - onset_ms)
        if onset_ms >= 120.0:
            min_offset = max(0.0, onset_ms - max(220.0, span_ms * 1.8))
            if offset < min_offset:
                offset = min_offset

    cons = max(0.0, _to_float(pred.get("target_cons_ms"), _to_float(row.get("current_cons"), 0.0)))
    cutoff_abs = _to_float(pred.get("target_cutoff_abs_ms"), offset + 10.0)
    cutoff_abs = max(cutoff_abs, offset + 2.0)
    tail_ms = _to_float(row.get("tail_ms"), 0.0)
    if tail_ms > 1.0:
        cutoff_abs = min(cutoff_abs, max(0.0, tail_ms - 1.0))
    pre = max(0.0, _to_float(pred.get("target_pre_ms"), _to_float(row.get("current_pre"), 0.0)))
    ovl = _to_float(pred.get("target_ovl_ms"), _to_float(row.get("current_ovl"), 0.0))
    cutoff = -(cutoff_abs - offset)
    try:
        return validate(offset, cons, cutoff, pre, ovl, alias_type=alias_type)
    except TypeError:
        return validate(offset, cons, cutoff, pre, ovl)


def apply_autofree_ml_to_oto_file(
    language: str,
    oto_path: str,
    tg_dir: str,
    wav_dir: str,
    custom_phonemes_path: str = "",
    callback=None,
    enabled: bool = True,
    format_override: Optional[str] = None,
    policy: Optional[str] = None,
    report: Optional[Dict[str, object]] = None,
) -> int:
    _ = custom_phonemes_path  # reserved
    policy_name = normalize_ml_policy(policy, enabled_default=enabled)
    required_policy = policy_name == "on"
    ml_report = report if isinstance(report, dict) else {}
    ml_report.setdefault("stage", "ml")
    ml_report.setdefault("route", AUTOFREE_BACKEND)
    ml_report.setdefault("policy", policy_name)
    ml_report.setdefault("status", "skipped")
    ml_report.setdefault("fallback_used", False)
    ml_report.setdefault("attempted_routes", [])
    ml_report.setdefault("missing_routes", [])
    ml_report.setdefault("invalid_routes", [])
    ml_report.setdefault("loaded_models", [])
    ml_report.setdefault("infer_failures", [])
    ml_report.setdefault("changed_lines", 0)
    ml_report.setdefault("model_confidence", 0.0)
    ml_report.setdefault("fallback_reason", "")

    if policy_name == "off":
        ml_report.update(make_runtime_report("ml", ML_POLICY_OFF, "ML 정책이 off로 설정되었습니다."))
        ml_report["status"] = "skipped"
        return 0
    if os.environ.get("UTOA_DISABLE_OTO_ML", "").strip().lower() in {"1", "true", "yes", "on"}:
        ml_report.update(make_runtime_report("ml", ML_DISABLED_ENV, "환경변수로 OTO ML이 비활성화되었습니다."))
        ml_report["status"] = "skipped"
        return 0
    if not oto_path or not os.path.isfile(oto_path):
        ml_report.update(make_runtime_report("ml", ML_INPUT_MISSING, f"OTO 파일이 없습니다: {oto_path}"))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0

    rows = parse_oto_rows(oto_path, language=str(language or "").strip().lower())
    if not rows:
        ml_report.update(make_runtime_report("ml", ML_INPUT_MISSING, "OTO 행을 읽지 못했습니다."))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0

    feature_rows, feature_stats = build_rows_for_inference(
        language=language,
        oto_path=oto_path,
        wav_dir=wav_dir,
        tg_dir=tg_dir,
        format_type_override=str(format_override or ""),
    )
    if not feature_rows:
        reason = "ML feature를 추출하지 못했습니다."
        if int(feature_stats.get("skip_token_missing", 0)) > 0:
            reason = "autofree token 누락으로 추론 대상이 없습니다."
        ml_report.update(make_runtime_report("ml", ML_NO_FEATURES, reason))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        ml_report["fallback_reason"] = "token_missing" if int(feature_stats.get("skip_token_missing", 0)) > 0 else ""
        return 0

    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [line.rstrip("\n") for line in f]
    rows_by_index = {int(row["line_index"]): row for row in rows}

    bundle_cache: Dict[str, Optional[OtoModelBundle]] = {}
    changed = 0
    conf_values: List[float] = []

    for feat in feature_rows:
        fmt = normalize_format_type(language, str(format_override or feat.get("format_type", ""))) or "general"
        if fmt not in ml_report["attempted_routes"]:
            ml_report["attempted_routes"].append(fmt)
        line_index = int(feat.get("line_index", -1))
        if line_index < 0 or line_index not in rows_by_index:
            continue

        if fmt not in bundle_cache:
            model_dir = _resolve_autofree_model_dir(language, fmt)
            if not model_dir:
                ml_report["missing_routes"].append({"format_type": fmt, "code": ML_MODEL_MISSING, "model_dir": ""})
                bundle_cache[fmt] = None
            else:
                bundle = load_oto_model_bundle(model_dir)
                ok, reason = validate_autofree_bundle_contract(bundle)
                if not ok:
                    ml_report["invalid_routes"].append(
                        {"format_type": fmt, "code": ML_BUNDLE_INVALID, "model_dir": model_dir, "reason": reason}
                    )
                    bundle = None
                else:
                    if model_dir not in ml_report["loaded_models"]:
                        ml_report["loaded_models"].append(model_dir)
                        _emit(callback, f"[OTO-ML] autofree 모델 로드: {model_dir}")
                bundle_cache[fmt] = bundle

        bundle = bundle_cache.get(fmt)
        if bundle is None:
            continue
        row = rows_by_index.get(line_index)
        if row is None:
            continue
        try:
            pred = predict_autofree_abs(bundle.payload, feat)
            conf = _to_float(pred.get("confidence"), 0.0)
            conf_values.append(conf)
            o2, c2, ct2, p2, ov2 = _apply_abs_guard(language, feat, pred)
        except Exception as exc:
            ml_report["infer_failures"].append({"line_index": line_index, "message": str(exc)})
            continue

        if (
            abs(o2 - _to_float(row.get("offset"), 0.0)) > 1e-6
            or abs(c2 - _to_float(row.get("cons"), 0.0)) > 1e-6
            or abs(ct2 - _to_float(row.get("cutoff"), 0.0)) > 1e-6
            or abs(p2 - _to_float(row.get("pre"), 0.0)) > 1e-6
            or abs(ov2 - _to_float(row.get("ovl"), 0.0)) > 1e-6
        ):
            changed += 1
        raw_lines[line_index] = f"{row['wav']}={row['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{p2:.2f},{ov2:.2f}"

    if changed > 0:
        with open(oto_path, "w", encoding="utf-8") as f:
            for line in raw_lines:
                f.write(line + "\n")
        _emit(callback, f"[OTO-ML] autofree 수치 보정 적용: {changed} lines")

    ml_report["changed_lines"] = int(changed)
    if conf_values:
        ml_report["model_confidence"] = float(sum(conf_values) / float(len(conf_values)))
    if int(feature_stats.get("skip_token_missing", 0)) > 0:
        ml_report["fallback_reason"] = "token_missing"
    if changed > 0:
        ml_report.update(make_runtime_report("ml", ML_APPLIED, f"{changed} lines adjusted"))
        ml_report["status"] = "applied"
        return changed
    if ml_report["loaded_models"]:
        if ml_report["infer_failures"]:
            ml_report.update(make_runtime_report("ml", ML_INFER_FAILED, "autofree 추론 일부 실패로 기본 계산을 유지했습니다."))
            ml_report["status"] = "fallback"
            ml_report["fallback_used"] = True
        else:
            ml_report.update(make_runtime_report("ml", ML_APPLIED, "변경이 필요하지 않았습니다."))
            ml_report["status"] = "no_change"
        return 0
    if ml_report["invalid_routes"]:
        ml_report.update(make_runtime_report("ml", ML_BUNDLE_INVALID, "autofree 번들이 계약과 맞지 않습니다."))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0
    if ml_report["missing_routes"]:
        ml_report.update(make_runtime_report("ml", ML_MODEL_MISSING, "autofree 모델이 없어 기본 계산으로 유지했습니다."))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0
    if ml_report["infer_failures"]:
        ml_report.update(make_runtime_report("ml", ML_INFER_FAILED, "autofree 추론 실패로 기본 계산으로 유지했습니다."))
        ml_report["status"] = "fallback"
        ml_report["fallback_used"] = True
        return 0
    ml_report.update(make_runtime_report("ml", ML_ROUTE_UNAVAILABLE, "autofree 경로에서 사용 가능한 모델이 없습니다."))
    ml_report["status"] = "skipped"
    return 0


__all__ = ["apply_autofree_ml_to_oto_file", "validate_autofree_bundle_contract"]
