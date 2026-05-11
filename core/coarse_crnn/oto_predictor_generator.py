from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from core.coarse_crnn.alias_role import classify_alias_role, is_diphthong as _is_diphthong, normalize_role
from core.coarse_crnn.audio import load_wav_mono
from core.coarse_crnn.oto_audio_candidates import AudioCandidates, compute_audio_candidates
from core.coarse_crnn.oto_inference import predict_oto_with_model
from core.coarse_crnn.oto_model import load_oto_checkpoint
from core.coarse_crnn.oto_targets import OtoAnchors, anchors_to_oto_params, oto_params_to_anchors
from core.coarse_crnn.training import resolve_torch_device
from core.no_mfa_oto_builder import resolve_no_mfa_source_oto
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_ml_features import classify_alias_type, detect_format_type
from core.oto_normalization import normalize_wav_key


DEFAULT_OTO_CRNN_MODEL_NAME = "oto_anchor_crnn_role_v2.pt"


@dataclass(frozen=True)
class _PredictRow:
    wav_rel: str
    wav_abs: str
    alias: str
    prev_alias: str
    next_alias: str
    row_index: int
    row_count: int
    is_special: bool = False
    prev_is_special: bool = False
    next_is_special: bool = False
    base_params: dict[str, float] | None = None


def generate_oto_with_crnn_predictor(
    *,
    wav_dir: str,
    out_path: str,
    source_oto_path: str,
    language: str,
    format_type: str = "",
    model_path: str = "",
    device: str = "auto",
    alias_suffix: str = "",
    callback: Callable[[str], None] | None = None,
    special_aliases: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[int, int, list[str]]:
    source = resolve_no_mfa_source_oto(wav_dir=wav_dir, source_hint=source_oto_path)
    if not source:
        return 0, 0, ["CRNN OTO 예측용 베이스 OTO를 찾지 못했습니다."]
    model_file = resolve_oto_crnn_model_path(model_path)
    if not model_file:
        return 0, 0, ["CRNN OTO 예측 모델을 찾지 못했습니다. 모델 경로를 지정해 주세요."]
    output_file = _normalize_output_oto_path(out_path)
    if not output_file:
        return 0, 0, ["출력 OTO 경로가 비어 있습니다."]

    rows, total, missing = _prepare_prediction_rows(
        wav_dir=wav_dir,
        source_oto_path=source,
        special_aliases=_normalize_special_aliases(special_aliases),
    )
    if not rows:
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            return 0, total, [f"CRNN OTO 예측 대상 WAV 매칭 실패: {sample}"]
        return 0, total, ["CRNN OTO 예측 대상 행이 비어 있습니다."]

    _log(callback, f"[OTO-CRNN] source={source}")
    _log(callback, f"[OTO-CRNN] model={model_file}")
    _log(callback, f"[OTO-CRNN] device={device} rows={len(rows)}/{total}")
    fmt = _resolve_effective_format(
        language=language,
        requested_format=format_type,
        aliases=[row.alias for row in rows],
        callback=callback,
    )

    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    try:
        model, config, _meta = load_oto_checkpoint(model_file, map_location=str(torch_device))
        model = model.to(torch_device).eval()
    except Exception as exc:
        return 0, total, [f"CRNN OTO 모델 로드 실패: {exc}"]

    out_lines: list[str] = []
    errors: list[str] = []
    guard_changed = 0
    low_conf_fallback_count = 0
    activity_fallback_count = 0
    candidate_snap_count = 0
    activity_profile_cache: dict[str, dict[str, float]] = {}
    audio_candidate_cache: dict[str, AudioCandidates | None] = {}
    audio_candidate_sequence_state: dict[str, int] = {}
    lang = str(language or "").strip().lower() or "korean"
    suffix = _normalize_alias_suffix(alias_suffix)
    for idx, row in enumerate(rows):
        try:
            pred = predict_oto_with_model(
                model=model,
                config=config,
                wav_path=row.wav_abs,
                language=lang,
                format_type=fmt,
                alias=row.alias,
                prev_alias=row.prev_alias,
                next_alias=row.next_alias,
                row_index_in_wav=row.row_index,
                file_row_count=row.row_count,
                device=str(torch_device),
                is_special=row.is_special,
                prev_is_special=row.prev_is_special,
                next_is_special=row.next_is_special,
            )
            alias = _apply_alias_suffix(row.alias, suffix)
            params, changed = _apply_conservative_right_boundary_guard(
                pred.params,
                language=lang,
                alias=row.alias,
                duration_ms=float(getattr(pred, "duration_ms", 0.0) or 0.0),
                is_special=row.is_special,
            )
            params, fallback_reason = _apply_low_confidence_fallback(
                predicted_params=params,
                predicted_confidence=float(getattr(pred, "confidence", 0.0) or 0.0),
                predicted_error_ms=getattr(pred, "predicted_error_ms", None),
                predicted_low_confidence=bool(getattr(pred, "low_confidence", False)),
                confidence_components=getattr(pred, "confidence_components", None),
                base_params=row.base_params,
                language=lang,
                alias=row.alias,
                duration_ms=float(getattr(pred, "duration_ms", 0.0) or 0.0),
                is_special=row.is_special,
            )
            if fallback_reason:
                low_conf_fallback_count += 1
            params, activity_reason = _apply_activity_window_fallback(
                predicted_anchors={
                    "offset": float(getattr(pred.anchors, "offset", 0.0)),
                    "overlap": float(getattr(pred.anchors, "overlap", 0.0)),
                    "preutterance": float(getattr(pred.anchors, "preutterance", 0.0)),
                    "consonant": float(getattr(pred.anchors, "consonant", 0.0)),
                    "cutoff": float(getattr(pred.anchors, "cutoff", 0.0)),
                },
                predicted_params=params,
                base_params=row.base_params,
                wav_path=row.wav_abs,
                sample_rate=16000,
                cache=activity_profile_cache,
                row_index=row.row_index,
                row_count=row.row_count,
            )
            if activity_reason:
                activity_fallback_count += 1
            params, snap_reason = _apply_audio_candidate_snap(
                predicted_params=params,
                wav_path=row.wav_abs,
                language=lang,
                format_type=fmt,
                alias=row.alias,
                duration_ms=float(getattr(pred, "duration_ms", 0.0) or 0.0),
                cache=audio_candidate_cache,
                sequence_state=audio_candidate_sequence_state,
                config=config,
                is_special=row.is_special,
            )
            if snap_reason:
                candidate_snap_count += 1
                params, _snap_guard_changed = _apply_conservative_right_boundary_guard(
                    params,
                    language=lang,
                    alias=row.alias,
                    duration_ms=float(getattr(pred, "duration_ms", 0.0) or 0.0),
                    is_special=row.is_special,
                )
            guard_changed += 1 if changed else 0
            out_lines.append(
                f"{row.wav_rel}={alias},"
                f"{float(params['offset']):.3f},"
                f"{float(params['consonant']):.3f},"
                f"{float(params['cutoff']):.3f},"
                f"{float(params['preutterance']):.3f},"
                f"{float(params['overlap']):.3f}"
            )
        except Exception as exc:
            errors.append(f"{row.wav_rel}={row.alias}: {exc}")
        if callback and (idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(rows)):
            _log(callback, f"[OTO-CRNN] rows={idx + 1}/{len(rows)}")

    if errors:
        return len(out_lines), total, errors[:20]
    if not out_lines:
        return 0, total, ["CRNN OTO 예측 결과가 비었습니다."]

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(out_lines).rstrip() + "\n")
    except OSError as exc:
        return 0, total, [f"CRNN OTO 출력 저장 실패: {output_file} ({exc})"]

    if missing:
        sample = ", ".join(sorted(missing)[:5])
        suffix_text = "..." if len(missing) > 5 else ""
        _log(callback, f"[OTO-CRNN] 매칭 실패 wav: {sample}{suffix_text}")
    if guard_changed:
        _log(callback, f"[OTO-CRNN] right-boundary guard adjusted rows={guard_changed}/{len(out_lines)}")
    if low_conf_fallback_count:
        _log(callback, f"[OTO-CRNN] low-confidence fallback rows={low_conf_fallback_count}/{len(out_lines)}")
    if activity_fallback_count:
        _log(callback, f"[OTO-CRNN] activity-window fallback rows={activity_fallback_count}/{len(out_lines)}")
    if candidate_snap_count:
        _log(callback, f"[OTO-CRNN] audio-candidate snap rows={candidate_snap_count}/{len(out_lines)}")
    _log(callback, f"[OTO-CRNN] written={len(out_lines)} total={total}")
    return len(out_lines), total, []


def resolve_oto_crnn_model_path(path_hint: str = "") -> str:
    candidates = []
    for raw in (path_hint, os.environ.get("UTOA_OTO_CRNN_MODEL_PATH", "")):
        raw_text = str(raw or "").strip()
        if raw_text:
            candidates.append(raw_text)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.extend(
        [
            os.path.join(root, "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
            os.path.join(root, "assets", "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
            os.path.join(root, "ml_workspace", "models", "coarse_crnn", DEFAULT_OTO_CRNN_MODEL_NAME),
        ]
    )
    for candidate in candidates:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
        if os.path.isfile(expanded):
            return expanded
    return ""


def _prepare_prediction_rows(
    *,
    wav_dir: str,
    source_oto_path: str,
    special_aliases: frozenset[str] | None = None,
) -> tuple[list[_PredictRow], int, set[str]]:
    wav_root = os.path.abspath(str(wav_dir or "").strip())
    exact, normalized = _build_wav_lookup(wav_root)
    raw_rows = []
    total = 0
    missing: set[str] = set()
    for raw in read_text_with_fallback(source_oto_path).splitlines():
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        total += 1
        wav_name = os.path.basename(str(parsed.get("wav", "") or "").replace("\\", "/").strip())
        mapped = exact.get(wav_name.lower(), "")
        if not mapped:
            mapped = normalized.get(normalize_wav_key(wav_name), "")
        if not mapped:
            missing.add(wav_name)
            continue
        raw_rows.append(
            {
                "wav_rel": mapped,
                "wav_abs": os.path.join(wav_root, mapped.replace("/", os.sep)),
                "alias": str(parsed.get("alias", "") or "").strip(),
                "base_params": _normalize_base_params(parsed),
            }
        )

    special_set: frozenset[str] = special_aliases or frozenset()

    per_wav: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        per_wav.setdefault(row["wav_rel"], []).append(row)

    out: list[_PredictRow] = []
    for wav_rel, items in per_wav.items():
        count = len(items)
        for idx, row in enumerate(items):
            prev_alias = str(items[idx - 1].get("alias", "") or "") if idx > 0 else ""
            next_alias = str(items[idx + 1].get("alias", "") or "") if idx + 1 < count else ""
            alias_text = str(row["alias"])
            out.append(
                _PredictRow(
                    wav_rel=wav_rel,
                    wav_abs=str(row["wav_abs"]),
                    alias=alias_text,
                    prev_alias=prev_alias,
                    next_alias=next_alias,
                    row_index=idx,
                    row_count=count,
                    is_special=_alias_is_special(alias_text, special_set),
                    prev_is_special=_alias_is_special(prev_alias, special_set),
                    next_is_special=_alias_is_special(next_alias, special_set),
                    base_params=dict(row.get("base_params") or {}) or None,
                )
            )
    return out, total, missing


def _normalize_special_aliases(
    special_aliases: set[str] | list[str] | tuple[str, ...] | None,
) -> frozenset[str]:
    if not special_aliases:
        return frozenset()
    cleaned: set[str] = set()
    for item in special_aliases:
        text = str(item or "").strip().lower()
        if text:
            cleaned.add(text)
    return frozenset(cleaned)


def _alias_is_special(alias: str, special_set: frozenset[str]) -> bool:
    if not special_set:
        return False
    text = str(alias or "").strip().lower()
    if not text:
        return False
    if text in special_set:
        return True
    # Match by individual token so a single special phoneme like "tt" tags
    # any alias that contains it (e.g. "a tt", "tt a", "a tt a").
    for token in text.split():
        if token in special_set:
            return True
    return False


def _build_wav_lookup(wav_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    if not os.path.isdir(wav_dir):
        return exact, normalized
    for current_root, _dirs, files in os.walk(wav_dir):
        for file_name in files:
            if not str(file_name).lower().endswith(".wav"):
                continue
            abs_path = os.path.join(current_root, file_name)
            rel_path = os.path.relpath(abs_path, wav_dir).replace("\\", "/")
            exact.setdefault(str(file_name).strip().lower(), rel_path)
            norm = normalize_wav_key(file_name)
            if norm:
                normalized.setdefault(norm, rel_path)
    return exact, normalized


def _apply_conservative_right_boundary_guard(
    params: dict[str, float],
    *,
    language: str,
    alias: str,
    duration_ms: float,
    is_special: bool = False,
) -> tuple[dict[str, float], bool]:
    if str(os.environ.get("UTOA_OTO_CRNN_RIGHT_GUARD_ENABLE", "1")).strip().lower() in {"0", "false", "off", "no"}:
        return dict(params), False

    out = {key: float(value) for key, value in dict(params).items()}
    pre = max(0.0, float(out.get("preutterance", 0.0)))
    ovl = max(0.0, min(float(out.get("overlap", 0.0)), pre))
    out["overlap"] = ovl

    alias_type = _safe_alias_type(language, alias)
    role = _safe_alias_role(language, alias, alias_type=alias_type, is_special=bool(is_special))
    diphthong = _safe_is_diphthong(language, alias)
    max_cons_gap, min_cons_gap, max_cut_gap, min_cut_gap = _right_guard_limits_for_role(
        role,
        alias_type=alias_type,
        is_diphthong=diphthong,
        is_special=bool(is_special),
    )
    max_cons_gap = _env_float("UTOA_OTO_CRNN_MAX_CONS_GAP_MS", max_cons_gap)
    max_cut_gap = _env_float("UTOA_OTO_CRNN_MAX_CUTOFF_GAP_MS", max_cut_gap)

    cons_raw = max(0.0, float(out.get("consonant", 0.0)))
    cons_min = pre + max(1.0, min_cons_gap)
    cons_max = pre + max(cons_min - pre, max_cons_gap)
    cons = _clamp(cons_raw, cons_min, cons_max)

    cutoff_mag_raw = abs(float(out.get("cutoff", 0.0)))
    cutoff_min = cons + max(1.0, min_cut_gap)
    cutoff_max = cons + max(cutoff_min - cons, max_cut_gap)
    offset = max(0.0, float(out.get("offset", 0.0)))
    if duration_ms > 1.0:
        cutoff_max = min(cutoff_max, max(cutoff_min, float(duration_ms) - offset - 1.0))
    cutoff_mag = _clamp(cutoff_mag_raw, cutoff_min, cutoff_max)

    out["consonant"] = cons
    out["cutoff"] = -cutoff_mag
    changed = (
        abs(cons - cons_raw) > 1e-6
        or abs(cutoff_mag - cutoff_mag_raw) > 1e-6
        or abs(ovl - float(params.get("overlap", 0.0))) > 1e-6
    )
    return out, changed


def _right_guard_limits(alias_type: str) -> tuple[float, float, float, float]:
    kind = str(alias_type or "").strip().lower()
    # return max_cons_gap, min_cons_gap, max_cut_gap, min_cut_gap
    if kind == "cv_head":
        return 92.0, 18.0, 150.0, 36.0
    if kind == "cv":
        return 78.0, 16.0, 130.0, 34.0
    if kind == "vcv":
        return 82.0, 16.0, 118.0, 30.0
    if kind == "vc":
        return 48.0, 10.0, 82.0, 22.0
    if kind == "vv":
        return 92.0, 16.0, 142.0, 32.0
    if kind == "br":
        return 42.0, 8.0, 72.0, 18.0
    return 82.0, 14.0, 128.0, 30.0


# Role-axis right-boundary limits. Mirrors `_right_guard_limits` shape but
# branches on alias_role so the same row gets the same guard regardless of
# voicebank format wrapper.
_ROLE_GUARD_LIMITS: dict[str, tuple[float, float, float, float]] = {
    "-cv":  (92.0, 18.0, 150.0, 36.0),
    "cv":   (78.0, 16.0, 130.0, 34.0),
    "cv-":  (60.0, 10.0, 110.0, 24.0),
    "v":    (36.0,  4.0,  92.0, 14.0),
    "v-":   (36.0,  4.0, 104.0, 18.0),
    "vc":   (48.0, 10.0,  82.0, 22.0),
    "vv":   (72.0, 14.0, 116.0, 26.0),
    "v-cv": (82.0, 16.0, 118.0, 30.0),
    "endbr":(40.0,  6.0,  78.0, 16.0),
    "br":   (34.0,  4.0,  64.0, 12.0),
    "other":(82.0, 14.0, 128.0, 30.0),
    "special":(88.0, 18.0, 134.0, 32.0),
}


def _right_guard_limits_for_role(
    role: str,
    *,
    alias_type: str = "",
    is_diphthong: bool = False,
    is_special: bool = False,
) -> tuple[float, float, float, float]:
    role_text = "special" if bool(is_special) else normalize_role(role)
    if role_text in _ROLE_GUARD_LIMITS:
        max_cons, min_cons, max_cut, min_cut = _ROLE_GUARD_LIMITS[role_text]
    else:
        # Fall back to the legacy alias_type table for any role this code
        # path hasn't enumerated yet — keeps unknown rows under the old guard.
        max_cons, min_cons, max_cut, min_cut = _right_guard_limits(alias_type)
    if bool(is_diphthong) and role_text in {"cv", "-cv", "cv-", "v-cv", "vv"}:
        max_cons *= 1.30
        max_cut *= 1.20
    if bool(is_special) and role_text in {"v", "v-"}:
        cv_max, cv_min, _cut_max, _cut_min = _ROLE_GUARD_LIMITS["cv"]
        max_cons = max(max_cons, cv_max)
        min_cons = max(min_cons, cv_min)
    return max_cons, min_cons, max_cut, min_cut


def _safe_alias_type(language: str, alias: str) -> str:
    try:
        return str(classify_alias_type(language, alias) or "other").strip().lower() or "other"
    except Exception:
        text = str(alias or "").strip()
        if text.startswith("- "):
            return "cv_head"
        if " " in text:
            return "vcv"
        return "cv" if text else "other"


def _safe_alias_role(
    language: str,
    alias: str,
    *,
    alias_type: str = "",
    is_special: bool = False,
) -> str:
    try:
        return classify_alias_role(
            language,
            alias,
            alias_type=alias_type,
            is_special=bool(is_special),
        )
    except Exception:
        return "special" if bool(is_special) else "other"


def _safe_is_diphthong(language: str, alias: str) -> bool:
    try:
        return bool(_is_diphthong(language, alias))
    except Exception:
        return False


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
    return raw not in {"0", "false", "off", "no"}


def _normalize_base_params(parsed: dict[str, Any] | None) -> dict[str, float]:
    row = dict(parsed or {})
    return {
        "offset": float(row.get("offset", 0.0) or 0.0),
        "consonant": float(row.get("cons", 0.0) or 0.0),
        "cutoff": float(row.get("cutoff", 0.0) or 0.0),
        "preutterance": float(row.get("pre", 0.0) or 0.0),
        "overlap": float(row.get("ovl", 0.0) or 0.0),
    }


def _apply_low_confidence_fallback(
    *,
    predicted_params: dict[str, float],
    predicted_confidence: float,
    predicted_error_ms: float | None,
    predicted_low_confidence: bool,
    confidence_components: dict[str, float] | None,
    base_params: dict[str, float] | None,
    language: str,
    alias: str,
    duration_ms: float,
    is_special: bool,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_LOW_CONF_FALLBACK_ENABLE", True):
        return dict(predicted_params), ""
    if not base_params:
        return dict(predicted_params), ""
    min_conf = _env_float("UTOA_OTO_CRNN_LOW_CONF_MIN_CONFIDENCE", 0.35)
    max_error_ms = _env_float("UTOA_OTO_CRNN_LOW_CONF_MAX_PREDICTED_ERROR_MS", 450.0)
    trigger_conf = float(predicted_confidence) < float(min_conf)
    calibrated_error_ms = None
    if predicted_error_ms is not None:
        calibrated_error_ms = float(predicted_error_ms)
    trigger_err = calibrated_error_ms is not None and float(calibrated_error_ms) > float(max_error_ms)
    if trigger_err and _env_bool("UTOA_OTO_CRNN_LOW_CONF_REQUIRE_LOW_HEATMAP_FOR_ERROR", True):
        heatmap_conf = None
        try:
            heatmap_conf = float(dict(confidence_components or {}).get("heatmap"))
        except Exception:
            heatmap_conf = None
        if heatmap_conf is not None and heatmap_conf >= _env_float("UTOA_OTO_CRNN_LOW_CONF_HEATMAP_MAX_FOR_ERROR", 0.35):
            trigger_err = False
    trigger_model = bool(predicted_low_confidence) and _env_bool("UTOA_OTO_CRNN_LOW_CONF_USE_MODEL_FLAG", False)
    trigger = bool(trigger_model or trigger_conf or trigger_err)
    if not trigger:
        return dict(predicted_params), ""

    base = {key: float(value) for key, value in dict(base_params).items()}
    model = {key: float(value) for key, value in dict(predicted_params).items()}
    conf_gap = max(0.0, float(min_conf) - float(predicted_confidence))
    conf_denom = max(float(min_conf), 1e-6)
    strength = min(1.0, conf_gap / conf_denom)
    base_weight = _clamp(0.70 + (0.30 * strength), 0.70, 1.00)
    fused = {
        "offset": (model["offset"] * (1.0 - base_weight)) + (base["offset"] * base_weight),
        "consonant": (model["consonant"] * (1.0 - base_weight)) + (base["consonant"] * base_weight),
        "cutoff": (model["cutoff"] * (1.0 - base_weight)) + (base["cutoff"] * base_weight),
        "preutterance": (model["preutterance"] * (1.0 - base_weight)) + (base["preutterance"] * base_weight),
        "overlap": (model["overlap"] * (1.0 - base_weight)) + (base["overlap"] * base_weight),
    }
    guarded, _ = _apply_conservative_right_boundary_guard(
        fused,
        language=language,
        alias=alias,
        duration_ms=duration_ms,
        is_special=is_special,
    )
    reason = "low_confidence"
    if trigger_err and trigger_conf:
        reason = "low_confidence+high_error"
    elif trigger_err:
        reason = "high_predicted_error"
    return guarded, reason


def _resolve_effective_format(
    *,
    language: str,
    requested_format: str,
    aliases: list[str],
    callback: Callable[[str], None] | None = None,
) -> str:
    req = str(requested_format or "").strip().lower()
    if req and req not in {"other", "general", "auto"}:
        _log(callback, f"[OTO-CRNN] format={req} (requested)")
        return req
    detected = ""
    try:
        detected = str(detect_format_type(str(language or "").strip().lower(), list(aliases or [])) or "").strip().lower()
    except Exception:
        detected = ""
    if detected:
        _log(callback, f"[OTO-CRNN] format={detected} (detected)")
        return detected
    fallback = "general" if req in {"", "auto"} else (req or "general")
    _log(callback, f"[OTO-CRNN] format={fallback} (fallback)")
    return fallback


def _apply_activity_window_fallback(
    *,
    predicted_anchors: dict[str, float],
    predicted_params: dict[str, float],
    base_params: dict[str, float] | None,
    wav_path: str,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
    row_index: int = 0,
    row_count: int = 1,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_ACTIVITY_FALLBACK_ENABLE", True):
        return dict(predicted_params), ""
    profile = _analyze_activity_profile(wav_path, sample_rate=sample_rate, cache=cache)
    if not profile:
        return dict(predicted_params), ""
    row_shifted = _shift_params_into_row_activity_window(
        predicted_anchors=predicted_anchors,
        duration_ms=float(profile.get("duration_ms", 0.0) or 0.0),
        active_start_ms=float(profile.get("active_start_ms", 0.0) or 0.0),
        active_end_ms=float(profile.get("active_end_ms", 0.0) or 0.0),
        row_index=row_index,
        row_count=row_count,
    )
    if row_shifted:
        return row_shifted, "activity_row_shift"
    if not _is_anchor_outside_activity(predicted_anchors, profile):
        return dict(predicted_params), ""
    shifted = _shift_params_into_activity_window(
        predicted_anchors=predicted_anchors,
        duration_ms=float(profile.get("duration_ms", 0.0) or 0.0),
        active_start_ms=float(profile.get("active_start_ms", 0.0) or 0.0),
        active_end_ms=float(profile.get("active_end_ms", 0.0) or 0.0),
    )
    if shifted:
        return shifted, "activity_shift"
    if base_params and _env_bool("UTOA_OTO_CRNN_ACTIVITY_ALLOW_BASE_FALLBACK", False):
        return {key: float(value) for key, value in dict(base_params).items()}, "activity_outlier"
    return dict(predicted_params), ""


def _apply_audio_candidate_snap(
    *,
    predicted_params: dict[str, float],
    wav_path: str,
    language: str,
    alias: str,
    format_type: str = "",
    duration_ms: float,
    cache: dict[str, AudioCandidates | None],
    sequence_state: dict[str, int] | None = None,
    config: Any | None = None,
    is_special: bool = False,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ENABLE", True):
        return dict(predicted_params), ""
    role = _safe_alias_role(language, alias, alias_type=_safe_alias_type(language, alias), is_special=bool(is_special))
    allowed_roles = _env_csv_set(
        "UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ROLES",
        {"-cv", "cv", "cv-", "v-cv", "special"},
    )
    role_text = "special" if bool(is_special) else normalize_role(role)
    if role_text not in allowed_roles:
        return dict(predicted_params), ""
    format_text = _normalize_format_key(format_type) or _normalize_format_key(
        _infer_alias_format_for_runtime(language=language, alias=alias)
    )
    blocked_pairs = _env_pair_set(
        "UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_BLOCK_FORMAT_ROLES",
        {("cvc", "v-cv"), ("cvvc", "v-cv"), ("vcv", "-cv")},
    )
    if (format_text, role_text) in blocked_pairs:
        return dict(predicted_params), ""

    candidates = _get_audio_candidates(wav_path, cache=cache, config=config)
    if candidates is None:
        return dict(predicted_params), ""

    try:
        anchors = oto_params_to_anchors(
            offset=float(predicted_params.get("offset", 0.0) or 0.0),
            consonant=float(predicted_params.get("consonant", 0.0) or 0.0),
            cutoff=float(predicted_params.get("cutoff", 0.0) or 0.0),
            preutterance=float(predicted_params.get("preutterance", 0.0) or 0.0),
            overlap=float(predicted_params.get("overlap", 0.0) or 0.0),
            duration_ms=float(duration_ms),
        )
    except Exception:
        return dict(predicted_params), ""

    target_ms = float(anchors.preutterance)
    max_delta = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_MAX_DELTA_MS", 55.0))
    min_strength = _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_MIN_STRENGTH", 0.20)
    active_start = float(getattr(candidates, "active_start_ms", 0.0) or 0.0)
    active_end = float(getattr(candidates, "active_end_ms", duration_ms) or duration_ms)
    active_pad = _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_ACTIVE_PAD_MS", 80.0)
    sequence_enabled = _env_bool("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_ENABLE", True)
    sequence_key = os.path.abspath(str(wav_path or "")).lower()
    min_peak_index = -1
    if sequence_enabled and sequence_state is not None:
        min_peak_index = int(sequence_state.get(sequence_key, -1))
        max_delta = max(0.0, _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_MAX_DELTA_MS", 100.0))

    peaks = []
    for peak_index, peak in enumerate(getattr(candidates, "onset_peaks", []) or []):
        if sequence_enabled and peak_index <= min_peak_index:
            continue
        time_ms = float(getattr(peak, "time_ms", 0.0) or 0.0)
        strength = float(getattr(peak, "strength", 0.0) or 0.0)
        if strength < min_strength:
            continue
        if time_ms < active_start - active_pad or time_ms > active_end + active_pad:
            continue
        delta = time_ms - target_ms
        if abs(delta) <= max_delta:
            if sequence_enabled:
                skipped = max(0, int(peak_index) - int(min_peak_index) - 1)
                jump_penalty = _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_JUMP_PENALTY", 0.20)
                score = abs(delta) - (30.0 * strength) + (jump_penalty * float(skipped))
            else:
                score = abs(delta)
            peaks.append((score, -strength, peak_index, time_ms, strength))
    if not peaks:
        return dict(predicted_params), ""

    _score, _neg_strength, peak_index, candidate_ms, strength = min(peaks)
    if sequence_enabled:
        blend = _clamp(_env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SEQUENCE_BLEND", 0.40), 0.0, 1.0)
    else:
        blend = _clamp(_env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_BLEND", 0.65), 0.0, 1.0)
    delta = (float(candidate_ms) - target_ms) * blend
    if abs(delta) < _env_float("UTOA_OTO_CRNN_AUDIO_CANDIDATE_SNAP_MIN_MOVE_MS", 4.0):
        return dict(predicted_params), ""

    shifted = OtoAnchors(
        offset=anchors.offset + delta,
        overlap=anchors.overlap + delta,
        preutterance=anchors.preutterance + delta,
        consonant=anchors.consonant + delta,
        cutoff=anchors.cutoff + delta,
    )
    if sequence_enabled and sequence_state is not None:
        sequence_state[sequence_key] = int(peak_index)
    return anchors_to_oto_params(shifted, duration_ms=float(duration_ms)), f"candidate_onset:{strength:.3f}"


def _get_audio_candidates(
    wav_path: str,
    *,
    cache: dict[str, AudioCandidates | None],
    config: Any | None = None,
) -> AudioCandidates | None:
    key = os.path.abspath(str(wav_path or "")).lower()
    if key in cache:
        return cache[key]
    try:
        candidates = compute_audio_candidates(
            str(wav_path),
            target_sr=int(getattr(config, "sample_rate", 16000) or 16000),
            n_mels=int(getattr(config, "n_mels", 64) or 64),
            frame_ms=float(getattr(config, "frame_ms", 25.0) or 25.0),
            hop_ms=float(getattr(config, "hop_ms", 10.0) or 10.0),
        )
    except Exception:
        candidates = None
    cache[key] = candidates
    return candidates


def _env_csv_set(name: str, default: set[str]) -> set[str]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return set(default)
    out = {normalize_role(item.strip()) for item in raw.split(",") if item.strip()}
    return out or set(default)


def _env_pair_set(name: str, default: set[tuple[str, str]]) -> set[tuple[str, str]]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return set(default)
    out: set[tuple[str, str]] = set()
    for item in raw.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if "|" not in text:
            continue
        left, right = text.split("|", 1)
        fmt = _normalize_format_key(left)
        role = normalize_role(right)
        if fmt and role:
            out.add((fmt, role))
    return out


def _normalize_format_key(value: object) -> str:
    return str(value or "").strip().lower() or "unknown"


def _infer_alias_format_for_runtime(*, language: str, alias: str) -> str:
    # Runtime generation already resolved one format for the whole batch, but
    # snap risk is role-and-bank-format dependent. Alias text gives a useful
    # local fallback when the batch format is broad or absent.
    try:
        return str(detect_format_type(str(language or "").strip().lower(), [str(alias or "")]) or "").strip().lower()
    except Exception:
        return ""


def _shift_params_into_row_activity_window(
    *,
    predicted_anchors: dict[str, float],
    duration_ms: float,
    active_start_ms: float,
    active_end_ms: float,
    row_index: int,
    row_count: int,
) -> dict[str, float]:
    count = max(1, int(row_count))
    if count <= 1 or not _env_bool("UTOA_OTO_CRNN_ACTIVITY_ROW_SHIFT_ENABLE", False):
        return {}
    duration = max(1.0, float(duration_ms))
    active_start = max(0.0, float(active_start_ms))
    active_end = min(duration, max(active_start, float(active_end_ms)))
    active_span = active_end - active_start
    if active_span <= 40.0:
        return {}

    idx = max(0, min(count - 1, int(row_index)))
    row_width = active_span / float(count)
    if row_width < _env_float("UTOA_OTO_CRNN_ACTIVITY_ROW_MIN_WIDTH_MS", 35.0):
        return {}
    row_start = active_start + (float(idx) * row_width)
    row_end = active_start + (float(idx + 1) * row_width)
    tolerance = _env_float("UTOA_OTO_CRNN_ACTIVITY_ROW_TOLERANCE_MS", 90.0)
    mid = _anchor_mid_ms(predicted_anchors)
    if row_start - tolerance <= mid <= row_end + tolerance:
        return {}

    inset_ratio = _clamp(_env_float("UTOA_OTO_CRNN_ACTIVITY_ROW_TARGET_RATIO", 0.35), 0.05, 0.95)
    target_mid = row_start + (row_width * inset_ratio)
    return _shift_params_to_target_mid(
        predicted_anchors=predicted_anchors,
        duration_ms=duration,
        target_mid_ms=target_mid,
    )


def _shift_params_into_activity_window(
    *,
    predicted_anchors: dict[str, float],
    duration_ms: float,
    active_start_ms: float,
    active_end_ms: float,
) -> dict[str, float]:
    duration = max(1.0, float(duration_ms))
    start_ms = max(0.0, float(active_start_ms))
    end_ms = min(duration, max(start_ms, float(active_end_ms)))
    if end_ms <= start_ms + 20.0:
        return {}
    mid = _anchor_mid_ms(predicted_anchors)
    tolerance_ms = _env_float("UTOA_OTO_CRNN_ACTIVITY_EARLY_TOLERANCE_MS", 80.0)
    if mid >= start_ms - max(0.0, float(tolerance_ms)):
        return {}
    target_mid = start_ms + _env_float("UTOA_OTO_CRNN_ACTIVITY_TARGET_INSET_MS", 35.0)
    return _shift_params_to_target_mid(
        predicted_anchors=predicted_anchors,
        duration_ms=duration,
        target_mid_ms=target_mid,
        max_shift_ms=_env_float("UTOA_OTO_CRNN_ACTIVITY_MAX_SHIFT_MS", 3000.0),
    )


def _shift_params_to_target_mid(
    *,
    predicted_anchors: dict[str, float],
    duration_ms: float,
    target_mid_ms: float,
    max_shift_ms: float | None = None,
) -> dict[str, float]:
    duration = max(1.0, float(duration_ms))
    anchors = {
        "offset": float(predicted_anchors.get("offset", 0.0) or 0.0),
        "overlap": float(predicted_anchors.get("overlap", predicted_anchors.get("offset", 0.0)) or 0.0),
        "preutterance": float(predicted_anchors.get("preutterance", 0.0) or 0.0),
        "consonant": float(predicted_anchors.get("consonant", 0.0) or 0.0),
        "cutoff": float(predicted_anchors.get("cutoff", 0.0) or 0.0),
    }
    mid = (anchors["preutterance"] + anchors["consonant"]) * 0.5
    delta = float(target_mid_ms) - float(mid)
    if abs(delta) <= 1e-6:
        return {}
    if max_shift_ms is not None:
        cap = max(0.0, float(max_shift_ms))
        delta = _clamp(delta, -cap, cap)
    if delta > 0.0:
        delta = min(delta, max(0.0, duration - anchors["cutoff"] - 1.0))
    else:
        delta = max(delta, -min(anchors.values()))
    if abs(delta) <= 1e-6:
        return {}
    shifted = OtoAnchors(
        offset=anchors["offset"] + delta,
        overlap=anchors["overlap"] + delta,
        preutterance=anchors["preutterance"] + delta,
        consonant=anchors["consonant"] + delta,
        cutoff=anchors["cutoff"] + delta,
    )
    return anchors_to_oto_params(shifted, duration_ms=duration)


def _anchor_mid_ms(predicted_anchors: dict[str, float]) -> float:
    pre = float(predicted_anchors.get("preutterance", 0.0) or 0.0)
    cons = float(predicted_anchors.get("consonant", 0.0) or 0.0)
    return (pre + cons) * 0.5


def _analyze_activity_profile(
    wav_path: str,
    *,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    key = os.path.abspath(str(wav_path or "")).lower()
    if key in cache:
        return dict(cache[key])
    try:
        samples, sr, duration_sec = load_wav_mono(str(wav_path), target_sr=int(sample_rate))
    except Exception:
        cache[key] = {}
        return {}
    duration_ms = max(1.0, float(duration_sec) * 1000.0)
    if samples is None or int(getattr(samples, "shape", [0])[0]) <= 0 or sr <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    data = np.asarray(samples, dtype=np.float32)
    if data.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)

    window_ms = _env_float("UTOA_OTO_CRNN_ACTIVITY_WINDOW_MS", 15.0)
    window_samples = max(1, int(round(float(sr) * max(1.0, float(window_ms)) / 1000.0)))
    if data.size <= window_samples:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)

    stride_samples = max(1, window_samples // 2)
    frame_count = 1 + int((data.size - window_samples) // stride_samples)
    rms = np.empty((frame_count,), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * stride_samples
        frame = data[start : start + window_samples]
        rms[idx] = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0

    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 1e-8:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    low = float(np.percentile(rms, _env_float("UTOA_OTO_CRNN_ACTIVITY_NOISE_PERCENTILE", 20.0)))
    high = float(np.percentile(rms, _env_float("UTOA_OTO_CRNN_ACTIVITY_SPEECH_PERCENTILE", 88.0)))
    dynamic = low + ((max(high, peak) - low) * _env_float("UTOA_OTO_CRNN_ACTIVITY_DYNAMIC_RATIO", 0.35))
    threshold = max(dynamic, peak * _env_float("UTOA_OTO_CRNN_ACTIVITY_PEAK_RATIO", 0.04), _env_float("UTOA_OTO_CRNN_ACTIVITY_ABS_FLOOR", 1e-5))
    min_run = max(1, int(round(_env_float("UTOA_OTO_CRNN_ACTIVITY_MIN_ACTIVE_MS", 30.0) / ((float(stride_samples) / float(sr)) * 1000.0))))
    active_frames = _sustained_frame_indices(rms >= threshold, min_run)
    if active_frames.size <= 0:
        threshold = max(low + ((peak - low) * 0.20), _env_float("UTOA_OTO_CRNN_ACTIVITY_ABS_FLOOR", 1e-5))
        active_frames = _sustained_frame_indices(rms >= threshold, min_run)
    if active_frames.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)

    stride_ms = (float(stride_samples) / float(sr)) * 1000.0
    start_ms = float(active_frames[0]) * stride_ms
    end_ms = (float(active_frames[-1]) * stride_ms) + float(window_ms)
    pad_ms = 25.0
    active_start = max(0.0, start_ms - pad_ms)
    active_end = min(duration_ms, end_ms + pad_ms)
    if active_end - active_start < 40.0:
        active_start = 0.0
        active_end = duration_ms
    profile = {
        "active_start_ms": float(active_start),
        "active_end_ms": float(active_end),
        "duration_ms": float(duration_ms),
    }
    cache[key] = profile
    return dict(profile)


def _sustained_frame_indices(mask: np.ndarray, min_run: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    run = max(1, int(min_run))
    if arr.size <= 0:
        return np.asarray([], dtype=np.int64)
    if run <= 1:
        return np.flatnonzero(arr)
    count = 0
    first = -1
    last = -1
    active = False
    for idx, value in enumerate(arr):
        if bool(value):
            count += 1
            if count >= run and not active:
                first = idx - run + 1
                active = True
            if active:
                last = idx
        else:
            count = 0
            if active:
                break
    if first < 0 or last < first:
        return np.asarray([], dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


def _is_anchor_outside_activity(predicted_anchors: dict[str, float], profile: dict[str, float]) -> bool:
    start_ms = float(profile.get("active_start_ms", 0.0) or 0.0)
    end_ms = float(profile.get("active_end_ms", 0.0) or 0.0)
    if end_ms <= start_ms + 20.0:
        return False
    offset = float(predicted_anchors.get("offset", 0.0) or 0.0)
    pre = float(predicted_anchors.get("preutterance", 0.0) or 0.0)
    cons = float(predicted_anchors.get("consonant", 0.0) or 0.0)
    cutoff = float(predicted_anchors.get("cutoff", 0.0) or 0.0)
    mid = (pre + cons) * 0.5
    if mid < (start_ms - 80.0) or mid > (end_ms + 80.0):
        return True
    if cutoff < (start_ms - 40.0):
        return True
    if offset > (end_ms + 40.0):
        return True
    return False


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _normalize_output_oto_path(out_path: str) -> str:
    raw = str(out_path or "").strip()
    if not raw:
        return ""
    if raw.endswith(("\\", "/")) or os.path.isdir(raw):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    return raw


def _normalize_alias_suffix(suffix: str) -> str:
    text = str(suffix or "").strip()
    if not text:
        return ""
    return text[1:] if text.startswith("_") else text


def _apply_alias_suffix(alias: str, suffix: str) -> str:
    text = str(alias or "").strip()
    if not text or not suffix:
        return text
    return f"{text}_{suffix}"


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callable(callback):
        callback(message)


__all__ = [
    "DEFAULT_OTO_CRNN_MODEL_NAME",
    "generate_oto_with_crnn_predictor",
    "resolve_oto_crnn_model_path",
]
