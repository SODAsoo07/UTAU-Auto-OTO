from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from core.coarse_crnn.alias_role import classify_alias_role, is_diphthong as _is_diphthong, normalize_role
from core.coarse_crnn.audio import load_wav_mono
from core.coarse_crnn.oto_inference import predict_oto_with_model
from core.coarse_crnn.oto_model import load_oto_checkpoint
from core.coarse_crnn.oto_targets import anchors_to_oto_params, oto_params_to_anchors
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
    kr_cvvc_role_fallback_count = 0
    silence_window_clamp_count = 0
    span_cap_count = 0
    onset_snap_count = 0
    activity_profile_cache: dict[str, dict[str, float]] = {}
    voicebank_activity_profile = _analyze_voicebank_activity_baseline(
        wav_dir=wav_dir,
        sample_rate=16000,
        cache=activity_profile_cache,
    )
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
                base_params=row.base_params,
                wav_path=row.wav_abs,
                sample_rate=16000,
                cache=activity_profile_cache,
                voicebank_profile=voicebank_activity_profile,
                language=lang,
                format_type=fmt,
                alias=row.alias,
                duration_ms=float(getattr(pred, "duration_ms", 0.0) or 0.0),
                is_special=row.is_special,
            )
            if fallback_reason:
                low_conf_fallback_count += 1
            pred_duration_ms = float(getattr(pred, "duration_ms", 0.0) or 0.0)
            predicted_anchors = _resolve_predicted_anchors(
                pred=pred,
                predicted_params=params,
                duration_ms=pred_duration_ms,
            )
            params, role_fallback_reason = _apply_korean_cvvc_role_fallback(
                predicted_anchors=predicted_anchors,
                predicted_params=params,
                base_params=row.base_params,
                wav_path=row.wav_abs,
                sample_rate=16000,
                cache=activity_profile_cache,
                voicebank_profile=voicebank_activity_profile,
                predicted_confidence=float(getattr(pred, "confidence", 0.0) or 0.0),
                predicted_error_ms=getattr(pred, "predicted_error_ms", None),
                predicted_low_confidence=bool(getattr(pred, "low_confidence", False)),
                language=lang,
                format_type=fmt,
                alias=row.alias,
                duration_ms=pred_duration_ms,
                is_special=row.is_special,
            )
            if role_fallback_reason:
                kr_cvvc_role_fallback_count += 1
                predicted_anchors = _resolve_predicted_anchors(
                    pred=pred,
                    predicted_params=params,
                    duration_ms=pred_duration_ms,
                )
            params, activity_reason = _apply_activity_window_fallback(
                predicted_anchors=predicted_anchors,
                predicted_params=params,
                base_params=row.base_params,
                wav_path=row.wav_abs,
                sample_rate=16000,
                cache=activity_profile_cache,
                voicebank_profile=voicebank_activity_profile,
                language=lang,
                format_type=fmt,
                alias=row.alias,
                duration_ms=pred_duration_ms,
                is_special=row.is_special,
            )
            if activity_reason:
                activity_fallback_count += 1
            params, clamp_reason = _apply_silence_window_clamp(
                predicted_params=params,
                wav_path=row.wav_abs,
                sample_rate=16000,
                cache=activity_profile_cache,
                voicebank_profile=voicebank_activity_profile,
                duration_ms=pred_duration_ms,
                language=lang,
                alias=row.alias,
                is_special=row.is_special,
            )
            if clamp_reason:
                silence_window_clamp_count += 1
            params, onset_reason = _apply_onset_voiced_snap(
                predicted_params=params,
                wav_path=row.wav_abs,
                sample_rate=16000,
                cache=activity_profile_cache,
                voicebank_profile=voicebank_activity_profile,
                duration_ms=pred_duration_ms,
                language=lang,
                alias=row.alias,
                is_special=row.is_special,
            )
            if onset_reason:
                onset_snap_count += 1
            params, span_reason = _apply_duration_span_cap(
                predicted_params=params,
                duration_ms=pred_duration_ms,
                language=lang,
                format_type=fmt,
                alias=row.alias,
                is_special=row.is_special,
            )
            if span_reason:
                span_cap_count += 1
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
    if kr_cvvc_role_fallback_count:
        _log(callback, f"[OTO-CRNN] korean-cvvc role fallback rows={kr_cvvc_role_fallback_count}/{len(out_lines)}")
    if activity_fallback_count:
        _log(callback, f"[OTO-CRNN] activity-window fallback rows={activity_fallback_count}/{len(out_lines)}")
    if silence_window_clamp_count:
        _log(callback, f"[OTO-CRNN] silence-window clamp rows={silence_window_clamp_count}/{len(out_lines)}")
    if onset_snap_count:
        _log(callback, f"[OTO-CRNN] onset-voiced snap rows={onset_snap_count}/{len(out_lines)}")
    if span_cap_count:
        _log(callback, f"[OTO-CRNN] duration-span cap rows={span_cap_count}/{len(out_lines)}")
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
        language=language,
        alias=alias,
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
    language: str = "",
    alias: str = "",
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
    # Tighten diphthong bounds by default to reduce stretched vowels.
    if bool(is_diphthong) and role_text in {"cv", "-cv", "cv-", "v-cv", "vv"}:
        max_cons *= _env_float("UTOA_OTO_CRNN_DIPHTHONG_CONS_CAP_GAIN", 0.92)
        max_cut *= _env_float("UTOA_OTO_CRNN_DIPHTHONG_CUTOFF_CAP_GAIN", 0.85)
    if bool(is_special) and role_text in {"v", "v-"}:
        cv_max, cv_min, _cut_max, _cut_min = _ROLE_GUARD_LIMITS["cv"]
        max_cons = max(max_cons, cv_max)
        min_cons = max(min_cons, cv_min)
    if _is_japanese_n_alias(alias=alias, language=language):
        max_cons = min(max_cons, _env_float("UTOA_OTO_CRNN_JP_N_MAX_CONS_GAP_MS", 42.0))
        max_cut = min(max_cut, _env_float("UTOA_OTO_CRNN_JP_N_MAX_CUTOFF_GAP_MS", 78.0))
        min_cons = max(min_cons, _env_float("UTOA_OTO_CRNN_JP_N_MIN_CONS_GAP_MS", 8.0))
        min_cut = max(min_cut, _env_float("UTOA_OTO_CRNN_JP_N_MIN_CUTOFF_GAP_MS", 20.0))
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


def _is_japanese_n_alias(*, alias: str, language: str) -> bool:
    lang = str(language or "").strip().lower()
    if not (lang.startswith("ja") or lang.startswith("jp") or lang == "japanese"):
        return False
    text = str(alias or "").strip()
    if not text:
        return False
    if text in {"ん", "ン", "N", "n"}:
        return True
    for token in text.replace("-", " ").split():
        if token in {"ん", "ン", "N", "n"}:
            return True
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


def _is_multi_signal_required(language: object, format_type: object) -> bool:
    """Plan B (0-order) slice-aware AND mode dispatcher.

    Returns True for slices where ``_apply_low_confidence_fallback`` should
    require BOTH a low-confidence signal AND a high-predicted-error signal
    before firing (i.e. confidence alone is not enough). False for slices
    where the legacy OR-trigger is preserved.

    Slice patterns are read from ``UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES``
    (comma-separated ``language/format`` pairs, ``*`` wildcard). Default
    list reflects the v2 0-order Plan B exploration result on baseline
    cvvcvcboost_e6: ``korean/cvc`` and ``japanese/cvvc`` are slices where
    multi-signal AND mode preserves 25-33%% of model output rows without
    breaching the effective preutterance accuracy floor. Korean CVVC stays
    in OR mode because its bad_rate (>=93%) leaves no headroom.
    """
    raw = str(os.environ.get("UTOA_OTO_CRNN_LOW_CONF_AND_MODE_SLICES", "") or "").strip()
    if raw == "":
        raw = "korean/cvc,japanese/cvvc"
    if raw.lower() in {"none", "off", "0", "false"}:
        return False
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    for pattern in raw.split(","):
        text = pattern.strip().lower()
        if not text:
            continue
        parts = text.split("/")
        while len(parts) < 2:
            parts.append("*")
        match_lang = parts[0] in {"*", lang}
        match_fmt = parts[1] in {"*", fmt}
        if match_lang and match_fmt:
            return True
    return False


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
    base_params: dict[str, float] | None,
    wav_path: str,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
    voicebank_profile: dict[str, float],
    language: str,
    format_type: str,
    alias: str,
    duration_ms: float,
    is_special: bool,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_LOW_CONF_FALLBACK_ENABLE", True):
        return dict(predicted_params), ""
    if not _base_params_fallback_usable(base_params):
        return dict(predicted_params), ""
    lang = str(language or "").strip().lower()
    min_conf = _env_float_lang("UTOA_OTO_CRNN_LOW_CONF_MIN_CONFIDENCE", lang, 0.54)
    max_error_ms = _env_float_lang("UTOA_OTO_CRNN_LOW_CONF_MAX_PREDICTED_ERROR_MS", lang, 95.0)
    use_model_flag = _env_bool("UTOA_OTO_CRNN_LOW_CONF_USE_MODEL_FLAG", False)
    trigger_conf = float(predicted_confidence) < float(min_conf)
    trigger_err = predicted_error_ms is not None and float(predicted_error_ms) > float(max_error_ms)
    trigger_model = bool(predicted_low_confidence) and bool(use_model_flag)
    # Plan B (0-order, [[CRNN-정확도-개선-방안-v2]]): in slice-AND mode, the
    # confidence-only branch is suppressed. Fallback fires only when (a) the
    # explicit model flag is on (rare; legacy override), (b) BOTH conf-low and
    # err-high agree, or (c) err-high alone (since predicted_error is a
    # learnt signal that already requires conf-head agreement implicitly via
    # the uncertainty/error fusion). This converts confidence into a
    # corroborating signal instead of a primary one for slices where the
    # confidence head is collapsed (good_mean ≈ bad_mean). The OR-mode is
    # preserved everywhere else for backward compatibility.
    if _is_multi_signal_required(language, format_type):
        trigger = bool(trigger_model or trigger_err or (trigger_conf and trigger_err))
    else:
        trigger = bool(trigger_model or trigger_conf or trigger_err)
    if not trigger:
        return dict(predicted_params), ""
    if not _source_oto_shape_usable(base_params, duration_ms=duration_ms):
        return dict(predicted_params), ""

    conf_gap = max(0.0, float(min_conf) - float(predicted_confidence))
    conf_denom = max(float(min_conf), 1e-6)
    conf_strength = min(1.0, conf_gap / conf_denom)
    err_strength = 0.0
    if predicted_error_ms is not None:
        err_band = max(10.0, float(max_error_ms) * 0.55)
        err_strength = _clamp((float(predicted_error_ms) - float(max_error_ms)) / err_band, 0.0, 1.0)
    strength = max(conf_strength, err_strength)
    min_base_weight = _clamp(_env_float_lang("UTOA_OTO_CRNN_LOW_CONF_BASE_WEIGHT_MIN", lang, 0.45), 0.05, 0.95)
    max_base_weight = _clamp(
        _env_float_lang("UTOA_OTO_CRNN_LOW_CONF_BASE_WEIGHT_MAX", lang, 0.82),
        min_base_weight,
        1.0,
    )
    base_weight = _clamp(
        min_base_weight + ((max_base_weight - min_base_weight) * strength),
        min_base_weight,
        max_base_weight,
    )
    fused = _blend_with_source_oto_shape(
        predicted_params=predicted_params,
        base_params=base_params,
        shape_weight=base_weight,
        cutoff_weight=_source_shape_cutoff_weight(
            shape_weight=base_weight,
            language=lang,
            format_type=format_type,
            alias=alias,
            is_special=is_special,
        ),
        duration_ms=duration_ms,
    )
    guarded, _ = _apply_conservative_right_boundary_guard(
        fused,
        language=language,
        alias=alias,
        duration_ms=duration_ms,
        is_special=is_special,
    )
    guarded, _ = _apply_duration_span_cap(
        predicted_params=guarded,
        duration_ms=duration_ms,
        language=lang,
        format_type=format_type,
        alias=alias,
        is_special=is_special,
    )
    reason = "low_confidence"
    if trigger_err and trigger_conf and trigger_model:
        reason = "low_confidence+high_error+model_flag"
    elif trigger_err and trigger_conf:
        reason = "low_confidence+high_error"
    elif trigger_err:
        reason = "high_predicted_error"
    elif trigger_model:
        reason = "low_confidence_model_flag"
    if _is_multi_signal_required(language, format_type):
        reason = f"{reason}+slice_and_mode"
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
    voicebank_profile: dict[str, float],
    language: str,
    format_type: str,
    alias: str,
    duration_ms: float,
    is_special: bool,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_ACTIVITY_FALLBACK_ENABLE", True):
        return dict(predicted_params), ""
    if not _base_params_fallback_usable(base_params):
        return dict(predicted_params), ""
    profile = _analyze_activity_profile(
        wav_path,
        sample_rate=sample_rate,
        cache=cache,
        voicebank_profile=voicebank_profile,
    )
    if not profile:
        return dict(predicted_params), ""
    if not _is_anchor_outside_activity(predicted_anchors, profile):
        return dict(predicted_params), ""
    if not _source_oto_shape_usable(base_params, duration_ms=duration_ms):
        return dict(predicted_params), ""
    lang = str(language or "").strip().lower()
    base_weight = _clamp(_env_float_lang("UTOA_OTO_CRNN_ACTIVITY_FALLBACK_BASE_WEIGHT", lang, 0.72), 0.20, 0.95)
    fused = _blend_with_source_oto_shape(
        predicted_params=predicted_params,
        base_params=base_params,
        shape_weight=base_weight,
        cutoff_weight=_source_shape_cutoff_weight(
            shape_weight=base_weight,
            language=lang,
            format_type=format_type,
            alias=alias,
            is_special=is_special,
        ),
        duration_ms=duration_ms,
    )
    guarded, _ = _apply_conservative_right_boundary_guard(
        fused,
        language=lang,
        alias=alias,
        duration_ms=max(1.0, float(duration_ms)),
        is_special=bool(is_special),
    )
    guarded, _ = _apply_duration_span_cap(
        predicted_params=guarded,
        duration_ms=max(1.0, float(duration_ms)),
        language=lang,
        format_type=format_type,
        alias=alias,
        is_special=bool(is_special),
    )
    return guarded, "activity_outlier"


def _apply_korean_cvvc_role_fallback(
    *,
    predicted_anchors: dict[str, float],
    predicted_params: dict[str, float],
    base_params: dict[str, float] | None,
    wav_path: str,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
    voicebank_profile: dict[str, float],
    predicted_confidence: float,
    predicted_error_ms: float | None,
    predicted_low_confidence: bool,
    language: str,
    format_type: str,
    alias: str,
    duration_ms: float,
    is_special: bool,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_KR_CVVC_ROLE_FALLBACK_ENABLE", True):
        return dict(predicted_params), ""
    if not _base_params_fallback_usable(base_params):
        return dict(predicted_params), ""
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    if not (lang.startswith("ko") or lang.startswith("kr") or lang == "korean"):
        return dict(predicted_params), ""
    if fmt != "cvvc":
        return dict(predicted_params), ""

    alias_type = _safe_alias_type(lang, alias)
    role = _safe_alias_role(lang, alias, alias_type=alias_type, is_special=bool(is_special))
    if role not in {"vc", "v-cv", "cv", "vv", "other", "-cv"}:
        return dict(predicted_params), ""

    profile = _analyze_activity_profile(
        wav_path,
        sample_rate=sample_rate,
        cache=cache,
        voicebank_profile=voicebank_profile,
    )
    activity_risk = bool(profile) and _is_anchor_outside_activity(predicted_anchors, profile)
    shift_risk = bool(profile) and _is_korean_cvvc_syllable_shift_risk(
        predicted_anchors=predicted_anchors,
        profile=profile,
        role=role,
    )
    min_conf = _env_float_lang("UTOA_OTO_CRNN_KR_CVVC_ROLE_MIN_CONFIDENCE", lang, 0.62)
    max_error_ms = _env_float_lang("UTOA_OTO_CRNN_KR_CVVC_ROLE_MAX_PREDICTED_ERROR_MS", lang, 85.0)
    trigger_conf = float(predicted_confidence) < float(min_conf)
    trigger_err = predicted_error_ms is not None and float(predicted_error_ms) > float(max_error_ms)
    use_model_flag = _env_bool("UTOA_OTO_CRNN_KR_CVVC_ROLE_USE_MODEL_FLAG", False)
    trigger_model = bool(predicted_low_confidence) and bool(use_model_flag)
    if not (activity_risk or shift_risk or trigger_conf or trigger_err or trigger_model):
        return dict(predicted_params), ""
    if not _source_oto_shape_usable(base_params, duration_ms=duration_ms):
        return dict(predicted_params), ""

    base_weight = _korean_cvvc_role_base_weight(role)
    if shift_risk or activity_risk:
        base_weight = max(
            base_weight,
            _env_float_lang("UTOA_OTO_CRNN_KR_CVVC_SHIFT_BASE_WEIGHT", lang, 0.88),
        )
    if trigger_err:
        base_weight = max(base_weight, _env_float_lang("UTOA_OTO_CRNN_KR_CVVC_ERROR_BASE_WEIGHT", lang, 0.82))
    base_weight = _clamp(base_weight, 0.35, 0.96)
    fused = _blend_with_source_oto_shape(
        predicted_params=predicted_params,
        base_params=base_params,
        shape_weight=base_weight,
        cutoff_weight=_source_shape_cutoff_weight(
            shape_weight=base_weight,
            language=lang,
            format_type=fmt,
            alias=alias,
            is_special=is_special,
            role=role,
        ),
        duration_ms=duration_ms,
    )
    guarded, _ = _apply_conservative_right_boundary_guard(
        fused,
        language=lang,
        alias=alias,
        duration_ms=max(1.0, float(duration_ms)),
        is_special=bool(is_special),
    )
    guarded, _ = _apply_duration_span_cap(
        predicted_params=guarded,
        duration_ms=max(1.0, float(duration_ms)),
        language=lang,
        format_type=fmt,
        alias=alias,
        is_special=bool(is_special),
    )
    reasons = [f"korean_cvvc_{role}_role"]
    if shift_risk:
        reasons.append("syllable_shift_guard")
    if activity_risk:
        reasons.append("activity_window_fallback")
    if trigger_err:
        reasons.append("high_predicted_error")
    if trigger_conf:
        reasons.append("low_confidence")
    if trigger_model:
        reasons.append("model_low_confidence")
    return guarded, "+".join(reasons)


def _korean_cvvc_role_base_weight(role: str) -> float:
    role_key = str(role or "other").strip().lower()
    defaults = {
        "vc": 0.86,
        "v-cv": 0.86,
        "other": 0.90,
        "cv": 0.76,
        "vv": 0.80,
        "-cv": 0.74,
    }
    env_name = "UTOA_OTO_CRNN_KR_CVVC_ROLE_BASE_WEIGHT_" + role_key.replace("-", "_").upper()
    return _clamp(_env_float(env_name, defaults.get(role_key, 0.78)), 0.35, 0.96)


def _is_korean_cvvc_syllable_shift_risk(
    *,
    predicted_anchors: dict[str, float],
    profile: dict[str, float],
    role: str,
) -> bool:
    start_ms = float(profile.get("active_start_ms", 0.0) or 0.0)
    end_ms = float(profile.get("active_end_ms", 0.0) or 0.0)
    if end_ms <= start_ms + 20.0:
        return False
    core_start = float(profile.get("active_core_start_ms", start_ms) or start_ms)
    core_end = float(profile.get("active_core_end_ms", end_ms) or end_ms)
    if core_end <= core_start + 20.0:
        core_start, core_end = start_ms, end_ms
    pre = float(predicted_anchors.get("preutterance", 0.0) or 0.0)
    cons = float(predicted_anchors.get("consonant", 0.0) or 0.0)
    cutoff = float(predicted_anchors.get("cutoff", 0.0) or 0.0)
    mid = (pre + cons) * 0.5
    role_key = str(role or "").strip().lower()
    margin = _env_float("UTOA_OTO_CRNN_KR_CVVC_SHIFT_MARGIN_MS", 52.0)
    if mid < core_start - margin or mid > core_end + margin:
        return True
    if role_key in {"vc", "v-cv"} and (pre > core_end - 14.0 or cons < core_start - 35.0):
        return True
    if role_key in {"cv", "other"} and pre > core_end - 20.0:
        return True
    if cutoff < core_start - 35.0:
        return True
    return False


def _apply_silence_window_clamp(
    *,
    predicted_params: dict[str, float],
    wav_path: str,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
    voicebank_profile: dict[str, float],
    duration_ms: float,
    language: str,
    alias: str,
    is_special: bool,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_SILENCE_CLAMP_ENABLE", True):
        return dict(predicted_params), ""
    profile = _analyze_activity_profile(
        wav_path,
        sample_rate=sample_rate,
        cache=cache,
        voicebank_profile=voicebank_profile,
    )
    if not profile:
        return dict(predicted_params), ""
    active_start = float(profile.get("active_start_ms", 0.0) or 0.0)
    active_end = float(profile.get("active_end_ms", 0.0) or 0.0)
    total_ms = max(1.0, float(duration_ms) or float(profile.get("duration_ms", 0.0) or 0.0))
    if active_end <= active_start + 25.0 or total_ms <= 1.0:
        return dict(predicted_params), ""

    params = {key: float(value) for key, value in dict(predicted_params).items()}
    anchors = oto_params_to_anchors(
        offset=params.get("offset", 0.0),
        consonant=params.get("consonant", 0.0),
        cutoff=params.get("cutoff", 0.0),
        preutterance=params.get("preutterance", 0.0),
        overlap=params.get("overlap", 0.0),
        duration_ms=total_ms,
    ).to_dict()
    if not _is_anchor_outside_activity(anchors, profile):
        return params, ""

    left_margin = _env_float("UTOA_OTO_CRNN_SILENCE_CLAMP_LEFT_MARGIN_MS", 35.0)
    right_margin = _env_float("UTOA_OTO_CRNN_SILENCE_CLAMP_RIGHT_MARGIN_MS", 45.0)
    max_shift = _env_float("UTOA_OTO_CRNN_SILENCE_CLAMP_MAX_SHIFT_MS", 280.0)
    lo = max(0.0, active_start - max(0.0, left_margin))
    hi = min(total_ms, active_end + max(0.0, right_margin))
    if hi <= lo + 30.0:
        return params, ""

    mid = (float(anchors["preutterance"]) + float(anchors["consonant"])) * 0.5
    target_mid = _clamp(mid, lo + 12.0, hi - 12.0)
    shift = _clamp(target_mid - mid, -abs(max_shift), abs(max_shift))
    if abs(shift) < 1e-3:
        return params, ""

    shifted = {
        "offset": float(anchors["offset"]) + shift,
        "overlap": float(anchors["overlap"]) + shift,
        "preutterance": float(anchors["preutterance"]) + shift,
        "consonant": float(anchors["consonant"]) + shift,
        "cutoff": float(anchors["cutoff"]) + shift,
    }
    clamped = anchors_to_oto_params(shifted, duration_ms=total_ms)
    guarded, _ = _apply_conservative_right_boundary_guard(
        clamped,
        language=language,
        alias=alias,
        duration_ms=total_ms,
        is_special=is_special,
    )
    return guarded, "silence_window_clamp"


def _apply_onset_voiced_snap(
    *,
    predicted_params: dict[str, float],
    wav_path: str,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
    voicebank_profile: dict[str, float],
    duration_ms: float,
    language: str,
    alias: str,
    is_special: bool,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_ONSET_VOICED_SNAP_ENABLE", True):
        return dict(predicted_params), ""
    profile = _analyze_activity_profile(
        wav_path,
        sample_rate=sample_rate,
        cache=cache,
        voicebank_profile=voicebank_profile,
    )
    if not profile:
        return dict(predicted_params), ""
    onset_ms = float(profile.get("onset_ms", -1.0) or -1.0)
    active_start = float(profile.get("active_start_ms", 0.0) or 0.0)
    active_end = float(profile.get("active_end_ms", 0.0) or 0.0)
    total_ms = max(1.0, float(duration_ms) or float(profile.get("duration_ms", 0.0) or 0.0))
    if onset_ms < 0.0 or active_end <= active_start + 20.0 or total_ms <= 1.0:
        return dict(predicted_params), ""

    lang = str(language or "").strip().lower()
    params = {key: float(value) for key, value in dict(predicted_params).items()}
    anchors = oto_params_to_anchors(
        offset=params.get("offset", 0.0),
        consonant=params.get("consonant", 0.0),
        cutoff=params.get("cutoff", 0.0),
        preutterance=params.get("preutterance", 0.0),
        overlap=params.get("overlap", 0.0),
        duration_ms=total_ms,
    ).to_dict()
    pre = float(anchors["preutterance"])
    cons = float(anchors["consonant"])
    cutoff = float(anchors["cutoff"])
    changed = False

    before_margin = _env_float_lang("UTOA_OTO_CRNN_ONSET_SNAP_BEFORE_MARGIN_MS", lang, 18.0)
    after_margin = _env_float_lang("UTOA_OTO_CRNN_ONSET_SNAP_AFTER_MARGIN_MS", lang, 60.0)
    max_shift = _env_float_lang("UTOA_OTO_CRNN_ONSET_SNAP_MAX_SHIFT_MS", lang, 140.0)
    target_pre = onset_ms + _env_float_lang("UTOA_OTO_CRNN_ONSET_TARGET_OFFSET_MS", lang, 6.0)
    if pre < onset_ms - max(0.0, before_margin) or pre > onset_ms + max(0.0, after_margin):
        target_pre = _clamp(target_pre, max(0.0, active_start + 1.0), min(total_ms, active_end - 12.0))
        shift = _clamp(target_pre - pre, -abs(max_shift), abs(max_shift))
        if abs(shift) >= 1e-3:
            anchors = {
                "offset": float(anchors["offset"]) + shift,
                "overlap": float(anchors["overlap"]) + shift,
                "preutterance": float(anchors["preutterance"]) + shift,
                "consonant": float(anchors["consonant"]) + shift,
                "cutoff": float(anchors["cutoff"]) + shift,
            }
            changed = True
            pre = float(anchors["preutterance"])
            cons = float(anchors["consonant"])
            cutoff = float(anchors["cutoff"])

    continuity = float(profile.get("continuity_score", 0.0) or 0.0)
    continuity_threshold = _env_float_lang("UTOA_OTO_CRNN_ONSET_CONTINUITY_MIN_SCORE", lang, 0.60)
    if continuity >= continuity_threshold:
        core_start = float(profile.get("active_core_start_ms", active_start) or active_start)
        core_end = float(profile.get("active_core_end_ms", active_end) or active_end)
        min_cons_gap = _env_float_lang("UTOA_OTO_CRNN_ONSET_MIN_CONS_GAP_MS", lang, 8.0)
        min_tail = _env_float_lang("UTOA_OTO_CRNN_ONSET_MIN_TAIL_MS", lang, 20.0)

        pre_new = _clamp(pre, max(0.0, core_start - 8.0), max(0.0, core_end - 20.0))
        cons_new = _clamp(cons, pre_new + max(1.0, min_cons_gap), max(pre_new + 2.0, core_end - 8.0))
        cutoff_new = max(cutoff, cons_new + max(1.0, min_tail))
        cutoff_new = min(cutoff_new, total_ms)
        if (abs(pre_new - pre) > 1e-3) or (abs(cons_new - cons) > 1e-3) or (abs(cutoff_new - cutoff) > 1e-3):
            anchors["preutterance"] = pre_new
            anchors["consonant"] = cons_new
            anchors["cutoff"] = cutoff_new
            changed = True

    if not changed:
        return params, ""
    snapped = anchors_to_oto_params(anchors, duration_ms=total_ms)
    guarded, _ = _apply_conservative_right_boundary_guard(
        snapped,
        language=lang,
        alias=alias,
        duration_ms=total_ms,
        is_special=bool(is_special),
    )
    return guarded, "onset_voiced_snap"


def _analyze_voicebank_activity_baseline(
    *,
    wav_dir: str,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    root = os.path.abspath(str(wav_dir or "").strip())
    if not root or not os.path.isdir(root):
        return {}
    key = f"__voicebank_profile__::{root.lower()}::{int(sample_rate)}"
    cached = cache.get(key)
    if isinstance(cached, dict) and cached:
        return dict(cached)

    max_files = int(max(8, _env_float("UTOA_OTO_CRNN_VB_PROFILE_MAX_WAVS", 64)))
    allow_noise_mult = _env_float("UTOA_OTO_CRNN_VB_ALLOW_NOISE_MULT", 2.5)
    allow_speech_mult = _env_float("UTOA_OTO_CRNN_VB_ALLOW_SPEECH_MULT", 0.24)
    reject_noise_mult = _env_float("UTOA_OTO_CRNN_VB_REJECT_NOISE_MULT", 3.4)
    reject_speech_mult = _env_float("UTOA_OTO_CRNN_VB_REJECT_SPEECH_MULT", 0.30)

    wav_paths: list[str] = []
    for current_root, _dirs, files in os.walk(root):
        for name in files:
            if str(name).lower().endswith(".wav"):
                wav_paths.append(os.path.join(current_root, name))
                if len(wav_paths) >= max_files:
                    break
        if len(wav_paths) >= max_files:
            break
    if not wav_paths:
        cache[key] = {}
        return {}

    rms_values: list[float] = []
    for wav_path in wav_paths:
        try:
            samples, sr, _dur = load_wav_mono(str(wav_path), target_sr=int(sample_rate))
        except Exception:
            continue
        if samples is None or sr <= 0:
            continue
        arr = np.asarray(samples, dtype=np.float32)
        frame = max(80, int(round(float(sr) * 0.008)))
        hop = max(40, int(round(float(sr) * 0.004)))
        if arr.size < frame:
            continue
        for start in range(0, int(arr.size) - frame + 1, hop):
            seg = arr[start : start + frame]
            rms_values.append(float(np.sqrt(np.mean(np.square(seg), dtype=np.float32))))
    if not rms_values:
        cache[key] = {}
        return {}

    rms = np.asarray(rms_values, dtype=np.float32)
    noise_floor = float(np.percentile(rms, 30.0))
    speech_ref = float(np.percentile(rms, 92.0))
    allow_thr = max(noise_floor * allow_noise_mult, speech_ref * allow_speech_mult, 1e-6)
    reject_thr = max(noise_floor * reject_noise_mult, speech_ref * reject_speech_mult, allow_thr + 1e-7)
    profile = {
        "noise_floor_rms": float(noise_floor),
        "speech_ref_rms": float(speech_ref),
        "allow_rms_threshold": float(allow_thr),
        "reject_rms_threshold": float(reject_thr),
        "sampled_wavs": int(len(wav_paths)),
    }
    cache[key] = profile
    return dict(profile)


def _analyze_activity_profile(
    wav_path: str,
    *,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
    voicebank_profile: dict[str, float] | None = None,
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
    arr = np.asarray(samples, dtype=np.float32)
    abs_samples = np.abs(arr)
    peak = float(np.max(abs_samples)) if abs_samples.size else 0.0
    if peak <= 1e-6:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    frame = max(80, int(round(float(sr) * 0.008)))
    hop = max(40, int(round(float(sr) * 0.004)))
    if arr.size < frame:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    frames = []
    zcr_frames = []
    flux_frames = []
    prev_spec = None
    window = np.hanning(frame).astype(np.float32)
    for start in range(0, int(arr.size) - frame + 1, hop):
        seg = arr[start : start + frame]
        frames.append(float(np.sqrt(np.mean(np.square(seg), dtype=np.float32))))
        signs = np.signbit(seg)
        zcr_frames.append(float(np.mean(signs[:-1] != signs[1:])) if seg.size > 1 else 0.0)
        spec = np.abs(np.fft.rfft(seg * window))
        if prev_spec is None:
            flux_frames.append(0.0)
        else:
            flux_frames.append(float(np.mean(np.maximum(0.0, spec - prev_spec))))
        prev_spec = spec
    env = np.asarray(frames, dtype=np.float32)
    if env.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    noise_floor_local = float(np.percentile(env, 35.0))
    strong_floor_local = float(np.percentile(env, 92.0))
    vb = dict(voicebank_profile or {})
    noise_floor = max(noise_floor_local, float(vb.get("noise_floor_rms", 0.0) or 0.0))
    strong_floor = max(strong_floor_local, float(vb.get("speech_ref_rms", 0.0) or 0.0))
    allow_thr = float(vb.get("allow_rms_threshold", 0.0) or 0.0)
    reject_thr = float(vb.get("reject_rms_threshold", 0.0) or 0.0)
    threshold = max(noise_floor * 2.8, strong_floor * 0.28, allow_thr, 1e-5)
    active_frames = env >= threshold
    min_run_frames = max(4, int(round(0.028 / max(float(hop) / float(sr), 1e-5))))
    active_frames = _enforce_min_run(active_frames, min_run_frames)
    active_idx = np.flatnonzero(active_frames)
    if active_idx.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    start_ms = (float(active_idx[0]) * float(hop) / float(sr)) * 1000.0
    end_ms = ((float(active_idx[-1]) * float(hop) + float(frame)) / float(sr)) * 1000.0
    zcr_arr = np.asarray(zcr_frames, dtype=np.float32) if zcr_frames else np.zeros_like(env)
    flux_arr = np.asarray(flux_frames, dtype=np.float32) if flux_frames else np.zeros_like(env)
    flux_floor = float(np.percentile(flux_arr, 45.0)) if flux_arr.size else 0.0
    flux_peak = float(np.percentile(flux_arr, 95.0)) if flux_arr.size else 0.0
    flux_thr = max(flux_floor + (flux_peak - flux_floor) * 0.20, 1e-7)
    cand_env_thr = max(noise_floor * 1.9, strong_floor * 0.22, 1e-6)
    onset_candidates = np.flatnonzero((env >= cand_env_thr) & (flux_arr >= flux_thr))
    if onset_candidates.size <= 0:
        onset_idx = int(active_idx[0])
    else:
        onset_idx = int(onset_candidates[0])
        for idx in onset_candidates.tolist():
            if float(zcr_arr[idx]) <= 0.38:
                onset_idx = int(idx)
                break
    onset_ms = (float(onset_idx) * float(hop) / float(sr)) * 1000.0
    pad_ms = 20.0
    active_start = max(0.0, start_ms - pad_ms)
    active_end = min(duration_ms, end_ms + pad_ms)
    if active_end - active_start < 40.0:
        active_start = 0.0
        active_end = duration_ms
    core_env_thr = max(noise_floor * 2.2, strong_floor * 0.30, reject_thr, 1e-6)
    core_flags = env >= core_env_thr
    core_flags = _enforce_min_run(core_flags, max(3, min_run_frames - 1))
    core_idx = np.flatnonzero(core_flags)
    if core_idx.size > 0:
        core_start_ms = (float(core_idx[0]) * float(hop) / float(sr)) * 1000.0
        core_end_ms = ((float(core_idx[-1]) * float(hop) + float(frame)) / float(sr)) * 1000.0
    else:
        core_start_ms = active_start
        core_end_ms = active_end
    run_slice = env[int(max(0, onset_idx)) : int(active_idx[-1]) + 1]
    cont_thr = max(noise_floor * 1.8, strong_floor * 0.22, 1e-6)
    continuity_score = float(np.mean(run_slice >= cont_thr)) if run_slice.size else 0.0
    profile = {
        "active_start_ms": float(active_start),
        "active_end_ms": float(active_end),
        "active_core_start_ms": float(max(0.0, core_start_ms)),
        "active_core_end_ms": float(min(duration_ms, core_end_ms)),
        "onset_ms": float(max(0.0, onset_ms)),
        "continuity_score": float(_clamp(continuity_score, 0.0, 1.0)),
        "duration_ms": float(duration_ms),
    }
    cache[key] = profile
    return dict(profile)


def _enforce_min_run(flags: np.ndarray, min_run_frames: int) -> np.ndarray:
    arr = np.asarray(flags, dtype=bool).copy()
    if arr.size <= 0 or int(min_run_frames) <= 1:
        return arr
    run_start = None
    for idx, on in enumerate(arr.tolist() + [False]):
        if on and run_start is None:
            run_start = idx
            continue
        if (not on) and run_start is not None:
            run_len = idx - run_start
            if run_len < int(min_run_frames):
                arr[run_start:idx] = False
            run_start = None
    return arr


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


def _base_params_fallback_usable(base_params: dict[str, float] | None) -> bool:
    if not isinstance(base_params, dict) or not base_params:
        return False
    values: list[float] = []
    for key in ("offset", "consonant", "cutoff", "preutterance", "overlap"):
        raw = base_params.get(key, 0.0)
        try:
            values.append(float(raw))
        except Exception:
            return False
    if not values:
        return False
    # Placeholder OTO (all zeros) is not a reliable fallback target.
    if all(abs(v) < 1e-6 for v in values):
        return False
    return True


def _source_oto_shape_usable(base_params: dict[str, float] | None, *, duration_ms: float) -> bool:
    if not _base_params_fallback_usable(base_params):
        return False
    total_ms = max(1.0, float(duration_ms) or 0.0)
    base = {key: float(value) for key, value in dict(base_params or {}).items()}
    pre = max(0.0, float(base.get("preutterance", 0.0)))
    cons = max(0.0, float(base.get("consonant", 0.0)))
    ovl = max(0.0, float(base.get("overlap", 0.0)))
    cutoff_span = abs(float(base.get("cutoff", 0.0)))
    min_pre = _env_float("UTOA_OTO_CRNN_SOURCE_SHAPE_MIN_PRE_MS", 8.0)
    min_cons = _env_float("UTOA_OTO_CRNN_SOURCE_SHAPE_MIN_CONS_MS", 20.0)
    min_cutoff = _env_float("UTOA_OTO_CRNN_SOURCE_SHAPE_MIN_CUTOFF_SPAN_MS", 35.0)
    if pre < min_pre or cons < min_cons or cutoff_span < min_cutoff:
        return False
    if ovl > pre + 40.0:
        return False
    if cons < pre + 1.0:
        return False
    if cutoff_span < cons + 1.0:
        return False
    if cons > total_ms * 0.98 or cutoff_span > total_ms * 1.20:
        return False
    return True


def _blend_with_source_oto_shape(
    *,
    predicted_params: dict[str, float],
    base_params: dict[str, float] | None,
    shape_weight: float,
    cutoff_weight: float | None = None,
    duration_ms: float,
) -> dict[str, float]:
    model = {key: float(value) for key, value in dict(predicted_params).items()}
    base = {key: float(value) for key, value in dict(base_params or {}).items()}
    weight = _clamp(float(shape_weight), 0.0, 1.0)
    tail_weight = _clamp(float(weight if cutoff_weight is None else cutoff_weight), 0.0, weight)
    source_shape = {
        "offset": float(model["offset"]),
        "consonant": max(1.0, float(base.get("consonant", model["consonant"]))),
        "cutoff": -max(1.0, abs(float(base.get("cutoff", model["cutoff"])))),
        "preutterance": max(0.0, float(base.get("preutterance", model["preutterance"]))),
        "overlap": max(0.0, float(base.get("overlap", model["overlap"]))),
    }
    fused = {
        "offset": float(model["offset"]),
        "consonant": (model["consonant"] * (1.0 - weight)) + (source_shape["consonant"] * weight),
        "cutoff": (model["cutoff"] * (1.0 - tail_weight)) + (source_shape["cutoff"] * tail_weight),
        "preutterance": (model["preutterance"] * (1.0 - weight)) + (source_shape["preutterance"] * weight),
        "overlap": (model["overlap"] * (1.0 - weight)) + (source_shape["overlap"] * weight),
    }
    anchors = oto_params_to_anchors(
        offset=fused["offset"],
        consonant=fused["consonant"],
        cutoff=fused["cutoff"],
        preutterance=fused["preutterance"],
        overlap=fused["overlap"],
        duration_ms=max(1.0, float(duration_ms)),
    )
    return anchors_to_oto_params(anchors, duration_ms=max(1.0, float(duration_ms)))


def _source_shape_cutoff_weight(
    *,
    shape_weight: float,
    language: str,
    format_type: str,
    alias: str,
    is_special: bool,
    role: str | None = None,
) -> float:
    weight = _clamp(float(shape_weight), 0.0, 1.0)
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    role_key = str(role or "").strip().lower()
    if not role_key:
        alias_type = _safe_alias_type(lang, alias)
        role_key = _safe_alias_role(lang, alias, alias_type=alias_type, is_special=bool(is_special))

    factor = _env_float_lang("UTOA_OTO_CRNN_SOURCE_SHAPE_CUTOFF_WEIGHT_FACTOR", lang, 0.45)
    if (lang.startswith("ko") or lang.startswith("kr") or lang == "korean") and fmt == "cvvc":
        defaults = {
            "v-cv": 0.12,
            "vv": 0.18,
            "v": 0.14,
            "vc": 0.34,
            "other": 0.34,
            "cv": 0.42,
            "-cv": 0.42,
            "br": 0.22,
            "endbr": 0.22,
        }
        role_factor_name = (
            "UTOA_OTO_CRNN_KR_CVVC_SOURCE_SHAPE_CUTOFF_FACTOR_"
            + str(role_key or "other").replace("-", "_").upper()
        )
        factor = _env_float(role_factor_name, defaults.get(role_key, factor))
    return _clamp(weight * _clamp(float(factor), 0.0, 1.0), 0.0, weight)


def _resolve_predicted_anchors(*, pred: Any, predicted_params: dict[str, float], duration_ms: float) -> dict[str, float]:
    anchors_obj = getattr(pred, "anchors", None)
    if anchors_obj is not None:
        try:
            return {
                "offset": float(getattr(anchors_obj, "offset", 0.0)),
                "preutterance": float(getattr(anchors_obj, "preutterance", 0.0)),
                "consonant": float(getattr(anchors_obj, "consonant", 0.0)),
                "cutoff": float(getattr(anchors_obj, "cutoff", 0.0)),
            }
        except Exception:
            pass
    dur = max(1.0, float(duration_ms))
    anchors = oto_params_to_anchors(
        offset=float(predicted_params.get("offset", 0.0)),
        consonant=float(predicted_params.get("consonant", 0.0)),
        cutoff=float(predicted_params.get("cutoff", 0.0)),
        preutterance=float(predicted_params.get("preutterance", 0.0)),
        overlap=float(predicted_params.get("overlap", 0.0)),
        duration_ms=dur,
    )
    return anchors.to_dict()


def _apply_duration_span_cap(
    *,
    predicted_params: dict[str, float],
    duration_ms: float,
    language: str,
    format_type: str,
    alias: str,
    is_special: bool,
) -> tuple[dict[str, float], str]:
    if not _env_bool("UTOA_OTO_CRNN_SPAN_CAP_ENABLE", True):
        return dict(predicted_params), ""
    dur = max(1.0, float(duration_ms))
    params = {key: float(value) for key, value in dict(predicted_params).items()}
    anchors = oto_params_to_anchors(
        offset=params.get("offset", 0.0),
        consonant=params.get("consonant", 0.0),
        cutoff=params.get("cutoff", 0.0),
        preutterance=params.get("preutterance", 0.0),
        overlap=params.get("overlap", 0.0),
        duration_ms=dur,
    ).to_dict()
    offset = float(anchors.get("offset", 0.0))
    cutoff = float(anchors.get("cutoff", 0.0))
    span = max(1.0, cutoff - offset)
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    ratio_default = 0.70 if lang == "japanese" else (0.66 if lang == "korean" else 0.72)
    ratio_default = _env_float_lang("UTOA_OTO_CRNN_MAX_SPAN_RATIO", lang, ratio_default)
    ratio = ratio_default
    if fmt == "cvvc":
        ratio = min(ratio, _env_float_lang("UTOA_OTO_CRNN_MAX_SPAN_RATIO_CVVC", lang, 0.62))
    elif fmt == "cvc":
        ratio = min(ratio, _env_float_lang("UTOA_OTO_CRNN_MAX_SPAN_RATIO_CVC", lang, 0.66))
    elif fmt == "vcv":
        ratio = min(ratio, _env_float_lang("UTOA_OTO_CRNN_MAX_SPAN_RATIO_VCV", lang, 0.72))
    elif fmt == "cv":
        ratio = min(ratio, _env_float_lang("UTOA_OTO_CRNN_MAX_SPAN_RATIO_CV", lang, 0.78))
    if _is_japanese_n_alias(alias=alias, language=lang):
        ratio = min(ratio, _env_float_lang("UTOA_OTO_CRNN_JP_N_MAX_SPAN_RATIO", lang, 0.58))
    if _safe_is_diphthong(lang, alias):
        ratio = min(ratio, _env_float_lang("UTOA_OTO_CRNN_DIPHTHONG_MAX_SPAN_RATIO", lang, 0.57))
    ratio = _clamp(ratio, 0.35, 0.95)
    max_span = max(35.0, dur * ratio)
    if span <= max_span:
        return params, ""
    cutoff_new = offset + max_span
    min_tail = _env_float_lang("UTOA_OTO_CRNN_SPAN_CAP_MIN_TAIL_MS", lang, 28.0)
    cutoff_new = max(cutoff_new, float(anchors.get("consonant", 0.0)) + max(1.0, min_tail))
    cutoff_new = min(cutoff_new, dur)
    if cutoff_new <= offset + 1.0:
        return params, ""
    capped_anchors = dict(anchors)
    capped_anchors["cutoff"] = float(cutoff_new)
    out = anchors_to_oto_params(capped_anchors, duration_ms=dur)
    guarded, _ = _apply_conservative_right_boundary_guard(
        out,
        language=lang,
        alias=alias,
        duration_ms=dur,
        is_special=bool(is_special),
    )
    return guarded, "duration_span_cap"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _env_float_lang(name: str, language: str, default: float) -> float:
    lang = str(language or "").strip().lower()
    suffix = ""
    if lang.startswith("ko") or lang.startswith("kr") or lang == "korean":
        suffix = "_KR"
    elif lang.startswith("ja") or lang.startswith("jp") or lang == "japanese":
        suffix = "_JA"
    if suffix:
        raw_lang = str(os.environ.get(f"{name}{suffix}", "") or "").strip()
        if raw_lang:
            try:
                return float(raw_lang)
            except Exception:
                pass
    return _env_float(name, default)


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
