from __future__ import annotations

import copy
import fnmatch
import json
import os
from typing import Dict, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_PROFILE: Dict[str, object] = {
    "blank": {
        "threshold": 0.55,
        "risky_threshold": 0.44,
        "severe_threshold": 0.58,
        "score_scale": 1.0,
        "score_bias": 0.0,
        "fallback_boost": 0.0,
    },
    "mel": {
        "threshold": 0.42,
        "score_scale": 1.0,
        "score_bias": 0.0,
        "fallback_penalty": 0.0,
    },
    "hysteresis": {
        "enabled": True,
        "min_run": 2,
        "blank": {
            "on_threshold": 0.58,
            "off_threshold": 0.50,
        },
        "mel": {
            "on_threshold": 0.42,
            "off_threshold": 0.48,
        },
    },
    "metrics": {
        "enabled": True,
    },
}


_EXTERNAL_CACHE = {
    "path": "",
    "mtime_ns": -1,
    "payload": None,
    "resolved": {},
}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _norm_token(value: object) -> str:
    return str(value or "").strip().lower()


def _default_profile_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "ml", "configs", "silence_reliability_profile.json")


def _resolve_profile_path() -> str:
    env_path = os.environ.get("UTOA_ML_SILENCE_PROFILE_PATH", "").strip()
    if env_path:
        return os.path.abspath(env_path)
    return _default_profile_path()


def _merge_dict(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(dict(out.get(key) or {}), value)
        else:
            out[key] = value
    return out


def _try_parse_payload(text: str, path: str) -> Optional[Dict[str, object]]:
    ext = str(os.path.splitext(path)[1] or "").strip().lower()
    payload = None
    if yaml is not None and ext in {".yaml", ".yml"}:
        try:
            payload = yaml.safe_load(text)
        except Exception:
            payload = None
    if payload is None:
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
    if payload is None and yaml is not None:
        try:
            payload = yaml.safe_load(text)
        except Exception:
            payload = None
    return payload if isinstance(payload, dict) else None


def _load_external_payload() -> Tuple[Dict[str, object], str, int]:
    path = _resolve_profile_path()
    if not path or not os.path.isfile(path):
        return {}, "", -1
    try:
        st = os.stat(path)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except OSError:
        return {}, "", -1

    if (
        _EXTERNAL_CACHE.get("payload") is not None
        and _EXTERNAL_CACHE.get("path") == path
        and int(_EXTERNAL_CACHE.get("mtime_ns", -1)) == mtime_ns
    ):
        return dict(_EXTERNAL_CACHE.get("payload") or {}), path, mtime_ns

    payload: Dict[str, object] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        parsed = _try_parse_payload(text, path)
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = {}

    _EXTERNAL_CACHE["path"] = path
    _EXTERNAL_CACHE["mtime_ns"] = mtime_ns
    _EXTERNAL_CACHE["payload"] = dict(payload)
    _EXTERNAL_CACHE["resolved"] = {}
    return payload, path, mtime_ns


def _context_profile_cache_key(row_context: Dict[str, object]) -> str:
    lang = _norm_token(row_context.get("language"))
    fmt = _norm_token(row_context.get("format_type"))
    alias = _norm_token(row_context.get("alias_type"))
    vb = _norm_token(row_context.get("voicebank_id"))
    wav = _norm_token(row_context.get("wav_norm") or row_context.get("wav"))
    return f"{lang}|{fmt}|{alias}|{vb}|{wav}"


def _match_selector_value(actual: object, expected: object) -> bool:
    a = _norm_token(actual)
    if isinstance(expected, (list, tuple, set)):
        return any(_match_selector_value(a, item) for item in expected)
    e = _norm_token(expected)
    if not e or e == "*":
        return True
    if "*" in e or "?" in e:
        return fnmatch.fnmatch(a, e)
    return a == e


def _rule_matches(row_context: Dict[str, object], selector: Dict[str, object]) -> bool:
    if not isinstance(selector, dict):
        return False
    for key in ("language", "format_type", "alias_type", "voicebank_id", "wav", "wav_norm"):
        if key not in selector:
            continue
        if not _match_selector_value(row_context.get(key), selector.get(key)):
            return False
    return True


def _apply_context_rules(
    row_context: Dict[str, object],
    base_profile: Dict[str, object],
    rule_list: object,
) -> Dict[str, object]:
    out = dict(base_profile)
    if not isinstance(rule_list, list):
        return out
    for rule in rule_list:
        if not isinstance(rule, dict):
            continue
        selector = rule.get("match")
        if selector is None:
            selector = {
                key: rule.get(key)
                for key in ("language", "format_type", "alias_type", "voicebank_id", "wav", "wav_norm")
                if key in rule
            }
        if not _rule_matches(row_context, selector):
            continue
        if isinstance(rule.get("overrides"), dict):
            override = dict(rule.get("overrides") or {})
        else:
            override = {
                k: v
                for k, v in rule.items()
                if k not in {"match", "language", "format_type", "alias_type", "voicebank_id", "wav", "wav_norm"}
            }
        out = _merge_dict(out, override)
    return out


def _apply_env_overrides(profile: Dict[str, object]) -> Dict[str, object]:
    out = dict(profile)

    def _env_float(name: str, current: Optional[float] = None) -> Optional[float]:
        raw = str(os.environ.get(name, "") or "").strip()
        if not raw:
            return current
        try:
            return float(raw)
        except Exception:
            return current

    def _env_flag(name: str, current: bool) -> bool:
        raw = str(os.environ.get(name, "") or "").strip().lower()
        if not raw:
            return bool(current)
        if raw in {"1", "true", "yes", "on", "y"}:
            return True
        if raw in {"0", "false", "no", "off", "n"}:
            return False
        return bool(current)

    blank = dict(out.get("blank") or {})
    mel = dict(out.get("mel") or {})
    hys = dict(out.get("hysteresis") or {})
    hys_blank = dict(hys.get("blank") or {})
    hys_mel = dict(hys.get("mel") or {})

    blank["threshold"] = _env_float("UTOA_ML_BLANK_RISK_THRESHOLD", _to_float(blank.get("threshold"), 0.55))
    blank["risky_threshold"] = _env_float(
        "UTOA_ML_BLANK_RISKY_THRESHOLD",
        _to_float(blank.get("risky_threshold"), 0.44),
    )
    blank["severe_threshold"] = _env_float(
        "UTOA_ML_BLANK_SEVERE_THRESHOLD",
        _to_float(blank.get("severe_threshold"), 0.58),
    )
    blank["score_scale"] = _env_float("UTOA_ML_BLANK_SCORE_SCALE", _to_float(blank.get("score_scale"), 1.0))
    blank["score_bias"] = _env_float("UTOA_ML_BLANK_SCORE_BIAS", _to_float(blank.get("score_bias"), 0.0))
    blank["fallback_boost"] = _env_float(
        "UTOA_ML_BLANK_FALLBACK_BOOST",
        _to_float(blank.get("fallback_boost"), 0.0),
    )

    mel["threshold"] = _env_float("UTOA_ML_MEL_UNRELIABLE_THRESHOLD", _to_float(mel.get("threshold"), 0.42))
    mel["score_scale"] = _env_float("UTOA_ML_MEL_SCORE_SCALE", _to_float(mel.get("score_scale"), 1.0))
    mel["score_bias"] = _env_float("UTOA_ML_MEL_SCORE_BIAS", _to_float(mel.get("score_bias"), 0.0))
    mel["fallback_penalty"] = _env_float(
        "UTOA_ML_MEL_FALLBACK_PENALTY",
        _to_float(mel.get("fallback_penalty"), 0.0),
    )

    hys["enabled"] = _env_flag("UTOA_ML_SILENCE_HYSTERESIS_ENABLE", bool(hys.get("enabled", True)))
    hys["min_run"] = _to_int(os.environ.get("UTOA_ML_SILENCE_HYSTERESIS_MIN_RUN", hys.get("min_run", 2)), 2)
    hys_blank["on_threshold"] = _env_float(
        "UTOA_ML_BLANK_HYSTERESIS_ON",
        _to_float(hys_blank.get("on_threshold"), 0.58),
    )
    hys_blank["off_threshold"] = _env_float(
        "UTOA_ML_BLANK_HYSTERESIS_OFF",
        _to_float(hys_blank.get("off_threshold"), 0.50),
    )
    hys_mel["on_threshold"] = _env_float(
        "UTOA_ML_MEL_HYSTERESIS_ON",
        _to_float(hys_mel.get("on_threshold"), 0.42),
    )
    hys_mel["off_threshold"] = _env_float(
        "UTOA_ML_MEL_HYSTERESIS_OFF",
        _to_float(hys_mel.get("off_threshold"), 0.48),
    )

    hys["blank"] = hys_blank
    hys["mel"] = hys_mel
    out["blank"] = blank
    out["mel"] = mel
    out["hysteresis"] = hys
    return out


def _resolve_voicebank_profile(row_context: Dict[str, object], payload: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    voicebanks = payload.get("voicebanks")
    if not isinstance(voicebanks, dict):
        return out
    vb = _norm_token(row_context.get("voicebank_id"))
    if not vb:
        return out
    for raw_key, raw_value in voicebanks.items():
        if not isinstance(raw_value, dict):
            continue
        if _match_selector_value(vb, raw_key):
            out = _merge_dict(out, raw_value)
            if isinstance(raw_value.get("contexts"), list):
                out = _apply_context_rules(row_context, out, raw_value.get("contexts"))
    return out


def resolve_silence_reliability_profile(row_context: Dict[str, object]) -> Dict[str, object]:
    payload, path, mtime_ns = _load_external_payload()
    cache_scope = f"{path}|{mtime_ns}"
    cache_key = _context_profile_cache_key(row_context)
    scoped = dict(_EXTERNAL_CACHE.get("resolved") or {})
    scoped_key = f"{cache_scope}|{cache_key}"
    if scoped_key in scoped:
        return copy.deepcopy(scoped[scoped_key])

    profile: Dict[str, object] = copy.deepcopy(DEFAULT_PROFILE)
    if isinstance(payload, dict):
        if isinstance(payload.get("defaults"), dict):
            profile = _merge_dict(profile, dict(payload.get("defaults") or {}))
        lang_map = payload.get("languages")
        lang = _norm_token(row_context.get("language"))
        if isinstance(lang_map, dict) and lang in lang_map and isinstance(lang_map.get(lang), dict):
            profile = _merge_dict(profile, dict(lang_map.get(lang) or {}))
        fmt_map = payload.get("formats")
        fmt = _norm_token(row_context.get("format_type"))
        if isinstance(fmt_map, dict):
            fmt_profile = fmt_map.get(fmt, fmt_map.get("*"))
            if isinstance(fmt_profile, dict):
                profile = _merge_dict(profile, dict(fmt_profile))
        alias_map = payload.get("alias_types")
        alias = _norm_token(row_context.get("alias_type"))
        if isinstance(alias_map, dict):
            alias_profile = alias_map.get(alias, alias_map.get("*"))
            if isinstance(alias_profile, dict):
                profile = _merge_dict(profile, dict(alias_profile))
        profile = _merge_dict(profile, _resolve_voicebank_profile(row_context, payload))
        profile = _apply_context_rules(row_context, profile, payload.get("contexts"))

    profile = _apply_env_overrides(profile)
    scoped[scoped_key] = copy.deepcopy(profile)
    _EXTERNAL_CACHE["resolved"] = scoped
    return profile


def resolve_reliability_thresholds(profile: Dict[str, object]) -> Dict[str, object]:
    blank = dict(profile.get("blank") or {})
    mel = dict(profile.get("mel") or {})
    hys = dict(profile.get("hysteresis") or {})
    hys_blank = dict(hys.get("blank") or {})
    hys_mel = dict(hys.get("mel") or {})

    blank_thr = _clamp01(_to_float(blank.get("threshold"), 0.55))
    blank_risky = _clamp01(_to_float(blank.get("risky_threshold"), max(0.0, blank_thr - 0.11)))
    blank_severe = _clamp01(_to_float(blank.get("severe_threshold"), max(blank_risky + 0.04, blank_thr + 0.03)))

    mel_thr = _clamp01(_to_float(mel.get("threshold"), 0.42))
    hys_enabled = bool(hys.get("enabled", True))
    min_run = max(1, _to_int(hys.get("min_run"), 2))

    blank_on = _clamp01(_to_float(hys_blank.get("on_threshold"), max(blank_thr, blank_risky)))
    blank_off = _clamp01(_to_float(hys_blank.get("off_threshold"), min(blank_on, blank_thr - 0.05)))
    if blank_off > blank_on:
        blank_off = blank_on

    mel_on = _clamp01(_to_float(hys_mel.get("on_threshold"), mel_thr))
    mel_off = _clamp01(_to_float(hys_mel.get("off_threshold"), max(mel_on, mel_thr + 0.06)))
    if mel_off < mel_on:
        mel_off = mel_on

    return {
        "blank_threshold": blank_thr,
        "blank_risky_threshold": blank_risky,
        "blank_severe_threshold": blank_severe,
        "mel_threshold": mel_thr,
        "hysteresis_enabled": hys_enabled,
        "hysteresis_min_run": min_run,
        "blank_hys_on": blank_on,
        "blank_hys_off": blank_off,
        "mel_hys_on": mel_on,
        "mel_hys_off": mel_off,
    }


def make_reliability_state_key(row_context: Dict[str, object]) -> str:
    voicebank = _norm_token(row_context.get("voicebank_id"))
    wav = _norm_token(row_context.get("wav_norm") or row_context.get("wav"))
    fmt = _norm_token(row_context.get("format_type"))
    alias = _norm_token(row_context.get("alias_type"))
    return f"{voicebank}|{wav}|{fmt}|{alias}"


def _step_hysteresis(
    *,
    value: float,
    prev_active: bool,
    prev_pending: int,
    on_threshold: float,
    off_threshold: float,
    high_is_risk: bool,
    min_run: int,
) -> Tuple[bool, int, bool]:
    if high_is_risk:
        candidate = bool(value >= (off_threshold if prev_active else on_threshold))
    else:
        candidate = bool(value <= (off_threshold if prev_active else on_threshold))

    if candidate == prev_active:
        return prev_active, 0, False

    pending = max(0, int(prev_pending)) + 1
    if pending >= max(1, int(min_run)):
        return candidate, 0, True
    return prev_active, pending, False


def apply_reliability_hysteresis(
    *,
    blank_risk_score: float,
    mel_reliability_score: float,
    profile: Dict[str, object],
    prev_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    thresholds = resolve_reliability_thresholds(profile)
    blank_direct = bool(blank_risk_score >= float(thresholds["blank_threshold"]))
    mel_direct = bool(mel_reliability_score < float(thresholds["mel_threshold"]))

    if not bool(thresholds.get("hysteresis_enabled", True)):
        state = {
            "blank": {"active": blank_direct, "pending": 0},
            "mel": {"active": mel_direct, "pending": 0},
        }
        return {
            **thresholds,
            "blank_flag": int(blank_direct),
            "mel_unreliable": bool(mel_direct),
            "blank_switched": False,
            "mel_switched": False,
            "state": state,
        }

    prev = dict(prev_state or {})
    prev_blank = dict(prev.get("blank") or {})
    prev_mel = dict(prev.get("mel") or {})
    blank_active, blank_pending, blank_switched = _step_hysteresis(
        value=float(blank_risk_score),
        prev_active=bool(prev_blank.get("active", False)),
        prev_pending=_to_int(prev_blank.get("pending"), 0),
        on_threshold=float(thresholds["blank_hys_on"]),
        off_threshold=float(thresholds["blank_hys_off"]),
        high_is_risk=True,
        min_run=int(thresholds["hysteresis_min_run"]),
    )
    mel_active, mel_pending, mel_switched = _step_hysteresis(
        value=float(mel_reliability_score),
        prev_active=bool(prev_mel.get("active", False)),
        prev_pending=_to_int(prev_mel.get("pending"), 0),
        on_threshold=float(thresholds["mel_hys_on"]),
        off_threshold=float(thresholds["mel_hys_off"]),
        high_is_risk=False,
        min_run=int(thresholds["hysteresis_min_run"]),
    )
    state = {
        "blank": {"active": bool(blank_active), "pending": int(blank_pending)},
        "mel": {"active": bool(mel_active), "pending": int(mel_pending)},
    }
    return {
        **thresholds,
        "blank_flag": int(blank_active),
        "mel_unreliable": bool(mel_active),
        "blank_switched": bool(blank_switched),
        "mel_switched": bool(mel_switched),
        "state": state,
    }


__all__ = [
    "DEFAULT_PROFILE",
    "apply_reliability_hysteresis",
    "make_reliability_state_key",
    "resolve_reliability_thresholds",
    "resolve_silence_reliability_profile",
]
