from __future__ import annotations

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class ConfidenceCalibration:
    enabled: bool = False
    scale: float = 1.0
    bias: float = 0.0
    min_clip: float = 0.0
    max_clip: float = 1.0


def calibration_from_env() -> ConfidenceCalibration:
    enabled = _env_bool("UTOA_OTO_CRNN_CONF_CALIB_ENABLE", False)
    scale = _env_float("UTOA_OTO_CRNN_CONF_CALIB_SCALE", 1.0)
    bias = _env_float("UTOA_OTO_CRNN_CONF_CALIB_BIAS", 0.0)
    min_clip = _env_float("UTOA_OTO_CRNN_CONF_CALIB_MIN", 0.0)
    max_clip = _env_float("UTOA_OTO_CRNN_CONF_CALIB_MAX", 1.0)
    if max_clip < min_clip:
        min_clip, max_clip = max_clip, min_clip
    return ConfidenceCalibration(
        enabled=enabled,
        scale=float(scale),
        bias=float(bias),
        min_clip=float(min_clip),
        max_clip=float(max_clip),
    )


def apply_calibration(value: float, calib: ConfidenceCalibration) -> float:
    v = _clamp01(float(value))
    if not calib.enabled:
        return v
    # Temperature-like remap on logit domain then logistic back to probability.
    x = _safe_logit(v)
    y = (x * float(calib.scale)) + float(calib.bias)
    p = _sigmoid(y)
    return _clamp(float(p), float(calib.min_clip), float(calib.max_clip))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_logit(p: float) -> float:
    c = _clamp(float(p), 1e-6, 1.0 - 1e-6)
    return math.log(c / (1.0 - c))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if raw is None:
        return float(default)
    text = str(raw).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except Exception:
        return float(default)


__all__ = ["ConfidenceCalibration", "apply_calibration", "calibration_from_env"]
