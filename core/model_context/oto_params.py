from __future__ import annotations

import os
import wave
from typing import Mapping

from core.model_context.schema import OTO_PARAM_ORDER
from core.model_context.types import AbsoluteOtoAnchors, OtoParamContext


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_oto_params(params: Mapping[str, object] | None) -> dict[str, float]:
    values = dict(params or {})
    return {
        "offset": _as_float(values.get("offset", values.get("off", 0.0))),
        "consonant": _as_float(values.get("consonant", values.get("cons", 0.0))),
        "cutoff": _as_float(values.get("cutoff", values.get("cut", 0.0))),
        "preutterance": _as_float(values.get("preutterance", values.get("pre", 0.0))),
        "overlap": _as_float(values.get("overlap", values.get("ovl", 0.0))),
    }


def safe_cutoff_abs_ms(
    offset_ms: float,
    cutoff: float,
    wav_duration_ms_value: float,
    *,
    convention: str = "positive_relative",
) -> float:
    offset = max(0.0, _as_float(offset_ms))
    duration = max(1.0, _as_float(wav_duration_ms_value, 1.0))
    cut = _as_float(cutoff)
    mode = str(convention or "positive_relative").strip().lower()
    if cut < 0.0:
        resolved = offset + abs(cut)
    elif mode in {"candidate", "legacy_candidate"}:
        rel = offset + cut
        candidates = [rel]
        if cut > offset:
            candidates.append(cut)
        candidates = [item for item in candidates if item <= duration + 2.0] or candidates
        resolved = min(candidates, key=lambda item: abs(float(item) - rel))
    elif mode in {"positive_tail", "tail"}:
        resolved = duration - cut
    elif mode in {"auto", "auto_tail"} and cut > max(300.0, duration * 0.45):
        resolved = duration - cut
    else:
        resolved = offset + cut
    if mode in {"auto", "auto_tail", "candidate", "legacy_candidate", "positive_tail", "tail"} and duration > 1.0:
        resolved = min(resolved, max(offset + 1.0, duration - 1.0))
    else:
        resolved = min(resolved, duration)
    return max(offset + 1.0, float(resolved))


def resolve_cutoff_abs_ms(*, offset_ms: float, cutoff: float, duration_ms: float) -> float:
    offset = max(0.0, _as_float(offset_ms))
    duration = max(0.0, _as_float(duration_ms))
    raw = _as_float(cutoff)
    value = offset + abs(raw) if raw < 0.0 else offset + raw
    if duration > 0.0:
        value = min(value, max(offset + 1.0, duration - 1.0))
    return max(offset + 1.0, value)


def source_params_to_context(
    params: Mapping[str, object] | None,
    *,
    duration_ms: float,
    cutoff_convention: str = "positive_relative",
) -> OtoParamContext:
    raw = normalize_oto_params(params)
    duration = max(1.0, _as_float(duration_ms, 1.0))
    warnings: list[str] = []
    offset_abs = max(0.0, raw["offset"])
    pre_abs_raw = offset_abs + max(0.0, raw["preutterance"])
    overlap_abs_raw = offset_abs + max(0.0, raw["overlap"])
    consonant_abs_raw = offset_abs + max(0.0, raw["consonant"])
    cutoff_abs_raw = safe_cutoff_abs_ms(offset_abs, raw["cutoff"], duration, convention=cutoff_convention)

    pre_abs = max(offset_abs, pre_abs_raw)
    overlap_abs = min(pre_abs, max(offset_abs, overlap_abs_raw))
    consonant_abs = max(pre_abs, consonant_abs_raw)
    cutoff_abs = min(duration, max(consonant_abs + 1.0, cutoff_abs_raw))

    if raw["cutoff"] >= 0.0:
        warnings.append(f"positive_cutoff:{cutoff_convention}")
    if (pre_abs, overlap_abs, consonant_abs, cutoff_abs) != (
        pre_abs_raw,
        min(pre_abs_raw, overlap_abs_raw),
        consonant_abs_raw,
        cutoff_abs_raw,
    ):
        warnings.append("anchor_order_repaired")
    if duration <= 1.0:
        warnings.append("low_duration")

    repaired = {
        "offset": offset_abs,
        "consonant": max(1.0, consonant_abs - offset_abs),
        "cutoff": -max(1.0, cutoff_abs - offset_abs),
        "preutterance": max(0.0, pre_abs - offset_abs),
        "overlap": max(0.0, overlap_abs - offset_abs),
    }
    usable = duration > 1.0 and cutoff_abs > consonant_abs and not all(abs(raw[key]) < 1e-6 for key in OTO_PARAM_ORDER)
    return OtoParamContext(
        raw_params=raw,
        repaired_params=repaired,
        anchors_ms=(offset_abs, overlap_abs, pre_abs, consonant_abs, cutoff_abs),
        duration_ms=duration,
        repair_warnings=tuple(warnings),
        usable_as_target=usable,
        reason="source" if usable else "unusable_source_params",
    )


def source_params_to_anchors(
    params: Mapping[str, object] | None,
    *,
    duration_ms: float,
    cutoff_convention: str = "positive_relative",
) -> AbsoluteOtoAnchors:
    return source_params_to_context(
        params,
        duration_ms=duration_ms,
        cutoff_convention=cutoff_convention,
    ).to_absolute_anchors()


def oto_row_to_absolute_anchors(
    *,
    offset: float,
    consonant: float,
    cutoff: float,
    preutterance: float,
    overlap: float,
    duration_ms: float,
) -> AbsoluteOtoAnchors:
    return source_params_to_anchors(
        {
            "offset": offset,
            "consonant": consonant,
            "cutoff": cutoff,
            "preutterance": preutterance,
            "overlap": overlap,
        },
        duration_ms=duration_ms,
        cutoff_convention="positive_relative",
    )


def absolute_anchors_to_oto_params(anchors: AbsoluteOtoAnchors, *, duration_ms: float) -> dict[str, float]:
    offset = max(0.0, float(anchors.offset_abs))
    pre = max(offset, float(anchors.pre_abs))
    ovl = min(pre, max(offset, float(anchors.overlap_abs)))
    cons = max(pre, float(anchors.consonant_abs))
    cutoff_abs = max(cons + 1.0, min(float(duration_ms), float(anchors.cutoff_abs)))
    return {
        "offset": offset,
        "consonant": max(1.0, cons - offset),
        "cutoff": -max(1.0, cutoff_abs - offset),
        "preutterance": max(0.0, pre - offset),
        "overlap": max(0.0, ovl - offset),
    }


def wav_duration_ms(path: str, *, missing_value: float = 0.0) -> float:
    if not path or not os.path.isfile(path):
        return float(missing_value)
    try:
        with wave.open(path, "rb") as handle:
            rate = max(1, int(handle.getframerate()))
            frames = max(0, int(handle.getnframes()))
            return 1000.0 * frames / float(rate)
    except Exception:
        try:
            from scipy.io import wavfile

            rate, data = wavfile.read(path)
            frames = int(getattr(data, "shape", [len(data)])[0])
            return 1000.0 * max(0, frames) / float(max(1, int(rate)))
        except Exception:
            return float(missing_value)


def wav_duration_100ns(path: str, *, missing_value_ms: float = 0.0) -> int:
    return int(round(wav_duration_ms(path, missing_value=missing_value_ms) * 10000.0))
