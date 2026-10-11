"""User style adjustments applied to rendered OTO rows (advanced settings).

The acoustic anchors decide where a phone is; these settings only change how
an OTO row is drawn around them, like the ratio parameters of the old AutoOTO:

- overlap ratio per alias group: overlap = ratio x preutterance (both measured
  from the offset). Hand-made OTOs commonly use 1/3 to 1/2.
- preutterance shift per alias group (ms, + = later), for bank conventions.
- fixed (consonant) extension and a cutoff margin (ms).

Every field defaults to "unchanged", so the default style writes the same
oto.ini as before.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Mapping

from .domain import RenderedOtoRow


CV_ROLES = frozenset({"cv", "cv_head", "cvvc", "v", "v_head", "v_blend"})
VC_ROLES = frozenset({"vc"})
VCV_ROLES = frozenset({"vcv", "vv"})


@dataclass(frozen=True)
class OtoStyle:
    cv_overlap_ratio: float | None = None
    vc_overlap_ratio: float | None = None
    vcv_overlap_ratio: float | None = None
    cv_pre_shift_ms: float = 0.0
    vc_pre_shift_ms: float = 0.0
    vcv_pre_shift_ms: float = 0.0
    consonant_extra_ms: float = 0.0
    cutoff_margin_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("cv_overlap_ratio", "vc_overlap_ratio", "vcv_overlap_ratio"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("cv_pre_shift_ms", "vc_pre_shift_ms", "vcv_pre_shift_ms"):
            value = getattr(self, name)
            if not (math.isfinite(value) and -200.0 <= value <= 200.0):
                raise ValueError(f"{name} must be between -200 and 200 ms")
        for name in ("consonant_extra_ms", "cutoff_margin_ms"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 500.0):
                raise ValueError(f"{name} must be between 0 and 500 ms")

    @property
    def is_default(self) -> bool:
        return self == OtoStyle()

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OtoStyle":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown OTO style fields: {', '.join(unknown)}")
        values: dict[str, float | None] = {}
        for name, value in payload.items():
            if value is None or value == "":
                values[name] = None if name.endswith("_ratio") else 0.0
            else:
                values[name] = float(value)
        return cls(**values)


def load_oto_style(path: Path) -> OtoStyle:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("OTO style preset must be a JSON object")
    return OtoStyle.from_dict(payload.get("oto_style", payload))


def save_oto_style(path: Path, style: OtoStyle) -> None:
    payload = {"schema_version": 1, "kind": "auto_oto_style_preset", "oto_style": style.to_dict()}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_oto_style(row: RenderedOtoRow, style: OtoStyle | None) -> RenderedOtoRow:
    if style is None or style.is_default:
        return row
    if row.role in CV_ROLES:
        ratio, shift = style.cv_overlap_ratio, style.cv_pre_shift_ms
    elif row.role in VC_ROLES:
        ratio, shift = style.vc_overlap_ratio, style.vc_pre_shift_ms
    elif row.role in VCV_ROLES:
        ratio, shift = style.vcv_overlap_ratio, style.vcv_pre_shift_ms
    else:
        return row  # breath, silence, release and special rows keep their template shape
    offset = row.offset
    pre_abs = offset + row.preutterance
    overlap_abs = offset + row.overlap
    fixed_abs = offset + row.consonant
    cutoff_abs = offset - row.cutoff if row.cutoff <= 0.0 else _file_end_from_absolute(row)
    end_abs = cutoff_abs  # never write past the template's end (it is inside the file)
    if shift:
        pre_abs = min(pre_abs + shift, end_abs)
        if pre_abs < offset:
            offset = max(0.0, pre_abs)
            pre_abs = offset
    if ratio is not None:
        overlap_abs = offset + ratio * (pre_abs - offset)
    overlap_abs = min(max(offset, overlap_abs), pre_abs)
    fixed_abs = min(max(fixed_abs + style.consonant_extra_ms, pre_abs), end_abs)
    cutoff_abs = min(max(end_abs - style.cutoff_margin_ms, fixed_abs, offset + 5.0), end_abs)
    absolute = dict(row.absolute)
    absolute.update(
        {
            "offset_abs": offset,
            "overlap_abs": overlap_abs,
            "pre_abs": pre_abs,
            "fixed_end_abs": fixed_abs,
            "cutoff_abs": cutoff_abs,
        }
    )
    return replace(
        row,
        offset=offset,
        overlap=overlap_abs - offset,
        preutterance=pre_abs - offset,
        consonant=fixed_abs - offset,
        cutoff=-(cutoff_abs - offset),
        absolute=absolute,
    )


# Korean VCV hand-OTO window around the preutterance, ms: (pre-offset, pre-overlap,
# fixed-pre, max cutoff-pre). Medians of 4 hand-timed Korean VCV banks
# (2026-10-11). The acoustic offset spans the whole previous vowel,
# but singers' OTOs start the window about one consonant+half-vowel before pre.
KOREAN_VCV_WINDOW_MS: Mapping[str, tuple[float | None, float, float, float]] = {
    "vcv": (195.0, 75.0, 110.0, 295.0),
    "coda_cv": (205.0, 76.0, 125.0, 396.0),  # n ga, l ba: syllabic coda into CV
    "vv": (125.0, 35.0, 104.0, 340.0),
    "head": (None, 70.0, 104.0, 278.0),  # - ga keeps its acoustic consonant-start offset
    "coda_stop": (105.0, 45.0, 52.0, 200.0),  # a k
    "coda_sonorant": (150.0, 54.0, 55.0, 410.0),  # a ng
    "release": (160.0, 67.0, 73.0, 250.0),  # a -
    "release_breath": (195.0, 80.0, 190.0, 420.0),  # ng h, a h
}
# Hand OTOs put a VV row's pre ~30 ms before the decoded vowel transition (signed
# median +21..+59 ms on 5 of 6 banks, 0 on the sixth; -30 ms raised pre<=30 ms on 5, kept 1).
KOREAN_VCV_VV_PRE_SHIFT_MS = -30.0
_KOREAN_SONORANT_LEFT = frozenset({"n", "m", "ng", "l"})
_KOREAN_STOP_CODAS = frozenset({"k", "t", "p"})


def korean_vcv_window_kind(row: RenderedOtoRow) -> str | None:
    parts = [part.casefold() for part in row.alias.split()]
    if row.role in {"cv_head", "v_head"} or (len(parts) == 2 and parts[0] == "-" and row.role in {"cv", "v"}):
        return "head"
    if len(parts) == 2 and row.role in {"vc", "release"}:
        if parts[1] in _KOREAN_STOP_CODAS:
            return "coda_stop"
        if parts[1] in _KOREAN_SONORANT_LEFT:
            return "coda_sonorant"
        if parts[1] == "-":
            return "release"
        if parts[1] in {"h", "r"}:
            return "release_breath"
        return None
    if row.role == "vv":
        return "vv"
    if row.role == "vcv":
        return "coda_cv" if parts and parts[0].casefold() in _KOREAN_SONORANT_LEFT else "vcv"
    return None


def apply_korean_vcv_window(row: RenderedOtoRow) -> RenderedOtoRow:
    """Draw a Korean VCV row with the hand-OTO window shape; pre stays acoustic (VV: see shift)."""
    kind = korean_vcv_window_kind(row)
    if kind is None or row.cutoff > 0.0:
        return row
    return _draw_window(row, KOREAN_VCV_WINDOW_MS[kind], KOREAN_VCV_VV_PRE_SHIFT_MS if kind == "vv" else 0.0)


# Korean CVC hand-OTO window, same fields as KOREAN_VCV_WINDOW_MS: medians over 35
# hand-made Korean CVC banks (2026-10-11). Generated offsets ran ~45 ms (CV) to
# ~130 ms (VC) early and fixed ~45 ms short on two hand-made banks of one reclist.
KOREAN_CVC_WINDOW_MS: Mapping[str, tuple[float | None, float, float, float]] = {
    "cv": (85.0, 48.0, 89.0, 232.0),
    "head": (90.0, 54.0, 98.0, 243.0),
    "v": (47.0, 20.0, 105.0, 329.0),  # wa, a
    "vc": (110.0, 59.0, 51.0, 76.0),  # a g: ends inside the next consonant
    "coda": (96.0, 51.0, 70.0, 212.0),  # ang, ap: glued V+coda
    "vv": (185.0, 63.0, 126.0, 246.0),
    "release": (146.0, 72.0, 91.0, 184.0),  # a R, n R
}


def korean_cvc_window_kind(row: RenderedOtoRow) -> str | None:
    parts = [part.casefold() for part in row.alias.split()]
    if any(ord(ch) > 0x2000 for ch in row.alias):
        return None  # kana duplicates keep their own template shape
    if row.role in {"cv_head", "v_head"} or (len(parts) == 2 and parts[0] == "-"):
        return "head"
    if row.role == "release" or (len(parts) == 2 and parts[1] in {"r", "-", "h"}):
        return "release"
    if row.role == "vc":
        return "coda" if len(parts) == 1 else "vc"
    return row.role if row.role in {"cv", "v", "vv"} else None


def apply_korean_cvc_window(row: RenderedOtoRow) -> RenderedOtoRow:
    """Draw a Korean CVC row with the hand-OTO window shape; pre stays acoustic."""
    kind = korean_cvc_window_kind(row)
    if kind is None or row.cutoff > 0.0:
        return row
    return _draw_window(row, KOREAN_CVC_WINDOW_MS[kind])


def _draw_window(row: RenderedOtoRow, shape: tuple[float | None, float, float, float],
                 pre_shift_ms: float = 0.0) -> RenderedOtoRow:
    lead, overlap_lead, fixed_len, cutoff_max = shape
    end_abs = row.offset - row.cutoff
    pre_abs = row.offset + row.preutterance
    if pre_shift_ms:
        pre_abs = max(row.offset, pre_abs + pre_shift_ms)
    offset = row.offset if lead is None else max(row.offset, pre_abs - lead)
    cutoff_abs = max(min(end_abs, pre_abs + cutoff_max), pre_abs + 5.0)
    fixed_abs = min(pre_abs + fixed_len, cutoff_abs)
    overlap_abs = min(max(pre_abs - overlap_lead, offset), pre_abs)
    absolute = dict(row.absolute)
    absolute.update({"offset_abs": offset, "overlap_abs": overlap_abs, "pre_abs": pre_abs,
                     "fixed_end_abs": fixed_abs, "cutoff_abs": cutoff_abs})
    return replace(row, offset=offset, overlap=overlap_abs - offset, preutterance=pre_abs - offset,
                   consonant=fixed_abs - offset, cutoff=-(cutoff_abs - offset), absolute=absolute)


def _file_end_from_absolute(row: RenderedOtoRow) -> float:
    value = row.absolute.get("cutoff_abs")
    if value is None:
        raise ValueError(f"row {row.alias!r} has a positive cutoff without an absolute end")
    return float(value)


__all__ = ["KOREAN_CVC_WINDOW_MS", "KOREAN_VCV_WINDOW_MS", "OtoStyle", "apply_korean_cvc_window", "apply_korean_vcv_window", "apply_oto_style", "load_oto_style", "save_oto_style"]
