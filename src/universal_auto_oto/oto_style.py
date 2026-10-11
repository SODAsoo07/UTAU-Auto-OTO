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


def _file_end_from_absolute(row: RenderedOtoRow) -> float:
    value = row.absolute.get("cutoff_abs")
    if value is None:
        raise ValueError(f"row {row.alias!r} has a positive cutoff without an absolute end")
    return float(value)


__all__ = ["OtoStyle", "apply_oto_style", "load_oto_style", "save_oto_style"]
