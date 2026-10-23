"""Per-alias-shape timing style for language-pack banks.

Templates put the preutterance on an acoustic landmark; how far from that
landmark each labeler sets offset/overlap/preutterance/consonant/cutoff is a
personal habit that is consistent within one labeler and different between
labelers (two hand-timed English VCCV banks: a style fitted on half of one bank
moved the other half from 56% to 74% preutterance within 30 ms, while a style
from the other labeler did not help).  A style is therefore data: a shift in ms
per alias shape and anchor, added after the template is evaluated.

The pack ships a default style (``style:`` in the pack YAML); ``fit_style``
learns a user's own style from a bank they already timed by hand.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

STYLE_ANCHORS: Mapping[str, str] = {
    "offset": "offset_abs",
    "overlap": "overlap_abs",
    "pre": "pre_abs",
    "fixed": "fixed_abs",
    "cutoff": "cutoff_abs",
}
NO_STYLE = "none"
MIN_ROWS_PER_SHAPE = 4


class PackStyleError(ValueError):
    pass


@dataclass(frozen=True)
class PackStyle:
    pack_id: str
    shapes: Mapping[str, Mapping[str, float]]
    source: str
    sha256: str

    def shifts_for(self, *keys: str) -> Mapping[str, float]:
        """Shifts keyed by template anchor name (offset_abs, ...) for the first key the style has.

        Keys go from specific to general: ``"VC-|son"`` (shape and the class of the
        consonant at the vowel boundary), then ``"VC-"``.
        """
        values = next((self.shapes[key] for key in keys if key in self.shapes), {})
        return {STYLE_ANCHORS[name]: float(value) for name, value in values.items() if value}


def parse_style(raw: object, *, pack_id: str, source: str) -> PackStyle:
    if not isinstance(raw, Mapping):
        raise PackStyleError(f"{source}: style must be a mapping of alias shapes")
    shapes: dict[str, dict[str, float]] = {}
    for shape, anchors in raw.items():
        if not isinstance(anchors, Mapping):
            raise PackStyleError(f"{source}: style for {shape!r} must be a mapping")
        parsed = {}
        for name, value in anchors.items():
            if name == "rows":
                continue
            if name not in STYLE_ANCHORS:
                raise PackStyleError(f"{source}: unknown style anchor {name!r} for {shape!r}")
            number = float(value)
            if not math.isfinite(number) or abs(number) > 1000.0:
                raise PackStyleError(f"{source}: style shift out of range: {shape} {name}={value}")
            parsed[name] = number
        shapes[str(shape)] = parsed
    text = yaml.safe_dump({"pack_id": pack_id, "shapes": shapes}, sort_keys=True)
    return PackStyle(pack_id, shapes, source, hashlib.sha256(text.encode("utf-8")).hexdigest())


def load_style_profile(path: Path) -> PackStyle:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise PackStyleError(f"{path}: schema_version must be 1")
    pack_id = str(payload.get("pack_id") or "")
    if not pack_id:
        raise PackStyleError(f"{path}: pack_id is required")
    return parse_style(payload.get("shapes") or {}, pack_id=pack_id, source=str(path))


@dataclass(frozen=True)
class StyleSample:
    keys: tuple[str, ...]  # specific to general, e.g. ("VC-|son", "VC-")
    generated: Mapping[str, float]
    reference: Mapping[str, float]


def fit_style(samples: Iterable[StyleSample], *, min_rows: int = MIN_ROWS_PER_SHAPE) -> dict[str, dict[str, float]]:
    """Median (reference - generated) per style key and anchor; keys with too few rows are left out."""
    by_shape: dict[str, list[StyleSample]] = {}
    for sample in samples:
        for key in sample.keys:
            by_shape.setdefault(key, []).append(sample)
    result: dict[str, dict[str, float]] = {}
    for shape, rows in sorted(by_shape.items()):
        if len(rows) < min_rows:
            continue
        shifts: dict[str, float] = {"rows": len(rows)}
        for name in STYLE_ANCHORS:
            deltas = [row.reference[name] - row.generated[name] for row in rows
                      if name in row.reference and name in row.generated]
            if deltas:
                shifts[name] = round(statistics.median(deltas), 1)
        result[shape] = shifts
    return result


def oto_absolute(values: Sequence[float], duration_ms: float | None) -> dict[str, float]:
    """oto.ini numbers (offset, consonant, cutoff, pre, overlap) -> absolute ms from the file start."""
    offset, consonant, cutoff, pre, overlap = (float(value) for value in values)
    result = {"offset": offset, "overlap": offset + overlap, "pre": offset + pre, "fixed": offset + consonant}
    if cutoff < 0:
        result["cutoff"] = offset - cutoff
    elif duration_ms is not None:
        result["cutoff"] = duration_ms - cutoff
    return result


def write_style_profile(path: Path, pack_id: str, shapes: Mapping[str, Mapping[str, float]],
                        provenance: Mapping[str, object]) -> None:
    payload = {"schema_version": 1, "pack_id": pack_id, "provenance": dict(provenance), "shapes": dict(shapes)}
    Path(path).write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


__all__ = [
    "NO_STYLE",
    "PackStyle",
    "PackStyleError",
    "STYLE_ANCHORS",
    "StyleSample",
    "fit_style",
    "load_style_profile",
    "oto_absolute",
    "parse_style",
    "write_style_profile",
]
