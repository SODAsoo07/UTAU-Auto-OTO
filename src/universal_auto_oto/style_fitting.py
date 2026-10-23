"""fit-style: learn a labeler's per-shape timing style from banks they already timed by hand.

Each bank is generated without any style (templates only, the bank's oto.ini
supplying only aliases); the style is the median gap between the hand-made and
generated anchors per alias shape.  Apply it with ``generate-voicebank
--style-profile``.
"""

from __future__ import annotations

import json
from collections import Counter
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import soundfile as sf

from .generator import GenerationOptions, generate_voicebank
from .generic_alias_binding import parse_pack_alias, style_keys
from .language_pack import find_language_pack
from .oto_io import read_text_auto
from .voicebank_project_adapter import VoicebankReviewRequired
from .pack_style import NO_STYLE, MIN_ROWS_PER_SHAPE, StyleSample, fit_style, oto_absolute, write_style_profile

FIT_ALIGNMENT_PROFILE = "cvvc_slot_rhythm_v16"  # used when the pack names no alignment profile


def fit_style_from_banks(
    banks: Sequence[Path],
    *,
    output: Path,
    format_hint: str,
    language_hint: str,
    work_dir: Path | None = None,
    workers: int = 0,
    min_rows: int = MIN_ROWS_PER_SHAPE,
) -> dict[str, object]:
    pack = find_language_pack(language_hint, format_hint)
    if pack is None:
        raise ValueError(f"no language pack for {language_hint}/{format_hint}; style fitting needs one")
    samples: list[StyleSample] = []
    per_bank = []
    with tempfile.TemporaryDirectory(prefix="autooto_style_") as scratch:
        root = Path(work_dir) if work_dir is not None else Path(scratch)
        for index, bank in enumerate(banks):
            bank = Path(bank).resolve()
            reference_path = bank / "oto.ini"
            if not reference_path.is_file():
                raise FileNotFoundError(f"hand-made oto.ini not found: {reference_path}")
            out = root / f"{index:02d}_{bank.name}"
            try:
                generate_voicebank(GenerationOptions(
                    bank=bank, output=out, base_oto=reference_path, format_hint=format_hint,
                    language_hint=language_hint, backend="routed_hsmm",
                    development_alignment_profile=pack.alignment_profile or FIT_ALIGNMENT_PROFILE, unmapped_alias_policy="exclude_row",
                    style_profile=NO_STYLE, force=True, alignment_workers=workers, write_timelines=False,
                ))
            except VoicebankReviewRequired as error:
                # e.g. a breath-only folder: no alias the pack can bind.
                per_bank.append({"bank": str(bank), "rows": 0, "skipped": str(error)})
                continue
            reference = _reference_rows(reference_path)
            used = 0
            for line in (out / "row_report.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                key = (Path(row["recording_wav"]).name, row["alias"])
                hand = reference.get(key)
                parsed = parse_pack_alias(pack, row["alias"])
                if hand is None or parsed is None:
                    continue
                absolute = row["absolute"]
                generated = {"offset": absolute["offset_abs"], "overlap": absolute["overlap_abs"],
                             "pre": absolute["pre_abs"], "fixed": absolute["fixed_end_abs"],
                             "cutoff": absolute["cutoff_abs"]}
                samples.append(StyleSample(style_keys(pack, parsed), generated,
                                           oto_absolute(hand, _duration_ms(bank / row["recording_wav"]))))
                used += 1
            per_bank.append({"bank": str(bank), "rows": used})
    if not samples:
        raise ValueError("no hand-timed row could be matched; check the format/language and the oto.ini files")
    shapes = fit_style(samples, min_rows=min_rows)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language_pack": pack.pack_id,
        "language_pack_sha256": pack.sha256,
        "alignment_profile": pack.alignment_profile or FIT_ALIGNMENT_PROFILE,
        "banks": per_bank,
        "rule": "median(hand - generated) per alias shape and anchor, generated without style",
    }
    write_style_profile(output, pack.pack_id, shapes, provenance)
    result: dict[str, object] = {"output": str(output), "shapes": len(shapes), "rows": len(samples), "banks": per_bank}
    template = template_like_reference(banks)
    if template is not None:
        result["warnings"] = [{"code": "template_like_reference", **template}]
    return result


# A hand-timed oto.ini has an offset per take (over half the rows distinct in five English banks);
# a reclist template copied without adjusting has a handful (10 values over 3,429 rows in one bank).
TEMPLATE_DISTINCT_OFFSET_SHARE = 0.10
TEMPLATE_MIN_ROWS = 50


def template_like_reference(banks: Sequence[Path]) -> dict[str, object] | None:
    """Details when the banks' oto.ini look like an unadjusted template rather than hand timing."""
    offsets: Counter[float] = Counter()
    for bank in banks:
        path = Path(bank) / "oto.ini"
        if path.is_file():
            offsets.update(round(values[0], 3) for values in _reference_rows(path).values())
    rows = sum(offsets.values())
    if rows < TEMPLATE_MIN_ROWS or len(offsets) / rows >= TEMPLATE_DISTINCT_OFFSET_SHARE:
        return None
    return {"rows": rows, "distinct_offsets": len(offsets),
            "most_common_offsets": [value for value, _count in offsets.most_common(3)]}


def _reference_rows(path: Path) -> dict[tuple[str, str], tuple[float, ...]]:
    text, _encoding, _newline = read_text_auto(path)
    rows: dict[tuple[str, str], tuple[float, ...]] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        wav, rest = line.split("=", 1)
        fields = rest.split(",")
        try:
            numbers = tuple(float(value) for value in fields[1:6])
        except ValueError:
            continue
        if len(numbers) == 5:
            rows.setdefault((wav.strip().lstrip("﻿"), fields[0].strip()), numbers)
    return rows


def _duration_ms(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return None
    return info.frames / info.samplerate * 1000.0


__all__ = ["fit_style_from_banks", "template_like_reference"]
