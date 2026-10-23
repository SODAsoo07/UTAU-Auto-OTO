"""Learn a labeler's style from a few checked rows instead of a whole hand-timed bank.

Generation picks, for every alias shape, up to SAMPLE_PER_SHAPE low-risk rows from different WAVs
and lists them in style_sample.csv. The user checks those rows in the generated oto.ini (fixing the
ones that are off, leaving the right ones alone) and fits: per shape, the median of (checked -
generated) per anchor, shrunk by n / (n + SHRINK_ROWS), is added to the style the generation used.
Unchanged rows count as "the generated value was right".

Low-risk rows keep alignment errors out of the style; the shrinkage keeps a five-row median from
overriding a good default. Simulated on five hand-timed English VCCV banks (2026-10-19), five rows
per shape (about 200 rows) moved preutterance within 30 ms by +0.1 to +7.1 points over the default
style, never lower; random rows without shrinkage lost up to 2.7 points on one bank.
"""

from __future__ import annotations

import csv
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .generic_alias_binding import parse_pack_alias
from .language_pack import LanguagePack, language_pack_by_id
from .pack_style import PackStyle, load_style_profile, oto_absolute, write_style_profile

SAMPLE_PER_SHAPE = 5
SHRINK_ROWS = 3.0
MIN_SHAPE_ROWS = 3
SAMPLE_FILE = "style_sample.csv"
ANCHORS = (("offset", "offset_abs"), ("overlap", "overlap_abs"), ("pre", "pre_abs"), ("fixed", "fixed_end_abs"),
           ("cutoff", "cutoff_abs"))
FIELDS = ("oto", "line", "wav", "alias", "shape", "wav_path", "language_pack", "style_source", "style_sha256",
          *(name for name, _key in ANCHORS))


def select_style_sample(pack: LanguagePack, reports: Sequence[Mapping[str, object]], risks: Sequence[float], *,
                        per_shape: int = SAMPLE_PER_SHAPE, seed: int = 0) -> list[int]:
    """Positions of the rows to check: per shape, from the lower-risk half, one row per WAV."""
    by_shape: dict[str, list[int]] = {}
    for position, report in enumerate(reports):
        parsed = parse_pack_alias(pack, str(report["alias"]))
        if parsed is not None:
            by_shape.setdefault(parsed.shape, []).append(position)
    chooser = random.Random(seed)
    chosen: list[int] = []
    for shape in sorted(by_shape):
        members = by_shape[shape]
        if len(members) < MIN_SHAPE_ROWS:
            continue
        pool = sorted(members, key=lambda p: (float(risks[p]), str(reports[p]["wav"]), str(reports[p]["alias"])))
        pool = pool[: max(per_shape, len(pool) // 2)]
        chooser.shuffle(pool)
        seen: set[str] = set()
        for position in pool:
            wav = str(reports[position]["wav"])
            if wav in seen:
                continue
            seen.add(wav)
            chosen.append(position)
            if len(seen) == per_shape:
                break
    return sorted(chosen)


def sample_rows(pack: LanguagePack, reports: Sequence[Mapping[str, object]], positions: Sequence[int],
                wav_paths: Mapping[int, Path], style: Mapping[str, object] | None) -> list[dict[str, object]]:
    rows = []
    for position in positions:
        report = reports[position]
        absolute = report["absolute"]
        rows.append({
            "oto": report["output_oto"], "line": report["oto_line"], "wav": report["wav"], "alias": report["alias"],
            "shape": parse_pack_alias(pack, str(report["alias"])).shape,
            "wav_path": str(wav_paths[int(report["recording_index"])]), "language_pack": pack.pack_id,
            "style_source": (style or {}).get("source", ""), "style_sha256": (style or {}).get("sha256", ""),
            **{name: round(float(absolute[key]), 3) for name, key in ANCHORS},
        })
    return rows


def write_style_sample(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _base_style(pack_id: str, source: str) -> PackStyle | None:
    if not source:
        return None
    if source.startswith("package:"):
        return language_pack_by_id(pack_id).style
    return load_style_profile(Path(source))


def _shape_rows(pack_id: str, generated: set[str]) -> dict[tuple[str, str], int]:
    """(generated oto.ini, shape) -> rows of that shape in the file, read before the user's edits matter
    (only aliases are counted)."""
    from .style_fitting import _reference_rows

    pack = language_pack_by_id(pack_id)
    counts: dict[tuple[str, str], int] = {}
    for path in generated:
        if not Path(path).is_file():
            continue
        for _wav, alias in _reference_rows(Path(path)):
            parsed = parse_pack_alias(pack, alias)
            if parsed is not None:
                counts[(path, parsed.shape)] = counts.get((path, parsed.shape), 0) + 1
    return counts


def _edited_rows(path: Path) -> dict[tuple[str, str], tuple[float, ...]]:
    from .style_fitting import _reference_rows

    return {(wav.casefold(), alias): values for (wav, alias), values in _reference_rows(path).items()}


def find_style_samples(output: Path) -> list[Path]:
    """style_sample.csv of a generation, or of every folder of a batch generation under ``output``."""
    output = Path(output)
    direct = output / SAMPLE_FILE
    return [direct] if direct.is_file() else sorted(output.rglob(SAMPLE_FILE))


def fit_style_from_sample(samples: Path | Sequence[Path], output: Path, *, edited: Path | None = None,
                          root: Path | None = None) -> dict[str, object]:
    """Write a style profile = the generation's style + the shrunk per-shape median correction.

    By default the generated oto.ini files listed in the samples are read, edited in place.
    ``edited`` is a checked oto.ini file, or a folder laid out like the generation output under
    ``root`` (default: the folder of the single sample), e.g. the voicebank the outputs were copied into.
    """
    from .style_fitting import _duration_ms

    samples = [Path(samples)] if isinstance(samples, (str, Path)) else [Path(item) for item in samples]
    if not samples:
        raise ValueError("no style_sample.csv given")
    rows = []
    for path in samples:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows += list(csv.DictReader(handle))
    sample = samples[0] if len(samples) == 1 else Path(root or samples[0].parent)
    if not rows:
        raise ValueError(f"{sample}: no sample rows")
    pack_ids = {row["language_pack"] for row in rows}
    sources = {(row["style_source"], row["style_sha256"]) for row in rows}
    if len(pack_ids) != 1 or len(sources) != 1:
        raise ValueError(f"{sample}: rows come from different packs or styles")
    pack_id = pack_ids.pop()
    source, sha256 = sources.pop()
    base = _base_style(pack_id, source)
    if base is not None and sha256 and base.sha256 != sha256:
        raise ValueError(f"the style used for generation changed since ({source}); regenerate before fitting")
    root = Path(root) if root is not None else samples[0].parent
    cache: dict[Path, dict[tuple[str, str], tuple[float, ...]]] = {}
    deltas: dict[tuple[str, str], dict[str, list[float]]] = {}
    found = changed = 0
    for row in rows:
        oto = Path(row["oto"])
        if edited is not None:
            edited_path = Path(edited)
            if edited_path.is_dir():
                try:
                    oto = edited_path / oto.relative_to(root)
                except ValueError:
                    oto = edited_path / oto.name
            else:
                oto = edited_path
        if oto not in cache:
            cache[oto] = _edited_rows(oto) if oto.is_file() else {}
        values = cache[oto].get((row["wav"].casefold(), row["alias"]))
        if values is None:
            continue
        found += 1
        checked = oto_absolute(values, _duration_ms(Path(row["wav_path"])))
        per_anchor = deltas.setdefault((row["shape"], row["oto"]), {})
        moved = False
        for name, _key in ANCHORS:
            if name in checked and row.get(name) not in (None, ""):
                difference = checked[name] - float(row[name])
                per_anchor.setdefault(name, []).append(difference)
                moved = moved or abs(difference) >= 1.0
        changed += moved
    if not found:
        raise ValueError("none of the sample rows were found in the checked oto.ini")
    # A shape can be timed differently in different folders (VC before a pause in CV_CVC, VC before a
    # consonant in VC), so each generated oto.ini gets its own shrunk median, weighted by how many
    # rows of the shape that file holds.
    shape_rows = _shape_rows(pack_id, {generated for _shape, generated in deltas})
    corrections: dict[str, dict[str, float]] = {}
    for shape in sorted({shape for shape, _generated in deltas}):
        totals: dict[str, list[float]] = {}
        for (key_shape, generated), per_anchor in deltas.items():
            if key_shape != shape:
                continue
            for name, values in per_anchor.items():
                weight = shape_rows.get((generated, shape)) or len(values)
                shrunk = statistics.median(values) * len(values) / (len(values) + SHRINK_ROWS)
                totals.setdefault(name, [0.0, 0.0])
                totals[name][0] += weight * shrunk
                totals[name][1] += weight
        corrections[shape] = {name: round(total / weight, 1) for name, (total, weight) in totals.items() if weight}
    shapes = {key: dict(anchors) for key, anchors in (base.shapes.items() if base else ())}
    for shape, correction in corrections.items():
        keys = [key for key in shapes if key == shape or key.startswith(shape + "|")] or [shape]
        for key in keys:
            anchors = shapes.setdefault(key, {})
            for name, value in correction.items():
                anchors[name] = round(float(anchors.get(name, 0.0)) + value, 1)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language_pack": pack_id,
        "base_style": {"source": source, "sha256": sha256},
        "samples": [str(path) for path in samples], "rows_checked": found, "rows_changed": changed,
        "rule": (f"base style + per shape: median(checked - generated) x n/(n+{SHRINK_ROWS:g}) per generated"
                 " oto.ini, averaged by that file's row count of the shape"),
        "corrections": corrections,
    }
    write_style_profile(Path(output), pack_id, shapes, provenance)
    return {"output": str(output), "rows_checked": found, "rows_changed": changed, "shapes_corrected": len(corrections),
            "corrections": corrections}


__all__ = ["SAMPLE_FILE", "SAMPLE_PER_SHAPE", "find_style_samples", "fit_style_from_sample", "sample_rows", "select_style_sample",
           "write_style_sample"]
