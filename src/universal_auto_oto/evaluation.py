from __future__ import annotations

import json
import os
import unicodedata
import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import file_sha256
from .oto_io import oto_cutoff_end_abs, read_text_auto, write_lines_atomic
from .planning import normalize_wav_key


@dataclass(frozen=True)
class OtoTimingRecord:
    wav: str
    wav_key: str
    alias: str
    occurrence: int
    offset_abs: float
    overlap_abs: float
    pre_abs: float
    fixed_abs: float
    cutoff_abs: float


class WavDirectoryIndex:
    """Resolve OTO WAV references without collapsing multipitch subfolders."""

    def __init__(self, root: Path, *, recursive: bool = True) -> None:
        self.root = root.resolve()
        iterator = self.root.rglob("*") if recursive else self.root.glob("*")
        self.paths = tuple(path.resolve() for path in iterator if path.is_file() and path.suffix.casefold() == ".wav")
        self._relative: dict[str, list[Path]] = defaultdict(list)
        self._basename: dict[str, list[Path]] = defaultdict(list)
        self._duration_cache: dict[str, float] = {}
        for path in self.paths:
            relative = path.relative_to(self.root)
            self._relative[_normalize_relative_reference(str(relative))].append(path)
            self._basename[normalize_wav_key(path.name)].append(path)

    def resolve(self, wav_name: str) -> Path:
        for reference_variant in _legacy_filename_variants(wav_name):
            literal_parts = [
                part
                for part in reference_variant.strip().replace("\\", "/").split("/")
                if part not in {"", "."}
            ]
            if literal_parts:
                literal = self.root.joinpath(*literal_parts)
                if literal.is_file():
                    return literal.resolve()
        relative_key = _normalize_relative_reference(wav_name)
        relative_matches = self._relative.get(relative_key, [])
        if len(relative_matches) == 1:
            return relative_matches[0]
        basename_matches = self._basename.get(normalize_wav_key(wav_name), [])
        if len(basename_matches) == 1:
            return basename_matches[0]
        pitch_prefixed_matches = self._pitch_prefixed_embedded_matches(wav_name)
        if len(pitch_prefixed_matches) == 1:
            return pitch_prefixed_matches[0]
        embedded_matches = self._embedded_basename_matches(wav_name)
        if len(embedded_matches) == 1:
            return embedded_matches[0]
        count = len(relative_matches) if relative_matches else len(basename_matches)
        if not count:
            count = len(pitch_prefixed_matches) or len(embedded_matches)
        raise FileNotFoundError(f"WAV 매핑이 유일하지 않습니다: {wav_name} -> {count}")

    def identity(self, wav_name: str) -> str:
        path = self.resolve(wav_name)
        return unicodedata.normalize("NFKC", os.path.normcase(str(path)))

    def reference(self, wav_name: str) -> str:
        """Return the actual root-relative WAV reference for a legacy OTO key."""

        return self.resolve(wav_name).relative_to(self.root).as_posix()

    def identity_for_path(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.suffix.casefold() != ".wav":
            raise FileNotFoundError(path)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"WAV index 밖의 경로입니다: {resolved}") from error
        return unicodedata.normalize("NFKC", os.path.normcase(str(resolved)))

    def duration(self, wav_name: str) -> float:
        identity = self.identity(wav_name)
        if identity not in self._duration_cache:
            info = sf.info(self.resolve(wav_name))
            self._duration_cache[identity] = 1000.0 * info.frames / info.samplerate
        return self._duration_cache[identity]

    def _embedded_basename_matches(self, wav_name: str) -> list[Path]:
        """Resolve one longest real WAV basename suffixed to a legacy metadata prefix."""

        raw = unicodedata.normalize("NFKC", wav_name.strip().replace("\\", "/"))
        tail = raw.rsplit("/", 1)[-1].casefold()
        matches = [path for path in self.paths if tail.endswith(path.name.casefold())]
        if not matches:
            return []
        longest = max(len(path.name) for path in matches)
        return [path for path in matches if len(path.name) == longest]

    def _pitch_prefixed_embedded_matches(self, wav_name: str) -> list[Path]:
        """Prefer a multipitch folder named before the first legacy comma."""

        raw = unicodedata.normalize("NFKC", wav_name.strip().replace("\\", "/"))
        if "," not in raw:
            return []
        pitch, _metadata = raw.split(",", 1)
        pitch = pitch.strip().casefold()
        if not pitch or "/" in pitch:
            return []
        matches = [
            path
            for path in self._embedded_basename_matches(raw)
            if path.relative_to(self.root).parts
            and path.relative_to(self.root).parts[0].casefold() == pitch
        ]
        return matches


def evaluate_oto(
    generated_oto: Path,
    reference_oto: Path,
    wav_dir: Path,
    *,
    output_json: Path | None = None,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    wav_dir = wav_dir.resolve()
    wav_index = WavDirectoryIndex(wav_dir)
    generated = read_oto_timing_records(
        generated_oto.resolve(),
        wav_index.duration,
        identity_resolver=wav_index.identity,
        canonical_wav_resolver=wav_index.reference,
    )
    reference = read_oto_timing_records(
        reference_oto.resolve(),
        wav_index.duration,
        identity_resolver=wav_index.identity,
        canonical_wav_resolver=wav_index.reference,
    )
    generated_map = {_record_key(row): row for row in generated}
    reference_map = {_record_key(row): row for row in reference}
    shared_keys = sorted(set(generated_map) & set(reference_map))
    errors: dict[str, list[float]] = defaultdict(list)
    by_role: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    comparison_rows: list[dict[str, object]] = []
    catastrophic_pre = 0
    for key in shared_keys:
        predicted = generated_map[key]
        target = reference_map[key]
        role = _alias_role(predicted.alias)
        for field in ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs"):
            delta = abs(float(getattr(predicted, field)) - float(getattr(target, field)))
            errors[field].append(delta)
            by_role[role][field].append(delta)
        comparison_rows.append(
            {
                "wav": predicted.wav,
                "alias": predicted.alias,
                "occurrence": predicted.occurrence,
                "alias_role": role,
                **{
                    f"{field}_generated_ms": float(getattr(predicted, field))
                    for field in ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs")
                },
                **{
                    f"{field}_reference_ms": float(getattr(target, field))
                    for field in ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs")
                },
                **{
                    f"{field}_error_ms": abs(float(getattr(predicted, field)) - float(getattr(target, field)))
                    for field in ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs")
                },
                "pre_error_gt_200ms": abs(predicted.pre_abs - target.pre_abs) > 200.0,
            }
        )
        if abs(predicted.pre_abs - target.pre_abs) > 200.0:
            catastrophic_pre += 1
    result: dict[str, Any] = {
        "generated_oto": str(generated_oto.resolve()),
        "reference_oto": str(reference_oto.resolve()),
        "wav_dir": str(wav_dir),
        "generated_rows": len(generated),
        "reference_rows": len(reference),
        "matched_rows": len(shared_keys),
        "generated_unmatched_rows": len(generated_map) - len(shared_keys),
        "reference_unmatched_rows": len(reference_map) - len(shared_keys),
        "metrics_ms": {field: _summarize(values) for field, values in sorted(errors.items())},
        "catastrophic_pre_error_gt_200ms": {
            "count": catastrophic_pre,
            "rate": catastrophic_pre / max(1, len(shared_keys)),
        },
        "by_alias_role": {
            role: {field: _summarize(values) for field, values in sorted(role_errors.items())}
            for role, role_errors in sorted(by_role.items())
        },
        "reference_kind": "manual_oto_convention_not_acoustic_landmark_gold",
    }
    if output_csv:
        _write_evaluation_rows(output_csv.resolve(), comparison_rows)
        result["output_csv"] = str(output_csv.resolve())
    if output_json:
        payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        write_lines_atomic(output_json.resolve(), payload.splitlines(), encoding="utf-8", newline="\n")
    return result


def _write_evaluation_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write exact-match comparison rows in a review-friendly deterministic order."""

    fieldnames = (
        "wav", "alias", "occurrence", "alias_role",
        "offset_abs_error_ms", "overlap_abs_error_ms", "pre_abs_error_ms", "fixed_abs_error_ms", "cutoff_abs_error_ms",
        "pre_error_gt_200ms",
        "offset_abs_generated_ms", "overlap_abs_generated_ms", "pre_abs_generated_ms", "fixed_abs_generated_ms", "cutoff_abs_generated_ms",
        "offset_abs_reference_ms", "overlap_abs_reference_ms", "pre_abs_reference_ms", "fixed_abs_reference_ms", "cutoff_abs_reference_ms",
    )
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["pre_abs_error_ms"]),
            str(row["wav"]),
            str(row["alias"]),
            int(row["occurrence"]),
        ),
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(ordered)
    write_lines_atomic(path, buffer.getvalue().splitlines(), encoding="utf-8-sig", newline="\n", backup_existing=False)


def read_oto_timing_records(
    path: Path,
    duration_resolver,
    *,
    identity_resolver=None,
    canonical_wav_resolver=None,
    strict: bool = True,
    issues: list[dict[str, Any]] | None = None,
) -> list[OtoTimingRecord]:
    text, _encoding, _newline = read_text_auto(path)
    occurrence_count: Counter[tuple[str, str]] = Counter()
    rows: list[OtoTimingRecord] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or "=" not in raw:
            continue
        try:
            wav, payload = raw.split("=", 1)
            fields = payload.split(",")
            if len(fields) != 6:
                raise ValueError(f"OTO 필드 수가 6이 아닙니다")
            alias = fields[0]
            offset, consonant, cutoff, preutterance, overlap = (
                _parse_oto_float(value) for value in fields[1:]
            )
            raw_wav = wav.strip()
            canonical_wav = canonical_wav_resolver(raw_wav) if canonical_wav_resolver else raw_wav
            duration = duration_resolver(canonical_wav)
            cutoff_abs = oto_cutoff_end_abs(duration_ms=duration, offset=offset, cutoff=cutoff)
            wav_key = identity_resolver(canonical_wav) if identity_resolver else normalize_wav_key(canonical_wav)
            base_key = (wav_key, alias)
            occurrence = occurrence_count[base_key]
            occurrence_count[base_key] += 1
            rows.append(
                OtoTimingRecord(
                    wav=canonical_wav,
                    wav_key=wav_key,
                    alias=alias,
                    occurrence=occurrence,
                    offset_abs=offset,
                    overlap_abs=offset + overlap,
                    pre_abs=offset + preutterance,
                    fixed_abs=offset + consonant,
                    cutoff_abs=cutoff_abs,
                )
            )
        except Exception as error:
            if strict:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if issues is not None:
                issues.append({"line": line_number, "error": str(error), "raw": raw[:200]})
    return rows


def read_voicebank_oto_timing_records(
    wav_root: Path,
    *,
    strict: bool = True,
    issues: list[dict[str, Any]] | None = None,
) -> list[OtoTimingRecord]:
    """Read every ``oto.ini`` in a multipitch voicebank with folder-local WAV lookup.

    Each OTO file is resolved against its own directory first, then canonicalized
    to a path relative to ``wav_root``.  This keeps duplicate basenames at
    different pitches distinct and supports legacy pitch-prefixed OTO keys.
    """

    wav_root = wav_root.resolve()
    root_index = WavDirectoryIndex(wav_root)
    rows: list[OtoTimingRecord] = []
    oto_paths = sorted(
        (path for path in wav_root.rglob("*") if path.is_file() and path.name.casefold() == "oto.ini"),
        key=lambda path: os.path.normcase(str(path.relative_to(wav_root))),
    )
    for oto_path in oto_paths:
        local_index = WavDirectoryIndex(oto_path.parent, recursive=False)
        local_issues: list[dict[str, Any]] = []

        def canonical_reference(raw_wav: str, *, _local_index=local_index) -> str:
            return _local_index.resolve(raw_wav).relative_to(wav_root).as_posix()

        parsed = read_oto_timing_records(
            oto_path,
            root_index.duration,
            identity_resolver=root_index.identity,
            canonical_wav_resolver=canonical_reference,
            strict=strict,
            issues=local_issues,
        )
        rows.extend(parsed)
        if issues is not None:
            for issue in local_issues:
                issues.append({"oto_path": oto_path.relative_to(wav_root).as_posix(), **issue})
    return rows


def canonicalize_voicebank_oto(wav_root: Path, output: Path, *, oto_root: Path | None = None) -> dict[str, Any]:
    """Write a comparison-only merged OTO while preserving every raw parameter field.

    The source OTO files are never changed.  Only the left-side WAV reference is
    rewritten to its actual root-relative path, so positive cutoff/right-blank
    conventions and any reviewed timing values remain exactly as authored.
    """

    wav_root = wav_root.resolve()
    oto_root = (oto_root or wav_root).resolve()
    output = output.resolve()
    root_index = WavDirectoryIndex(wav_root)
    oto_paths = sorted(
        (path for path in oto_root.rglob("*") if path.is_file() and path.name.casefold() == "oto.ini"),
        key=lambda path: os.path.normcase(str(path.relative_to(oto_root))),
    )
    source_hashes = {path.relative_to(oto_root).as_posix(): file_sha256(path) for path in oto_paths}
    lines: list[str] = []
    rewritten_rows = 0
    for oto_path in oto_paths:
        local_index = WavDirectoryIndex(wav_root / oto_path.parent.relative_to(oto_root), recursive=False)
        text, _encoding, _newline = read_text_auto(oto_path)
        for line_number, raw in enumerate(text.splitlines(), 1):
            if not raw.strip() or "=" not in raw:
                lines.append(raw)
                continue
            wav, payload = raw.split("=", 1)
            try:
                canonical_wav = local_index.resolve(wav.strip()).relative_to(wav_root).as_posix()
            except Exception as error:
                raise ValueError(f"{oto_path}:{line_number}: {error}") from error
            lines.append(f"{canonical_wav}={payload}")
            rewritten_rows += 1
    write_lines_atomic(output, lines, encoding="utf-8", newline="\n", backup_existing=False)
    changed = {
        relative: digest
        for relative, digest in source_hashes.items()
        if not (oto_root / relative).is_file() or file_sha256(oto_root / relative) != digest
    }
    if changed:
        raise RuntimeError(f"source OTO changed while canonicalizing: {sorted(changed)}")
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "source_oto_file_sha256": source_hashes,
        "oto_file_count": len(oto_paths),
        "oto_row_count": rewritten_rows,
        "oto_root": str(oto_root),
        "source_files_modified": False,
    }


# Backward-compatible private alias.  New provenance-sensitive importers use
# the public name so they do not need to depend on an implementation detail.
_read_timing_records = read_oto_timing_records


def _record_key(row: OtoTimingRecord) -> tuple[str, str, int]:
    return row.wav_key, row.alias, row.occurrence


def _parse_oto_float(value: str) -> float:
    stripped = value.strip()
    return float(stripped) if stripped else 0.0


def _normalize_relative_reference(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value.strip().replace("\\", "/"))
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts:
        return ""
    filename = parts[-1]
    stem = Path(filename).stem
    suffix = Path(filename).suffix.casefold()
    parts[-1] = f"{normalize_wav_key(stem)}{suffix}"
    return "/".join(part.casefold() for part in parts)


def _legacy_filename_variants(value: str) -> tuple[str, ...]:
    variants = [value]
    for source_encoding, target_encoding in (("cp932", "cp949"), ("cp949", "cp932")):
        try:
            converted = value.encode(source_encoding).decode(target_encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if converted not in variants:
            variants.append(converted)
    return tuple(variants)


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mae": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mae": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _alias_role(alias: str) -> str:
    value = alias.strip().casefold()
    parts = value.split()
    if value.startswith("br"):
        return "breath"
    if value.startswith("-"):
        return "cv_head"
    if len(parts) >= 2 and parts[-1] == "r":
        return "release"
    if len(parts) == 2 and parts[0] == parts[1]:
        return "vv"
    if len(parts) >= 2:
        return "vc"
    return "cv_or_cvc"
