from __future__ import annotations

import csv
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .types import EVENT_LABELS, FRAME_LABELS, MANUAL_SOURCE, REJECTED_LABEL_SOURCES

WAV_EXTENSIONS = {".wav"}


class ManifestValidationError(ValueError):
    """Raised when a gold manifest row is not suitable for PoC training."""


@dataclass(frozen=True)
class ScaffoldSummary:
    rows: int
    labelled_rows: int
    needs_manual_labels: int


def wav_duration_ms(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
    if sample_rate <= 0:
        raise ManifestValidationError(f"Invalid sample rate in wav: {path}")
    return 1000.0 * float(frames) / float(sample_rate)


def load_manifest_jsonl(
    path: str | Path,
    *,
    require_manual: bool = True,
    require_labels: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(f"{source_path}:{line_no}: invalid JSON: {exc}") from exc
            validate_manifest_row(row, require_manual=require_manual, require_labels=require_labels)
            rows.append(row)
    return rows


def write_manifest_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def validate_manifest_row(
    row: dict,
    *,
    require_manual: bool = True,
    require_labels: bool = False,
) -> None:
    if not isinstance(row, dict):
        raise ManifestValidationError("manifest row must be an object")
    wav_path = row.get("wav_path")
    if not wav_path:
        raise ManifestValidationError("manifest row requires wav_path")
    source = str(row.get("label_source") or row.get("source") or MANUAL_SOURCE).strip().lower()
    if source in REJECTED_LABEL_SOURCES:
        raise ManifestValidationError(
            f"{wav_path}: rejected label source '{source}'. Use hand-labelled manual_gold rows."
        )
    if require_manual and source != MANUAL_SOURCE:
        raise ManifestValidationError(f"{wav_path}: label_source must be '{MANUAL_SOURCE}', got '{source}'")
    frame_labels = row.get("frame_labels") or []
    events = row.get("events") or []
    if require_labels and not frame_labels:
        raise ManifestValidationError(f"{wav_path}: at least one frame label segment is required")
    for segment in frame_labels:
        _validate_frame_segment(wav_path, segment)
    for event in events:
        _validate_event(wav_path, event)


def _validate_frame_segment(wav_path: object, segment: object) -> None:
    if not isinstance(segment, dict):
        raise ManifestValidationError(f"{wav_path}: frame label segment must be an object")
    label = str(segment.get("label") or "")
    if label not in FRAME_LABELS:
        raise ManifestValidationError(f"{wav_path}: invalid frame label '{label}'")
    start_ms = float(segment.get("start_ms", -1.0))
    end_ms = float(segment.get("end_ms", -1.0))
    if start_ms < 0.0 or end_ms <= start_ms:
        raise ManifestValidationError(f"{wav_path}: invalid frame label span {start_ms}-{end_ms} ms")
    confidence = float(segment.get("confidence", 1.0))
    if confidence < 0.0 or confidence > 1.0:
        raise ManifestValidationError(f"{wav_path}: frame label confidence must be 0..1")


def _validate_event(wav_path: object, event: object) -> None:
    if not isinstance(event, dict):
        raise ManifestValidationError(f"{wav_path}: event must be an object")
    label = str(event.get("label") or "")
    if label not in EVENT_LABELS:
        raise ManifestValidationError(f"{wav_path}: invalid event label '{label}'")
    time_ms = float(event.get("time_ms", -1.0))
    if time_ms < 0.0:
        raise ManifestValidationError(f"{wav_path}: invalid event time {time_ms} ms")
    confidence = float(event.get("confidence", 1.0))
    if confidence < 0.0 or confidence > 1.0:
        raise ManifestValidationError(f"{wav_path}: event confidence must be 0..1")


def build_goldset_scaffold(
    wav_dir: str | Path,
    *,
    language: str,
    format_type: str,
    aliases_path: str | Path | None = None,
    relative_to: str | Path | None = None,
) -> tuple[list[dict], ScaffoldSummary]:
    root = Path(wav_dir)
    alias_map = load_alias_sidecar(aliases_path) if aliases_path else {}
    base = Path(relative_to).resolve() if relative_to else None
    rows: list[dict] = []
    for wav_path in sorted(root.rglob("*")):
        if wav_path.suffix.lower() not in WAV_EXTENSIONS:
            continue
        resolved = wav_path.resolve()
        row_wav_path = _rel_or_abs_path(resolved, base)
        alias_info = alias_map.get(wav_path.name) or alias_map.get(wav_path.stem) or {}
        aliases = list(alias_info.get("aliases") or [])
        expected_phones = list(alias_info.get("expected_phones") or [])
        rows.append(
            {
                "schema_version": 1,
                "row_id": wav_path.stem,
                "wav_name": wav_path.name,
                "wav_path": row_wav_path,
                "duration_ms": wav_duration_ms(resolved),
                "language": language,
                "format_type": format_type,
                "label_source": MANUAL_SOURCE,
                "aliases": aliases,
                "expected_phones": expected_phones,
                "frame_labels": [],
                "events": [],
                "needs_manual_labels": True,
            }
        )
    summary = ScaffoldSummary(rows=len(rows), labelled_rows=0, needs_manual_labels=len(rows))
    return rows, summary


def load_alias_sidecar(path: str | Path) -> dict[str, dict[str, list[str]]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return _load_alias_jsonl(source)
    if suffix in {".csv", ".tsv"}:
        return _load_alias_table(source, delimiter="\t" if suffix == ".tsv" else ",")
    return _load_alias_plain(source)


def _load_alias_jsonl(path: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("wav_name") or row.get("wav_path") or row.get("row_id") or "")
            if not key:
                continue
            result[Path(key).name] = {
                "aliases": _listify(row.get("aliases")),
                "expected_phones": _listify(row.get("expected_phones") or row.get("phones")),
            }
    return result


def _load_alias_table(path: Path, *, delimiter: str) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            key = row.get("wav_name") or row.get("wav_path") or row.get("row_id")
            if not key:
                continue
            result[Path(key).name] = {
                "aliases": _split_list(row.get("aliases")),
                "expected_phones": _split_list(row.get("expected_phones") or row.get("phones")),
            }
    return result


def _load_alias_plain(path: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split(",", maxsplit=2)
            key = parts[0].strip()
            aliases = _split_list(parts[1]) if len(parts) >= 2 else []
            phones = _split_list(parts[2]) if len(parts) >= 3 else []
            result[Path(key).name] = {"aliases": aliases, "expected_phones": phones}
    return result


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    if "|" in value:
        return [part.strip() for part in value.split("|") if part.strip()]
    return [part.strip() for part in value.split() if part.strip()]


def _listify(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _split_list(value)
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                phone = item.get("phone") or item.get("label")
                if phone:
                    out.append(str(phone))
            else:
                out.append(str(item))
        return out
    return [str(value)]


def _rel_or_abs_path(path: Path, base: Path | None) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
