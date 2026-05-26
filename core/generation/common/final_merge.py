from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from core.generation.common.oto_encoding import detect_text_file_encoding, write_oto_lines_with_encoding
from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.generation.common.validation_splitter import (
    SPLIT_ATTENTION_ONLY,
    SPLIT_CLEAN,
    SPLIT_FIX_REQUIRED,
    ValidationRecord,
    load_validation_records_with_errors,
)


FINAL_MERGE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MergeResult:
    ok: bool
    final_path: str
    report_path: str
    unresolved_row_ids: tuple[str, ...]
    row_count: int
    attention_only_acknowledged: bool = False
    attention_only_unacknowledged_count: int = 0
    merge_input_errors: tuple[str, ...] = ()
    validation_errors: tuple[str, ...] = ()


def merge_split_oto(
    *,
    validation_jsonl_path: str,
    final_path: str,
    edited_fix_required_path: str = "",
    edited_attention_only_path: str = "",
    allow_unresolved: bool = False,
    acknowledge_attention_only: bool = False,
    language: str = "",
    preferred_encoding_source: str = "",
) -> MergeResult:
    loaded_records, validation_errors = load_validation_records_with_errors(validation_jsonl_path, require_rows=True)
    records = sorted(loaded_records, key=lambda item: item.line_index)
    fix_records = [r for r in records if r.split == SPLIT_FIX_REQUIRED]
    attention_records = [r for r in records if r.split == SPLIT_ATTENTION_ONLY]
    edited_fix = _read_ini_lines(edited_fix_required_path)
    edited_attention = _read_ini_lines(edited_attention_only_path)

    replacements: dict[str, str] = {}
    unresolved: list[str] = []
    merge_input_errors: list[str] = []
    if not validation_errors:
        _load_replacements(
            replacements,
            unresolved,
            merge_input_errors,
            fix_records,
            edited_fix,
            require_changed=True,
            split_name=SPLIT_FIX_REQUIRED,
        )
        _load_replacements(
            replacements,
            unresolved,
            merge_input_errors,
            attention_records,
            edited_attention,
            require_changed=False,
            split_name=SPLIT_ATTENTION_ONLY,
        )

    final_lines: list[str] = []
    merged_rows: list[dict[str, object]] = []
    if not validation_errors:
        for record in records:
            if record.split == SPLIT_CLEAN:
                _append_final_line(final_lines, merged_rows, record, record.raw_line, "clean")
                continue
            if record.row_id in replacements:
                output_source = f"edited_{record.split}"
                _append_final_line(final_lines, merged_rows, record, replacements[record.row_id], output_source)
                continue
            if record.split == SPLIT_ATTENTION_ONLY:
                _append_final_line(
                    final_lines,
                    merged_rows,
                    record,
                    record.raw_line,
                    "attention_only_original",
                )
                continue
            unresolved.append(record.row_id)
            if allow_unresolved:
                _append_final_line(
                    final_lines,
                    merged_rows,
                    record,
                    record.raw_line,
                    "fix_required_unresolved_original",
                )

    dedup_unresolved = tuple(dict.fromkeys(unresolved))
    dedup_input_errors = tuple(dict.fromkeys(merge_input_errors))
    dedup_validation_errors = tuple(dict.fromkeys(validation_errors))
    attention_unacknowledged_count = len(attention_records) if attention_records and not acknowledge_attention_only else 0
    ok = (allow_unresolved or not dedup_unresolved) and not dedup_input_errors and not dedup_validation_errors
    os.makedirs(os.path.dirname(os.path.abspath(final_path)) or ".", exist_ok=True)
    report_path = os.path.splitext(final_path)[0] + ".merge_report.json"
    validation_jsonl_sha256 = _sha256_file(validation_jsonl_path) if os.path.isfile(str(validation_jsonl_path or "")) else ""
    report = {
        "schema_version": FINAL_MERGE_SCHEMA_VERSION,
        "ok": bool(ok),
        "allow_unresolved": bool(allow_unresolved),
        "validation_jsonl_path": os.path.abspath(validation_jsonl_path),
        "validation_jsonl_sha256": validation_jsonl_sha256,
        "validation_error_count": len(dedup_validation_errors),
        "validation_errors": list(dedup_validation_errors),
        "final_path": os.path.abspath(final_path),
        "row_count": len(final_lines),
        "merged_row_count": len(merged_rows),
        "merged_rows": merged_rows,
        "unresolved_count": len(dedup_unresolved),
        "unresolved_row_ids": list(dedup_unresolved),
        "merge_input_error_count": len(dedup_input_errors),
        "merge_input_errors": list(dedup_input_errors),
        "attention_only_count": len(attention_records),
        "attention_only_acknowledged": bool(acknowledge_attention_only),
        "attention_only_unacknowledged_count": int(attention_unacknowledged_count),
        "final_sha256": "",
        "final_size_bytes": 0,
        "final_write_encoding": "",
    }

    if ok:
        preferred = detect_text_file_encoding(preferred_encoding_source) if preferred_encoding_source else ""
        encoding = write_oto_lines_with_encoding(final_path, final_lines, language=language, preferred_encoding=preferred)
        report["final_sha256"] = _sha256_file(final_path)
        report["final_size_bytes"] = os.path.getsize(final_path)
        report["final_write_encoding"] = encoding

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return MergeResult(
        ok=ok,
        final_path=final_path,
        report_path=report_path,
        unresolved_row_ids=dedup_unresolved,
        row_count=len(final_lines),
        attention_only_acknowledged=bool(acknowledge_attention_only),
        attention_only_unacknowledged_count=attention_unacknowledged_count,
        merge_input_errors=dedup_input_errors,
        validation_errors=dedup_validation_errors,
    )


def _load_replacements(
    replacements: dict[str, str],
    unresolved: list[str],
    errors: list[str],
    records: Sequence[ValidationRecord],
    edited_lines: Sequence[str],
    *,
    require_changed: bool,
    split_name: str,
) -> None:
    for idx, record in enumerate(records):
        if idx >= len(edited_lines):
            if split_name == SPLIT_FIX_REQUIRED:
                unresolved.append(record.row_id)
            continue
        line = edited_lines[idx]
        if not _line_matches_record(line, record):
            errors.append(f"{split_name}.row_mismatch:{record.row_id}")
            if split_name == SPLIT_FIX_REQUIRED:
                unresolved.append(record.row_id)
            continue
        if require_changed and _normalize_line(line) == _normalize_line(record.raw_line):
            unresolved.append(record.row_id)
            continue
        replacements[record.row_id] = line
    for extra_idx in range(len(records), len(edited_lines)):
        errors.append(f"{split_name}.extra_row:{extra_idx}")


def _append_final_line(
    final_lines: list[str],
    merged_rows: list[dict[str, object]],
    record: ValidationRecord,
    line: str,
    output_source: str,
) -> None:
    output_index = len(final_lines)
    output_line = str(line or "")
    final_lines.append(output_line)
    merged_rows.append(_merged_row_lineage(record, output_index, output_line, output_source))


def _merged_row_lineage(
    record: ValidationRecord,
    output_index: int,
    output_line: str,
    output_source: str,
) -> dict[str, object]:
    return {
        "output_index": int(output_index),
        "line_index": int(record.line_index),
        "row_id": record.row_id,
        "split": record.split,
        "output_source": str(output_source or ""),
        "wav": record.wav,
        "alias": record.alias,
        "occurrence_index": int(record.occurrence_index),
        "role_family": record.role_family,
        "reasons": list(record.reasons),
        "slot_index": int(record.slot_index),
        "left_slot_index": int(record.left_slot_index),
        "right_slot_index": int(record.right_slot_index),
        "source_row_index": int(record.source_row_index),
        "source_timing_trusted": bool(record.source_timing_trusted),
        "validation_raw_line_sha256": _line_sha256(record.raw_line),
        "line_sha256": _line_sha256(output_line),
        "changed_from_validation": _normalize_line(output_line) != _normalize_line(record.raw_line),
    }


def _read_ini_lines(path: str) -> tuple[str, ...]:
    if not path or not os.path.isfile(path):
        return ()
    text = read_text_with_fallback(path)
    return tuple(line.rstrip("\r\n") for line in text.splitlines() if line.strip())


def _normalize_line(line: str) -> str:
    return str(line or "").strip().replace(" ", "")


def _line_matches_record(line: str, record: ValidationRecord) -> bool:
    parsed = parse_oto_line(line)
    if not parsed:
        return False
    return str(parsed.get("wav", "") or "") == record.wav and str(parsed.get("alias", "") or "") == record.alias


def _line_sha256(line: str) -> str:
    return hashlib.sha256(str(line or "").encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = [
    "FINAL_MERGE_SCHEMA_VERSION",
    "MergeResult",
    "merge_split_oto",
]
