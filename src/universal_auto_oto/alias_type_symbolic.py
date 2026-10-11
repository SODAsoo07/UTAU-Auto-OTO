from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .domain import AcousticSubtype, EdgeRole, StructuralRole, TransitionType
from .oto_io import write_lines_atomic


CREATION_METHOD = "alias_notation_phone_inventory_rules"
DEFAULT_LABELS_FILENAME = "symbolic_alias_type_labels.jsonl"
DEFAULT_REVIEW_QUEUE_FILENAME = "symbolic_alias_type_review_queue.csv"
DEFAULT_REPORT_FILENAME = "symbolic_alias_type_export_report.json"
REVIEW_FIELDS = (
    "row_index",
    "row_key",
    "recording_wav",
    "alias",
    "source_index",
    "phone_indices",
    "structural_role",
    "acoustic_subtype",
    "transition_type",
    "edge_role",
    "confidence",
    "review_reasons",
    "corrected_structural_role",
    "corrected_acoustic_subtype",
    "corrected_transition_type",
    "corrected_edge_role",
    "status",
    "note",
)


def export_symbolic_alias_types(
    row_report_jsonl: Path,
    output_labels_jsonl: Path,
    review_queue_csv: Path,
    *,
    output_json: Path | None = None,
    overwrite: bool = False,
    backup_existing: bool = False,
) -> dict[str, object]:
    """Export deterministic symbolic labels without treating them as manual Gold."""

    source = row_report_jsonl.resolve()
    labels_path = output_labels_jsonl.resolve()
    queue_path = review_queue_csv.resolve()
    report_path = output_json.resolve() if output_json is not None else None
    targets = [labels_path, queue_path, *(tuple([report_path]) if report_path is not None else ())]
    if len(set(targets)) != len(targets):
        raise ValueError("symbolic AliasType output paths must be distinct")
    for target in targets:
        if target.exists() and not overwrite:
            raise FileExistsError(target)

    source_rows = _load_jsonl(source)
    source_sha256 = _sha256(source)
    labels: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    seen_row_keys: set[str] = set()

    for row_index, source_row in enumerate(source_rows):
        row_key = _row_key(source_row, row_index)
        if row_key in seen_row_keys:
            raise ValueError(f"row report identity is duplicated: {row_key}")
        seen_row_keys.add(row_key)
        if source_row.get("alias_type_label_source") != CREATION_METHOD:
            raise ValueError(f"row_report[{row_index}] lacks symbolic AliasType provenance")
        if source_row.get("alias_type_uses_oto_timing") is not False:
            raise ValueError(f"row_report[{row_index}] does not prove OTO timing independence")

        alias_type = source_row.get("alias_type")
        if not isinstance(alias_type, Mapping):
            raise ValueError(f"row_report[{row_index}].alias_type must be a mapping")
        normalized = _validated_alias_type(alias_type, row_index)
        reasons = _string_sequence(alias_type.get("reasons"), f"row_report[{row_index}].alias_type.reasons")
        review_reasons = _review_reasons(normalized, reasons)
        recording_wav = _nonempty(
            source_row.get("recording_wav") or source_row.get("wav"),
            f"row_report[{row_index}].recording_wav",
        )
        alias = _alias(source_row.get("alias"), f"row_report[{row_index}].alias")
        phone_indices = _phone_indices(source_row.get("phone_indices"), row_index)
        label = {
            "schema_version": 1,
            "reference_kind": "symbolic_alias_type_label",
            "creation_method": CREATION_METHOD,
            "gold_eligible": False,
            "automatic_gold_created": False,
            "uses_oto_timing": False,
            "source_row_report": str(source),
            "source_row_report_sha256": source_sha256,
            "alias_type_rule_set_sha256": _sha_value(
                source_row.get("alias_type_rule_set_sha256"),
                f"row_report[{row_index}].alias_type_rule_set_sha256",
            ),
            "row_index": row_index,
            "row_key": row_key,
            "recording_wav": recording_wav,
            "alias": alias,
            "source_index": source_row.get("source_index", row_index),
            "phone_indices": phone_indices,
            **normalized,
            "reasons": list(reasons),
            "recommended_template_id": source_row.get("recommended_template_id"),
            "needs_review": bool(review_reasons),
            "review_reasons": list(review_reasons),
        }
        labels.append(label)
        if review_reasons:
            review_rows.append(
                {
                    **label,
                    "phone_indices": json.dumps(phone_indices, separators=(",", ":")),
                    "review_reasons": ";".join(review_reasons),
                    "corrected_structural_role": "",
                    "corrected_acoustic_subtype": "",
                    "corrected_transition_type": "",
                    "corrected_edge_role": "",
                    "status": "pending",
                    "note": "",
                }
            )

    _write_jsonl(labels_path, labels, backup_existing=backup_existing)
    _write_csv(queue_path, review_rows, backup_existing=backup_existing)
    report = {
        "schema_version": 1,
        "purpose": "symbolic_alias_type_export",
        "passed": True,
        "classification_complete": not review_rows,
        "reference_kind": "symbolic_alias_type_label",
        "creation_method": CREATION_METHOD,
        "gold_eligible": False,
        "automatic_gold_created": False,
        "uses_oto_timing": False,
        "source_row_report": str(source),
        "source_row_report_sha256": source_sha256,
        "output_labels": str(labels_path),
        "output_labels_sha256": _sha256(labels_path),
        "review_queue": str(queue_path),
        "review_queue_sha256": _sha256(queue_path),
        "row_count": len(labels),
        "automatic_row_count": len(labels) - len(review_rows),
        "review_row_count": len(review_rows),
        "review_reason_counts": dict(
            sorted(Counter(reason for label in labels for reason in label["review_reasons"]).items())
        ),
        "next_action": (
            "review only the rows in review_queue"
            if review_rows
            else "no symbolic AliasType review is required; manual Gold remains a separate optional accuracy gate"
        ),
    }
    if report_path is not None:
        _write_report(report_path, report, backup_existing=backup_existing)
        report["output_json"] = str(report_path)
        report["output_json_sha256"] = _sha256(report_path)
    return report


def _validated_alias_type(value: Mapping[str, object], row_index: int) -> dict[str, object]:
    try:
        structural_role = StructuralRole(str(value["structural_role"])).value
        acoustic_subtype = AcousticSubtype(str(value["acoustic_subtype"])).value
        transition_type = TransitionType(str(value["transition_type"])).value
        edge_role = EdgeRole(str(value["edge_role"])).value
    except (KeyError, ValueError) as error:
        raise ValueError(f"row_report[{row_index}].alias_type is invalid: {error}") from error
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"row_report[{row_index}].alias_type.confidence must be within 0..1")
    return {
        "structural_role": structural_role,
        "acoustic_subtype": acoustic_subtype,
        "transition_type": transition_type,
        "edge_role": edge_role,
        "confidence": float(confidence),
    }


def _review_reasons(alias_type: Mapping[str, object], reasons: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    if alias_type["structural_role"] == StructuralRole.UNKNOWN.value:
        result.append("unknown_structural_role")
    if alias_type["acoustic_subtype"] == AcousticSubtype.UNKNOWN.value:
        result.append("unknown_acoustic_subtype")
    if float(alias_type["confidence"]) < 1.0:
        result.append("symbolic_confidence_below_1")
    if any(reason == "scope_excluded:CC_CCV" for reason in reasons):
        result.append("scope_excluded_CC_CCV")
    if any(reason == "unsupported_structural_role" for reason in reasons):
        result.append("unsupported_structural_role")
    if any(reason.startswith("no_subtype_rule_match:") for reason in reasons):
        result.append("no_subtype_rule_match")
    return tuple(dict.fromkeys(result))


def _row_key(row: Mapping[str, object], index: int) -> str:
    return "|".join(
        (
            _nonempty(row.get("recording_wav") or row.get("wav"), f"row_report[{index}].recording_wav"),
            _alias(row.get("alias"), f"row_report[{index}].alias"),
            str(row.get("source_index", index)),
        )
    )


def _alias(value: object, path: str) -> str:
    # An empty OTO alias is valid; WAV + source_index still distinguishes the row.
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _phone_indices(value: object, row_index: int) -> list[int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise ValueError(f"row_report[{row_index}].phone_indices must be non-negative integers")
    return list(value)


def _string_sequence(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{path} must be a string sequence")
    return tuple(value)


def _sha_value(value: object, path: str) -> str:
    text = _nonempty(value, path).casefold()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{path} must be a SHA-256 digest")
    return text


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} line {line_number} must be a mapping")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    backup_existing: bool = False,
) -> None:
    write_lines_atomic(
        path,
        (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in rows),
        encoding="utf-8",
        newline="\n",
        backup_existing=backup_existing,
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    backup_existing: bool = False,
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(REVIEW_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in REVIEW_FIELDS})
    write_lines_atomic(
        path,
        stream.getvalue().splitlines(),
        encoding="utf-8-sig",
        newline="\r\n",
        backup_existing=backup_existing,
    )


def _write_report(path: Path, payload: Mapping[str, object], *, backup_existing: bool = False) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    write_lines_atomic(
        path,
        text.splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=backup_existing,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
