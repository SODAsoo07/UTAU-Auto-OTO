from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from universal_auto_oto.alias_type_symbolic import export_symbolic_alias_types


RULE_SHA = "a" * 64


def _row(alias: str, source_index: int, alias_type: dict[str, object]) -> dict[str, object]:
    return {
        "recording_wav": "x.wav",
        "alias": alias,
        "source_index": source_index,
        "phone_indices": [0, 1],
        "alias_type": alias_type,
        "alias_type_label_source": "alias_notation_phone_inventory_rules",
        "alias_type_uses_oto_timing": False,
        "alias_type_rule_set_sha256": RULE_SHA,
        "recommended_template_id": "builtin:cv_hard",
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_symbolic_export_fills_certain_rows_and_queues_only_uncertainty(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(
        source,
        [
            _row(
                "- ka",
                0,
                {
                    "structural_role": "CV",
                    "acoustic_subtype": "hard",
                    "transition_type": "plain",
                    "edge_role": "head",
                    "confidence": 1.0,
                    "reasons": ["edge_role_source:alias_head_marker", "rule:cv_hard"],
                },
            ),
            _row(
                "x a",
                1,
                {
                    "structural_role": "CV",
                    "acoustic_subtype": "unknown",
                    "transition_type": "unknown",
                    "edge_role": "medial",
                    "confidence": 0.0,
                    "reasons": ["no_subtype_rule_match:missing=none"],
                },
            ),
        ],
    )
    labels = tmp_path / "labels.jsonl"
    queue = tmp_path / "queue.csv"
    report_path = tmp_path / "report.json"

    report = export_symbolic_alias_types(source, labels, queue, output_json=report_path)

    exported = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    assert exported[0]["edge_role"] == "head"
    assert exported[0]["needs_review"] is False
    assert exported[0]["uses_oto_timing"] is False
    assert exported[0]["gold_eligible"] is False
    assert exported[0]["automatic_gold_created"] is False
    with queue.open("r", encoding="utf-8-sig", newline="") as stream:
        queued = list(csv.DictReader(stream))
    assert [row["alias"] for row in queued] == ["x a"]
    assert queued[0]["status"] == "pending"
    assert report["row_count"] == 2
    assert report["automatic_row_count"] == 1
    assert report["review_row_count"] == 1
    assert report["classification_complete"] is False


def test_symbolic_export_writes_header_only_queue_when_all_rows_are_certain(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_rows(
        source,
        [
            _row(
                "- ka",
                0,
                {
                    "structural_role": "CV",
                    "acoustic_subtype": "hard",
                    "transition_type": "plain",
                    "edge_role": "head",
                    "confidence": 1.0,
                    "reasons": ["rule:cv_hard"],
                },
            )
        ],
    )

    report = export_symbolic_alias_types(source, tmp_path / "labels.jsonl", tmp_path / "queue.csv")

    with (tmp_path / "queue.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        assert list(csv.DictReader(stream)) == []
    assert report["classification_complete"] is True
    assert report["review_row_count"] == 0


def test_symbolic_export_rejects_duplicate_row_identity(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    row = _row(
        "ka",
        0,
        {
            "structural_role": "CV",
            "acoustic_subtype": "hard",
            "transition_type": "plain",
            "edge_role": "isolated",
            "confidence": 1.0,
            "reasons": ["rule:cv_hard"],
        },
    )
    _write_rows(source, [row, row])

    with pytest.raises(ValueError, match="identity is duplicated"):
        export_symbolic_alias_types(source, tmp_path / "labels.jsonl", tmp_path / "queue.csv")
