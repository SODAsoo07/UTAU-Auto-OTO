"""Unit tests for ml.scripts.coarse_crnn.aggregate_ab_labels (Phase E)."""
from __future__ import annotations

import csv
import os
import sys
import tempfile

import pytest


# Make the scripts package importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.scripts.coarse_crnn.aggregate_ab_labels import (  # noqa: E402
    DEFECT_COLUMNS,
    _normalize_defect,
    aggregate,
    read_label_rows,
    render_markdown,
)


def _write_csv(rows: list[dict[str, str]], path: str) -> None:
    headers = [
        "case_id",
        "voicebank_id",
        "language",
        "wav_name",
        "alias",
        "version_id",
        *DEFECT_COLUMNS,
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _row(
    *,
    case_id: str,
    vb: str,
    lang: str,
    version: str,
    defects: dict[str, int] | None = None,
) -> dict[str, str]:
    out = {
        "case_id": case_id,
        "voicebank_id": vb,
        "language": lang,
        "wav_name": f"{case_id}.wav",
        "alias": "a k",
        "version_id": version,
    }
    for col in DEFECT_COLUMNS:
        out[col] = str((defects or {}).get(col, 0))
    out["notes"] = ""
    return out


def test_normalize_defect_treats_blank_as_zero():
    assert _normalize_defect("") == 0
    assert _normalize_defect(None) == 0
    assert _normalize_defect("0") == 0
    assert _normalize_defect("false") == 0
    assert _normalize_defect("no") == 0


def test_normalize_defect_treats_other_values_as_one():
    assert _normalize_defect("1") == 1
    assert _normalize_defect("true") == 1
    assert _normalize_defect("yes") == 1
    assert _normalize_defect("audible click") == 1


def test_aggregate_single_version_no_defects():
    rows = [_row(case_id=f"c{i}", vb="vb1", lang="korean", version="V1") for i in range(10)]
    result = aggregate(rows)
    assert result["row_count"] == 10
    stats = result["overall"]["V1"]
    for col in DEFECT_COLUMNS:
        assert stats[col] == 0.0
    assert stats["any_defect"] == 0.0
    assert stats["count"] == 10


def test_aggregate_per_category_rate():
    rows = [
        _row(case_id="c1", vb="vb1", lang="korean", version="V1", defects={"click": 1}),
        _row(case_id="c2", vb="vb1", lang="korean", version="V1", defects={"click": 1}),
        _row(case_id="c3", vb="vb1", lang="korean", version="V1", defects={"vc_misalign": 1}),
        _row(case_id="c4", vb="vb1", lang="korean", version="V1"),
    ]
    result = aggregate(rows)
    stats = result["overall"]["V1"]
    assert stats["click"] == 0.5  # 2 of 4
    assert stats["vc_misalign"] == 0.25
    assert stats["any_defect"] == 0.75  # c1, c2, c3


def test_aggregate_any_defect_counts_rows_not_defects():
    """A row with multiple defects still counts once toward 'any_defect'."""
    rows = [
        _row(case_id="c1", vb="vb1", lang="korean", version="V1", defects={"click": 1, "vc_misalign": 1}),
        _row(case_id="c2", vb="vb1", lang="korean", version="V1"),
    ]
    result = aggregate(rows)
    stats = result["overall"]["V1"]
    assert stats["click"] == 0.5
    assert stats["vc_misalign"] == 0.5
    assert stats["any_defect"] == 0.5  # only c1 has any defect


def test_aggregate_per_language_breakdown():
    rows = [
        _row(case_id="ko1", vb="vb1", lang="korean", version="V1", defects={"vc_misalign": 1}),
        _row(case_id="ko2", vb="vb1", lang="korean", version="V1", defects={"vc_misalign": 1}),
        _row(case_id="ja1", vb="vb2", lang="japanese", version="V1"),
        _row(case_id="ja2", vb="vb2", lang="japanese", version="V1"),
    ]
    result = aggregate(rows)
    ko = result["per_language"]["V1"]["korean"]
    ja = result["per_language"]["V1"]["japanese"]
    assert ko["any_defect"] == 1.0
    assert ja["any_defect"] == 0.0
    assert ko["vc_misalign"] == 1.0
    assert ja["vc_misalign"] == 0.0


def test_aggregate_per_voicebank_split():
    rows = [
        _row(case_id="c1", vb="vb1", lang="korean", version="V1", defects={"click": 1}),
        _row(case_id="c2", vb="vb1", lang="korean", version="V1"),
        _row(case_id="c3", vb="vb2", lang="korean", version="V1", defects={"click": 1}),
        _row(case_id="c4", vb="vb2", lang="korean", version="V1", defects={"click": 1}),
    ]
    result = aggregate(rows)
    vb1 = result["per_voicebank"]["V1"]["vb1"]
    vb2 = result["per_voicebank"]["V1"]["vb2"]
    assert vb1["click"] == 0.5
    assert vb2["click"] == 1.0


def test_aggregate_deltas_vs_reference():
    rows = []
    # V1 (reference): 50% defect
    rows.append(_row(case_id="c1", vb="vb1", lang="korean", version="V1", defects={"click": 1}))
    rows.append(_row(case_id="c2", vb="vb1", lang="korean", version="V1"))
    # V2 (improved): 0% defect
    rows.append(_row(case_id="c1", vb="vb1", lang="korean", version="V2"))
    rows.append(_row(case_id="c2", vb="vb1", lang="korean", version="V2"))
    result = aggregate(rows, reference_version="V1")
    deltas = result["deltas_vs_reference"]
    assert "V2" in deltas
    assert deltas["V2"]["click"] == pytest.approx(-0.5)
    assert deltas["V2"]["any_defect"] == pytest.approx(-0.5)


def test_aggregate_handles_empty_rows():
    result = aggregate([])
    assert result["row_count"] == 0
    assert result["overall"] == {}


def test_read_label_rows_round_trip(tmp_path):
    csv_path = str(tmp_path / "labels.csv")
    rows_in = [
        _row(case_id="c1", vb="vb1", lang="korean", version="V1", defects={"click": 1}),
        _row(case_id="c2", vb="vb1", lang="korean", version="V1"),
    ]
    _write_csv(rows_in, csv_path)
    rows_out = read_label_rows(csv_path)
    assert len(rows_out) == 2
    assert rows_out[0]["case_id"] == "c1"
    assert rows_out[0]["click"] == "1"


def test_read_label_rows_rejects_missing_columns(tmp_path):
    csv_path = str(tmp_path / "bad.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("case_id,foo\nc1,x\n")
    with pytest.raises(SystemExit):
        read_label_rows(csv_path)


def test_render_markdown_contains_required_sections(tmp_path):
    rows = [
        _row(case_id="ko1", vb="vb1", lang="korean", version="V1", defects={"vc_misalign": 1}),
        _row(case_id="ja1", vb="vb2", lang="japanese", version="V1"),
        _row(case_id="ko1", vb="vb1", lang="korean", version="V2"),
        _row(case_id="ja1", vb="vb2", lang="japanese", version="V2"),
    ]
    result = aggregate(rows, reference_version="V1")
    md = render_markdown(run_id="test_run", aggregated=result, labels_path="labels.csv")
    assert "전체 결함 비율" in md
    assert "언어별 결함 비율" in md
    assert "보이스뱅크별 결함 비율" in md
    assert "기준 버전 대비 변화" in md
    assert "V1" in md and "V2" in md
    # Korean and Japanese language tags should appear
    assert "korean" in md
    assert "japanese" in md
    # KO-JA gap column should be present when both languages exist
    assert "KO-JA pp 격차" in md


def test_render_markdown_no_reference_omits_delta_section():
    rows = [
        _row(case_id="c1", vb="vb1", lang="korean", version="V1"),
    ]
    result = aggregate(rows)  # no reference
    md = render_markdown(run_id="test_run", aggregated=result, labels_path="labels.csv")
    assert "기준 버전 대비 변화" not in md


def test_render_markdown_handles_single_language():
    """No KO-JA gap column when only one language is present."""
    rows = [
        _row(case_id="c1", vb="vb1", lang="korean", version="V1"),
    ]
    result = aggregate(rows)
    md = render_markdown(run_id="test_run", aggregated=result, labels_path="labels.csv")
    assert "KO-JA pp 격차" not in md
