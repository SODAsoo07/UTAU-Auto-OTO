from core.generation.review_queue import (
    build_review_queue_rows,
    export_review_queue_if_enabled,
    update_review_queue_runtime_report,
)


def test_build_review_queue_rows_filters_actionable_modes():
    rows = build_review_queue_rows(
        [
            {"apply_mode": "full_apply", "alias": "a"},
            {"apply_mode": " conservative_apply ", "alias": "b"},
            {"apply_mode": "template_preserve", "alias": "c"},
            {"apply_mode": "review_required", "alias": "d"},
            {"apply_mode": "", "alias": "e"},
        ]
    )

    assert [row["alias"] for row in rows] == ["b", "c", "d"]


def test_export_review_queue_if_enabled_writes_filtered_rows():
    logs = []
    captured = {}

    def write_jsonl(path, rows):
        captured["path"] = path
        captured["rows"] = list(rows)
        return len(rows)

    written = export_review_queue_if_enabled(
        review_queue_export=True,
        review_queue_path="review.jsonl",
        row_apply_decisions=[
            {"apply_mode": "full_apply", "alias": "a"},
            {"apply_mode": "review_required", "alias": "b"},
        ],
        log_fn=logs.append,
        write_jsonl_records_fn=write_jsonl,
    )

    assert written == 1
    assert captured == {
        "path": "review.jsonl",
        "rows": [{"apply_mode": "review_required", "alias": "b"}],
    }
    assert any("review queue exported: 1 rows" in line for line in logs)


def test_export_review_queue_if_enabled_skips_disabled_export():
    called = False

    def write_jsonl(_path, _rows):
        nonlocal called
        called = True
        return 0

    written = export_review_queue_if_enabled(
        review_queue_export=False,
        review_queue_path="review.jsonl",
        row_apply_decisions=[{"apply_mode": "review_required"}],
        log_fn=lambda _msg: None,
        write_jsonl_records_fn=write_jsonl,
    )

    assert written == 0
    assert called is False


def test_update_review_queue_runtime_report_records_counts_and_path():
    report = {}

    update_review_queue_runtime_report(
        report,
        row_apply_mode_counts={"full_apply": 2, "review_required": 1},
        review_queue_export=True,
        row_apply_decisions=[{"apply_mode": "review_required"}],
        review_queue_path="review.jsonl",
    )

    assert report == {
        "row_apply_mode_counts": {"full_apply": 2, "review_required": 1},
        "review_queue_path": "review.jsonl",
    }
