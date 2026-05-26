from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, List

from core.generator_finish import write_jsonl_records


REVIEW_QUEUE_APPLY_MODES = frozenset(
    {
        "conservative_apply",
        "template_preserve",
        "review_required",
    }
)


def build_review_queue_rows(
    row_apply_decisions: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for row in row_apply_decisions:
        apply_mode = str(row.get("apply_mode", "")).strip().lower()
        if apply_mode in REVIEW_QUEUE_APPLY_MODES:
            rows.append(row)
    return rows


def export_review_queue_if_enabled(
    *,
    review_queue_export: bool,
    review_queue_path: str,
    row_apply_decisions: Sequence[Mapping[str, Any]],
    log_fn: Callable[[str], None],
    write_jsonl_records_fn: Callable[[str, Sequence[object]], int] = write_jsonl_records,
) -> int:
    if not review_queue_export:
        return 0
    try:
        review_rows = build_review_queue_rows(row_apply_decisions)
        if not review_rows:
            return 0
        written = write_jsonl_records_fn(review_queue_path, review_rows)
        log_fn(f"[REVIEW] review queue exported: {written} rows -> {review_queue_path}")
        return int(written)
    except Exception as exc:
        log_fn(f"[REVIEW] review queue export failed: {exc}")
        return 0


def update_review_queue_runtime_report(
    runtime_report: Dict[str, Any],
    *,
    row_apply_mode_counts: Mapping[str, int],
    review_queue_export: bool,
    row_apply_decisions: Sequence[Mapping[str, Any]],
    review_queue_path: str,
) -> None:
    runtime_report["row_apply_mode_counts"] = {
        str(key): int(value)
        for key, value in dict(row_apply_mode_counts).items()
    }
    if review_queue_export and row_apply_decisions:
        runtime_report["review_queue_path"] = str(review_queue_path)


__all__ = [
    "REVIEW_QUEUE_APPLY_MODES",
    "build_review_queue_rows",
    "export_review_queue_if_enabled",
    "update_review_queue_runtime_report",
]
