from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class GeneratorFinishContext:
    log_fn: Callable[[str], None]
    anchor_stats: Mapping[str, object]
    anchor_log_path: str
    anchor_log_dir: str
    cleanup_timing_jsonl: bool
    timing_jsonl_prefix: str


def write_oto_lines(out_path: str, lines: Sequence[str]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def cleanup_timing_anchor_jsonl_files(log_dir: str, prefix: str) -> tuple[int, int]:
    if not log_dir or not os.path.isdir(log_dir):
        return 0, 0
    removed = 0
    failed = 0
    try:
        names = os.listdir(log_dir)
    except Exception:
        return 0, 1
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".jsonl"):
            continue
        path = os.path.join(log_dir, name)
        try:
            os.remove(path)
            removed += 1
        except Exception:
            failed += 1
    return removed, failed


def write_jsonl_records(path: str, rows: Sequence[object]) -> int:
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def finalize_generator_finish(context: GeneratorFinishContext) -> None:
    anchor_locked = int(context.anchor_stats.get("anchor_locked_count", 0) or 0)
    if anchor_locked > 0:
        context.log_fn(
            "[AnchorLock] summary: "
            f"anchor_locked_count={anchor_locked}, "
            f"cutoff_clamped_count={int(context.anchor_stats.get('cutoff_clamped_count', 0) or 0)}, "
            f"vc_cutoff_leak_guard_count={int(context.anchor_stats.get('vc_cutoff_leak_guard_count', 0) or 0)}"
        )
        context.log_fn(f"[AnchorLock] detail log: {context.anchor_log_path}")

    if not context.cleanup_timing_jsonl:
        return

    removed_count, failed_count = cleanup_timing_anchor_jsonl_files(
        context.anchor_log_dir,
        context.timing_jsonl_prefix,
    )
    if removed_count > 0:
        context.log_fn(f"[AnchorLock] timing jsonl cleanup: {removed_count} files")
    if failed_count > 0:
        context.log_fn(f"[AnchorLock] timing jsonl cleanup failed: {failed_count} files")


__all__ = [
    "GeneratorFinishContext",
    "cleanup_timing_anchor_jsonl_files",
    "finalize_generator_finish",
    "write_jsonl_records",
    "write_oto_lines",
]
