from __future__ import annotations


def apply_anchor_record(anchor_store, anchor_record):
    if anchor_store is None or anchor_record is None:
        return False
    idx, payload = anchor_record
    anchor_store[idx] = payload
    return True


def commit_row_artifacts(
    final_lines,
    rows,
    *,
    anchor_store=None,
    anchor_record=None,
    log_fn=None,
    messages=None,
):
    apply_anchor_record(anchor_store, anchor_record)
    if log_fn is not None:
        for message in messages or ():
            if message:
                log_fn(message)
    final_lines.extend(rows or [])


__all__ = [
    "apply_anchor_record",
    "commit_row_artifacts",
]
