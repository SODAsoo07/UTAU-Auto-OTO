from __future__ import annotations

import bisect
from typing import Any, Dict, Iterable, List


def build_interval_lookup(intervals: Iterable[Any] | None) -> Dict[str, Any]:
    seq = list(intervals or [])
    starts: List[float] = []
    ends: List[float] = []
    indexed = True
    prev_start = float("-inf")
    prev_end = float("-inf")
    for item in seq:
        try:
            start = float(getattr(item, "minTime", 0.0))
            end = float(getattr(item, "maxTime", 0.0))
        except Exception:
            start = 0.0
            end = 0.0
        starts.append(start)
        ends.append(end)
        if start < prev_start or end < prev_end:
            indexed = False
        prev_start = start
        prev_end = end
    return {
        "intervals": seq,
        "starts": starts,
        "ends": ends,
        "indexed": bool(indexed),
    }


def intervals_within_bounds(
    lookup: Dict[str, Any] | None,
    start_time: float,
    end_time: float,
    *,
    eps: float = 1e-6,
) -> List[Any]:
    data = lookup or {}
    intervals = data.get("intervals") or []
    if not intervals:
        return []

    lo = float(start_time) - float(eps)
    hi = float(end_time) + float(eps)
    if hi < lo:
        lo, hi = hi, lo

    if bool(data.get("indexed")):
        starts = data.get("starts") or []
        ends = data.get("ends") or []
        left = bisect.bisect_left(ends, lo)
        right = bisect.bisect_right(starts, hi)
        out = []
        for idx in range(left, right):
            if starts[idx] >= lo and ends[idx] <= hi:
                out.append(intervals[idx])
        return out

    out = []
    for item in intervals:
        try:
            start = float(getattr(item, "minTime", 0.0))
            end = float(getattr(item, "maxTime", 0.0))
        except Exception:
            continue
        if start >= lo and end <= hi:
            out.append(item)
    return out
