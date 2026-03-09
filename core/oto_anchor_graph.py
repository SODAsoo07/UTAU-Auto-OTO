from __future__ import annotations


def build_adjacent_anchor_graph(planned_indices):
    clean = []
    for raw in planned_indices or []:
        try:
            clean.append(int(raw))
        except Exception:
            continue

    pairs = []
    for idx, prev_idx in enumerate(clean[:-1]):
        next_idx = None
        for probe in clean[idx + 1:]:
            if int(probe) > int(prev_idx):
                next_idx = int(probe)
                break
        pairs.append(
            {
                "prev_idx": int(prev_idx),
                "next_idx": next_idx,
                "valid": bool(next_idx is not None),
            }
        )

    return {
        "planned_indices": clean,
        "pairs": pairs,
        "pair_count": len(pairs),
    }


def resolve_bridge_anchor_pair(
    graph,
    bridge_seq_idx,
    *,
    realized_anchor_by_idx=None,
    estimated_anchor_by_idx=None,
    local_prev_idx=None,
    local_next_idx=None,
):
    realized = dict(realized_anchor_by_idx or {})
    estimated = dict(estimated_anchor_by_idx or {})
    pairs = list((graph or {}).get("pairs") or [])

    def _pick_anchor(idx):
        if idx is None:
            return None
        return realized.get(int(idx)) or estimated.get(int(idx))

    pair = None
    try:
        seq_idx = int(bridge_seq_idx)
    except Exception:
        seq_idx = -1
    if 0 <= seq_idx < len(pairs):
        candidate = pairs[seq_idx]
        if candidate and candidate.get("valid"):
            pair = candidate

    prev_idx = pair.get("prev_idx") if pair else local_prev_idx
    next_idx = pair.get("next_idx") if pair else local_next_idx
    prev_anchor = _pick_anchor(prev_idx)
    next_anchor = _pick_anchor(next_idx)

    if (
        (prev_anchor is None or next_anchor is None)
        and local_prev_idx is not None
        and local_next_idx is not None
    ):
        local_prev_anchor = _pick_anchor(local_prev_idx)
        local_next_anchor = _pick_anchor(local_next_idx)
        if local_prev_anchor is not None and local_next_anchor is not None:
            prev_idx = int(local_prev_idx)
            next_idx = int(local_next_idx)
            prev_anchor = local_prev_anchor
            next_anchor = local_next_anchor
            pair = None

    return {
        "prev_idx": prev_idx,
        "next_idx": next_idx,
        "prev_anchor": prev_anchor,
        "next_anchor": next_anchor,
        "source": "planned" if pair else ("local" if prev_idx is not None and next_idx is not None else ""),
    }


__all__ = [
    "build_adjacent_anchor_graph",
    "resolve_bridge_anchor_pair",
]
