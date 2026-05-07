from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict


def prepare_kr_general_row_bounds(
    *,
    is_vc: bool,
    current_w_idx: int,
    selected_w_idx: int,
    bridge_pair: Mapping[str, Any],
    syllables_info: Sequence[Mapping[str, Any]],
    prepare_cv_bounds_fn: Callable[..., tuple[Any, Any, Any, Any, Any, Any]],
    prepare_vc_bounds_fn: Callable[..., tuple[Any, Any, Any, Any, Any, Any]],
) -> Dict[str, Any]:
    bridge_next_idx = None
    current_idx = int(current_w_idx)
    selected_idx = int(selected_w_idx)

    if not is_vc:
        current_idx = max(current_idx, selected_idx)
        (
            selected_idx,
            curr_phones,
            c_start,
            c_end,
            n_start,
            n_end,
        ) = prepare_cv_bounds_fn(syllables_info, selected_idx)
    else:
        selected_idx = current_idx
        if bridge_pair.get("next_idx") is not None:
            bridge_next_idx = int(bridge_pair["next_idx"])
        if bridge_pair.get("prev_idx") is not None:
            current_idx = int(bridge_pair["prev_idx"])
        (
            current_idx,
            curr_phones,
            c_start,
            c_end,
            n_start,
            n_end,
        ) = prepare_vc_bounds_fn(
            syllables_info,
            current_idx,
            next_w_idx=bridge_next_idx,
        )

    return {
        "current_w_idx": int(current_idx),
        "selected_w_idx": int(selected_idx),
        "bridge_next_idx": bridge_next_idx,
        "curr_phones": curr_phones,
        "c_start": float(c_start),
        "c_end": float(c_end),
        "n_start": float(n_start),
        "n_end": float(n_end),
    }


__all__ = ["prepare_kr_general_row_bounds"]
