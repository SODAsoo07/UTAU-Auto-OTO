"""
VC↔CV pairing logic.

학습 시 인접 VC↔CV 행 간의 일관성 손실을 계산하기 위한 페어링 맵을 구축합니다.
"""

from __future__ import annotations

import bisect
import os
from typing import Dict, List, Tuple


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _build_vc_cv_pair_map(df) -> Dict[int, int]:
    if "alias_type" not in df.columns or "row_index_in_wav" not in df.columns:
        return {}
    group_col = "wav_norm" if "wav_norm" in df.columns else ("wav" if "wav" in df.columns else "")
    if not group_col:
        return {}
    max_gap = max(1, _env_int("UTOA_ML_VC_CV_MAX_GAP", 5))
    pair_map: Dict[int, int] = {}
    for _, group in df.groupby(group_col):
        if len(group) <= 1:
            continue
        ordered = []
        for pos, row in group.iterrows():
            try:
                row_idx = float(row.get("row_index_in_wav", 0.0) or 0.0)
            except Exception:
                row_idx = 0.0
            alias_type = str(row.get("alias_type", "") or "").strip().lower()
            ordered.append((row_idx, int(pos), alias_type))
        ordered.sort(key=lambda x: x[0])
        cv_positions = [(row_idx, pos) for row_idx, pos, a_type in ordered if a_type in {"cv", "cv_head"}]
        vc_positions = [(row_idx, pos) for row_idx, pos, a_type in ordered if a_type == "vc"]
        if not cv_positions or not vc_positions:
            continue
        cv_row_indices = [row_idx for row_idx, _ in cv_positions]
        vc_row_indices = [row_idx for row_idx, _ in vc_positions]

        def _nearest(indices, positions, value):
            insert_at = bisect.bisect_left(indices, value)
            chosen = None
            if insert_at < len(positions):
                cand_idx, cand_pos = positions[insert_at]
                if abs(cand_idx - value) <= max_gap:
                    chosen = cand_pos
            if chosen is None and insert_at > 0:
                cand_idx, cand_pos = positions[insert_at - 1]
                if abs(cand_idx - value) <= max_gap:
                    chosen = cand_pos
            return chosen

        # VC -> nearest CV
        for row_idx, pos, a_type in ordered:
            if a_type != "vc":
                continue
            chosen = _nearest(cv_row_indices, cv_positions, row_idx)
            if chosen is not None:
                pair_map[int(pos)] = int(chosen)

        # CV (and CV head) -> nearest VC
        for row_idx, pos, a_type in ordered:
            if a_type not in {"cv", "cv_head"}:
                continue
            chosen = _nearest(vc_row_indices, vc_positions, row_idx)
            if chosen is not None:
                pair_map[int(pos)] = int(chosen)
    return pair_map


def _batch_pair_positions(batch_indices: List[int], pair_map: Dict[int, int]) -> Tuple[List[int], List[int]]:
    if not pair_map:
        return [], []
    pos_map = {int(idx): pos for pos, idx in enumerate(batch_indices)}
    src_pos = []
    dst_pos = []
    for idx, pos in pos_map.items():
        other = pair_map.get(idx)
        if other is None:
            continue
        pos2 = pos_map.get(other)
        if pos2 is None:
            continue
        src_pos.append(pos)
        dst_pos.append(pos2)
    return src_pos, dst_pos
