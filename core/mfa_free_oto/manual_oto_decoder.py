from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CandidateOption:
    candidate_index: int
    time_ms: float
    score: float
    order_norm: float = 0.0
    slot_pos_norm: float = 0.0


@dataclass(frozen=True)
class JointAnchorOption:
    anchor_indices: Mapping[str, int]
    anchor_times_ms: Mapping[str, float]
    score: float
    time_order_norm: float = 0.0
    slot_pos_norm: float = 0.0

    @property
    def center_time_ms(self) -> float:
        if "preutterance" in self.anchor_times_ms:
            return float(self.anchor_times_ms["preutterance"])
        if self.anchor_times_ms:
            return float(sum(self.anchor_times_ms.values()) / len(self.anchor_times_ms))
        return 0.0


def decode_monotonic_candidate_indices(
    options_by_row: Sequence[Sequence[CandidateOption]],
    *,
    order_penalty: float = 0.35,
    time_backward_tolerance_ms: float = 15.0,
    time_backward_penalty: float = 8.0,
    large_jump_penalty_per_ms: float = 0.0002,
) -> list[int | None]:
    """Select one candidate per row while preferring nondecreasing time.

    This is deliberately small and dependency-free: the scorer remains a local
    candidate ranker, while this decoder handles the recurrent UTAU failure mode
    where independent rows choose candidates from another slot in the same wav.
    """
    rows = [tuple(options) for options in options_by_row]
    if not rows:
        return []

    dp: list[list[float]] = []
    back: list[list[int | None]] = []
    for row_idx, options in enumerate(rows):
        if not options:
            dp.append([])
            back.append([])
            continue
        local_scores = [
            float(option.score) - float(order_penalty) * abs(float(option.order_norm) - float(option.slot_pos_norm))
            for option in options
        ]
        if row_idx == 0:
            dp.append(local_scores)
            back.append([None] * len(options))
            continue

        prev_options = rows[row_idx - 1]
        prev_scores = dp[row_idx - 1]
        current_scores: list[float] = []
        current_back: list[int | None] = []
        for option_idx, option in enumerate(options):
            best_score: float | None = None
            best_prev: int | None = None
            for prev_idx, prev_option in enumerate(prev_options):
                if prev_idx >= len(prev_scores):
                    continue
                transition = 0.0
                backward_ms = float(prev_option.time_ms) - float(option.time_ms)
                if backward_ms > float(time_backward_tolerance_ms):
                    transition -= float(time_backward_penalty) + 0.01 * backward_ms
                else:
                    transition -= float(large_jump_penalty_per_ms) * abs(float(option.time_ms) - float(prev_option.time_ms))
                score = float(prev_scores[prev_idx]) + local_scores[option_idx] + transition
                if best_score is None or score > best_score:
                    best_score = score
                    best_prev = prev_idx
            if best_score is None:
                best_score = local_scores[option_idx]
            current_scores.append(float(best_score))
            current_back.append(best_prev)
        dp.append(current_scores)
        back.append(current_back)

    selected: list[int | None] = [None] * len(rows)
    last_with_options = None
    for idx in range(len(rows) - 1, -1, -1):
        if dp[idx]:
            last_with_options = idx
            break
    if last_with_options is None:
        return selected

    current_idx = max(range(len(dp[last_with_options])), key=lambda idx: dp[last_with_options][idx])
    for row_idx in range(last_with_options, -1, -1):
        if not rows[row_idx]:
            selected[row_idx] = None
            continue
        option = rows[row_idx][current_idx]
        selected[row_idx] = int(option.candidate_index)
        prev_idx = back[row_idx][current_idx]
        if prev_idx is None:
            break
        current_idx = int(prev_idx)
    return selected


def build_joint_anchor_options(
    options_by_anchor: Mapping[str, Sequence[CandidateOption]],
    *,
    anchors: Sequence[str] = ("offset", "overlap", "preutterance", "fixed_end"),
    top_per_anchor: int = 6,
    max_options: int = 80,
    anchor_order_tolerance_ms: float = 8.0,
    anchor_order_penalty: float = 0.08,
    anchor_span_soft_max_ms: float = 650.0,
    anchor_span_penalty: float = 0.001,
    slot_order_penalty: float = 1.25,
) -> list[JointAnchorOption]:
    trimmed: list[tuple[str, tuple[CandidateOption, ...]]] = []
    for anchor in anchors:
        values = tuple(sorted(options_by_anchor.get(anchor, ()), key=lambda option: float(option.score), reverse=True))
        if not values:
            return []
        trimmed.append((str(anchor), values[: max(1, int(top_per_anchor))]))

    out: list[JointAnchorOption] = []
    for combo in product(*(values for _, values in trimmed)):
        anchor_indices = {anchor: int(option.candidate_index) for (anchor, _), option in zip(trimmed, combo, strict=False)}
        anchor_times = {anchor: float(option.time_ms) for (anchor, _), option in zip(trimmed, combo, strict=False)}
        score = float(sum(float(option.score) for option in combo))
        time_order_norm = float(sum(float(option.order_norm) for option in combo) / max(1, len(combo)))
        slot_pos_norm = float(sum(float(option.slot_pos_norm) for option in combo) / max(1, len(combo)))
        score -= float(slot_order_penalty) * abs(time_order_norm - slot_pos_norm)
        order_violation = 0.0
        previous_time: float | None = None
        for anchor in anchors:
            current = anchor_times.get(str(anchor))
            if current is None:
                continue
            if previous_time is not None:
                order_violation += max(0.0, previous_time - current - float(anchor_order_tolerance_ms))
            previous_time = current
        if order_violation > 0.0:
            score -= float(anchor_order_penalty) * order_violation
        if "offset" in anchor_times and "fixed_end" in anchor_times:
            span = float(anchor_times["fixed_end"]) - float(anchor_times["offset"])
            if span < -float(anchor_order_tolerance_ms):
                score -= float(anchor_order_penalty) * abs(span)
            elif span > float(anchor_span_soft_max_ms):
                score -= float(anchor_span_penalty) * (span - float(anchor_span_soft_max_ms))
        out.append(
            JointAnchorOption(
                anchor_indices=anchor_indices,
                anchor_times_ms=anchor_times,
                score=score,
                time_order_norm=time_order_norm,
                slot_pos_norm=slot_pos_norm,
            )
        )
    return sorted(out, key=lambda option: float(option.score), reverse=True)[: max(1, int(max_options))]


def decode_joint_anchor_lattice(
    options_by_row: Sequence[Sequence[JointAnchorOption]],
    *,
    time_backward_tolerance_ms: float = 20.0,
    time_backward_penalty: float = 8.0,
    large_jump_penalty_per_ms: float = 0.0002,
    slot_order_penalty: float = 1.0,
    slot_transition_penalty: float = 1.6,
    slot_backward_tolerance: float = 0.03,
    slot_backward_penalty: float = 3.0,
) -> list[JointAnchorOption | None]:
    rows = [tuple(options) for options in options_by_row]
    if not rows:
        return []

    dp: list[list[float]] = []
    back: list[list[int | None]] = []
    for row_idx, options in enumerate(rows):
        if not options:
            dp.append([])
            back.append([])
            continue
        local_scores = [
            float(option.score) - float(slot_order_penalty) * abs(float(option.time_order_norm) - float(option.slot_pos_norm))
            for option in options
        ]
        if row_idx == 0:
            dp.append(local_scores)
            back.append([None] * len(options))
            continue

        prev_options = rows[row_idx - 1]
        prev_scores = dp[row_idx - 1]
        current_scores: list[float] = []
        current_back: list[int | None] = []
        for option_idx, option in enumerate(options):
            best_score: float | None = None
            best_prev: int | None = None
            for prev_idx, prev_option in enumerate(prev_options):
                if prev_idx >= len(prev_scores):
                    continue
                transition = 0.0
                backward_ms = float(prev_option.center_time_ms) - float(option.center_time_ms)
                if backward_ms > float(time_backward_tolerance_ms):
                    transition -= float(time_backward_penalty) + 0.01 * backward_ms
                else:
                    transition -= float(large_jump_penalty_per_ms) * abs(
                        float(option.center_time_ms) - float(prev_option.center_time_ms)
                    )
                slot_backward = float(prev_option.time_order_norm) - float(option.time_order_norm)
                if slot_backward > float(slot_backward_tolerance):
                    transition -= float(slot_backward_penalty) + 3.0 * slot_backward
                expected_delta = float(option.slot_pos_norm) - float(prev_option.slot_pos_norm)
                selected_delta = float(option.time_order_norm) - float(prev_option.time_order_norm)
                transition -= float(slot_transition_penalty) * abs(selected_delta - expected_delta)
                score = float(prev_scores[prev_idx]) + local_scores[option_idx] + transition
                if best_score is None or score > best_score:
                    best_score = score
                    best_prev = prev_idx
            if best_score is None:
                best_score = local_scores[option_idx]
            current_scores.append(float(best_score))
            current_back.append(best_prev)
        dp.append(current_scores)
        back.append(current_back)

    selected: list[JointAnchorOption | None] = [None] * len(rows)
    last_with_options = None
    for idx in range(len(rows) - 1, -1, -1):
        if dp[idx]:
            last_with_options = idx
            break
    if last_with_options is None:
        return selected

    current_idx = max(range(len(dp[last_with_options])), key=lambda idx: dp[last_with_options][idx])
    for row_idx in range(last_with_options, -1, -1):
        if not rows[row_idx]:
            selected[row_idx] = None
            continue
        selected[row_idx] = rows[row_idx][current_idx]
        prev_idx = back[row_idx][current_idx]
        if prev_idx is None:
            break
        current_idx = int(prev_idx)
    return selected


__all__ = [
    "CandidateOption",
    "JointAnchorOption",
    "build_joint_anchor_options",
    "decode_joint_anchor_lattice",
    "decode_monotonic_candidate_indices",
]
