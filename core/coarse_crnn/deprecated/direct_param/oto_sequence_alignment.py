from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.oto_audio_candidates import AudioCandidates
from core.coarse_crnn.oto_targets import OtoAnchors, anchors_to_oto_params, oto_params_to_anchors
from core.coarse_crnn.slot_graph import slot_type_for_role


SEQUENCE_BOUNDARY_KINDS = ("vowel_onset", "vowel_transition", "vowel_tail", "silence_boundary")


@dataclass(frozen=True)
class SequenceBoundaryCandidate:
    time_ms: float
    score: float
    kind: str
    source: str


def decode_sequence_alignment_for_states(
    *,
    row_states: list[dict[str, Any]],
    language: str,
    role_resolver: Callable[[Any], str],
    candidate_getter: Callable[[str], AudioCandidates | None],
    max_shift_ms: float = 260.0,
    anchor_blend: float = 0.60,
    boundary_blend: float = 0.72,
    min_move_ms: float = 4.0,
    model_score_blend: float = 0.35,
    allowed_roles: set[str] | None = None,
) -> tuple[int, int]:
    """Align OTO rows to wav-level boundary candidates by role slots.

    This is intentionally not a raw row-order aligner. CVVC banks often place
    all CV rows before VC/VV rows in oto.ini, while the actual audio timeline
    interleaves anchors and boundaries. The alignment therefore maps anchor
    rows first, then places boundary rows inside adjacent anchor windows.
    """
    moved = 0
    held = 0
    groups: dict[str, list[dict[str, Any]]] = {}
    for state in row_states:
        row = state.get("row")
        if row is None:
            continue
        groups.setdefault(str(getattr(row, "wav_abs", "")), []).append(state)

    for wav_abs, states in groups.items():
        states.sort(key=lambda state: int(getattr(state.get("row"), "row_index", 0)))
        audio = candidate_getter(str(wav_abs))
        if audio is None:
            held += len(states)
            continue
        candidates = collect_sequence_boundary_candidates(audio)
        if not candidates:
            held += len(states)
            continue

        annotated = [_annotate_state(state, role_resolver=role_resolver) for state in states]
        anchor_items = [item for item in annotated if item.slot_type == "anchor"]
        boundary_items = [item for item in annotated if item.slot_type == "boundary"]
        final_items = [item for item in annotated if item.slot_type in {"final", "silence"}]

        anchor_times: list[float] = []
        if anchor_items:
            anchor_candidates = [cand for cand in candidates if _candidate_role_score(cand, "cv") > 0.0]
            path = _align_anchor_items(anchor_items, anchor_candidates, model_score_blend=model_score_blend)
            if path:
                for item, cand_idx in zip(anchor_items, path):
                    if not _role_allowed(item.role, allowed_roles):
                        anchor_times.append(float(_state_anchor_time(item.state)))
                        continue
                    cand = anchor_candidates[cand_idx]
                    changed = _move_state_to_candidate(
                        item.state,
                        item.role,
                        cand,
                        blend=anchor_blend,
                        max_shift_ms=max_shift_ms,
                        min_move_ms=min_move_ms,
                    )
                    anchor_times.append(float(_state_anchor_time(item.state)))
                    if changed:
                        moved += 1
                    else:
                        held += 1
            else:
                held += len(anchor_items)
                anchor_times = sorted(_state_anchor_time(item.state) for item in anchor_items)
        anchor_times = sorted(anchor_times)

        for ordinal, item in enumerate(boundary_items):
            if not _role_allowed(item.role, allowed_roles):
                continue
            lo, hi = _boundary_window_for_ordinal(ordinal, anchor_times, item.state)
            cand = _best_candidate_for_role(
                candidates,
                role=item.role,
                state=item.state,
                target_ms=_state_anchor_time(item.state),
                lo=lo,
                hi=hi,
                model_score_blend=model_score_blend,
            )
            if cand is None:
                held += 1
                continue
            changed = _move_state_to_candidate(
                item.state,
                item.role,
                cand,
                blend=boundary_blend,
                max_shift_ms=max_shift_ms,
                min_move_ms=min_move_ms,
            )
            if changed:
                moved += 1
            else:
                held += 1

        if final_items:
            final_candidates = [cand for cand in candidates if cand.kind in {"vowel_tail", "silence_boundary"}]
            for item in final_items:
                if not _role_allowed(item.role, allowed_roles):
                    continue
                lo = anchor_times[-1] if anchor_times else 0.0
                hi = _state_duration(item.state)
                cand = _best_candidate_for_role(
                    final_candidates,
                    role=item.role,
                    state=item.state,
                    target_ms=_state_anchor_time(item.state),
                    lo=lo,
                    hi=hi,
                    model_score_blend=model_score_blend,
                )
                if cand is None:
                    held += 1
                    continue
                changed = _move_state_to_candidate(
                    item.state,
                    item.role,
                    cand,
                    blend=boundary_blend,
                    max_shift_ms=max_shift_ms,
                    min_move_ms=min_move_ms,
                )
                if changed:
                    moved += 1
                else:
                    held += 1

    return moved, held


@dataclass(frozen=True)
class _AlignedState:
    state: dict[str, Any]
    role: str
    slot_type: str


def collect_sequence_boundary_candidates(audio: AudioCandidates) -> list[SequenceBoundaryCandidate]:
    duration = max(1.0, float(getattr(audio, "duration_ms", 0.0) or 1.0))
    out: list[SequenceBoundaryCandidate] = []

    def add(time_ms: float, score: float, kind: str, source: str) -> None:
        t = max(0.0, min(duration, float(time_ms)))
        s = max(0.0, min(1.0, float(score)))
        out.append(SequenceBoundaryCandidate(time_ms=t, score=s, kind=str(kind), source=str(source)))

    active_start = float(getattr(audio, "active_start_ms", 0.0) or 0.0)
    active_end = float(getattr(audio, "active_end_ms", duration) or duration)
    if active_end > active_start:
        add(active_start, 0.35, "silence_boundary", "active_start")
        add(active_end, 0.35, "silence_boundary", "active_end")

    for peak in list(getattr(audio, "onset_peaks", []) or []):
        add(float(getattr(peak, "time_ms", 0.0) or 0.0), float(getattr(peak, "strength", 0.0) or 0.0), "vowel_onset", "onset")

    for peak in list(getattr(audio, "mel_delta_peaks", []) or []):
        add(
            float(getattr(peak, "time_ms", 0.0) or 0.0),
            float(getattr(peak, "strength", 0.0) or 0.0),
            "vowel_transition",
            "mel_delta",
        )

    for seg in list(getattr(audio, "stable_vowel_segments", []) or []):
        conf = float(getattr(seg, "vowel_confidence", 0.0) or 0.0)
        voiced = float(getattr(seg, "voiced_score", 0.0) or 0.0)
        stable = float(getattr(seg, "stability_score", 0.0) or 0.0)
        score = max(0.05, min(1.0, (0.45 * conf) + (0.35 * voiced) + (0.20 * stable)))
        add(float(getattr(seg, "start_ms", 0.0) or 0.0), score * 0.75, "vowel_onset", "vowel_start")
        add(float(getattr(seg, "center_ms", 0.0) or 0.0), score * 0.72, "vowel_transition", "vowel_center")
        add(float(getattr(seg, "end_ms", 0.0) or 0.0), score, "vowel_tail", "vowel_end")

    out.sort(key=lambda cand: (cand.time_ms, -cand.score, cand.kind, cand.source))
    return _merge_nearby_candidates(out, merge_ms=10.0)


def _annotate_state(state: dict[str, Any], *, role_resolver: Callable[[Any], str]) -> _AlignedState:
    row = state.get("row")
    role = normalize_role(role_resolver(row))
    return _AlignedState(state=state, role=role, slot_type=slot_type_for_role(role))


def parse_sequence_role_allowlist(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    if str(raw).strip().lower() in {"*", "all", "any"}:
        return None
    roles = {
        normalize_role(part.strip())
        for part in str(raw).replace(";", ",").split(",")
        if part.strip()
    }
    return roles or None


def _role_allowed(role: str, allowed_roles: set[str] | None) -> bool:
    if not allowed_roles:
        return True
    return normalize_role(role) in allowed_roles


def _merge_nearby_candidates(
    candidates: list[SequenceBoundaryCandidate],
    *,
    merge_ms: float,
) -> list[SequenceBoundaryCandidate]:
    merged: list[SequenceBoundaryCandidate] = []
    for cand in candidates:
        if merged and abs(float(cand.time_ms) - float(merged[-1].time_ms)) <= float(merge_ms):
            prev = merged[-1]
            if _candidate_merge_rank(cand) > _candidate_merge_rank(prev):
                merged[-1] = cand
            continue
        merged.append(cand)
    return merged


def _candidate_merge_rank(cand: SequenceBoundaryCandidate) -> tuple[float, int]:
    kind_rank = {"vowel_onset": 4, "vowel_transition": 3, "vowel_tail": 2, "silence_boundary": 1}
    return float(cand.score), int(kind_rank.get(cand.kind, 0))


def _align_anchor_items(
    items: list[_AlignedState],
    candidates: list[SequenceBoundaryCandidate],
    *,
    model_score_blend: float,
) -> list[int]:
    if not items or len(candidates) < len(items):
        return []
    n_rows = len(items)
    n_cands = len(candidates)
    inf = float("inf")
    dp = np.full((n_rows, n_cands), inf, dtype=np.float64)
    prev = np.full((n_rows, n_cands), -1, dtype=np.int32)
    for j, cand in enumerate(candidates):
        dp[0, j] = _anchor_cost(
            items[0],
            cand,
            row_idx=0,
            row_count=n_rows,
            cand_idx=j,
            cand_count=n_cands,
            model_score_blend=model_score_blend,
        )
    for i in range(1, n_rows):
        for j in range(i, n_cands):
            base = _anchor_cost(
                items[i],
                candidates[j],
                row_idx=i,
                row_count=n_rows,
                cand_idx=j,
                cand_count=n_cands,
                model_score_blend=model_score_blend,
            )
            best_score = inf
            best_prev = -1
            for k in range(i - 1, j):
                if candidates[j].time_ms < candidates[k].time_ms + _min_gap_for_role(items[i].role):
                    continue
                skip_penalty = float(j - k - 1) * 18.0
                score = float(dp[i - 1, k]) + base + skip_penalty
                if score < best_score:
                    best_score = score
                    best_prev = k
            if best_prev >= 0:
                dp[i, j] = best_score
                prev[i, j] = best_prev
    last = int(np.argmin(dp[n_rows - 1]))
    if not np.isfinite(dp[n_rows - 1, last]):
        return []
    path = [0] * n_rows
    j = last
    for i in range(n_rows - 1, -1, -1):
        path[i] = int(j)
        j = int(prev[i, j]) if i > 0 else -1
    if any(path[i] <= path[i - 1] for i in range(1, len(path))):
        return []
    return path


def _anchor_cost(
    item: _AlignedState,
    cand: SequenceBoundaryCandidate,
    *,
    row_idx: int,
    row_count: int,
    cand_idx: int,
    cand_count: int,
    model_score_blend: float,
) -> float:
    target = _state_anchor_time(item.state)
    distance = abs(float(cand.time_ms) - target)
    role_score = _combined_candidate_role_score(
        cand,
        item.role,
        state=item.state,
        model_score_blend=model_score_blend,
    )
    row_ratio = float(row_idx) / float(max(1, row_count - 1))
    cand_ratio = float(cand_idx) / float(max(1, cand_count - 1))
    index_cost = abs(row_ratio - cand_ratio) * 35.0
    return distance + index_cost - (role_score * 55.0)


def _best_candidate_for_role(
    candidates: list[SequenceBoundaryCandidate],
    *,
    role: str,
    state: dict[str, Any],
    target_ms: float,
    lo: float,
    hi: float,
    model_score_blend: float,
) -> SequenceBoundaryCandidate | None:
    role_text = normalize_role(role)
    best: tuple[float, SequenceBoundaryCandidate] | None = None
    for cand in candidates:
        t = float(cand.time_ms)
        if t < float(lo) or t > float(hi):
            continue
        role_score = _combined_candidate_role_score(
            cand,
            role_text,
            state=state,
            model_score_blend=model_score_blend,
        )
        if role_score <= 0.0:
            continue
        center = (float(lo) + float(hi)) * 0.5
        width = max(1.0, float(hi) - float(lo))
        center_penalty = abs(t - center) / width * 12.0
        distance_weight = 0.35 if slot_type_for_role(role_text) == "boundary" else 1.0
        score = (abs(t - float(target_ms)) * distance_weight) + center_penalty - (role_score * 58.0)
        if best is None or score < best[0]:
            best = (score, cand)
    return best[1] if best else None


def _combined_candidate_role_score(
    cand: SequenceBoundaryCandidate,
    role: str,
    *,
    state: dict[str, Any],
    model_score_blend: float,
) -> float:
    acoustic = _candidate_role_score(cand, role)
    model = _state_model_candidate_score(state, cand)
    blend = _clamp(float(model_score_blend), 0.0, 1.0)
    if model is None:
        return acoustic
    return (acoustic * (1.0 - blend)) + (float(model) * blend)


def _state_model_candidate_score(state: dict[str, Any], cand: SequenceBoundaryCandidate) -> float | None:
    classes = [str(item) for item in (state.get("sequence_candidate_classes") or [])]
    if not classes or cand.kind not in classes:
        return None
    times = np.asarray(state.get("sequence_candidate_times_ms") or [], dtype=np.float32)
    scores = np.asarray(state.get("sequence_candidate_scores") or [], dtype=np.float32)
    if times.ndim != 1 or scores.ndim != 2 or scores.shape[0] != times.shape[0] or scores.shape[0] <= 0:
        return None
    cls_idx = classes.index(cand.kind)
    if cls_idx < 0 or cls_idx >= scores.shape[1]:
        return None
    idx = int(np.searchsorted(times, float(cand.time_ms)))
    if idx <= 0:
        nearest = 0
    elif idx >= int(times.shape[0]):
        nearest = int(times.shape[0]) - 1
    else:
        left = idx - 1
        right = idx
        nearest = left if abs(float(times[left]) - float(cand.time_ms)) <= abs(float(times[right]) - float(cand.time_ms)) else right
    if abs(float(times[nearest]) - float(cand.time_ms)) > 35.0:
        return None
    return max(0.0, min(1.0, float(scores[nearest, cls_idx])))


def _candidate_role_score(cand: SequenceBoundaryCandidate, role: str) -> float:
    role_text = normalize_role(role)
    kind = str(cand.kind)
    score = float(cand.score)
    if role_text in {"-cv", "cv", "v", "special"}:
        if kind == "vowel_onset":
            return score
        if kind == "silence_boundary" and role_text == "-cv":
            return score * 0.45
        return 0.0
    if role_text == "vc":
        if kind == "vowel_tail":
            return score
        if kind == "vowel_transition":
            return score * 0.45
        return 0.0
    if role_text in {"vv", "v-cv"}:
        if kind == "vowel_transition":
            return score
        if kind == "vowel_tail":
            return score * 0.82
        if kind == "vowel_onset" and role_text == "v-cv":
            return score * 0.70
        return 0.0
    if role_text in {"cv-", "v-", "endbr"}:
        if kind in {"vowel_tail", "silence_boundary"}:
            return score
        return 0.0
    if role_text == "br":
        return score if kind == "silence_boundary" else 0.0
    return score * 0.35


def _boundary_window_for_ordinal(
    ordinal: int,
    anchor_times: list[float],
    state: dict[str, Any],
) -> tuple[float, float]:
    duration = _state_duration(state)
    if not anchor_times:
        return 0.0, duration
    left_idx = max(0, min(int(ordinal), len(anchor_times) - 1))
    right_idx = min(left_idx + 1, len(anchor_times) - 1)
    lo = max(0.0, float(anchor_times[left_idx]) - 90.0)
    hi = duration if right_idx == left_idx else min(duration, float(anchor_times[right_idx]) + 110.0)
    if hi < lo + 12.0:
        return 0.0, duration
    return lo, hi


def _move_state_to_candidate(
    state: dict[str, Any],
    role: str,
    cand: SequenceBoundaryCandidate,
    *,
    blend: float,
    max_shift_ms: float,
    min_move_ms: float,
) -> bool:
    duration = _state_duration(state)
    params = dict(state.get("params") or {})
    try:
        anchors = oto_params_to_anchors(
            offset=float(params.get("offset", 0.0) or 0.0),
            consonant=float(params.get("consonant", 0.0) or 0.0),
            cutoff=float(params.get("cutoff", 0.0) or 0.0),
            preutterance=float(params.get("preutterance", 0.0) or 0.0),
            overlap=float(params.get("overlap", 0.0) or 0.0),
            duration_ms=duration,
        )
    except Exception:
        return False
    delta = (float(cand.time_ms) - float(anchors.preutterance)) * _clamp(float(blend), 0.0, 1.0)
    if abs(delta) > float(max_shift_ms):
        state["sequence_alignment_hold"] = True
        state["sequence_alignment_reason"] = f"hold:max_shift:{cand.kind}:{cand.score:.3f}"
        return False
    if abs(delta) < float(min_move_ms):
        state["sequence_alignment_hold"] = True
        state["sequence_alignment_reason"] = f"hold:min_move:{cand.kind}:{cand.score:.3f}"
        return False
    shifted = _shift_anchors_for_role(anchors, delta=delta, role=role)
    state["params"] = anchors_to_oto_params(shifted, duration_ms=duration)
    state["sequence_aligned"] = True
    state["sequence_alignment_reason"] = f"{cand.kind}:{cand.source}:{cand.score:.3f}"
    return True


def _shift_anchors_for_role(anchors: OtoAnchors, *, delta: float, role: str) -> OtoAnchors:
    role_text = normalize_role(role)
    if slot_type_for_role(role_text) == "boundary" and role_text != "v-cv":
        return OtoAnchors(
            offset=float(anchors.offset),
            overlap=float(anchors.overlap) + float(delta),
            preutterance=float(anchors.preutterance) + float(delta),
            consonant=float(anchors.consonant) + float(delta),
            cutoff=float(anchors.cutoff) + float(delta),
        )
    return OtoAnchors(
        offset=float(anchors.offset) + float(delta),
        overlap=float(anchors.overlap) + float(delta),
        preutterance=float(anchors.preutterance) + float(delta),
        consonant=float(anchors.consonant) + float(delta),
        cutoff=float(anchors.cutoff) + float(delta),
    )


def _state_anchor_time(state: dict[str, Any]) -> float:
    duration = _state_duration(state)
    params = dict(state.get("params") or {})
    try:
        anchors = oto_params_to_anchors(
            offset=float(params.get("offset", 0.0) or 0.0),
            consonant=float(params.get("consonant", 0.0) or 0.0),
            cutoff=float(params.get("cutoff", 0.0) or 0.0),
            preutterance=float(params.get("preutterance", 0.0) or 0.0),
            overlap=float(params.get("overlap", 0.0) or 0.0),
            duration_ms=duration,
        )
        return float(anchors.preutterance)
    except Exception:
        return 0.0


def _state_duration(state: dict[str, Any]) -> float:
    return max(1.0, float(state.get("duration_ms", 0.0) or 1.0))


def _min_gap_for_role(role: str) -> float:
    role_text = normalize_role(role)
    if role_text == "-cv":
        return 24.0
    if role_text in {"cv", "v", "special"}:
        return 14.0
    return 10.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


__all__ = [
    "SEQUENCE_BOUNDARY_KINDS",
    "SequenceBoundaryCandidate",
    "collect_sequence_boundary_candidates",
    "decode_sequence_alignment_for_states",
    "parse_sequence_role_allowlist",
]
