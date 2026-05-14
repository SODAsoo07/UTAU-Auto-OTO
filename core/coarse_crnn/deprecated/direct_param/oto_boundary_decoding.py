from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.oto_audio_candidates import AudioCandidates
from core.coarse_crnn.oto_targets import OtoAnchors, anchors_to_oto_params, oto_params_to_anchors
from core.coarse_crnn.slot_graph import slot_type_for_role


BOUNDARY_SLOT_CLASSES = ("none", "vowel_tail", "vowel_transition", "next_onset")


@dataclass(frozen=True)
class BoundaryCandidate:
    time_ms: float
    score: float
    source: str


def boundary_slot_label_for_role(role: object) -> str:
    role_text = normalize_role(role)
    if role_text == "vc":
        return "vowel_tail"
    if role_text == "vv":
        return "vowel_transition"
    if role_text == "v-cv":
        return "next_onset"
    return "none"


def boundary_slot_target_id(role: object) -> int:
    label = boundary_slot_label_for_role(role)
    try:
        return BOUNDARY_SLOT_CLASSES.index(label)
    except ValueError:
        return 0


def collect_boundary_candidates(
    candidates: AudioCandidates | None,
    *,
    role: object,
    duration_ms: float,
) -> list[BoundaryCandidate]:
    if candidates is None:
        return []
    role_text = normalize_role(role)
    duration = max(1.0, float(duration_ms or getattr(candidates, "duration_ms", 0.0) or 1.0))
    out: list[BoundaryCandidate] = []
    active_start = max(0.0, float(getattr(candidates, "active_start_ms", 0.0) or 0.0))
    active_end = min(duration, float(getattr(candidates, "active_end_ms", duration) or duration))

    def add(time_ms: float, score: float, source: str) -> None:
        t = max(0.0, min(duration, float(time_ms)))
        s = max(0.0, min(1.0, float(score)))
        out.append(BoundaryCandidate(time_ms=t, score=s, source=str(source)))

    if active_end > active_start:
        add(active_start, 0.18, "active_start")
        add(active_end, 0.18, "active_end")

    for peak in list(getattr(candidates, "onset_peaks", []) or []):
        strength = float(getattr(peak, "strength", 0.0) or 0.0)
        if strength <= 0.0:
            continue
        add(float(getattr(peak, "time_ms", 0.0) or 0.0), strength, "onset")

    for peak in list(getattr(candidates, "mel_delta_peaks", []) or []):
        strength = float(getattr(peak, "strength", 0.0) or 0.0)
        if strength <= 0.0:
            continue
        # Mel-delta peaks are especially useful for VV and voiced transitions.
        mult = 1.15 if role_text in {"vv", "v-cv"} else 0.85
        add(float(getattr(peak, "time_ms", 0.0) or 0.0), min(1.0, strength * mult), "mel_delta")

    for seg in list(getattr(candidates, "stable_vowel_segments", []) or []):
        conf = float(getattr(seg, "vowel_confidence", 0.0) or 0.0)
        voiced = float(getattr(seg, "voiced_score", 0.0) or 0.0)
        stable = float(getattr(seg, "stability_score", 0.0) or 0.0)
        score = max(0.05, min(1.0, (0.45 * conf) + (0.35 * voiced) + (0.20 * stable)))
        if role_text == "vc":
            add(float(getattr(seg, "end_ms", 0.0) or 0.0), score, "vowel_end")
        elif role_text == "vv":
            add(float(getattr(seg, "center_ms", 0.0) or 0.0), score * 0.80, "vowel_center")
            add(float(getattr(seg, "end_ms", 0.0) or 0.0), score, "vowel_end")
        elif role_text == "v-cv":
            add(float(getattr(seg, "start_ms", 0.0) or 0.0), score * 0.75, "vowel_start")
            add(float(getattr(seg, "end_ms", 0.0) or 0.0), score * 0.80, "vowel_end")

    out.sort(key=lambda item: (item.time_ms, -item.score, item.source))
    merged: list[BoundaryCandidate] = []
    merge_ms = 12.0
    for cand in out:
        if merged and abs(float(cand.time_ms) - float(merged[-1].time_ms)) <= merge_ms:
            prev = merged[-1]
            if cand.score > prev.score:
                merged[-1] = cand
            continue
        merged.append(cand)
    return merged


def correct_boundary_params_from_candidates(
    *,
    predicted_params: dict[str, float],
    audio_candidates: AudioCandidates | None,
    role: object,
    duration_ms: float,
    prev_anchor_ms: float | None = None,
    next_anchor_ms: float | None = None,
    max_delta_ms: float = 260.0,
    blend: float = 0.72,
) -> tuple[dict[str, float], str]:
    role_text = normalize_role(role)
    if slot_type_for_role(role_text) != "boundary":
        return dict(predicted_params), ""
    duration = max(1.0, float(duration_ms))
    try:
        anchors = oto_params_to_anchors(
            offset=float(predicted_params.get("offset", 0.0) or 0.0),
            consonant=float(predicted_params.get("consonant", 0.0) or 0.0),
            cutoff=float(predicted_params.get("cutoff", 0.0) or 0.0),
            preutterance=float(predicted_params.get("preutterance", 0.0) or 0.0),
            overlap=float(predicted_params.get("overlap", 0.0) or 0.0),
            duration_ms=duration,
        )
    except Exception:
        return dict(predicted_params), ""
    target = float(anchors.preutterance)
    candidates = collect_boundary_candidates(audio_candidates, role=role_text, duration_ms=duration)
    if not candidates:
        return dict(predicted_params), ""

    lo, hi = _anchor_window(prev_anchor_ms, next_anchor_ms, duration_ms=duration, role=role_text)
    best: tuple[float, BoundaryCandidate] | None = None
    for cand in candidates:
        t = float(cand.time_ms)
        if t < lo or t > hi:
            continue
        delta = abs(t - target)
        if delta > max(0.0, float(max_delta_ms)):
            continue
        role_bonus = _source_role_bonus(cand.source, role_text)
        center_penalty = _window_center_penalty(t, lo, hi)
        score = delta - (42.0 * float(cand.score)) - role_bonus + center_penalty
        if best is None or score < best[0]:
            best = (score, cand)
    if best is None:
        return dict(predicted_params), ""
    cand = best[1]
    delta = (float(cand.time_ms) - target) * max(0.0, min(1.0, float(blend)))
    if abs(delta) < 4.0:
        return dict(predicted_params), ""
    shifted = _shift_boundary_anchors(anchors, delta=delta, role=role_text)
    return anchors_to_oto_params(shifted, duration_ms=duration), f"boundary_candidate:{cand.source}:{cand.score:.3f}"


def decode_boundary_slot_graph_for_states(
    *,
    row_states: list[dict[str, Any]],
    language: str,
    role_resolver: Callable[[Any], str],
    candidate_getter: Callable[[str], AudioCandidates | None],
    max_delta_ms: float = 260.0,
    blend: float = 0.72,
) -> tuple[int, int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for state in row_states:
        row = state.get("row")
        if row is None:
            continue
        groups.setdefault(str(getattr(row, "wav_abs", "")), []).append(state)

    moved = 0
    held = 0
    for wav_abs, states in groups.items():
        states.sort(key=lambda state: int(getattr(state.get("row"), "row_index", 0)))
        audio_candidates = candidate_getter(str(wav_abs))
        anchor_points = _anchor_points_for_states(states, role_resolver=role_resolver)
        boundary_states: list[tuple[dict[str, Any], str, float | None, float | None]] = []
        for state in states:
            row = state.get("row")
            role = normalize_role(role_resolver(row))
            if slot_type_for_role(role) != "boundary":
                continue
            prev_anchor, next_anchor = _nearest_anchor_window(state, anchor_points)
            boundary_states.append((state, role, prev_anchor, next_anchor))
        if not boundary_states:
            continue
        if not anchor_points:
            held += len(boundary_states)
            continue
        # Weak DP: only enforce order inside the same anchor window. This avoids
        # forcing VC rows to be one-to-one with syllable rows while still
        # preventing multiple boundary rows in one window from crossing.
        by_window: dict[tuple[int, int], list[tuple[dict[str, Any], str, float | None, float | None]]] = {}
        for item in boundary_states:
            key = _anchor_window_key(item[2], item[3], anchor_points)
            by_window.setdefault(key, []).append(item)
        for items in by_window.values():
            prev_time: float | None = None
            for state, role, prev_anchor, next_anchor in items:
                params = dict(state.get("params") or {})
                duration = max(1.0, float(state.get("duration_ms", 0.0) or 1.0))
                corrected, reason = correct_boundary_params_from_candidates(
                    predicted_params=params,
                    audio_candidates=audio_candidates,
                    role=role,
                    duration_ms=duration,
                    prev_anchor_ms=prev_anchor,
                    next_anchor_ms=next_anchor,
                    max_delta_ms=max_delta_ms,
                    blend=blend,
                )
                if not reason:
                    held += 1
                    continue
                try:
                    new_anchors = oto_params_to_anchors(
                        offset=float(corrected.get("offset", 0.0) or 0.0),
                        consonant=float(corrected.get("consonant", 0.0) or 0.0),
                        cutoff=float(corrected.get("cutoff", 0.0) or 0.0),
                        preutterance=float(corrected.get("preutterance", 0.0) or 0.0),
                        overlap=float(corrected.get("overlap", 0.0) or 0.0),
                        duration_ms=duration,
                    )
                except Exception:
                    held += 1
                    continue
                if prev_time is not None and float(new_anchors.preutterance) < float(prev_time) + 8.0:
                    held += 1
                    continue
                prev_time = float(new_anchors.preutterance)
                state["params"] = corrected
                state["boundary_slot_graph"] = True
                state["boundary_slot_graph_reason"] = reason
                moved += 1
    return moved, held


def _anchor_window(
    prev_anchor_ms: float | None,
    next_anchor_ms: float | None,
    *,
    duration_ms: float,
    role: str,
) -> tuple[float, float]:
    duration = max(1.0, float(duration_ms))
    pad = 110.0 if normalize_role(role) in {"vc", "vv"} else 150.0
    lo = 0.0 if prev_anchor_ms is None else max(0.0, float(prev_anchor_ms) - pad)
    hi = duration if next_anchor_ms is None else min(duration, float(next_anchor_ms) + pad)
    if hi < lo + 12.0:
        return 0.0, duration
    return lo, hi


def _source_role_bonus(source: str, role: str) -> float:
    src = str(source or "")
    role_text = normalize_role(role)
    if role_text == "vc" and src == "vowel_end":
        return 20.0
    if role_text == "vv" and src in {"mel_delta", "vowel_end", "vowel_center"}:
        return 18.0
    if role_text == "v-cv" and src == "onset":
        return 22.0
    return 0.0


def _window_center_penalty(time_ms: float, lo: float, hi: float) -> float:
    width = max(1.0, float(hi) - float(lo))
    center = (float(lo) + float(hi)) * 0.5
    return abs(float(time_ms) - center) / width * 8.0


def _shift_boundary_anchors(anchors: OtoAnchors, *, delta: float, role: str) -> OtoAnchors:
    role_text = normalize_role(role)
    if role_text == "v-cv":
        return OtoAnchors(
            offset=float(anchors.offset) + float(delta),
            overlap=float(anchors.overlap) + float(delta),
            preutterance=float(anchors.preutterance) + float(delta),
            consonant=float(anchors.consonant) + float(delta),
            cutoff=float(anchors.cutoff) + float(delta),
        )
    return OtoAnchors(
        offset=float(anchors.offset),
        overlap=float(anchors.overlap) + float(delta),
        preutterance=float(anchors.preutterance) + float(delta),
        consonant=float(anchors.consonant) + float(delta),
        cutoff=float(anchors.cutoff) + float(delta),
    )


def _anchor_points_for_states(
    states: list[dict[str, Any]],
    *,
    role_resolver: Callable[[Any], str],
) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for state in states:
        row = state.get("row")
        role = normalize_role(role_resolver(row))
        if slot_type_for_role(role) != "anchor":
            continue
        try:
            duration = max(1.0, float(state.get("duration_ms", 0.0) or 1.0))
            params = dict(state.get("params") or {})
            anchors = oto_params_to_anchors(
                offset=float(params.get("offset", 0.0) or 0.0),
                consonant=float(params.get("consonant", 0.0) or 0.0),
                cutoff=float(params.get("cutoff", 0.0) or 0.0),
                preutterance=float(params.get("preutterance", 0.0) or 0.0),
                overlap=float(params.get("overlap", 0.0) or 0.0),
                duration_ms=duration,
            )
        except Exception:
            continue
        out.append((int(getattr(row, "row_index", 0)), float(anchors.preutterance)))
    out.sort(key=lambda item: item[1])
    return out


def _nearest_anchor_window(
    state: dict[str, Any],
    anchor_points: list[tuple[int, float]],
) -> tuple[float | None, float | None]:
    if not anchor_points:
        return None, None
    try:
        duration = max(1.0, float(state.get("duration_ms", 0.0) or 1.0))
        params = dict(state.get("params") or {})
        anchors = oto_params_to_anchors(
            offset=float(params.get("offset", 0.0) or 0.0),
            consonant=float(params.get("consonant", 0.0) or 0.0),
            cutoff=float(params.get("cutoff", 0.0) or 0.0),
            preutterance=float(params.get("preutterance", 0.0) or 0.0),
            overlap=float(params.get("overlap", 0.0) or 0.0),
            duration_ms=duration,
        )
        target = float(anchors.preutterance)
    except Exception:
        target = float(np.median([point[1] for point in anchor_points]))
    times = [point[1] for point in anchor_points]
    prev = None
    next_ = None
    for t in times:
        if t <= target:
            prev = t
        elif next_ is None:
            next_ = t
            break
    if prev is None and times:
        prev = times[0]
    if next_ is None and len(times) >= 2:
        next_ = times[-1]
    return prev, next_


def _anchor_window_key(
    prev_anchor: float | None,
    next_anchor: float | None,
    anchor_points: list[tuple[int, float]],
) -> tuple[int, int]:
    values = [point[1] for point in anchor_points]
    def idx(value: float | None) -> int:
        if value is None:
            return -1
        for i, t in enumerate(values):
            if abs(float(t) - float(value)) <= 1e-4:
                return i
        return -1
    return idx(prev_anchor), idx(next_anchor)


__all__ = [
    "BOUNDARY_SLOT_CLASSES",
    "BoundaryCandidate",
    "boundary_slot_label_for_role",
    "boundary_slot_target_id",
    "collect_boundary_candidates",
    "correct_boundary_params_from_candidates",
    "decode_boundary_slot_graph_for_states",
]
