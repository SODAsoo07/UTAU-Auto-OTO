from __future__ import annotations

import numpy as np

from core.coarse_crnn.labels import COARSE_LABELS, LABEL_TO_ID, coarse_for_phone
from core.coarse_crnn.deprecated.coarse_alignment.types import PhoneSegment, Segment


def greedy_segments(
    probs: np.ndarray,
    *,
    labels: list[str] | tuple[str, ...] = COARSE_LABELS,
    hop_sec: float,
    min_segment_sec: float = 0.015,
) -> list[Segment]:
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim != 2 or p.shape[0] <= 0:
        return []
    ids = np.argmax(p, axis=1)
    out: list[Segment] = []
    start = 0
    cur = int(ids[0])
    for idx in range(1, int(ids.shape[0]) + 1):
        if idx < int(ids.shape[0]) and int(ids[idx]) == cur:
            continue
        label = str(labels[cur])
        if label != "BLANK":
            score = float(np.mean(p[start:idx, cur]))
            out.append(Segment(label=label, start=float(start) * hop_sec, end=float(idx) * hop_sec, score=score))
        if idx < int(ids.shape[0]):
            start = idx
            cur = int(ids[idx])
    return _merge_short_segments(out, min_segment_sec=min_segment_sec)


def _merge_short_segments(segments: list[Segment], *, min_segment_sec: float) -> list[Segment]:
    if len(segments) <= 1:
        return segments
    out = list(segments)
    idx = 0
    while idx < len(out):
        seg = out[idx]
        if (float(seg.end) - float(seg.start)) >= float(min_segment_sec) or len(out) <= 1:
            idx += 1
            continue
        if idx > 0:
            prev = out[idx - 1]
            out[idx - 1] = Segment(prev.label, prev.start, seg.end, prev.score)
            out.pop(idx)
        else:
            nxt = out[idx + 1]
            out[idx + 1] = Segment(nxt.label, seg.start, nxt.end, nxt.score)
            out.pop(idx)
    return out


def viterbi_align_phones(
    probs: np.ndarray,
    phones: list[str],
    *,
    labels: list[str] | tuple[str, ...] = COARSE_LABELS,
    hop_sec: float,
    language: str,
    duration_sec: float,
    active_range_sec: tuple[float, float] | None = None,
    boundary_priors_sec: list[tuple[float, float]] | None = None,
    boundary_probs: np.ndarray | None = None,
) -> list[PhoneSegment]:
    p = np.asarray(probs, dtype=np.float32)
    phone_list = [str(x).strip() for x in phones if str(x).strip()]
    if p.ndim != 2 or p.shape[0] <= 0 or not phone_list:
        return []
    frame_count = int(p.shape[0])
    label_to_id = {str(label): idx for idx, label in enumerate(labels)}
    phone_coarse = [coarse_for_phone(phone, language=language) for phone in phone_list]
    crop_start, crop_end = _active_frame_range(
        p,
        label_to_id=label_to_id,
        phone_coarse=phone_coarse,
        hop_sec=hop_sec,
        active_range_sec=active_range_sec,
    )
    if crop_end - crop_start >= len(phone_list):
        p_work = p[crop_start:crop_end]
        boundary_probs_work = None if boundary_probs is None else np.asarray(boundary_probs, dtype=np.float32)[crop_start:crop_end]
        time_offset = float(crop_start) * float(hop_sec)
        use_sil_edges = False
    else:
        p_work = p
        boundary_probs_work = None if boundary_probs is None else np.asarray(boundary_probs, dtype=np.float32)
        time_offset = 0.0
        use_sil_edges = True
    frame_count = int(p_work.shape[0])
    state_coarse = ["SIL"] + phone_coarse + ["SIL"] if use_sil_edges and frame_count >= len(phone_list) + 2 else list(phone_coarse)
    state_offset = 1 if len(state_coarse) == len(phone_list) + 2 else 0
    state_count = len(state_coarse)
    if frame_count < state_count:
        return _uniform_phone_segments(phone_list, language=language, duration_sec=duration_sec)

    eps = 1e-7
    logp = np.log(np.maximum(p_work, eps))
    scores = np.zeros((frame_count, state_count), dtype=np.float32)
    for state_idx, coarse in enumerate(state_coarse):
        label_idx = int(label_to_id.get(coarse, label_to_id.get("SP", 0)))
        scores[:, state_idx] = logp[:, label_idx]
        scores[:, state_idx] += _state_score_bias(coarse)

    neg = -1.0e12
    dp = np.full((frame_count, state_count), neg, dtype=np.float32)
    back = np.zeros((frame_count, state_count), dtype=np.int16)
    dp[0, 0] = scores[0, 0]
    for t in range(1, frame_count):
        max_state = min(state_count - 1, t)
        for s in range(0, max_state + 1):
            stay = dp[t - 1, s]
            adv = dp[t - 1, s - 1] - 0.08 if s > 0 else neg
            if adv > stay:
                dp[t, s] = adv + scores[t, s]
                back[t, s] = 1
            else:
                dp[t, s] = stay + scores[t, s]
                back[t, s] = 0

    states = np.zeros((frame_count,), dtype=np.int32)
    s = state_count - 1
    for t in range(frame_count - 1, -1, -1):
        states[t] = s
        if t > 0 and back[t, s] == 1:
            s -= 1
            if s < 0:
                s = 0

    out: list[PhoneSegment] = []
    for phone_idx, phone in enumerate(phone_list):
        state_idx = phone_idx + state_offset
        frames = np.where(states == state_idx)[0]
        if frames.size <= 0:
            continue
        start_f = int(frames[0])
        end_f = int(frames[-1]) + 1
        coarse = phone_coarse[phone_idx]
        label_idx = int(label_to_id.get(coarse, 0))
        confidence = float(np.mean(p_work[start_f:end_f, label_idx])) if end_f > start_f else 0.0
        out.append(
            PhoneSegment(
                phone=phone,
                start=max(0.0, time_offset + float(start_f) * hop_sec),
                end=min(float(duration_sec), time_offset + float(end_f) * hop_sec),
                coarse=coarse,
                confidence=confidence,
            )
        )
    out = _apply_duration_prior_blend(
        out,
        phone_list=phone_list,
        phone_coarse=phone_coarse,
        active_start=time_offset,
        active_end=min(float(duration_sec), time_offset + float(frame_count) * float(hop_sec)),
        boundary_priors_sec=boundary_priors_sec,
    )
    out = _snap_internal_boundaries(
        out,
        probs=p_work,
        boundary_probs=boundary_probs_work,
        label_to_id=label_to_id,
        hop_sec=hop_sec,
        time_offset=time_offset,
    )
    return _repair_monotonic(out, duration_sec=duration_sec)


def _active_frame_range(
    probs: np.ndarray,
    *,
    label_to_id: dict[str, int],
    phone_coarse: list[str],
    hop_sec: float,
    active_range_sec: tuple[float, float] | None = None,
) -> tuple[int, int]:
    n = int(probs.shape[0])
    if n <= 0:
        return 0, 0
    if any(coarse == "SIL" for coarse in phone_coarse):
        return 0, n
    if active_range_sec is not None:
        try:
            start_sec, end_sec = float(active_range_sec[0]), float(active_range_sec[1])
            start = max(0, min(n, int(round(start_sec / max(float(hop_sec), 1e-6)))))
            end = max(start + 1, min(n, int(round(end_sec / max(float(hop_sec), 1e-6)))))
            if end - start >= max(3, len(phone_coarse)):
                return start, end
        except Exception:
            pass
    sil_idx = label_to_id.get("SIL")
    blank_idx = label_to_id.get("BLANK")
    non_sil = np.ones((n,), dtype=np.float32)
    if sil_idx is not None:
        non_sil -= probs[:, int(sil_idx)]
    if blank_idx is not None:
        non_sil -= probs[:, int(blank_idx)] * 0.5
    expected = np.zeros((n,), dtype=np.float32)
    for coarse in set(phone_coarse):
        idx = label_to_id.get(coarse)
        if idx is not None:
            expected = np.maximum(expected, probs[:, int(idx)])
    activity = np.maximum(non_sil, expected).astype(np.float32)
    if n >= 7:
        kernel = np.ones((7,), dtype=np.float32) / 7.0
        activity = np.convolve(activity, kernel, mode="same").astype(np.float32)
    threshold = max(0.22, min(0.55, float(np.mean(activity) + 0.20 * np.std(activity))))
    active = np.where(activity >= threshold)[0]
    if active.size <= 0:
        return 0, n
    start = int(active[0])
    end = int(active[-1]) + 1
    margin = max(2, int(round(0.060 / max(float(hop_sec), 1e-6))))
    start = max(0, start - margin)
    end = min(n, end + margin)
    if end - start < max(3, len(phone_coarse)):
        return 0, n
    return start, end


def _state_score_bias(coarse: str) -> float:
    if coarse == "V":
        return 0.06
    if coarse == "SIL":
        return 0.02
    if coarse in {"C_OBS", "C_FR", "C_SON"}:
        return 0.03
    return 0.0


def _duration_prior_seconds(coarse: str) -> float:
    return {
        "V": 0.090,
        "C_OBS": 0.045,
        "C_FR": 0.060,
        "C_SON": 0.055,
        "SP": 0.050,
        "BR": 0.120,
        "SIL": 0.100,
    }.get(str(coarse), 0.060)


def _apply_duration_prior_blend(
    segments: list[PhoneSegment],
    *,
    phone_list: list[str],
    phone_coarse: list[str],
    active_start: float,
    active_end: float,
    boundary_priors_sec: list[tuple[float, float]] | None = None,
) -> list[PhoneSegment]:
    if len(segments) != len(phone_list) or active_end <= active_start:
        return segments
    prior_bounds = _coerce_boundary_priors(boundary_priors_sec, expected=len(phone_list))
    if prior_bounds is None:
        priors = np.asarray([_duration_prior_seconds(coarse) for coarse in phone_coarse], dtype=np.float64)
        total = float(np.sum(priors))
        if total <= 0.0:
            return segments
        prior_durations = priors / total * (float(active_end) - float(active_start))
        prior_bounds = [float(active_start)]
        for dur in prior_durations:
            prior_bounds.append(prior_bounds[-1] + float(dur))
    mean_conf = float(np.mean([float(seg.confidence or 0.0) for seg in segments])) if segments else 0.0
    # Low-confidence paths are often dragged by silence or one dominant vowel.
    # Pull them more strongly toward duration priors, but keep acoustic evidence
    # dominant when confidence is already decent.
    alpha = max(0.18, min(0.55, 0.62 - mean_conf))
    out: list[PhoneSegment] = []
    for idx, seg in enumerate(segments):
        p_start = prior_bounds[idx]
        p_end = prior_bounds[idx + 1]
        start = (float(seg.start) * (1.0 - alpha)) + (float(p_start) * alpha)
        end = (float(seg.end) * (1.0 - alpha)) + (float(p_end) * alpha)
        out.append(PhoneSegment(seg.phone, start, end, seg.coarse, seg.confidence))
    return out


def _coerce_boundary_priors(priors: list[tuple[float, float]] | None, *, expected: int) -> list[float] | None:
    if priors is None or len(priors) != int(expected):
        return None
    bounds: list[float] = []
    for idx, item in enumerate(priors):
        try:
            start = float(item[0])
            end = float(item[1])
        except Exception:
            return None
        if end <= start:
            return None
        if idx == 0:
            bounds.append(start)
        bounds.append(end)
    for idx in range(1, len(bounds)):
        if bounds[idx] < bounds[idx - 1]:
            return None
    return bounds


def _snap_internal_boundaries(
    segments: list[PhoneSegment],
    *,
    probs: np.ndarray,
    boundary_probs: np.ndarray | None = None,
    label_to_id: dict[str, int],
    hop_sec: float,
    time_offset: float,
    search_sec: float = 0.080,
) -> list[PhoneSegment]:
    if len(segments) <= 1 or probs.ndim != 2 or probs.shape[0] <= 2:
        return segments
    out = [PhoneSegment(seg.phone, seg.start, seg.end, seg.coarse, seg.confidence) for seg in segments]
    n = int(probs.shape[0])
    win = max(2, int(round(float(search_sec) / max(float(hop_sec), 1e-6))))
    for idx in range(len(out) - 1):
        left = out[idx]
        right = out[idx + 1]
        left_id = label_to_id.get(left.coarse)
        right_id = label_to_id.get(right.coarse)
        if left_id is None or right_id is None:
            continue
        boundary = (float(left.end) + float(right.start)) * 0.5
        center = int(round((boundary - float(time_offset)) / max(float(hop_sec), 1e-6)))
        lo = max(1, center - win)
        hi = min(n - 1, center + win)
        if hi <= lo:
            continue
        best_c = center
        best_score = -1e9
        for c in range(lo, hi + 1):
            l0 = max(0, c - 2)
            r1 = min(n, c + 2)
            left_score = float(np.mean(probs[l0:c, int(left_id)])) if c > l0 else float(probs[c - 1, int(left_id)])
            right_score = float(np.mean(probs[c:r1, int(right_id)])) if r1 > c else float(probs[c, int(right_id)])
            # Keep the original boundary as a soft anchor so noisy posterior does
            # not move boundaries across an entire short phone.
            shift_penalty = abs(c - center) * 0.006
            boundary_score = 0.0
            if boundary_probs is not None and int(boundary_probs.shape[0]) > c:
                boundary_score = float(boundary_probs[c]) * 0.22
            score = left_score + right_score + boundary_score - shift_penalty
            if score > best_score:
                best_score = score
                best_c = c
        snapped = float(time_offset) + float(best_c) * float(hop_sec)
        min_left = float(left.start) + 0.004
        max_right = float(right.end) - 0.004
        snapped = max(min_left, min(max_right, snapped))
        out[idx] = PhoneSegment(left.phone, left.start, snapped, left.coarse, left.confidence)
        out[idx + 1] = PhoneSegment(right.phone, snapped, right.end, right.coarse, right.confidence)
    return out


def _uniform_phone_segments(phones: list[str], *, language: str, duration_sec: float) -> list[PhoneSegment]:
    n = max(1, len(phones))
    out: list[PhoneSegment] = []
    for idx, phone in enumerate(phones):
        start = float(duration_sec) * float(idx) / float(n)
        end = float(duration_sec) * float(idx + 1) / float(n)
        out.append(PhoneSegment(phone=phone, start=start, end=end, coarse=coarse_for_phone(phone, language=language), confidence=0.0))
    return out


def _repair_monotonic(segments: list[PhoneSegment], *, duration_sec: float) -> list[PhoneSegment]:
    out: list[PhoneSegment] = []
    cursor = 0.0
    for seg in segments:
        start = max(cursor, min(float(duration_sec), float(seg.start)))
        end = max(start + 0.001, min(float(duration_sec), float(seg.end)))
        out.append(PhoneSegment(seg.phone, start, end, seg.coarse, seg.confidence))
        cursor = end
    return out


__all__ = ["greedy_segments", "viterbi_align_phones"]

