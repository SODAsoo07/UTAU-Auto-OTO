from __future__ import annotations

import numpy as np

from core.coarse_crnn.labels import COARSE_LABELS, LABEL_TO_ID, coarse_for_phone
from core.coarse_crnn.types import PhoneSegment, Segment


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
) -> list[PhoneSegment]:
    p = np.asarray(probs, dtype=np.float32)
    phone_list = [str(x).strip() for x in phones if str(x).strip()]
    if p.ndim != 2 or p.shape[0] <= 0 or not phone_list:
        return []
    frame_count = int(p.shape[0])
    label_to_id = {str(label): idx for idx, label in enumerate(labels)}
    phone_coarse = [coarse_for_phone(phone, language=language) for phone in phone_list]
    state_coarse = ["SIL"] + phone_coarse + ["SIL"] if frame_count >= len(phone_list) + 2 else list(phone_coarse)
    state_offset = 1 if len(state_coarse) == len(phone_list) + 2 else 0
    state_count = len(state_coarse)
    if frame_count < state_count:
        return _uniform_phone_segments(phone_list, language=language, duration_sec=duration_sec)

    eps = 1e-7
    logp = np.log(np.maximum(p, eps))
    scores = np.zeros((frame_count, state_count), dtype=np.float32)
    for state_idx, coarse in enumerate(state_coarse):
        label_idx = int(label_to_id.get(coarse, label_to_id.get("SP", 0)))
        scores[:, state_idx] = logp[:, label_idx]

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
        confidence = float(np.mean(p[start_f:end_f, label_idx])) if end_f > start_f else 0.0
        out.append(
            PhoneSegment(
                phone=phone,
                start=max(0.0, float(start_f) * hop_sec),
                end=min(float(duration_sec), float(end_f) * hop_sec),
                coarse=coarse,
                confidence=confidence,
            )
        )
    return _repair_monotonic(out, duration_sec=duration_sec)


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
