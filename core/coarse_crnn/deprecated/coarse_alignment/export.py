from __future__ import annotations

import json
import os

from core.coarse_crnn.deprecated.coarse_alignment.types import PhoneSegment, Segment


def write_phone_lab(path: str, phone_segments: list[PhoneSegment]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for seg in phone_segments:
            handle.write(f"{float(seg.start):.6f} {float(seg.end):.6f} {seg.phone}\n")


def write_debug_json(path: str, *, coarse_segments: list[Segment], phone_segments: list[PhoneSegment], meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "meta": dict(meta or {}),
        "coarse_segments": [seg.__dict__ for seg in coarse_segments],
        "phone_segments": [seg.__dict__ for seg in phone_segments],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_textgrid(
    path: str,
    *,
    duration_sec: float,
    phone_segments: list[PhoneSegment],
    coarse_segments: list[Segment] | None = None,
    transcript: str = "",
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    coarse_segments = list(coarse_segments or [])
    try:
        import textgrid  # type: ignore

        tg = textgrid.TextGrid(minTime=0.0, maxTime=float(duration_sec))
        phone_tier = textgrid.IntervalTier(name="phones", minTime=0.0, maxTime=float(duration_sec))
        _add_intervals(phone_tier, [(s.start, s.end, s.phone) for s in phone_segments], float(duration_sec))
        tg.append(phone_tier)
        coarse_tier = textgrid.IntervalTier(name="coarse", minTime=0.0, maxTime=float(duration_sec))
        _add_intervals(coarse_tier, [(s.start, s.end, s.label) for s in coarse_segments], float(duration_sec))
        tg.append(coarse_tier)
        if transcript:
            word_tier = textgrid.IntervalTier(name="transcript", minTime=0.0, maxTime=float(duration_sec))
            word_tier.add(0.0, float(duration_sec), str(transcript))
            tg.append(word_tier)
        tg.write(str(path))
        return
    except Exception:
        pass

    with open(path, "w", encoding="utf-8") as handle:
        handle.write('File type = "ooTextFile"\n')
        handle.write('Object class = "TextGrid"\n\n')
        handle.write("xmin = 0\n")
        handle.write(f"xmax = {float(duration_sec):.6f}\n")
        handle.write("tiers? <exists>\n")
        handle.write("size = 1\n")
        handle.write("item []:\n")
        handle.write('    item [1]:\n        class = "IntervalTier"\n        name = "phones"\n')
        handle.write(f"        xmin = 0\n        xmax = {float(duration_sec):.6f}\n")
        handle.write(f"        intervals: size = {len(phone_segments)}\n")
        for idx, seg in enumerate(phone_segments, start=1):
            handle.write(f"        intervals [{idx}]:\n")
            handle.write(f"            xmin = {float(seg.start):.6f}\n")
            handle.write(f"            xmax = {float(seg.end):.6f}\n")
            handle.write(f'            text = "{seg.phone}"\n')


def _add_intervals(tier, rows: list[tuple[float, float, str]], duration_sec: float) -> None:
    cursor = 0.0
    idx = 0
    for start, end, label in rows:
        s = max(cursor, min(float(duration_sec), float(start)))
        e = max(s, min(float(duration_sec), float(end)))
        if s > cursor:
            tier.add(cursor, s, "")
        if e > s:
            tier.add(s, e, str(label or ""))
        cursor = max(cursor, e)
        idx += 1
    if cursor < float(duration_sec):
        tier.add(cursor, float(duration_sec), "")


__all__ = ["write_debug_json", "write_phone_lab", "write_textgrid"]

