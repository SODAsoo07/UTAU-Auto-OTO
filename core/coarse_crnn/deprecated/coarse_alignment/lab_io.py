from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from core.coarse_crnn.labels import LABEL_TO_ID, coarse_for_phone, coarse_id_for_phone
from core.coarse_crnn.deprecated.coarse_alignment.types import Segment
from core.textio_utils import read_text_auto


def _coerce_time_scale(values: list[float]) -> float:
    if not values:
        return 1.0
    max_value = max(float(v) for v in values)
    if max_value > 100000.0:
        return 1.0 / 10000000.0
    if max_value > 1000.0:
        return 0.001
    return 1.0


def parse_timed_lab(path: str) -> list[Segment]:
    text, _enc, err = read_text_auto(path)
    if err or text is None:
        return []
    rows: list[tuple[float, float, str]] = []
    raw_times: list[float] = []
    for raw in text.splitlines():
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except Exception:
            continue
        label = " ".join(parts[2:]).strip()
        if end <= start:
            continue
        rows.append((start, end, label))
        raw_times.extend([start, end])
    if not rows:
        return []
    scale = _coerce_time_scale(raw_times)
    return [Segment(label=label, start=float(start) * scale, end=float(end) * scale) for start, end, label in rows]


def parse_textgrid(path: str, *, tier_names: tuple[str, ...] = ("phones", "phone", "phoneme")) -> list[Segment]:
    try:
        import textgrid  # type: ignore
    except Exception:
        return []
    try:
        tg = textgrid.TextGrid.fromFile(str(path))
    except Exception:
        return []
    preferred = {name.lower() for name in tier_names}
    picked = None
    for tier in tg.tiers:
        if str(getattr(tier, "name", "") or "").strip().lower() in preferred:
            picked = tier
            break
    if picked is None and getattr(tg, "tiers", None):
        picked = tg.tiers[-1]
    if picked is None:
        return []
    out: list[Segment] = []
    for interval in picked:
        try:
            start = float(interval.minTime)
            end = float(interval.maxTime)
            label = str(interval.mark or "").strip()
        except Exception:
            continue
        if end <= start:
            continue
        out.append(Segment(label=label, start=start, end=end))
    return out


def find_alignment_label(wav_path: str) -> tuple[str, str]:
    stem = os.path.splitext(str(wav_path))[0]
    for suffix, fmt in ((".TextGrid", "textgrid"), (".textgrid", "textgrid"), (".lab", "timed_lab"), (".sinsy.lab", "timed_lab")):
        candidate = stem + suffix
        if os.path.isfile(candidate):
            return os.path.abspath(candidate), fmt
    return "", ""


def load_aligned_segments(label_path: str, label_format: str = "") -> list[Segment]:
    fmt = str(label_format or "").strip().lower()
    if fmt == "textgrid" or str(label_path).lower().endswith(".textgrid"):
        return parse_textgrid(label_path)
    return parse_timed_lab(label_path)


def read_lab_transcript(path: str) -> str:
    text, _enc, err = read_text_auto(path)
    if err or text is None:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return ""
    if len(lines) == 1 and len(lines[0].split()) < 3:
        return lines[0]
    return ""


def read_transcript_for_wav(wav_path: str) -> str:
    lab = os.path.splitext(str(wav_path))[0] + ".lab"
    if os.path.isfile(lab):
        transcript = read_lab_transcript(lab)
        if transcript:
            return transcript
    return os.path.splitext(os.path.basename(str(wav_path)))[0]


def frame_targets_from_segments(
    segments: list[Segment],
    *,
    frame_count: int,
    hop_sec: float,
    duration_sec: float,
    language: str,
    ignore_index: int = -100,
) -> np.ndarray:
    n = int(frame_count)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    targets = np.full((n,), LABEL_TO_ID["SIL"], dtype=np.int64)
    covered = np.zeros((n,), dtype=bool)
    for seg in segments or []:
        start = max(0.0, float(seg.start))
        end = min(float(duration_sec), float(seg.end))
        if end <= start:
            continue
        s = max(0, min(n, int(round(start / max(hop_sec, 1e-6)))))
        e = max(s + 1, min(n, int(round(end / max(hop_sec, 1e-6)))))
        label_id = coarse_id_for_phone(seg.label, language=language)
        targets[s:e] = int(label_id)
        covered[s:e] = True
    if not bool(np.any(covered)):
        targets[:] = ignore_index
    return targets


def label_compatibility_ratio(segments: list[Segment], *, language: str = "") -> float:
    if not segments:
        return 0.0
    usable = 0
    total = 0
    for seg in segments:
        label = str(seg.label or "").strip()
        if not label:
            continue
        total += 1
        if coarse_for_phone(label, language=language) in LABEL_TO_ID:
            usable += 1
    return float(usable) / float(max(1, total))


def iter_wavs(root: str):
    base = Path(root)
    if not base.exists():
        return
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            yield str(path)


__all__ = [
    "find_alignment_label",
    "frame_targets_from_segments",
    "iter_wavs",
    "label_compatibility_ratio",
    "load_aligned_segments",
    "parse_textgrid",
    "parse_timed_lab",
    "read_transcript_for_wav",
]

