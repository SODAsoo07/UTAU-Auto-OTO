from __future__ import annotations

import os
from dataclasses import dataclass

from core.oto_mapping_confidence import evaluate_index_plan
from core.oto_mapping_plan import build_monotonic_index_plan


@dataclass
class SinsyLabelEntry:
    start_ms: float
    end_ms: float
    label: str


def _coerce_time_scale(values):
    if not values:
        return 1.0
    max_value = max(float(v) for v in values)
    if max_value > 100000.0:
        return 1.0 / 10000.0
    if max_value > 1000.0:
        return 1.0
    return 1000.0


def resolve_sinsy_label_path(tg_path="", real_wav_name="", explicit_path=""):
    explicit = str(explicit_path or "").strip()
    if explicit:
        return os.path.abspath(explicit) if os.path.isfile(explicit) else ""
    tg_dir = os.path.dirname(str(tg_path or "").strip())
    wav_base = os.path.splitext(str(real_wav_name or "").strip())[0]
    tg_base = os.path.splitext(os.path.basename(str(tg_path or "").strip()))[0]
    for stem in [wav_base, tg_base]:
        if not stem or not tg_dir:
            continue
        for suffix in [".lab", ".sinsy.lab", ".syll.lab"]:
            candidate = os.path.join(tg_dir, stem + suffix)
            if os.path.isfile(candidate):
                return candidate
    return ""


def load_sinsy_label_entries(*, tg_path="", real_wav_name="", explicit_path=""):
    path = resolve_sinsy_label_path(tg_path=tg_path, real_wav_name=real_wav_name, explicit_path=explicit_path)
    if not path:
        return []
    rows = []
    raw_times = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
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
            if not label or end <= start:
                continue
            raw_times.extend([start, end])
            rows.append((start, end, label))
    if not rows:
        return []
    scale = _coerce_time_scale(raw_times)
    return [
        SinsyLabelEntry(start_ms=float(start) * scale, end_ms=float(end) * scale, label=str(label))
        for start, end, label in rows
    ]


def _candidate_bounds_ms(syl_info):
    phones = list((syl_info or {}).get("phones") or [])
    if phones:
        try:
            return float(phones[0].minTime) * 1000.0, float(phones[-1].maxTime) * 1000.0
        except Exception:
            pass
    start_time = (syl_info or {}).get("start_time")
    end_time = (syl_info or {}).get("end_time")
    if start_time is not None and end_time is not None:
        try:
            return float(start_time) * 1000.0, float(end_time) * 1000.0
        except Exception:
            pass
    return None, None


def _label_to_candidate_indices(label_entries, syllables_info):
    pairs = []
    for entry in label_entries or []:
        best_idx = None
        best_key = None
        label_center = (float(entry.start_ms) + float(entry.end_ms)) * 0.5
        for idx, syl in enumerate(syllables_info or []):
            start_ms, end_ms = _candidate_bounds_ms(syl)
            if start_ms is None or end_ms is None or end_ms <= start_ms:
                continue
            overlap = max(0.0, min(end_ms, float(entry.end_ms)) - max(start_ms, float(entry.start_ms)))
            center = (start_ms + end_ms) * 0.5
            center_delta = abs(center - label_center)
            key = (overlap, -center_delta)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is not None:
            pairs.append(int(best_idx))
        else:
            pairs.append(None)
    return pairs


def build_sinsy_guided_anchor_plan(
    *,
    expected_tokens,
    syllables_info,
    label_entries,
    normalize_expected_fn,
    normalize_label_fn,
    label_match_score_fn,
):
    if not expected_tokens or not syllables_info or not label_entries:
        return {"indices": None, "score_rows": [], "meta": evaluate_index_plan([], []), "source": "none"}

    token_list = [normalize_expected_fn(t) for t in (expected_tokens or []) if normalize_expected_fn(t)]
    if not token_list:
        return {"indices": None, "score_rows": [], "meta": evaluate_index_plan([], []), "source": "none"}

    label_to_syllable = _label_to_candidate_indices(label_entries, syllables_info)
    usable = []
    for entry, idx in zip(label_entries, label_to_syllable):
        if idx is None:
            continue
        usable.append((entry, int(idx), normalize_label_fn(entry.label)))
    if len(usable) < len(token_list):
        return {"indices": None, "score_rows": [], "meta": evaluate_index_plan([], []), "source": "none"}

    label_count = len(usable)
    target_count = len(token_list)
    score_rows = []
    for i, target_tok in enumerate(token_list):
        ideal = 0.0 if target_count <= 1 else (float(i) * float(max(label_count - 1, 0)) / float(target_count - 1))
        row = []
        for j, (_entry, mapped_idx, label_tok) in enumerate(usable):
            score = float(label_match_score_fn(target_tok, label_tok)) - (abs(float(j) - ideal) * 8.0)
            if label_tok == target_tok:
                score += 80.0
            if not (0 <= mapped_idx < len(syllables_info)):
                score -= 160.0
            row.append(float(score))
        score_rows.append(row)

    label_indices = build_monotonic_index_plan(score_rows)
    if not label_indices:
        return {"indices": None, "score_rows": score_rows, "meta": evaluate_index_plan(score_rows, []), "source": "none"}

    actual_indices = []
    for label_idx in label_indices:
        if not (0 <= int(label_idx) < len(usable)):
            return {"indices": None, "score_rows": score_rows, "meta": evaluate_index_plan(score_rows, []), "source": "none"}
        actual_indices.append(int(usable[int(label_idx)][1]))
    meta = evaluate_index_plan(score_rows, label_indices)
    return {
        "indices": actual_indices,
        "score_rows": score_rows,
        "meta": meta,
        "source": "sinsy_labels",
    }


__all__ = [
    "SinsyLabelEntry",
    "build_sinsy_guided_anchor_plan",
    "load_sinsy_label_entries",
    "resolve_sinsy_label_path",
]
