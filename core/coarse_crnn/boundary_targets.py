from __future__ import annotations

import os
import re
import wave
from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np

from core.coarse_crnn.alias_role import SPECIAL_ROLE, classify_alias_role, normalize_role
from core.coarse_crnn.boundary_types import (
    ANCHOR_ROLES,
    BOUNDARY_LABELS,
    AbsoluteOtoAnchors,
    OtoRowSpec,
    label_sigma_ms,
)
from core.coarse_crnn.lang import normalize_language
from core.coarse_crnn.labels import coarse_for_phone
from core.coarse_crnn.lang import phones_from_text
from core.coarse_crnn.slot_graph import filename_order_tokens
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def oto_row_to_absolute_anchors(*, offset: float, consonant: float, cutoff: float, preutterance: float, overlap: float, duration_ms: float) -> AbsoluteOtoAnchors:
    offset_abs = max(0.0, float(offset))
    pre_abs = max(offset_abs, offset_abs + max(0.0, float(preutterance)))
    overlap_abs = min(pre_abs, max(offset_abs, offset_abs + max(0.0, float(overlap))))
    consonant_abs = max(pre_abs, offset_abs + max(0.0, float(consonant)))
    if float(cutoff) < 0.0:
        cutoff_abs = max(consonant_abs + 1.0, float(duration_ms) + float(cutoff))
    else:
        cutoff_abs = max(consonant_abs + 1.0, offset_abs + float(cutoff))
    duration = max(1.0, float(duration_ms))
    cutoff_abs = min(max(consonant_abs + 1.0, cutoff_abs), duration)
    return AbsoluteOtoAnchors(
        offset_abs=offset_abs,
        overlap_abs=overlap_abs,
        pre_abs=pre_abs,
        consonant_abs=consonant_abs,
        cutoff_abs=cutoff_abs,
        confidence=1.0,
        reason="source",
    )


def absolute_anchors_to_oto_params(anchors: AbsoluteOtoAnchors, *, duration_ms: float) -> dict[str, float]:
    offset = max(0.0, float(anchors.offset_abs))
    pre = max(offset, float(anchors.pre_abs))
    ovl = min(pre, max(offset, float(anchors.overlap_abs)))
    cons = max(pre, float(anchors.consonant_abs))
    cutoff_abs = max(cons + 1.0, min(float(duration_ms), float(anchors.cutoff_abs)))
    return {
        "offset": offset,
        "preutterance": max(0.0, pre - offset),
        "overlap": max(0.0, ovl - offset),
        "consonant": max(1.0, cons - offset),
        "cutoff": cutoff_abs - float(duration_ms),
    }


def boundary_events_for_row(spec: OtoRowSpec, anchors: AbsoluteOtoAnchors) -> list[tuple[str, float, float]]:
    role = normalize_role(spec.role)
    if role == "-cv":
        return [
            ("syllable_onset", anchors.offset_abs, label_sigma_ms("syllable_onset")),
            ("vowel_start", anchors.pre_abs, label_sigma_ms("vowel_start")),
        ]
    if role in {"cv", SPECIAL_ROLE}:
        return [
            ("consonant_onset", anchors.offset_abs, label_sigma_ms("consonant_onset")),
            ("vowel_start", anchors.pre_abs, label_sigma_ms("vowel_start")),
            ("vowel_stable", 0.5 * (anchors.pre_abs + anchors.consonant_abs), label_sigma_ms("vowel_stable")),
        ]
    if role == "v":
        return [
            ("vowel_start", anchors.pre_abs, label_sigma_ms("vowel_start")),
            ("vowel_stable", 0.5 * (anchors.pre_abs + anchors.consonant_abs), label_sigma_ms("vowel_stable")),
        ]
    if role == "vc":
        return [
            ("vowel_end", anchors.pre_abs, label_sigma_ms("vowel_end")),
            ("transition_peak", anchors.pre_abs, label_sigma_ms("transition_peak")),
            ("next_onset", anchors.consonant_abs, label_sigma_ms("next_onset")),
        ]
    if role == "vv":
        return [
            ("vowel_end", anchors.pre_abs, label_sigma_ms("vowel_end")),
            ("transition_peak", anchors.pre_abs, label_sigma_ms("transition_peak")),
            ("vowel_start", anchors.consonant_abs, label_sigma_ms("vowel_start")),
        ]
    if role == "v-cv":
        return [
            ("transition_peak", anchors.pre_abs, label_sigma_ms("transition_peak")),
            ("next_onset", anchors.consonant_abs, label_sigma_ms("next_onset")),
            ("consonant_onset", anchors.consonant_abs, label_sigma_ms("consonant_onset")),
        ]
    if role in {"v-", "cv-"}:
        return [
            ("vowel_end", anchors.pre_abs, label_sigma_ms("vowel_end")),
            ("silence_boundary", anchors.cutoff_abs, label_sigma_ms("silence_boundary")),
        ]
    if role in {"br", "endbr"}:
        return [("silence_boundary", anchors.pre_abs, label_sigma_ms("silence_boundary"))]
    return [
        ("vowel_start", anchors.pre_abs, label_sigma_ms("vowel_start")),
        ("transition_peak", anchors.consonant_abs, label_sigma_ms("transition_peak")),
    ]


def build_boundary_target_map(rows: list[tuple[OtoRowSpec, AbsoluteOtoAnchors]], *, duration_ms: float, hop_ms: float, frame_count: int | None = None) -> tuple[list[float], np.ndarray]:
    duration = max(1.0, float(duration_ms))
    hop = max(2.0, float(hop_ms))
    if frame_count is None:
        frame_count = max(1, int(round(duration / hop)))
    times = (np.arange(frame_count, dtype=np.float32) * hop).tolist()
    target = np.zeros((int(frame_count), len(BOUNDARY_LABELS)), dtype=np.float32)
    label_to_idx = {name: idx for idx, name in enumerate(BOUNDARY_LABELS)}
    for spec, anchors in rows:
        for label, center_ms, sigma_ms in boundary_events_for_row(spec, anchors):
            idx = label_to_idx[label]
            sigma = max(4.0, float(sigma_ms))
            center = float(center_ms)
            for frame_idx in range(int(frame_count)):
                t = frame_idx * hop
                dist = (t - center) / sigma
                value = float(np.exp(-0.5 * dist * dist))
                if value > target[frame_idx, idx]:
                    target[frame_idx, idx] = value
    return times, target


def load_row_specs_from_source_oto(
    *,
    source_oto_path: str,
    wav_dir: str,
    language: str,
    format_type: str = "",
    alias_suffix: str = "",
    special_aliases: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[OtoRowSpec]]:
    source = os.path.abspath(str(source_oto_path or ""))
    if not source or not os.path.isfile(source):
        return {}
    special = {str(item).strip() for item in (special_aliases or []) if str(item).strip()}
    lang = normalize_language(language) or "korean"
    text = read_text_with_fallback(source)
    by_wav_rel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_idx, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "").strip()
        if not wav_name:
            continue
        by_wav_rel[wav_name].append(
            {
                "line_index": int(line_idx),
                "alias": str(parsed.get("alias", "") or ""),
                "offset": float(parsed.get("offset", 0.0) or 0.0),
                "cons": float(parsed.get("cons", 0.0) or 0.0),
                "cutoff": float(parsed.get("cutoff", 0.0) or 0.0),
                "pre": float(parsed.get("pre", 0.0) or 0.0),
                "ovl": float(parsed.get("ovl", 0.0) or 0.0),
            }
        )

    out: dict[str, list[OtoRowSpec]] = {}
    for wav_name, rows in by_wav_rel.items():
        wav_path = os.path.join(os.path.abspath(wav_dir), wav_name)
        if not os.path.isfile(wav_path):
            alt = os.path.join(os.path.dirname(source), wav_name)
            wav_path = alt if os.path.isfile(alt) else wav_path
        if not os.path.isfile(wav_path):
            continue
        duration = _wav_duration_ms(wav_path)
        slot_tokens = filename_order_tokens(wav_name)
        slot_count = max(1, len(slot_tokens))
        working: list[OtoRowSpec] = []
        anchor_cursor = 0
        for idx, row in enumerate(rows):
            alias = str(row["alias"] or "")
            components = _parse_alias_components(alias, lang)
            is_special = alias in special
            role = classify_alias_role(
                lang,
                alias,
                alias_type=(components.get("left_vowel") or ""),
                transition_type="",
                is_special=is_special,
            )
            role = normalize_role(role)
            if role in ANCHOR_ROLES:
                slot_index = min(anchor_cursor, slot_count - 1)
                anchor_cursor += 1
            elif role in {"vc", "vv", "v-cv"}:
                slot_index = min(max(0, anchor_cursor - 1), slot_count - 1)
            elif role in {"v-", "cv-", "br", "endbr"}:
                slot_index = max(0, min(slot_count - 1, max(0, anchor_cursor - 1)))
            else:
                slot_index = min(idx, slot_count - 1)
            working.append(
                OtoRowSpec(
                    wav_name=wav_name,
                    wav_path=os.path.abspath(wav_path),
                    alias=alias,
                    role=role,
                    slot_index=int(slot_index),
                    slot_count=int(slot_count),
                    prev_alias="",
                    next_alias="",
                    language=lang,
                    format_type=str(format_type or "other"),
                    line_index=int(row["line_index"]),
                    duration_ms=float(duration),
                    source_params={
                        "offset": float(row["offset"]),
                        "consonant": float(row["cons"]),
                        "cutoff": float(row["cutoff"]),
                        "preutterance": float(row["pre"]),
                        "overlap": float(row["ovl"]),
                    },
                    alias_suffix=str(alias_suffix or ""),
                    meta={
                        "left_vowel": components.get("left_vowel") or "",
                        "right_vowel": components.get("right_vowel") or "",
                        "right_consonant": components.get("right_consonant") or "",
                    },
                )
            )
        patched: list[OtoRowSpec] = []
        for idx, spec in enumerate(working):
            prev_alias = working[idx - 1].alias if idx > 0 else ""
            next_alias = working[idx + 1].alias if idx + 1 < len(working) else ""
            patched.append(replace(spec, prev_alias=prev_alias, next_alias=next_alias))
        out[os.path.abspath(wav_path)] = patched
    return out


def training_rows_to_wav_groups(rows: list[dict[str, Any]]) -> dict[str, list[tuple[OtoRowSpec, AbsoluteOtoAnchors]]]:
    grouped: dict[str, list[tuple[OtoRowSpec, AbsoluteOtoAnchors]]] = defaultdict(list)
    by_wav: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        wav_path = os.path.abspath(str(row.get("audio", "") or ""))
        if not wav_path:
            continue
        by_wav[wav_path].append(row)
    for wav_path, wav_rows in by_wav.items():
        wav_rows.sort(key=lambda item: int(item.get("line_index", 0) or 0))
        slot_count = max(1, len(filename_order_tokens(str(wav_rows[0].get("wav", "") or ""))))
        for idx, row in enumerate(wav_rows):
            role = normalize_role(row.get("alias_role", "other"))
            spec = OtoRowSpec(
                wav_name=str(row.get("wav", "") or os.path.basename(wav_path)),
                wav_path=wav_path,
                alias=str(row.get("alias", "") or ""),
                role=role,
                slot_index=min(idx, slot_count - 1),
                slot_count=slot_count,
                prev_alias=str(row.get("prev_alias", "") or ""),
                next_alias=str(row.get("next_alias", "") or ""),
                language=str(row.get("language", "") or "korean"),
                format_type=str(row.get("format_type", "") or "other"),
                line_index=int(row.get("line_index", idx) or idx),
                duration_ms=float(row.get("duration_ms", 0.0) or 0.0),
                source_params={},
            )
            duration = max(1.0, float(row.get("duration_ms", 0.0) or 0.0))
            anchors = oto_row_to_absolute_anchors(
                offset=float(row.get("target_offset_ms", row.get("offset", 0.0)) or 0.0),
                consonant=float(row.get("target_consonant_ms", row.get("cons", 0.0)) or 0.0),
                cutoff=float(row.get("target_cutoff", row.get("cutoff", 0.0)) or 0.0),
                preutterance=float(row.get("target_preutterance_ms", row.get("pre", 0.0)) or 0.0),
                overlap=float(row.get("target_overlap_ms", row.get("ovl", 0.0)) or 0.0),
                duration_ms=duration,
            )
            grouped[wav_path].append((spec, anchors))
    return grouped


def _wav_duration_ms(path: str) -> float:
    if not path or not os.path.isfile(path):
        return 0.0
    try:
        with wave.open(str(path), "rb") as wf:
            frames = int(wf.getnframes() or 0)
            sr = int(wf.getframerate() or 0)
        return float(frames) * 1000.0 / float(sr) if frames > 0 and sr > 0 else 0.0
    except Exception:
        return 0.0


def _parse_alias_components(alias: str, language: str) -> dict[str, str | None]:
    text = str(alias or "").strip()
    if not text:
        return {
            "left_vowel": None,
            "right_vowel": None,
            "right_consonant": None,
        }
    phones = _alias_phones(text, language=language)
    coarse = [coarse_for_phone(phone, language=language) for phone in phones]
    left_vowel = None
    right_vowel = None
    right_consonant = None
    for phone, cls in zip(phones, coarse):
        if cls == "V":
            left_vowel = str(phone).lower()
            break
    if phones:
        last_phone = str(phones[-1]).lower()
        last_cls = coarse[-1] if coarse else ""
        if last_cls == "V":
            right_vowel = last_phone
        elif str(last_cls).startswith("C_"):
            right_consonant = last_phone
    return {
        "left_vowel": left_vowel,
        "right_vowel": right_vowel,
        "right_consonant": right_consonant,
    }


def _alias_phones(alias: str, *, language: str) -> list[str]:
    text = str(alias or "").strip()
    if not text:
        return []
    if re.search(r"\s+", text):
        return [token for token in re.split(r"\s+", text) if token]
    tokens = phones_from_text(text, language)
    return list(tokens or [text])


__all__ = [
    "absolute_anchors_to_oto_params",
    "boundary_events_for_row",
    "build_boundary_target_map",
    "load_row_specs_from_source_oto",
    "oto_row_to_absolute_anchors",
    "training_rows_to_wav_groups",
]
