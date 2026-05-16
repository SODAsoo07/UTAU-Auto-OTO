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
        # Runtime convention in this project: negative cutoff is offset-relative.
        cutoff_abs = max(consonant_abs + 1.0, offset_abs + abs(float(cutoff)))
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
        # Runtime convention: store cutoff as negative offset-relative distance.
        "cutoff": -max(1.0, cutoff_abs - offset),
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
        filename_slot_count = max(1, len(slot_tokens))
        pre_roles: list[str] = []
        for row in rows:
            alias = str(row["alias"] or "")
            alias_type = _infer_alias_type(alias, language=lang)
            transition_type = _infer_transition_type(alias, language=lang)
            is_special = alias in special
            role = classify_alias_role(
                lang,
                alias,
                alias_type=alias_type,
                transition_type=transition_type,
                is_special=is_special,
            )
            pre_roles.append(normalize_role(role))
        slot_count = _resolve_slot_count(
            filename_slot_count=filename_slot_count,
            row_roles=pre_roles,
            row_count=len(rows),
        )
        anchor_role_count = sum(1 for role in pre_roles if normalize_role(role) in ANCHOR_ROLES)
        low_anchor_mode = bool(slot_count > 1 and anchor_role_count <= 1 and len(rows) > 1)
        # When the source OTO has fewer rows than the filename's syllable token
        # count (e.g. a wav like `_jjeuNG'jjeuN'jjeuM'jjeuL'jjeu'jjeo'jje'jja.wav`
        # whose OTO only lists the 4 plain-CV variants `jjeu`/`jjeo`/`jje`/`jja`),
        # the row-index-to-slot mapping collapses everything to the first 4
        # slots, leaving `jjeu` anchored ~2s before its real position. Try to
        # rescue these by matching the alias text to a filename token.
        sparse_rows_mode = bool(
            filename_slot_count > 1
            and len(rows) < filename_slot_count
            and _token_slot_match_enabled()
        )
        # When the filename encodes multiple syllables in token form, row order
        # in the source OTO generally tracks syllable order through the wav (a
        # 10-row, 8-mora wav typically lists NG g, N g, M g, L g, geu, eu g,
        # geo, eo g, ge, e g — all in syllable-progression order). The legacy
        # cursor-based slot assignment for vc/vv/v-cv roles pins every
        # transition row to slot 0, anchoring them at the start of the wav and
        # producing the systematic 200~1400 ms early bias reported in the
        # 2026-05-16 listening test. Use idx-based slot mapping for these wavs
        # so each row lands near its natural syllable position.
        filename_aware_idx_mode = bool(
            filename_slot_count > 1
            and not low_anchor_mode
            and _token_slot_match_enabled()
        )
        working: list[OtoRowSpec] = []
        anchor_cursor = 0
        for idx, row in enumerate(rows):
            alias = str(row["alias"] or "")
            components = _parse_alias_components(alias, lang)
            alias_type = _infer_alias_type(alias, language=lang)
            transition_type = _infer_transition_type(alias, language=lang)
            is_special = alias in special
            role = classify_alias_role(
                lang,
                alias,
                alias_type=alias_type,
                transition_type=transition_type,
                is_special=is_special,
            )
            role = normalize_role(role)
            matched_slot: int | None = None
            if sparse_rows_mode:
                matched_slot = _match_alias_to_filename_token(alias, slot_tokens)
            if matched_slot is not None:
                slot_index = matched_slot
            elif filename_aware_idx_mode:
                # Project row order to slot order by ratio, not raw idx.
                # This avoids early collapse when row_count != slot_count.
                slot_index = _project_row_index_to_slot(
                    row_index=int(idx),
                    row_count=len(rows),
                    slot_count=int(slot_count),
                )
            elif low_anchor_mode:
                # Filename-token extraction can collapse to slot_count=1 for JP kana-rich names.
                # In low-anchor rows, fall back to row-order slot projection to avoid full collapse.
                base_slot = _project_row_index_to_slot(
                    row_index=int(idx),
                    row_count=len(rows),
                    slot_count=int(slot_count),
                )
                parser_miss_mode = bool(int(filename_slot_count) <= 1)
                if role in {"vc", "vv", "v-cv"}:
                    # Left-shifting transition rows is only safe when filename
                    # tokenization itself failed (parser miss). On sparse KO
                    # CVVC rows with a valid multi-token filename, forcing -1
                    # causes systematic one-syllable early placement.
                    # Exception: KO coda-bridge aliases (NG/N/M/L + onset) are
                    # acoustically left-leaning transitions and tend to land
                    # late by one slot without a small left shift.
                    coda_bridge = _is_korean_coda_bridge_alias(alias, language=lang)
                    slot_index = (
                        max(0, min(slot_count - 1, base_slot - 1))
                        if (parser_miss_mode or coda_bridge)
                        else max(0, min(slot_count - 1, base_slot))
                    )
                elif role in {"v-", "cv-", "br", "endbr"}:
                    slot_index = (
                        max(0, min(slot_count - 1, base_slot - 1))
                        if parser_miss_mode
                        else max(0, min(slot_count - 1, base_slot))
                    )
                else:
                    slot_index = max(0, min(slot_count - 1, base_slot))
            elif role in ANCHOR_ROLES:
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
                    # Source OTO is treated as alias identity/order only.
                    source_params={},
                    alias_suffix=str(alias_suffix or ""),
                    meta={
                        "left_vowel": components.get("left_vowel") or "",
                        "right_vowel": components.get("right_vowel") or "",
                        "right_consonant": components.get("right_consonant") or "",
                        "alias_type": alias_type,
                        "transition_type": transition_type,
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
                source_params={
                    "offset": float(row.get("target_offset_ms", row.get("offset", 0.0)) or 0.0),
                    "consonant": float(row.get("target_consonant_ms", row.get("cons", 0.0)) or 0.0),
                    "cutoff": float(row.get("target_cutoff", row.get("cutoff", 0.0)) or 0.0),
                    "preutterance": float(row.get("target_preutterance_ms", row.get("pre", 0.0)) or 0.0),
                    "overlap": float(row.get("target_overlap_ms", row.get("ovl", 0.0)) or 0.0),
                },
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


def _infer_alias_type(alias: str, *, language: str) -> str:
    text = str(alias or "").strip()
    if not text:
        return "other"
    raw = text.lower()
    if raw in {"r", "br", "pau", "sil", "rest"}:
        return "br"
    if raw.startswith("-"):
        return "cv_head"
    if raw.endswith("-"):
        return "cv_tail"
    split_tokens = [token for token in re.split(r"\s+", text) if token]
    if len(split_tokens) == 2:
        left_token = split_tokens[0]
        right_token = split_tokens[1]
        left_vowel_like = _is_vowel_like_token(left_token, language=language)
        right_vowel_like = _is_vowel_like_token(right_token, language=language)
        left_coarse = coarse_for_phone(left_token, language=language)
        right_coarse = coarse_for_phone(right_token, language=language)
        left_is_c = _is_consonant_like_token(left_token, language=language) or str(left_coarse).startswith("C_")
        right_is_c = _is_consonant_like_token(right_token, language=language) or str(right_coarse).startswith("C_")

        if left_vowel_like:
            if right_vowel_like:
                return "vv"
            if _looks_japanese_kana_token(right_token):
                # V + kana token is typically VV/VCV transition (e.g., "a あ", "i か")
                return "vv" if _is_japanese_vowel_token(right_token) else "vcv"
            # KO CVVC core case: V + C transition alias ("a g", "i n", "eo dy"...)
            if right_is_c:
                return "vc"
            right_first = str(right_token[:1] or "").lower()
            if right_first in {"a", "i", "u", "e", "o"}:
                return "vv"
            return "vcv"

        # KO CVVC coda-bridge aliases often appear as Coda + Onset
        # (e.g., "NG g", "N ny", "L d"). These behave closer to V-CV bridge
        # rows than pure VC rows: forcing `vc` made many rows choose too-early
        # vowel-end candidates and caused one-step syllable shift regressions
        # on real 8-mora banks. Keep them in the V-CV lane.
        if left_is_c and right_is_c and _is_korean_coda_token(left_token):
            return "vcv"

    phones = _alias_phones(text, language=language)
    coarse = [coarse_for_phone(phone, language=language) for phone in phones]
    if not coarse:
        return "other"
    if len(coarse) == 1:
        return "vowel" if coarse[0] == "V" else "other"
    if coarse[0] == "V" and coarse[-1].startswith("C_"):
        return "vc"
    if coarse[0] == "V" and coarse[-1] == "V":
        return "vv"
    if coarse[0].startswith("C_") and coarse[-1] == "V":
        return "cv"
    return "other"


def _infer_transition_type(alias: str, *, language: str) -> str:
    alias_type = _infer_alias_type(alias, language=language)
    if alias_type == "br":
        return "br"
    if alias_type in {"vv"}:
        return "vv"
    if alias_type in {"vcv"}:
        return "cv"
    if alias_type in {"vc"}:
        return "vc"
    if alias_type in {"cv", "cv_head", "cv_tail"}:
        return "cv"
    if alias_type == "vowel":
        return "vowel"
    return "other"


def _is_vowel_like_token(token: str, *, language: str) -> bool:
    text = str(token or "").strip()
    if not text:
        return False
    low = text.lower()
    if low in {
        "a", "i", "u", "e", "o",
        "eo", "eu", "ae", "oe", "ui",
        "ya", "ye", "yo", "yu", "wa", "we", "wi", "wo",
        "yae", "yeo", "wae", "weo",
    }:
        return True
    if language == "japanese":
        return _is_japanese_vowel_token(text)
    return False


def _is_consonant_like_token(token: str, *, language: str) -> bool:
    text = str(token or "").strip().lower()
    if not text:
        return False
    if language != "korean":
        return False
    # Romanized KO CVVC onset/coda tokens are often consonant clusters with no
    # a/e/i/o/u nucleus (g, gy, gw, ny, bw, rw, ssy, NG/N/M/L, ...).
    if re.fullmatch(r"[a-z]+", text) and not any(ch in {"a", "e", "i", "o", "u"} for ch in text):
        return True
    return text in {"ng", "n", "m", "l", "r"}


def _is_korean_coda_token(token: str) -> bool:
    low = str(token or "").strip().lower()
    return low in {"ng", "n", "m", "l"}


def _is_korean_coda_bridge_alias(alias: str, *, language: str) -> bool:
    if str(language or "").strip().lower() not in {"korean", "ko", "kor"}:
        return False
    parts = [token for token in re.split(r"\s+", str(alias or "").strip()) if token]
    if len(parts) != 2:
        return False
    left_token, right_token = parts
    left_is_c = _is_consonant_like_token(left_token, language="korean") or str(
        coarse_for_phone(left_token, language="korean")
    ).startswith("C_")
    right_is_c = _is_consonant_like_token(right_token, language="korean") or str(
        coarse_for_phone(right_token, language="korean")
    ).startswith("C_")
    return bool(left_is_c and right_is_c and _is_korean_coda_token(left_token))


def _looks_japanese_kana_token(token: str) -> bool:
    text = str(token or "").strip()
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        if not (
            0x3040 <= code <= 0x309F  # Hiragana
            or 0x30A0 <= code <= 0x30FF  # Katakana
            or ch in {"ー", "・"}
        ):
            return False
    return True


def _is_japanese_vowel_token(token: str) -> bool:
    text = str(token or "").strip()
    if not text:
        return False
    vowels = {
        "あ", "い", "う", "え", "お",
        "ぁ", "ぃ", "ぅ", "ぇ", "ぉ",
        "ア", "イ", "ウ", "エ", "オ",
        "ァ", "ィ", "ゥ", "ェ", "ォ",
        "を", "ヲ",
    }
    return all(ch in vowels for ch in text)


def _token_slot_match_enabled() -> bool:
    raw = str(os.environ.get("UTOA_BOUNDARY_TOKEN_SLOT_MATCH_ENABLE", "") or "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


def _normalize_for_token_match(text: str) -> str:
    return str(text or "").strip().lower().replace(" ", "").replace("\t", "")


def _match_alias_to_filename_token(alias: str, tokens: list[str]) -> int | None:
    """Match a single-token alias to its position in the filename's token list.

    Returns the matched slot index, or ``None`` when:
    - the alias contains whitespace (compound aliases like "NG g" or "eu g" are
      transitions across slots and have no single owning token),
    - the normalized alias text doesn't equal any filename token.

    Used by `load_row_specs_from_source_oto` only when the source OTO has fewer
    rows than the filename's syllable tokens (sparse case where idx-based
    mapping collapses everything to the first N slots and pushes the last
    syllables 1-2 seconds out of position).
    """
    text = str(alias or "").strip()
    if not text or re.search(r"\s", text):
        return None
    norm = _normalize_for_token_match(text)
    if not norm:
        return None
    for slot_idx, token in enumerate(tokens or []):
        if _normalize_for_token_match(token) == norm:
            return int(slot_idx)
    return None


def _project_row_index_to_slot(*, row_index: int, row_count: int, slot_count: int) -> int:
    """Project a row index to slot index by relative position."""
    slots = max(1, int(slot_count))
    rows = max(1, int(row_count))
    if slots <= 1 or rows <= 1:
        return 0
    idx = max(0, min(int(rows - 1), int(row_index)))
    ratio = float(idx) / float(max(1, rows - 1))
    projected = int(round(ratio * float(max(0, slots - 1))))
    return max(0, min(slots - 1, projected))


def _resolve_slot_count(*, filename_slot_count: int, row_roles: list[str], row_count: int) -> int:
    base = max(1, int(filename_slot_count))
    if base > 1:
        return base
    anchors = sum(1 for role in row_roles if normalize_role(role) in ANCHOR_ROLES)
    if anchors > 1:
        return min(max(2, anchors), max(1, int(row_count)))
    count = max(1, int(row_count))
    if count <= 1:
        return 1
    vv_count = sum(1 for role in row_roles if normalize_role(role) == "vv")
    vcv_count = sum(1 for role in row_roles if normalize_role(role) == "v-cv")
    other_count = sum(1 for role in row_roles if normalize_role(role) == "other")
    v_count = sum(1 for role in row_roles if normalize_role(role) == "v")
    transition_count = sum(1 for role in row_roles if normalize_role(role) in {"vc", "vv", "v-cv"})
    if vv_count >= 1 and vcv_count == 0 and other_count == 0:
        # Pure vowel chains: rows usually map to sequence transitions directly.
        return min(count, max(2, v_count + vv_count))
    if vcv_count >= 1 and other_count >= 1:
        # Alternating CV-like + transition rows: infer slots from transition edges.
        return min(count, max(2, max(other_count, vcv_count + 1)))
    if transition_count >= max(2, count // 2):
        return min(count, max(2, transition_count + 1))
    if count >= 4:
        # Conservative fallback for parser-miss filenames.
        return min(count, max(2, int(round((count + 1) / 2.0))))
    return base


__all__ = [
    "absolute_anchors_to_oto_params",
    "boundary_events_for_row",
    "build_boundary_target_map",
    "load_row_specs_from_source_oto",
    "oto_row_to_absolute_anchors",
    "training_rows_to_wav_groups",
]
