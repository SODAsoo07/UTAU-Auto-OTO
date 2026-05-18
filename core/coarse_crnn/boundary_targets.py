from __future__ import annotations

import os
import re
import wave
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from core.coarse_crnn.alias_role import SPECIAL_ROLE, classify_alias_role, normalize_role
from core.coarse_crnn.boundary_types import (
    ANCHOR_ROLES,
    BOUNDARY_LABELS,
    PHONE_AWARE_CONSONANT_FAMILY_LABELS,
    PHONE_AWARE_CONSONANT_LABELS,
    PHONE_AWARE_CVS_LABELS,
    PHONE_AWARE_IGNORE_INDEX,
    PHONE_AWARE_VOWEL_GLIDE_LABELS,
    PHONE_AWARE_VOWEL_LABELS,
    PHONE_AWARE_VOWEL_NUCLEUS_LABELS,
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


@dataclass(frozen=True)
class PhoneAwareTargetMap:
    cvs_target: np.ndarray
    cvs_mask: np.ndarray
    consonant_target: np.ndarray
    consonant_mask: np.ndarray
    vowel_target: np.ndarray
    vowel_mask: np.ndarray
    consonant_family_target: np.ndarray
    consonant_family_mask: np.ndarray
    vowel_nucleus_target: np.ndarray
    vowel_nucleus_mask: np.ndarray
    vowel_glide_target: np.ndarray
    vowel_glide_mask: np.ndarray


def boundary_events_for_row(spec: OtoRowSpec, anchors: AbsoluteOtoAnchors) -> list[tuple[str, float, float]]:
    """Return (label_name, center_ms, sigma_ms) for the boundary scorer
    targets of one row.

    2026-05-18 audio-driven window edges: `syllable_onset` and
    `silence_boundary` are now painted for ALL non-silence roles, anchored at
    the audio-driven `syllable_audio_start_abs` / `silence_audio_end_abs` when
    set, otherwise falling back to the source-OTO `offset_abs` / `cutoff_abs`.
    This fixes the systematic "window too tight" failure where the previous
    target set never supervised cutoff for cv/vc/vv/v-cv rows, leaving the
    `silence_boundary` head un-trained and the decoder forced to use
    `transition_peak`/`next_onset` (clustered around the consonant burst) as
    a proxy. The audio-driven anchors are set by `_inject_audio_anchors_into_rows`
    inside the dataset loader; for rows where the wav isn't available
    (eval-only paths) the source-OTO anchors are used as before.
    """
    role = normalize_role(spec.role)
    syllable_onset_ms = anchors.syllable_audio_start_abs if anchors.syllable_audio_start_abs is not None else anchors.offset_abs
    silence_boundary_ms = anchors.silence_audio_end_abs if anchors.silence_audio_end_abs is not None else anchors.cutoff_abs
    syllable_onset_sigma = label_sigma_ms("syllable_onset")
    silence_boundary_sigma = label_sigma_ms("silence_boundary")
    if role == "-cv":
        return [
            ("syllable_onset", syllable_onset_ms, syllable_onset_sigma),
            ("vowel_start", anchors.pre_abs, label_sigma_ms("vowel_start")),
            ("silence_boundary", silence_boundary_ms, silence_boundary_sigma),
        ]
    if role in {"cv", SPECIAL_ROLE}:
        return [
            ("syllable_onset", syllable_onset_ms, syllable_onset_sigma),
            ("consonant_onset", anchors.offset_abs, label_sigma_ms("consonant_onset")),
            ("vowel_start", anchors.pre_abs, label_sigma_ms("vowel_start")),
            ("vowel_stable", 0.5 * (anchors.pre_abs + anchors.consonant_abs), label_sigma_ms("vowel_stable")),
            ("silence_boundary", silence_boundary_ms, silence_boundary_sigma),
        ]
    if role == "v":
        return [
            ("syllable_onset", syllable_onset_ms, syllable_onset_sigma),
            ("vowel_start", anchors.pre_abs, label_sigma_ms("vowel_start")),
            ("vowel_stable", 0.5 * (anchors.pre_abs + anchors.consonant_abs), label_sigma_ms("vowel_stable")),
            ("silence_boundary", silence_boundary_ms, silence_boundary_sigma),
        ]
    if role == "vc":
        return [
            ("syllable_onset", syllable_onset_ms, syllable_onset_sigma),
            ("vowel_end", anchors.pre_abs, label_sigma_ms("vowel_end")),
            ("transition_peak", anchors.pre_abs, label_sigma_ms("transition_peak")),
            ("next_onset", anchors.consonant_abs, label_sigma_ms("next_onset")),
            ("silence_boundary", silence_boundary_ms, silence_boundary_sigma),
        ]
    if role == "vv":
        return [
            ("syllable_onset", syllable_onset_ms, syllable_onset_sigma),
            ("vowel_end", anchors.pre_abs, label_sigma_ms("vowel_end")),
            ("transition_peak", anchors.pre_abs, label_sigma_ms("transition_peak")),
            ("vowel_start", anchors.consonant_abs, label_sigma_ms("vowel_start")),
            ("silence_boundary", silence_boundary_ms, silence_boundary_sigma),
        ]
    if role == "v-cv":
        return [
            ("syllable_onset", syllable_onset_ms, syllable_onset_sigma),
            ("transition_peak", anchors.pre_abs, label_sigma_ms("transition_peak")),
            ("next_onset", anchors.consonant_abs, label_sigma_ms("next_onset")),
            ("consonant_onset", anchors.consonant_abs, label_sigma_ms("consonant_onset")),
            ("silence_boundary", silence_boundary_ms, silence_boundary_sigma),
        ]
    if role in {"v-", "cv-"}:
        return [
            ("vowel_end", anchors.pre_abs, label_sigma_ms("vowel_end")),
            ("silence_boundary", silence_boundary_ms, silence_boundary_sigma),
        ]
    if role in {"br", "endbr"}:
        return [("silence_boundary", anchors.pre_abs, silence_boundary_sigma)]
    return [
        ("syllable_onset", syllable_onset_ms, syllable_onset_sigma),
        ("vowel_start", anchors.pre_abs, label_sigma_ms("vowel_start")),
        ("transition_peak", anchors.consonant_abs, label_sigma_ms("transition_peak")),
        ("silence_boundary", silence_boundary_ms, silence_boundary_sigma),
    ]


def compute_per_row_audio_extents(
    *,
    features: np.ndarray,
    hop_ms: float,
    rows: list[tuple[OtoRowSpec, AbsoluteOtoAnchors]],
    energy_floor_percentile: float = 0.15,
    voiced_percentile: float = 0.70,
    rel_db_threshold: float = 1.5,
    search_left_ms: float = 250.0,
    search_right_ms: float = 350.0,
    gap_between_rows_ms: float = 25.0,
) -> list[tuple[float, float]]:
    """Per-row (audio_start_ms, audio_end_ms) extracted from the wav's log-mel
    energy profile. Used to relabel `syllable_onset` / `silence_boundary` at
    the natural acoustic edges of each syllable instead of the source-OTO's
    (often tight) offset/cutoff.

    Algorithm (per row):
      1. Compute per-frame energy = mean of log-mel bins.
      2. Per-wav baseline:
         - `floor_db`   = `energy_floor_percentile` percentile (noise floor).
         - `voiced_db`  = `voiced_percentile` percentile (typical voice level).
      3. Threshold = `voiced_db - rel_db_threshold` (log-magnitude, ~15 dB
         below voiced peak by default). This avoids classifying low-energy
         vowel decay tails as silence.
      4. Per row, search backward from `anchors.offset_abs` up to
         `search_left_ms` for the first frame whose energy crosses below
         threshold — the frame just after that is `audio_start`. Cap by the
         previous row's `audio_end + gap_between_rows_ms` so adjacent rows
         don't overlap.
      5. Symmetric forward search from `anchors.cutoff_abs` for `audio_end`,
         capped by the next row's offset_abs.
      6. If the search fails (energy never crosses threshold), fall back to
         `offset_abs` / `cutoff_abs`.

    Returns one (start_ms, end_ms) tuple per input row, in the input order.
    """
    n_frames = int(features.shape[0]) if features is not None else 0
    if n_frames <= 0 or not rows:
        return [(float(spec_anchor[1].offset_abs), float(spec_anchor[1].cutoff_abs)) for spec_anchor in rows]
    energy = np.asarray(features, dtype=np.float32).mean(axis=1)
    if energy.size == 0:
        return [(float(spec_anchor[1].offset_abs), float(spec_anchor[1].cutoff_abs)) for spec_anchor in rows]
    floor_db = float(np.quantile(energy, max(0.01, min(0.40, float(energy_floor_percentile)))))
    voiced_db = float(np.quantile(energy, max(0.40, min(0.95, float(voiced_percentile)))))
    if not np.isfinite(voiced_db) or voiced_db - floor_db < 0.2:
        # Wav looks nearly silent or monotonous — fall back to source OTO.
        return [(float(spec_anchor[1].offset_abs), float(spec_anchor[1].cutoff_abs)) for spec_anchor in rows]
    threshold = float(voiced_db - max(0.1, float(rel_db_threshold)))
    hop = max(2.0, float(hop_ms))

    def _ms_to_frame(ms: float) -> int:
        return int(round(float(ms) / hop))

    def _frame_to_ms(frame: int) -> float:
        return float(frame) * hop

    # Sort row indices by anchor offset to apply neighbor caps.
    sorted_idx = sorted(range(len(rows)), key=lambda i: float(rows[i][1].offset_abs))
    audio_starts: dict[int, float] = {}
    audio_ends: dict[int, float] = {}
    for order_pos, row_idx in enumerate(sorted_idx):
        _spec, anchors = rows[row_idx]
        offset_frame = max(0, min(n_frames - 1, _ms_to_frame(anchors.offset_abs)))
        cutoff_frame = max(0, min(n_frames - 1, _ms_to_frame(anchors.cutoff_abs)))

        # Backward search bound: previous row's audio_end + gap (if any).
        prev_bound_frame = 0
        if order_pos > 0:
            prev_idx = sorted_idx[order_pos - 1]
            prev_end_ms = audio_ends.get(prev_idx, float(rows[prev_idx][1].cutoff_abs))
            prev_bound_frame = max(prev_bound_frame, _ms_to_frame(prev_end_ms + float(gap_between_rows_ms)))
        prev_bound_frame = max(prev_bound_frame, _ms_to_frame(float(anchors.offset_abs) - float(search_left_ms)))
        prev_bound_frame = max(0, min(offset_frame, prev_bound_frame))

        # Forward search bound: next row's offset_abs - gap (if any).
        next_bound_frame = n_frames - 1
        if order_pos + 1 < len(sorted_idx):
            next_idx = sorted_idx[order_pos + 1]
            next_offset_ms = float(rows[next_idx][1].offset_abs)
            next_bound_frame = min(next_bound_frame, _ms_to_frame(next_offset_ms - float(gap_between_rows_ms)))
        next_bound_frame = min(next_bound_frame, _ms_to_frame(float(anchors.cutoff_abs) + float(search_right_ms)))
        next_bound_frame = max(cutoff_frame, min(n_frames - 1, next_bound_frame))

        # Search backward from offset for first frame where energy drops below
        # threshold (silence). audio_start = the frame just after that.
        start_frame = offset_frame
        for f in range(offset_frame, prev_bound_frame - 1, -1):
            if energy[f] < threshold:
                start_frame = min(offset_frame, f + 1)
                break
        else:
            # Never crossed threshold; audio is voiced all the way to prev bound.
            start_frame = prev_bound_frame

        # Search forward from cutoff for first frame where energy drops below
        # threshold. audio_end = the frame just before that.
        end_frame = cutoff_frame
        for f in range(cutoff_frame, next_bound_frame + 1):
            if energy[f] < threshold:
                end_frame = max(cutoff_frame, f - 1)
                break
        else:
            end_frame = next_bound_frame

        # Safety: never narrow the window relative to source OTO. Only widen.
        start_frame = min(start_frame, offset_frame)
        end_frame = max(end_frame, cutoff_frame)

        audio_starts[row_idx] = _frame_to_ms(start_frame)
        audio_ends[row_idx] = _frame_to_ms(end_frame)

    return [(audio_starts[i], audio_ends[i]) for i in range(len(rows))]


def inject_audio_anchors_into_rows(
    rows: list[tuple[OtoRowSpec, AbsoluteOtoAnchors]],
    extents: list[tuple[float, float]],
) -> list[tuple[OtoRowSpec, AbsoluteOtoAnchors]]:
    """Return a new rows list where each AbsoluteOtoAnchors carries the
    audio-driven `syllable_audio_start_abs` / `silence_audio_end_abs` fields
    populated from `extents`. The original anchors are kept unchanged; this
    only adds the two optional fields used by `boundary_events_for_row`."""
    from dataclasses import replace as _dc_replace

    if not rows or not extents or len(rows) != len(extents):
        return rows
    updated: list[tuple[OtoRowSpec, AbsoluteOtoAnchors]] = []
    for (spec, anchors), (start_ms, end_ms) in zip(rows, extents):
        new_anchors = _dc_replace(
            anchors,
            syllable_audio_start_abs=float(start_ms),
            silence_audio_end_abs=float(end_ms),
        )
        updated.append((spec, new_anchors))
    return updated


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


def build_phone_aware_target_map(
    rows: list[tuple[OtoRowSpec, AbsoluteOtoAnchors]],
    *,
    duration_ms: float,
    hop_ms: float,
    frame_count: int | None = None,
) -> tuple[list[float], PhoneAwareTargetMap]:
    duration = max(1.0, float(duration_ms))
    hop = max(2.0, float(hop_ms))
    if frame_count is None:
        frame_count = max(1, int(round(duration / hop)))
    frames = int(frame_count)
    times = (np.arange(frames, dtype=np.float32) * hop).tolist()
    cvs_target = np.full((frames,), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
    consonant_target = np.full((frames,), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
    vowel_target = np.full((frames,), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
    consonant_family_target = np.full((frames,), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
    vowel_nucleus_target = np.full((frames,), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
    vowel_glide_target = np.full((frames,), PHONE_AWARE_IGNORE_INDEX, dtype=np.int64)
    cvs_mask = np.zeros((frames,), dtype=np.float32)
    consonant_mask = np.zeros((frames,), dtype=np.float32)
    vowel_mask = np.zeros((frames,), dtype=np.float32)
    consonant_family_mask = np.zeros((frames,), dtype=np.float32)
    vowel_nucleus_mask = np.zeros((frames,), dtype=np.float32)
    vowel_glide_mask = np.zeros((frames,), dtype=np.float32)
    for spec, anchors in rows:
        _paint_phone_targets(
            spec=spec,
            anchors=anchors,
            hop_ms=hop,
            frame_count=frames,
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
        )
    return times, PhoneAwareTargetMap(
        cvs_target=cvs_target,
        cvs_mask=cvs_mask,
        consonant_target=consonant_target,
        consonant_mask=consonant_mask,
        vowel_target=vowel_target,
        vowel_mask=vowel_mask,
        consonant_family_target=consonant_family_target,
        consonant_family_mask=consonant_family_mask,
        vowel_nucleus_target=vowel_nucleus_target,
        vowel_nucleus_mask=vowel_nucleus_mask,
        vowel_glide_target=vowel_glide_target,
        vowel_glide_mask=vowel_glide_mask,
    )


def phone_aware_role_family(spec: OtoRowSpec) -> str:
    role = normalize_role(spec.role)
    alias = str(spec.alias or "")
    lang = normalize_language(spec.language) or "korean"
    if role == "vc":
        if _is_korean_coda_bridge_alias(alias, language=lang):
            return "coda_bridge"
        left, right = _two_alias_tokens(alias)
        if _is_vowel_like_token(left, language=lang) and _is_consonant_like_token(right, language=lang):
            return "vc_coda"
        return "vc_onset"
    if role == "v-cv":
        return "v-cv"
    if role == "vv":
        return "vv"
    if role in {"-cv", "cv", "v", SPECIAL_ROLE}:
        return role
    if role in {"v-", "cv-", "br", "endbr"}:
        return "silence"
    return "other"


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
            # Two-token transition aliases (v-cv, vc) like "u k" in 8-mora
            # tense+coda wavs systematically land on slot 0 because the
            # single-token matcher above skips whitespace aliases and the
            # legacy cursor logic for transition roles doesn't advance when
            # CV anchors are sparse. Opt-in compound matching recovers them
            # by checking adjacent filename-token pairs.
            if (
                matched_slot is None
                and _compound_token_slot_match_enabled()
                and filename_slot_count > 1
                and role in {"vc", "v-cv"}
            ):
                matched_slot = _match_compound_alias_to_token_transition(alias, slot_tokens)
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
            lang = normalize_language(str(row.get("language", "") or "korean")) or "korean"
            alias = str(row.get("alias", "") or "")
            components = _parse_alias_components(alias, lang)
            spec = OtoRowSpec(
                wav_name=str(row.get("wav", "") or os.path.basename(wav_path)),
                wav_path=wav_path,
                alias=alias,
                role=role,
                slot_index=min(idx, slot_count - 1),
                slot_count=slot_count,
                prev_alias=str(row.get("prev_alias", "") or ""),
                next_alias=str(row.get("next_alias", "") or ""),
                language=lang,
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
                meta={
                    "left_vowel": components.get("left_vowel") or "",
                    "right_vowel": components.get("right_vowel") or "",
                    "right_consonant": components.get("right_consonant") or "",
                    "alias_type": _infer_alias_type(alias, language=lang),
                    "transition_type": _infer_transition_type(alias, language=lang),
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
    lang = normalize_language(language) or ""
    if lang == "korean":
        ko = _parse_korean_alias_components(text)
        if any(ko.values()):
            return ko
    if lang == "japanese":
        ja = _parse_japanese_alias_components(text)
        if any(ja.values()):
            return ja
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


def _parse_korean_alias_components(alias: str) -> dict[str, str | None]:
    parts = [token for token in re.split(r"\s+", str(alias or "").strip()) if token]
    left_vowel = None
    right_vowel = None
    right_consonant = None
    if len(parts) >= 2:
        left = _strip_color_suffix(parts[0])
        right = _strip_color_suffix(parts[-1])
        left_vowel = _ko_token_vowel(left)
        if _is_vowel_like_token(right, language="korean"):
            right_vowel = _normalize_phone_identity(right)
        elif _is_consonant_like_token(right, language="korean") or str(coarse_for_phone(right, language="korean")).startswith("C_"):
            right_consonant = _ko_token_onset(right) or _normalize_phone_identity(right)
            right_vowel = _ko_token_vowel(right)
        else:
            onset, vowel, _coda = _split_ko_roman_syllable(right)
            right_consonant = _normalize_phone_identity(onset)
            right_vowel = vowel
        return {
            "left_vowel": left_vowel,
            "right_vowel": right_vowel,
            "right_consonant": right_consonant,
        }

    if len(parts) == 1:
        token = _strip_color_suffix(parts[0])
        if _is_vowel_like_token(token, language="korean"):
            left_vowel = _normalize_phone_identity(token)
            right_vowel = left_vowel
        else:
            onset, vowel, _coda = _split_ko_roman_syllable(token)
            right_consonant = _normalize_phone_identity(onset)
            right_vowel = vowel
            left_vowel = vowel
    return {
        "left_vowel": left_vowel,
        "right_vowel": right_vowel,
        "right_consonant": right_consonant,
    }


def _parse_japanese_alias_components(alias: str) -> dict[str, str | None]:
    parts = [_strip_phone_decorations(token) for token in re.split(r"\s+", str(alias or "").strip()) if token]
    left_vowel = None
    right_vowel = None
    right_consonant = None
    if len(parts) >= 2:
        left = parts[0]
        right = parts[-1]
        _lo, left_vowel = _split_japanese_token(left)
        right_consonant, right_vowel = _split_japanese_token(right)
        if not right_consonant and _is_consonant_like_token(right, language="japanese"):
            right_consonant = _normalize_phone_identity(right)
        return {
            "left_vowel": left_vowel if left_vowel in PHONE_AWARE_VOWEL_LABELS else None,
            "right_vowel": right_vowel if right_vowel in PHONE_AWARE_VOWEL_LABELS else None,
            "right_consonant": right_consonant or None,
        }

    if len(parts) == 1:
        right_consonant, right_vowel = _split_japanese_token(parts[0])
        left_vowel = right_vowel
    return {
        "left_vowel": left_vowel if left_vowel in PHONE_AWARE_VOWEL_LABELS else None,
        "right_vowel": right_vowel if right_vowel in PHONE_AWARE_VOWEL_LABELS else None,
        "right_consonant": right_consonant or None,
    }


def _split_japanese_token(token: str) -> tuple[str, str]:
    raw = _strip_phone_decorations(token)
    if not raw or raw == "-":
        return "", ""
    syllable = ""
    if _looks_japanese_kana_token(raw):
        try:
            from core.ja_lab_generator import parse_ja_filename
            parsed = list(parse_ja_filename(raw) or [])
        except Exception:
            parsed = []
        syllable = str(parsed[0] or "") if parsed else ""
    else:
        syllables = [item for item in re.split(r"[_'\-/|]+", raw) if item]
        syllable = syllables[0] if syllables else raw
    syllable = _strip_phone_decorations(syllable).lower()
    if not syllable:
        return "", ""
    try:
        from core.ja_lab_generator import split_ja_romaji_syllable
        onset, vowel = split_ja_romaji_syllable(syllable)
    except Exception:
        onset, vowel = _split_simple_cv_token(syllable)
    onset = _normalize_phone_identity(onset)
    vowel = _normalize_phone_identity(vowel)
    if vowel == "n":
        vowel = ""
        if not onset:
            onset = "n"
    if vowel == "cl":
        vowel = ""
        if not onset:
            onset = "cl"
    return onset, vowel


def _paint_phone_targets(
    *,
    spec: OtoRowSpec,
    anchors: AbsoluteOtoAnchors,
    hop_ms: float,
    frame_count: int,
    cvs_target: np.ndarray,
    cvs_mask: np.ndarray,
    consonant_target: np.ndarray,
    consonant_mask: np.ndarray,
    vowel_target: np.ndarray,
    vowel_mask: np.ndarray,
    consonant_family_target: np.ndarray,
    consonant_family_mask: np.ndarray,
    vowel_nucleus_target: np.ndarray,
    vowel_nucleus_mask: np.ndarray,
    vowel_glide_target: np.ndarray,
    vowel_glide_mask: np.ndarray,
) -> None:
    role = normalize_role(spec.role)
    family = phone_aware_role_family(spec)
    lang = normalize_language(spec.language) or "korean"
    identity_enabled = _phone_identity_enabled(lang)
    meta = dict(spec.meta or {})
    left, right = _two_alias_tokens(str(spec.alias or ""))
    left_vowel = _normalize_phone_identity(str(meta.get("left_vowel") or ""))
    right_vowel = _normalize_phone_identity(str(meta.get("right_vowel") or ""))
    right_consonant = _normalize_phone_identity(str(meta.get("right_consonant") or ""))
    if not left_vowel and _is_vowel_like_token(left, language=lang):
        left_vowel = _normalize_phone_identity(left)
    if not right_vowel and _is_vowel_like_token(right, language=lang):
        right_vowel = _normalize_phone_identity(right)
    if not right_consonant and _is_consonant_like_token(right, language=lang):
        right_consonant = _normalize_phone_identity(right)
    if lang == "korean":
        if not left and not right:
            one_onset, one_vowel, _one_coda = _split_ko_roman_syllable(str(spec.alias or ""))
            if not right_consonant:
                right_consonant = _normalize_phone_identity(one_onset)
            if not right_vowel:
                right_vowel = one_vowel
            if not left_vowel:
                left_vowel = one_vowel
        if not left_vowel:
            left_vowel = _ko_token_vowel(left)
        if not right_vowel:
            right_vowel = _ko_token_vowel(right)
        if not right_consonant:
            right_consonant = _ko_token_onset(right)
    elif lang == "japanese":
        if not left and not right:
            one_onset, one_vowel = _split_japanese_token(str(spec.alias or ""))
            if not right_consonant:
                right_consonant = one_onset
            if not right_vowel:
                right_vowel = one_vowel
            if not left_vowel:
                left_vowel = one_vowel
        if not left_vowel:
            _lo, left_vowel = _split_japanese_token(left)
        if not right_vowel or not right_consonant:
            onset, vowel = _split_japanese_token(right)
            if not right_consonant:
                right_consonant = onset
            if not right_vowel:
                right_vowel = vowel
    left_consonant = _normalize_phone_identity(left) if _is_consonant_like_token(left, language=lang) else ""

    if family == "silence":
        _paint_cvs(cvs_target, cvs_mask, label="silence", center_ms=anchors.cutoff_abs, sigma_ms=30.0, hop_ms=hop_ms, frame_count=frame_count)
        return

    if role in {"-cv", "cv", SPECIAL_ROLE}:
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="consonant",
            token=right_consonant,
            center_ms=anchors.offset_abs,
            sigma_ms=18.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="vowel",
            token=right_vowel or left_vowel,
            center_ms=anchors.pre_abs,
            sigma_ms=24.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        return

    if role == "v":
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="vowel",
            token=right_vowel or left_vowel,
            center_ms=anchors.pre_abs,
            sigma_ms=28.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        return

    if family == "coda_bridge":
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="consonant",
            token=left_consonant,
            center_ms=anchors.pre_abs,
            sigma_ms=22.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="consonant",
            token=right_consonant or _clean_phone_token(right),
            center_ms=anchors.consonant_abs,
            sigma_ms=22.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        return

    if family in {"vc_coda", "vc_onset", "v-cv"}:
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="vowel",
            token=left_vowel,
            center_ms=max(0.0, anchors.pre_abs - 12.0),
            sigma_ms=18.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="consonant",
            token=right_consonant,
            center_ms=anchors.consonant_abs,
            sigma_ms=22.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        return

    if family == "vv":
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="vowel",
            token=left_vowel,
            center_ms=anchors.pre_abs,
            sigma_ms=20.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )
        _paint_phone_class(
            cvs_target=cvs_target,
            cvs_mask=cvs_mask,
            consonant_target=consonant_target,
            consonant_mask=consonant_mask,
            vowel_target=vowel_target,
            vowel_mask=vowel_mask,
            consonant_family_target=consonant_family_target,
            consonant_family_mask=consonant_family_mask,
            vowel_nucleus_target=vowel_nucleus_target,
            vowel_nucleus_mask=vowel_nucleus_mask,
            vowel_glide_target=vowel_glide_target,
            vowel_glide_mask=vowel_glide_mask,
            label="vowel",
            token=right_vowel,
            center_ms=anchors.consonant_abs,
            sigma_ms=20.0,
            identity_enabled=identity_enabled,
            hop_ms=hop_ms,
            frame_count=frame_count,
        )


def _paint_phone_class(
    *,
    cvs_target: np.ndarray,
    cvs_mask: np.ndarray,
    consonant_target: np.ndarray,
    consonant_mask: np.ndarray,
    vowel_target: np.ndarray,
    vowel_mask: np.ndarray,
    consonant_family_target: np.ndarray,
    consonant_family_mask: np.ndarray,
    vowel_nucleus_target: np.ndarray,
    vowel_nucleus_mask: np.ndarray,
    vowel_glide_target: np.ndarray,
    vowel_glide_mask: np.ndarray,
    label: str,
    token: str,
    center_ms: float,
    sigma_ms: float,
    hop_ms: float,
    frame_count: int,
    identity_enabled: bool = True,
) -> None:
    _paint_cvs(
        cvs_target,
        cvs_mask,
        label=label,
        center_ms=center_ms,
        sigma_ms=_cvs_sigma_ms(label, sigma_ms),
        hop_ms=hop_ms,
        frame_count=frame_count,
    )
    if not bool(identity_enabled):
        return
    if label == "consonant":
        idx = _phone_label_index(token, PHONE_AWARE_CONSONANT_LABELS)
        if idx is not None:
            _paint_index_target(consonant_target, consonant_mask, idx=idx, center_ms=center_ms, sigma_ms=sigma_ms, hop_ms=hop_ms, frame_count=frame_count)
        family_idx = _phone_label_index(consonant_family_identity(token), PHONE_AWARE_CONSONANT_FAMILY_LABELS)
        if family_idx is not None:
            _paint_index_target(
                consonant_family_target,
                consonant_family_mask,
                idx=family_idx,
                center_ms=center_ms,
                sigma_ms=sigma_ms,
                hop_ms=hop_ms,
                frame_count=frame_count,
            )
    elif label == "vowel":
        idx = _phone_label_index(token, PHONE_AWARE_VOWEL_LABELS)
        if idx is not None:
            _paint_index_target(vowel_target, vowel_mask, idx=idx, center_ms=center_ms, sigma_ms=sigma_ms, hop_ms=hop_ms, frame_count=frame_count)
        nucleus_idx = _phone_label_index(vowel_nucleus_identity(token), PHONE_AWARE_VOWEL_NUCLEUS_LABELS)
        if nucleus_idx is not None:
            _paint_index_target(
                vowel_nucleus_target,
                vowel_nucleus_mask,
                idx=nucleus_idx,
                center_ms=center_ms,
                sigma_ms=sigma_ms,
                hop_ms=hop_ms,
                frame_count=frame_count,
            )
        glide_idx = _phone_label_index(vowel_glide_identity(token), PHONE_AWARE_VOWEL_GLIDE_LABELS)
        if glide_idx is not None:
            _paint_index_target(
                vowel_glide_target,
                vowel_glide_mask,
                idx=glide_idx,
                center_ms=center_ms,
                sigma_ms=sigma_ms,
                hop_ms=hop_ms,
                frame_count=frame_count,
            )


def _phone_identity_enabled(language: str) -> bool:
    raw = str(os.environ.get("UTOA_BOUNDARY_PHONE_AUX_IDENTITY_LANGS", "") or "").strip().lower()
    if not raw:
        allowed = {"japanese", "korean"}
    elif raw in {"*", "all"}:
        return True
    else:
        allowed = {item.strip() for item in raw.split(",") if item.strip()}
    return normalize_language(language) in allowed


def _paint_cvs(target: np.ndarray, mask: np.ndarray, *, label: str, center_ms: float, sigma_ms: float, hop_ms: float, frame_count: int) -> None:
    try:
        idx = PHONE_AWARE_CVS_LABELS.index(str(label))
    except ValueError:
        return
    _paint_index_target(target, mask, idx=idx, center_ms=center_ms, sigma_ms=sigma_ms, hop_ms=hop_ms, frame_count=frame_count)


def _cvs_sigma_ms(label: str, sigma_ms: float) -> float:
    if str(label) == "vowel":
        return max(float(sigma_ms), 46.0)
    if str(label) == "consonant":
        return max(float(sigma_ms), 38.0)
    if str(label) == "silence":
        return max(float(sigma_ms), 54.0)
    return float(sigma_ms)


def _paint_index_target(target: np.ndarray, mask: np.ndarray, *, idx: int, center_ms: float, sigma_ms: float, hop_ms: float, frame_count: int) -> None:
    sigma = max(6.0, float(sigma_ms))
    radius = max(float(hop_ms), sigma * 1.75)
    center = float(center_ms)
    for frame_idx in range(int(frame_count)):
        t = frame_idx * float(hop_ms)
        if abs(t - center) <= radius:
            target[frame_idx] = int(idx)
            mask[frame_idx] = 1.0


def consonant_family_identity(token: str) -> str:
    clean = _normalize_phone_identity(token)
    if not clean:
        return ""
    if clean in {"cl"} or clean.endswith("cl"):
        return "closure"
    if clean in {"m", "my", "n", "ng", "ny"}:
        return "nasal"
    if clean in {"l", "r", "ry"}:
        return "liquid"
    if clean in {"s", "sh", "ss", "f", "v", "z"}:
        return "fricative"
    if clean in {"c", "ch", "dz", "j", "jj", "ts"}:
        return "affricate"
    if clean in {"w", "y"}:
        return "glide"
    if clean in {"h", "hy", "kh", "ph", "th"}:
        return "aspirate"
    if clean in {
        "b",
        "bb",
        "by",
        "d",
        "dd",
        "dy",
        "g",
        "gg",
        "gy",
        "gw",
        "k",
        "kk",
        "ky",
        "p",
        "pp",
        "py",
        "t",
        "tt",
        "ty",
    }:
        return "plosive"
    return "other"


def vowel_nucleus_identity(token: str) -> str:
    clean = _normalize_phone_identity(token)
    if not clean:
        return ""
    if clean in {"a", "wa", "ya"}:
        return "a"
    if clean in {"ae", "e", "oe", "we", "wae", "yae", "ye"}:
        return "e"
    if clean in {"i", "ui", "wi"}:
        return "i"
    if clean in {"o", "wo", "yo"}:
        return "o"
    if clean in {"u", "yu"}:
        return "u"
    if clean in {"eo", "weo", "yeo"}:
        return "eo"
    if clean == "eu":
        return "eu"
    return "other"


def vowel_glide_identity(token: str) -> str:
    clean = _normalize_phone_identity(token)
    if not clean:
        return ""
    if clean == "ui":
        return "ui"
    if clean.startswith("y"):
        return "y"
    if clean.startswith("w"):
        return "w"
    if clean in {"a", "ae", "e", "eo", "eu", "i", "o", "oe", "u"}:
        return "none"
    return "other"


def _phone_label_index(token: str, labels: tuple[str, ...]) -> int | None:
    clean = _normalize_phone_identity(token)
    if not clean:
        return None
    try:
        return tuple(labels).index(clean)
    except ValueError:
        return None


_PHONE_IDENTITY_ALIASES: dict[str, str] = {
    "eui": "ui",
    "kcl": "k",
    "pcl": "p",
    "tcl": "t",
    "ua": "wa",
    "ue": "we",
    "uo": "wo",
}


def _normalize_phone_identity(token: str) -> str:
    clean = _clean_phone_token(token)
    clean = _PHONE_IDENTITY_ALIASES.get(clean, clean)
    return _canonical_consonant_identity(clean)


def _canonical_consonant_identity(token: str) -> str:
    clean = str(token or "").strip().lower()
    if not clean or any(ch in {"a", "e", "i", "o", "u"} for ch in clean):
        return clean
    labels = set(PHONE_AWARE_CONSONANT_LABELS)
    if clean in labels:
        return clean
    if clean.endswith("cl") and clean[:-2] in labels:
        return clean[:-2]
    for coda in ("ng", "n", "m", "l", "r", "k", "p", "t"):
        if clean.startswith(coda) and len(clean) > len(coda):
            tail = _canonical_consonant_identity(clean[len(coda):])
            if tail in labels:
                return tail
    for onset in sorted(labels, key=len, reverse=True):
        if len(onset) <= 0 or onset in {"y", "w"}:
            continue
        if clean.startswith(onset) and len(clean) > len(onset):
            tail = clean[len(onset):]
            if tail and set(tail).issubset({"y", "w", "h"}):
                return onset
    if clean.startswith("hh"):
        return "h"
    return clean


def _strip_phone_decorations(token: str) -> str:
    text = _strip_color_suffix(str(token or "").strip())
    return _strip_pitch_suffix(text)


def _strip_pitch_suffix(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return text
    # A raw vowel token may arrive as "a3" after earlier lowercasing. Keep the
    # vowel and drop only the octave/dedup suffix.
    m_vowel_digit = re.fullmatch(r"([aeiounAEIOUN])\d+(?:[-_]*\d+)?[A-Za-z]*", text)
    if m_vowel_digit:
        return m_vowel_digit.group(1)
    # General UTAU pitch suffix: kC4P, あA3, a3-1, ra4. Remove the final note
    # marker only when there is real phone text before it.
    m = re.match(r"(?s)^(.+?)([A-Ga-g](?:#|b)?\d+(?:[-_]*\d+)?[A-Za-z]*)$", text)
    if m and str(m.group(1) or "").strip():
        return str(m.group(1)).strip()
    return text


def _split_simple_cv_token(token: str) -> tuple[str, str]:
    text = _clean_phone_token(token)
    if not text:
        return "", ""
    if text in {"a", "i", "u", "e", "o"}:
        return "", text
    for vowel in ("a", "i", "u", "e", "o"):
        if text.endswith(vowel):
            return text[: -len(vowel)], vowel
    return text, ""


def _clean_phone_token(token: str) -> str:
    text = _strip_phone_decorations(str(token or "").strip())
    text = text.strip("-_ ").lower()
    return text


def _two_alias_tokens(alias: str) -> tuple[str, str]:
    parts = [token for token in re.split(r"\s+", str(alias or "").strip()) if token]
    if len(parts) >= 2:
        return _strip_phone_decorations(parts[0]), _strip_phone_decorations(parts[1])
    return "", ""


def _alias_phones(alias: str, *, language: str) -> list[str]:
    text = " ".join(_strip_phone_decorations(token) for token in re.split(r"\s+", str(alias or "").strip()) if token)
    if not text:
        return []
    if re.search(r"\s+", text):
        return [token for token in re.split(r"\s+", text) if token]
    tokens = phones_from_text(text, language)
    return list(tokens or [text])


_KO_UPPERCASE_CODAS: frozenset[str] = frozenset({"N", "M", "L", "NG", "H", "T", "P", "K", "R"})


def normalize_ko_coda_in_alias(alias: str) -> str:
    """Lower-case KO uppercase coda tokens in 2-token aliases.

    Training data has 13,734 lowercase coda-bridge rows (``ng g``/``l h``/
    ``m b``) but only 377 uppercase (``NG g``/``L H``/``M b``) variants —
    the same acoustic transition labeled with different alias case
    conventions across voicebanks. Normalizing the uppercase form to
    lowercase at both training preprocessing AND inference unifies the
    distribution and gives the uppercase pattern 38× more supervision
    (2026-05-17 coda-bridge wall analysis).

    Only touches the LEFT token when it's a known KO coda; preserves the
    right token's case (the next-syllable onset). Single-token aliases and
    Japanese/other languages pass through unchanged.

    Examples:
    - ``"NG g"`` → ``"ng g"``
    - ``"L H"`` → ``"l h"`` (both tokens are codas/aspirate but only left
      gets normalized — right is the next-syllable onset)
    - ``"a bw"`` → ``"a bw"`` (left is vowel, unchanged)
    - ``"NG"`` → ``"NG"`` (single token, unchanged)
    - ``"ga"`` → ``"ga"`` (single token, unchanged)
    """
    text = str(alias or "").strip()
    if not text:
        return text
    parts = text.split()
    if len(parts) != 2:
        return text
    left, right = parts
    left_clean = left.split("_", 1)[0]
    if left_clean in _KO_UPPERCASE_CODAS:
        # Preserve any trailing color/voicebank suffix on the left token
        suffix = left[len(left_clean):]
        return f"{left_clean.lower()}{suffix} {right}"
    return text


def _infer_alias_type(alias: str, *, language: str) -> str:
    # P4-a: normalize KO uppercase codas at inference so they hit the same
    # alias-type / role lane as the 13.7k lowercase coda-bridge training rows.
    if str(language or "").strip().lower() == "korean":
        alias = normalize_ko_coda_in_alias(alias)
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

        # KO CVVC coda-bridge aliases (``NG g``/``N ny``/``L d``). Historically
        # routed to 'vcv' role because forcing 'vc' caused one-step slot
        # regressions on 8-mora banks. After the compound-token slot matcher
        # (``UTOA_BOUNDARY_COMPOUND_TOKEN_SLOT_MATCH_ENABLE``) shipped in
        # 2026-05-17, slot mapping for these is reliable, and the 'vcv' lane
        # produced the biggest single source of error in raw-model output
        # (152 rows, |Δ|=410ms, contributing 65.5ms = 31% of ALL |Δ|).
        # Route to 'vc' (which uses the compound matcher) by default; opt-in
        # via ``UTOA_BOUNDARY_KO_CODA_BRIDGE_ROLE_VCV=on`` to restore the old
        # behavior for diagnostic comparison.
        if left_is_c and right_is_c and _is_korean_coda_token(left_token):
            if str(os.environ.get("UTOA_BOUNDARY_KO_CODA_BRIDGE_ROLE_VCV", "") or "").strip().lower() in {"on", "1", "true", "yes"}:
                return "vcv"
            return "vc"

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
    normalized = _normalize_phone_identity(text)
    if normalized in PHONE_AWARE_CONSONANT_LABELS:
        return True
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


def _compound_token_slot_match_enabled() -> bool:
    """Whether two-token transition aliases should match adjacent filename tokens.

    Opt-in (default off) because matching ``u k`` style v-cv aliases against
    every adjacent token pair can produce ambiguous matches in long vowel-only
    wavs. Turn on for voicebanks with multi-mora wavs whose v-cv/vc rows are
    currently pinned to slot 0 (the dominant mis-mapping signature in the
    2026-05-17 val gate run).
    """
    raw = str(os.environ.get("UTOA_BOUNDARY_COMPOUND_TOKEN_SLOT_MATCH_ENABLE", "") or "").strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


def _strip_color_suffix(text: str) -> str:
    """Strip Japanese voicebank color tags like ``_W``/``_X`` from an alias side.

    Aliases shipped with multi-color JP banks often carry a single-letter
    suffix after an underscore (``b a_W``, ``hi_W``); the color does not
    participate in the underlying phone identity used for slot matching.
    """
    raw = str(text or "").strip()
    if "_" in raw:
        head, _, _tail = raw.partition("_")
        head = head.strip()
        if head:
            return head
    return raw


_KO_VOWEL_CORES_LONGEST_FIRST: tuple[str, ...] = (
    "eui", "yae",
    "weo", "wae", "yeo",
    "eu", "eo", "ae", "oe", "ui",
    "ya", "yu", "yo", "ye",
    "wa", "wi", "we", "wo",
    "ua", "ue", "uo",
    "a", "i", "u", "e", "o",
)
_KO_CODA_SUFFIXES: tuple[str, ...] = ("ng", "n", "m", "l", "k", "t", "p", "h", "r")


def _split_ko_roman_syllable(token: str) -> tuple[str, str, str]:
    """Split a KO romanized syllable token into onset/vowel/coda.

    Examples: ``bwa`` -> (``b``, ``wa``, ````), ``geuNG`` -> (``g``, ``eu``,
    ``ng``), ``eui`` -> (````, ``ui``, ````). Returns empty fields when the
    token is not a plausible romanized Korean syllable.
    """
    low = _clean_phone_token(token)
    if not low or not low.isalpha():
        return "", "", ""
    best: tuple[int, str] | None = None
    for vowel in _KO_VOWEL_CORES_LONGEST_FIRST:
        idx = low.find(vowel)
        if idx < 0:
            continue
        if best is None or idx < best[0] or (idx == best[0] and len(vowel) > len(best[1])):
            best = (idx, vowel)
    if best is None:
        return "", "", ""
    start, raw_vowel = best
    end = start + len(raw_vowel)
    onset = low[:start]
    vowel = _normalize_phone_identity(raw_vowel)
    coda = low[end:]
    if coda and coda not in _KO_CODA_SUFFIXES:
        return "", "", ""
    return onset, vowel, coda


def _ko_token_onset(token: str) -> str:
    onset, _vowel, _coda = _split_ko_roman_syllable(token)
    if onset:
        return _normalize_phone_identity(onset)
    _onset, vowel, _coda = _split_ko_roman_syllable(token)
    if vowel:
        return ""
    clean = _normalize_phone_identity(token)
    if _is_consonant_like_token(clean, language="korean") or str(coarse_for_phone(clean, language="korean")).startswith("C_"):
        return clean
    return ""


def _ko_token_vowel(token: str) -> str:
    if _is_vowel_like_token(token, language="korean"):
        return _normalize_phone_identity(token)
    _onset, vowel, _coda = _split_ko_roman_syllable(token)
    return vowel


def _trailing_bare_vowel(token: str) -> str:
    """Return the trailing KO vowel core of ``token`` ONLY when no coda follows.

    The compound alias matcher uses this to distinguish a bare CV like ``kku``
    (returns ``"u"``) from a coda-bearing ``kkung``/``kkul`` (returns ``""``)
    and from a different-vowel ``kkeu`` (returns ``"eu"``). Without this
    rejection, ``u k`` matches every coda variant in an 8-mora wav because
    every ``kkuNG``/``kkuL`` literally string-ends in ``"u"`` once the coda
    letters are part of the suffix check.
    """
    low = str(token or "").strip().lower()
    if not low or not low.isalpha():
        return ""
    for coda in _KO_CODA_SUFFIXES:
        if low.endswith(coda) and len(low) > len(coda):
            return ""
    for vc in _KO_VOWEL_CORES_LONGEST_FIRST:
        if low.endswith(vc):
            return vc
    return ""


def _match_compound_alias_to_token_transition(
    alias: str,
    tokens: list[str],
) -> int | None:
    """Match a 2-token transition alias to an adjacent bare-CV → CV boundary.

    For aliases like ``u k`` (v-cv) or ``a m`` (vc), find slot ``N`` such that
    ``tokens[N]`` is a bare CV ending in the alias's left vowel core AND
    ``tokens[N+1]`` begins with the alias's right consonant onset. Returns
    slot ``N`` on a unique adjacency, else ``None`` (caller falls back).

    Coda-bearing tokens are excluded on the left side so an alias like
    ``u k`` selects the bare ``kku`` transition rather than every ``kku*``
    variant. Vowel-core extraction is longest-first to keep ``eu`` distinct
    from ``u`` (so ``u k`` does not also fire on ``kkeu``).
    """
    text = str(alias or "").strip()
    if not text:
        return None
    parts = re.split(r"\s+", text)
    if len(parts) != 2:
        return None
    left = _strip_color_suffix(parts[0]).lower()
    right = _strip_color_suffix(parts[1]).lower()
    if not left or not right:
        return None
    right_onset = ""
    for ch in right:
        if ch.isalpha():
            right_onset += ch
            if len(right_onset) >= 2:
                break
        else:
            break
    if not right_onset:
        return None
    matches: list[int] = []
    for idx in range(len(tokens) - 1):
        cur_vowel = _trailing_bare_vowel(tokens[idx])
        if cur_vowel != left:
            continue
        nxt = _normalize_for_token_match(tokens[idx + 1])
        if not nxt:
            continue
        if nxt.startswith(right_onset):
            matches.append(idx)
        elif nxt.startswith(right_onset[:1]):
            matches.append(idx)
    if len(matches) == 1:
        return int(matches[0])
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
    "build_phone_aware_target_map",
    "compute_per_row_audio_extents",
    "inject_audio_anchors_into_rows",
    "consonant_family_identity",
    "load_row_specs_from_source_oto",
    "normalize_ko_coda_in_alias",
    "oto_row_to_absolute_anchors",
    "PhoneAwareTargetMap",
    "phone_aware_role_family",
    "training_rows_to_wav_groups",
    "vowel_glide_identity",
    "vowel_nucleus_identity",
]
