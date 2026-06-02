from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Mapping

from core.generation.common.oto_alias_family import (
    alias_family,
    alias_token_is_consonant_like,
    alias_token_is_vowel,
)
from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.mfa_free_oto.kana import parse_kana_slots, parse_kana_text

_PURE_VOWEL_SEQUENCE_CONSONANT_MS = 420.0
_PURE_VOWEL_SEQUENCE_CUTOFF_MS = 600.0
_PURE_VOWEL_SEQUENCE_PREUTTERANCE_MS = 300.0
_PURE_VOWEL_SEQUENCE_OVERLAP_MS = 100.0
_PURE_VOWEL_SEQUENCE_TOLERANCE_MS = 1e-3
_PURE_VOWEL_ONSET_SEQUENCE_CONSONANT_MS = 420.0
_PURE_VOWEL_ONSET_SEQUENCE_CUTOFF_MS = 600.0
_PURE_VOWEL_ONSET_SEQUENCE_PREUTTERANCE_MS = 250.0
_PURE_VOWEL_ONSET_SEQUENCE_OVERLAP_MS = 83.0
_PURE_VOWEL_ONSET_SEQUENCE_V_CONSONANT_MS = 125.0
_PURE_VOWEL_ONSET_SEQUENCE_V_CUTOFF_MS = 360.0
_PURE_VOWEL_ONSET_SEQUENCE_V_PREUTTERANCE_MS = 0.0
_PURE_VOWEL_ONSET_SEQUENCE_V_OVERLAP_MS = 0.0
_HEADLESS_VC_ONSET_SEQUENCE_CONSONANT_MS = 300.0
_HEADLESS_VC_ONSET_SEQUENCE_CUTOFF_MS = 330.0
_HEADLESS_VC_ONSET_SEQUENCE_PREUTTERANCE_MS = 250.0
_HEADLESS_VC_ONSET_SEQUENCE_OVERLAP_MS = 83.0
_REGULAR_PAIR_ONSET_PAIR_SPAN_MS = 340.0
_REGULAR_PAIR_ONSET_STEP_MS = 500.0
_REGULAR_PAIR_ONSET_TOLERANCE_MS = 1e-3
_CV_FIXED_EXPANSION_RESTORE_MIN_CONSONANT_MS = 120.0
_CV_FIXED_EXPANSION_RESTORE_MIN_CUTOFF_MS = 250.0
_JAPANESE_CVVC_CV_PREUTTERANCE_CAP_MS = 70.0
_JAPANESE_CVVC_CV_OVERLAP_CAP_MS = 25.0
_JAPANESE_CVVC_CV_HEAD_VOWEL_PREUTTERANCE_CAP_MS = 90.0
_JAPANESE_CVVC_CV_HEAD_VOWEL_OVERLAP_CAP_MS = 45.0
_JAPANESE_CVVC_CV_NEXT_TRANSITION_CUTOFF_PAD_MS = 80.0
_JAPANESE_CVVC_CV_NEXT_TRANSITION_CUTOFF_MIN_EXCESS_MS = 120.0
_JAPANESE_CVVC_PITCHED_CV_NEXT_TRANSITION_CUTOFF_PAD_MS = 120.0
_JAPANESE_CVVC_PITCHED_CV_NEXT_TRANSITION_CUTOFF_MIN_EXCESS_MS = 200.0
_JAPANESE_CVVC_PITCHED_REGULAR_PAIR_CV_CUTOFF_MIN_MS = 330.0
_JAPANESE_CVVC_PITCHED_REGULAR_PAIR_CV_CUTOFF_MAX_MS = 420.0
_JAPANESE_CVVC_VC_CUTOFF_ORDER_TRIGGER_TAIL_MS = 5.0
_JAPANESE_CVVC_VC_CUTOFF_ORDER_MIN_TAIL_MS = 45.0
_JAPANESE_CVVC_UNDERSCORE_VC_POSITIVE_OFFSET_SHIFT_MIN_MS = 40.0
_JAPANESE_CVVC_UNDERSCORE_VC_POSITIVE_OFFSET_SHIFT_MAX_MS = 130.0
_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_STEP_MS = 500.0
_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_TRIGGER_GAP_MS = 650.0
_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_MIN_ROWS = 4
_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_MAX_SHIFT_MS = 650.0
_JAPANESE_CVVC_ROMAJI_VC_FOLLOWING_CV_GAP_TRIGGER_MS = 380.0
_JAPANESE_CVVC_ROMAJI_VC_FOLLOWING_CV_TARGET_GAP_MS = 200.0
_JAPANESE_CVVC_ROMAJI_VC_FOLLOWING_CV_MAX_SHIFT_MS = 560.0
_JAPANESE_CVVC_PITCH_MISSING_CV_SLOT_SHIFT_MS = 500.0
_JAPANESE_CVVC_PITCH_MISSING_CV_SLOT_COMPRESSED_GAP_MS = 650.0
_JAPANESE_CVVC_PITCH_MISSING_CV_SLOT_MAX_STEPS = 1
_JAPANESE_CVVC_PITCH_HEADLESS_INITIAL_SLOT_SHIFT_MS = 500.0
_JAPANESE_CVVC_PITCH_HEADLESS_INITIAL_SLOT_CV_MAX_MS = 1200.0
_JAPANESE_CVVC_PITCH_HEADLESS_INITIAL_SLOT_MAX_ROWS = 5
_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_SHIFT_MS = -380.0
_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_FIRST_MIN_MS = 1800.0
_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_GAP_MIN_MS = 430.0
_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_GAP_MAX_MS = 570.0
_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_MIN_ROWS = 3
_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_MAX_ROWS = 8
_JAPANESE_CVVC_UNDERSCORE_KANA_SLOT1_SHIFT_MS = 500.0
_JAPANESE_CVVC_UNDERSCORE_KANA_SLOT1_HEAD_MAX_MS = 1050.0
_JAPANESE_CVVC_UNDERSCORE_KANA_SLOT1_HEAD_GAP_MAX_MS = 180.0
_JAPANESE_CVVC_UNDERSCORE_KANA_SLOT1_NEXT_GAP_MIN_MS = 550.0
_JAPANESE_CVVC_UNDERSCORE_KANA_DELAYED_SLOT_SHIFT_MS = -500.0
_JAPANESE_CVVC_UNDERSCORE_KANA_DELAYED_SLOT_GAP_MIN_MS = 460.0
_JAPANESE_CVVC_UNDERSCORE_KANA_DELAYED_SLOT_MIN_ML_LATE_MS = 250.0
_TERMINAL_STANDALONE_VOWEL_OFFSET_AFTER_TERMINAL_MS = 100.0
_TERMINAL_STANDALONE_VOWEL_CONSONANT_MS = 125.0
_TERMINAL_STANDALONE_VOWEL_CUTOFF_MS = 260.0
_TERMINAL_STANDALONE_VOWEL_PREUTTERANCE_MS = 30.0
_TERMINAL_STANDALONE_VOWEL_OVERLAP_MS = 20.0
_TERMINAL_STANDALONE_VOWEL_TERMINAL_AFTER_V_PAD_MS = 80.0
_TERMINAL_STANDALONE_VOWEL_TERMINAL_BEFORE_COMPANION_TRIGGER_MS = 20.0
_TERMINAL_STANDALONE_VOWEL_TOLERANCE_MS = 1e-3
_STANDALONE_RELEASE_R_MIN_OFFSET_MS = 1400.0
_STANDALONE_RELEASE_R_NON_OPEN_CUTOFF_CAP_MS = 420.0
_STANDALONE_RELEASE_R_NON_OPEN_LEFT_TOKENS = frozenset({"i", "u", "e", "o"})
_VOWEL_TOKEN_KEYS = {
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "e",
    "o": "o",
    "n": "n",
    "\u3042": "a",
    "\u3044": "i",
    "\u3046": "u",
    "\u3048": "e",
    "\u304a": "o",
    "\u3093": "n",
    "\u30a2": "a",
    "\u30a4": "i",
    "\u30a6": "u",
    "\u30a8": "e",
    "\u30aa": "o",
    "\u30f3": "n",
}
_ALIAS_ATTACHED_PITCH_SUFFIX_RE = re.compile(r"[A-Ga-g](?:#|b)?[0-8]$")


@dataclass(frozen=True)
class _OtoLineRecord:
    key: tuple[str, str, int]
    line_index: int
    line: str
    wav: str
    alias: str
    offset: float
    consonant: float
    cutoff: float
    preutterance: float
    overlap: float


def apply_no_mfa_lightgbm_safety_filter(
    *,
    pre_oto_path: str,
    post_oto_path: str,
    language: str,
    format_type: str,
    max_overlap_increase_ms: float = 5.0,
) -> dict[str, object]:
    """Conservative guard for applying existing OTO-ML to HSMM/rule OTO.

    The legacy LightGBM models were trained as general OTO refiners, not as a
    targeted residual layer for MFA-free HSMM rows. Keep families where the
    deterministic Japanese CVVC rules are more reliable, and prevent large
    overlap style changes while still allowing timing/length corrections.
    """
    language_norm = str(language or "").strip().lower()
    format_norm = str(format_type or "").strip().lower()
    if language_norm != "japanese" or format_norm != "cvvc":
        return {"enabled": False, "reason": "scope_not_japanese_cvvc"}
    pre_path = Path(pre_oto_path)
    post_path = Path(post_oto_path)
    if not pre_path.is_file() or not post_path.is_file():
        return {"enabled": False, "reason": "missing_oto"}

    pre_records = _load_oto_line_records(pre_path)
    post_text = read_text_with_fallback(str(post_path))
    post_lines = post_text.splitlines()
    post_records = _load_oto_line_records(post_path)
    max_overlap_increase = max(0.0, float(max_overlap_increase_ms))
    pure_vowel_sequence_restore_keys = _pure_vowel_sequence_restore_keys(pre_records)
    pure_vowel_onset_sequence_restore_keys = _pure_vowel_onset_sequence_restore_keys(pre_records)
    headless_vc_onset_sequence_restore_keys = _headless_vc_onset_sequence_restore_keys(pre_records)
    regular_pair_onset_sequence_restore_keys = _regular_pair_onset_sequence_restore_keys(pre_records)
    terminal_release_r_full_restore_keys = _terminal_release_r_full_restore_keys(pre_records)
    terminal_release_r_terminal_copy_restore_keys = _terminal_release_r_terminal_copy_restore_keys(pre_records)
    standalone_release_r_offset_restore_keys = _standalone_release_r_offset_restore_keys(pre_records)
    terminal_standalone_vowel_restore_keys = _terminal_standalone_vowel_restore_keys(pre_records)
    cv_next_transition_records = _cv_next_transition_records(post_records)

    restored_terminal_rows = 0
    restored_breath_rows = 0
    restored_pure_vowel_sequence_rows = 0
    restored_pure_vowel_onset_sequence_rows = 0
    restored_middle_dot_transition_rows = 0
    restored_headless_vc_onset_sequence_rows = 0
    restored_regular_pair_onset_sequence_rows = 0
    restored_terminal_release_r_rows = 0
    restored_terminal_release_r_terminal_copy_rows = 0
    restored_standalone_release_r_offset_rows = 0
    capped_standalone_release_r_cutoff_rows = 0
    capped_pitch_regular_pair_cv_pre_overlap_rows = 0
    clamped_pitch_regular_pair_cv_cutoff_rows = 0
    restored_terminal_standalone_vowel_rows = 0
    retimed_terminal_standalone_vowel_companion_rows = 0
    restored_initial_standalone_vowel_rows = 0
    restored_standalone_vowel_offset_cutoff_rows = 0
    restored_cv_head_cutoff_rows = 0
    restored_spaced_consonant_rows = 0
    restored_vv_overlap_rows = 0
    restored_terminal_vc_overlap_rows = 0
    restored_cv_fixed_cutoff_rows = 0
    repaired_vc_cutoff_order_rows = 0
    restored_underscore_vc_positive_offset_shift_rows = 0
    regularized_following_cv_block_offset_rows = 0
    retimed_romaji_vc_following_cv_rows = 0
    shifted_pitch_missing_cv_slot_rows = 0
    shifted_pitch_headless_initial_slot_rows = 0
    shifted_pitch_katakana_n_grid_rows = 0
    shifted_underscore_kana_slot1_rows = 0
    shifted_underscore_kana_delayed_slot_rows = 0
    capped_cv_next_transition_cutoff_rows = 0
    capped_cv_head_vowel_pre_overlap_rows = 0
    capped_cv_pre_overlap_rows = 0
    capped_overlap_rows = 0
    repaired_parameter_order_rows = 0
    changed_line_indices: set[int] = set()
    for key, post_record in post_records.items():
        pre_record = pre_records.get(key)
        if pre_record is None:
            continue
        if key in pure_vowel_sequence_restore_keys:
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_pure_vowel_sequence_rows += 1
            continue
        if key in pure_vowel_onset_sequence_restore_keys:
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_pure_vowel_onset_sequence_rows += 1
            continue
        if _should_restore_middle_dot_transition(pre_record, post_record):
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_middle_dot_transition_rows += 1
            continue
        if key in headless_vc_onset_sequence_restore_keys:
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_headless_vc_onset_sequence_rows += 1
            continue
        if key in regular_pair_onset_sequence_restore_keys:
            adjusted_line = _replace_oto_offset_and_cutoff(
                post_lines[post_record.line_index],
                offset=pre_record.offset,
                cutoff=pre_record.cutoff,
            )
            allowed_overlap = pre_record.overlap + max_overlap_increase
            if post_record.overlap > allowed_overlap + 1e-6:
                adjusted_line = _replace_oto_overlap(adjusted_line, allowed_overlap)
                capped_overlap_rows += 1
            adjusted_line, cutoff_repaired = _repair_vc_cutoff_order_line(adjusted_line)
            if cutoff_repaired:
                repaired_vc_cutoff_order_rows += 1
            if alias_family(post_record.alias) == "cv" and _alias_has_attached_pitch_suffix(post_record.alias):
                capped_line = _clamp_pitch_regular_pair_cv_cutoff(adjusted_line)
                if capped_line != adjusted_line:
                    adjusted_line = capped_line
                    clamped_pitch_regular_pair_cv_cutoff_rows += 1
                capped_line = _cap_cv_preutterance_and_overlap(adjusted_line, post_record)
                if capped_line != adjusted_line:
                    adjusted_line = capped_line
                    capped_cv_pre_overlap_rows += 1
                    capped_pitch_regular_pair_cv_pre_overlap_rows += 1
            if adjusted_line != post_lines[post_record.line_index]:
                post_lines[post_record.line_index] = adjusted_line
                changed_line_indices.add(post_record.line_index)
                restored_regular_pair_onset_sequence_rows += 1
            continue
        if key in terminal_release_r_full_restore_keys:
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_terminal_release_r_rows += 1
            continue
        if key in terminal_release_r_terminal_copy_restore_keys:
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_terminal_release_r_terminal_copy_rows += 1
            continue
        if key in standalone_release_r_offset_restore_keys:
            adjusted_line = _replace_oto_offset(
                post_lines[post_record.line_index],
                offset=pre_record.offset,
            )
            if _should_cap_standalone_release_r_cutoff(post_record):
                adjusted_line = _replace_oto_cutoff(
                    adjusted_line,
                    -float(_STANDALONE_RELEASE_R_NON_OPEN_CUTOFF_CAP_MS),
                )
                capped_standalone_release_r_cutoff_rows += 1
            if adjusted_line != post_lines[post_record.line_index]:
                post_lines[post_record.line_index] = adjusted_line
                changed_line_indices.add(post_record.line_index)
                restored_standalone_release_r_offset_rows += 1
            continue
        if key in terminal_standalone_vowel_restore_keys:
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_terminal_standalone_vowel_rows += 1
            continue
        family = alias_family(post_record.alias)
        if family == "v":
            if _should_restore_initial_standalone_vowel_profile(pre_record, post_record):
                if post_lines[post_record.line_index] != pre_record.line:
                    post_lines[post_record.line_index] = pre_record.line
                    changed_line_indices.add(post_record.line_index)
                    restored_initial_standalone_vowel_rows += 1
                continue
            adjusted_line = _replace_oto_offset_and_cutoff(
                post_lines[post_record.line_index],
                offset=pre_record.offset,
                cutoff=pre_record.cutoff,
            )
            if adjusted_line != post_lines[post_record.line_index]:
                post_lines[post_record.line_index] = adjusted_line
                changed_line_indices.add(post_record.line_index)
                restored_standalone_vowel_offset_cutoff_rows += 1
            continue
        if family == "cv_head":
            adjusted_line = _replace_oto_cutoff(
                post_lines[post_record.line_index],
                pre_record.cutoff,
            )
            if adjusted_line != post_lines[post_record.line_index]:
                post_lines[post_record.line_index] = adjusted_line
                changed_line_indices.add(post_record.line_index)
                restored_cv_head_cutoff_rows += 1
            continue
        if family == "terminal_v_dash":
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_terminal_rows += 1
            continue
        if family == "policy_breath":
            if post_lines[post_record.line_index] != pre_record.line:
                post_lines[post_record.line_index] = pre_record.line
                changed_line_indices.add(post_record.line_index)
                restored_breath_rows += 1
            continue
        if family == "spaced":
            adjusted_line = _replace_oto_consonant(
                post_lines[post_record.line_index],
                consonant=pre_record.consonant,
            )
            if adjusted_line != post_lines[post_record.line_index]:
                post_lines[post_record.line_index] = adjusted_line
                changed_line_indices.add(post_record.line_index)
                restored_spaced_consonant_rows += 1
            continue
        if family == "cv":
            adjusted_line = post_lines[post_record.line_index]
            if _should_restore_cv_fixed_cutoff(pre_record, post_record):
                adjusted_line = _replace_oto_fixed_and_cutoff(
                    adjusted_line,
                    consonant=pre_record.consonant,
                    cutoff=pre_record.cutoff,
                )
                if adjusted_line != post_lines[post_record.line_index]:
                    restored_cv_fixed_cutoff_rows += 1
            capped_line = _cap_cv_preutterance_and_overlap(adjusted_line, post_record)
            if capped_line != adjusted_line:
                capped_cv_pre_overlap_rows += 1
                adjusted_line = capped_line
            next_transition = cv_next_transition_records.get(key)
            capped_line = _cap_cv_next_transition_cutoff(adjusted_line, next_transition)
            if capped_line != adjusted_line:
                capped_cv_next_transition_cutoff_rows += 1
                adjusted_line = capped_line
            if adjusted_line != post_lines[post_record.line_index]:
                post_lines[post_record.line_index] = adjusted_line
                changed_line_indices.add(post_record.line_index)
            continue
        if family not in {"vc", "vv"}:
            continue
        adjusted_line = post_lines[post_record.line_index]
        restored_overlap_to_pre = False
        if family == "vc":
            adjusted_line, cutoff_repaired = _repair_vc_cutoff_order_line(adjusted_line)
            if cutoff_repaired:
                repaired_vc_cutoff_order_rows += 1
            if _should_restore_underscore_vc_positive_offset_shift(pre_record, post_record):
                capped_line = _replace_oto_offset(adjusted_line, offset=pre_record.offset)
                if capped_line != adjusted_line:
                    adjusted_line = capped_line
                    restored_underscore_vc_positive_offset_shift_rows += 1
            if _is_terminal_vc_alias(post_record.alias):
                capped_line = _replace_oto_overlap(adjusted_line, pre_record.overlap)
                if capped_line != adjusted_line:
                    adjusted_line = capped_line
                    restored_terminal_vc_overlap_rows += 1
                    restored_overlap_to_pre = True
        if family == "vv":
            capped_line = _replace_oto_overlap(adjusted_line, pre_record.overlap)
            if capped_line != adjusted_line:
                adjusted_line = capped_line
                restored_vv_overlap_rows += 1
                restored_overlap_to_pre = True
        allowed_overlap = pre_record.overlap + max_overlap_increase
        if not restored_overlap_to_pre and post_record.overlap > allowed_overlap + 1e-6:
            capped_line = _replace_oto_overlap(adjusted_line, allowed_overlap)
            if capped_line != adjusted_line:
                adjusted_line = capped_line
                capped_overlap_rows += 1
        if adjusted_line != post_lines[post_record.line_index]:
            post_lines[post_record.line_index] = adjusted_line
            changed_line_indices.add(post_record.line_index)

    for post_record in post_records.values():
        capped_line = _cap_cv_head_vowel_preutterance_and_overlap(post_lines[post_record.line_index])
        if capped_line != post_lines[post_record.line_index]:
            post_lines[post_record.line_index] = capped_line
            changed_line_indices.add(post_record.line_index)
            capped_cv_head_vowel_pre_overlap_rows += 1

    retimed_terminal_indices = _retime_terminal_v_dash_from_standalone_vowel_companions(post_lines)
    if retimed_terminal_indices:
        changed_line_indices.update(retimed_terminal_indices)
        retimed_terminal_standalone_vowel_companion_rows += len(retimed_terminal_indices)

    regularized_cv_indices = _regularize_romaji_following_cv_block_offsets(post_lines)
    if regularized_cv_indices:
        changed_line_indices.update(regularized_cv_indices)
        regularized_following_cv_block_offset_rows += len(regularized_cv_indices)

    retimed_romaji_vc_indices = _retime_romaji_vc_before_following_cv_offsets(post_lines)
    if retimed_romaji_vc_indices:
        changed_line_indices.update(retimed_romaji_vc_indices)
        retimed_romaji_vc_following_cv_rows += len(retimed_romaji_vc_indices)

    shifted_pitch_indices = _shift_pitch_missing_cv_slot_offsets(post_lines)
    if shifted_pitch_indices:
        changed_line_indices.update(shifted_pitch_indices)
        shifted_pitch_missing_cv_slot_rows += len(shifted_pitch_indices)

    shifted_headless_initial_indices = _shift_pitch_headless_initial_slot_offsets(post_lines)
    if shifted_headless_initial_indices:
        changed_line_indices.update(shifted_headless_initial_indices)
        shifted_pitch_headless_initial_slot_rows += len(shifted_headless_initial_indices)

    shifted_katakana_n_indices = _shift_pitch_katakana_n_grid_offsets(post_lines)
    if shifted_katakana_n_indices:
        changed_line_indices.update(shifted_katakana_n_indices)
        shifted_pitch_katakana_n_grid_rows += len(shifted_katakana_n_indices)

    shifted_slot1_indices = _shift_underscore_kana_collapsed_slot1_offsets(post_lines)
    if shifted_slot1_indices:
        changed_line_indices.update(shifted_slot1_indices)
        shifted_underscore_kana_slot1_rows += len(shifted_slot1_indices)

    shifted_delayed_slot_indices = _shift_underscore_kana_delayed_slot_offsets(post_lines, pre_records)
    if shifted_delayed_slot_indices:
        changed_line_indices.update(shifted_delayed_slot_indices)
        shifted_underscore_kana_delayed_slot_rows += len(shifted_delayed_slot_indices)

    for line_index, line in enumerate(list(post_lines)):
        repaired_line, repaired = _repair_oto_parameter_order_line(line)
        if not repaired:
            continue
        post_lines[line_index] = repaired_line
        changed_line_indices.add(line_index)
        repaired_parameter_order_rows += 1

    if changed_line_indices:
        post_path.write_text("\n".join(post_lines) + "\n", encoding="utf-8")
    return {
        "enabled": True,
        "restored_terminal_rows": int(restored_terminal_rows),
        "restored_breath_rows": int(restored_breath_rows),
        "restored_pure_vowel_sequence_rows": int(restored_pure_vowel_sequence_rows),
        "restored_pure_vowel_onset_sequence_rows": int(restored_pure_vowel_onset_sequence_rows),
        "restored_middle_dot_transition_rows": int(restored_middle_dot_transition_rows),
        "restored_headless_vc_onset_sequence_rows": int(restored_headless_vc_onset_sequence_rows),
        "restored_regular_pair_onset_sequence_rows": int(restored_regular_pair_onset_sequence_rows),
        "restored_terminal_release_r_rows": int(restored_terminal_release_r_rows),
        "restored_terminal_release_r_terminal_copy_rows": int(restored_terminal_release_r_terminal_copy_rows),
        "restored_standalone_release_r_offset_rows": int(restored_standalone_release_r_offset_rows),
        "capped_standalone_release_r_cutoff_rows": int(capped_standalone_release_r_cutoff_rows),
        "capped_pitch_regular_pair_cv_pre_overlap_rows": int(capped_pitch_regular_pair_cv_pre_overlap_rows),
        "clamped_pitch_regular_pair_cv_cutoff_rows": int(clamped_pitch_regular_pair_cv_cutoff_rows),
        "restored_terminal_standalone_vowel_rows": int(restored_terminal_standalone_vowel_rows),
        "retimed_terminal_standalone_vowel_companion_rows": int(
            retimed_terminal_standalone_vowel_companion_rows
        ),
        "restored_initial_standalone_vowel_rows": int(restored_initial_standalone_vowel_rows),
        "restored_standalone_vowel_offset_cutoff_rows": int(
            restored_standalone_vowel_offset_cutoff_rows
        ),
        "restored_cv_head_cutoff_rows": int(restored_cv_head_cutoff_rows),
        "restored_spaced_consonant_rows": int(restored_spaced_consonant_rows),
        "restored_vv_overlap_rows": int(restored_vv_overlap_rows),
        "restored_terminal_vc_overlap_rows": int(restored_terminal_vc_overlap_rows),
        "restored_cv_fixed_cutoff_rows": int(restored_cv_fixed_cutoff_rows),
        "repaired_vc_cutoff_order_rows": int(repaired_vc_cutoff_order_rows),
        "restored_underscore_vc_positive_offset_shift_rows": int(
            restored_underscore_vc_positive_offset_shift_rows
        ),
        "regularized_following_cv_block_offset_rows": int(
            regularized_following_cv_block_offset_rows
        ),
        "retimed_romaji_vc_following_cv_rows": int(retimed_romaji_vc_following_cv_rows),
        "shifted_pitch_missing_cv_slot_rows": int(shifted_pitch_missing_cv_slot_rows),
        "shifted_pitch_headless_initial_slot_rows": int(
            shifted_pitch_headless_initial_slot_rows
        ),
        "shifted_pitch_katakana_n_grid_rows": int(shifted_pitch_katakana_n_grid_rows),
        "shifted_underscore_kana_slot1_rows": int(shifted_underscore_kana_slot1_rows),
        "shifted_underscore_kana_delayed_slot_rows": int(
            shifted_underscore_kana_delayed_slot_rows
        ),
        "capped_cv_next_transition_cutoff_rows": int(capped_cv_next_transition_cutoff_rows),
        "capped_cv_head_vowel_pre_overlap_rows": int(capped_cv_head_vowel_pre_overlap_rows),
        "capped_cv_pre_overlap_rows": int(capped_cv_pre_overlap_rows),
        "capped_overlap_rows": int(capped_overlap_rows),
        "repaired_parameter_order_rows": int(repaired_parameter_order_rows),
        "changed_rows": int(len(changed_line_indices)),
        "max_overlap_increase_ms": float(max_overlap_increase),
    }


def _load_oto_line_records(path: Path) -> dict[tuple[str, str, int], _OtoLineRecord]:
    text = read_text_with_fallback(str(path))
    occurrences: dict[tuple[str, str], int] = {}
    records: dict[tuple[str, str, int], _OtoLineRecord] = {}
    for line_index, raw_line in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw_line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"]).strip()
        alias = str(parsed["alias"]).strip()
        base_key = (wav.lower(), alias)
        occurrence = int(occurrences.get(base_key, 0))
        occurrences[base_key] = occurrence + 1
        key = (base_key[0], base_key[1], occurrence)
        records[key] = _OtoLineRecord(
            key=key,
            line_index=line_index,
            line=str(raw_line).rstrip("\r\n"),
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
    return records


def _pure_vowel_sequence_restore_keys(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[tuple[str, str, int]]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    for record in records.values():
        grouped.setdefault(record.wav.lower(), []).append(record)

    restore_keys: set[tuple[str, str, int]] = set()
    for wav, wav_records in grouped.items():
        ordered = sorted(wav_records, key=lambda record: record.line_index)
        if len(ordered) < 4:
            continue
        wav_name = Path(wav.replace("\\", "/")).name
        if not wav_name.startswith("_"):
            continue
        head = ordered[0]
        terminal = ordered[-1]
        middle = ordered[1:-1]
        if alias_family(head.alias) != "cv_head":
            continue
        if alias_family(terminal.alias) != "terminal_v_dash":
            continue
        if not middle or not all(_is_pure_vowel_sequence_middle(record) for record in middle):
            continue
        restore_keys.add(head.key)
        restore_keys.update(record.key for record in middle)
    return restore_keys


def _pure_vowel_onset_sequence_restore_keys(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[tuple[str, str, int]]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    for record in records.values():
        grouped.setdefault(record.wav.lower(), []).append(record)

    restore_keys: set[tuple[str, str, int]] = set()
    for wav, wav_records in grouped.items():
        ordered = sorted(wav_records, key=lambda record: record.line_index)
        if len(ordered) < 3:
            continue
        wav_name = Path(wav.replace("\\", "/")).name
        if not wav_name.startswith("_"):
            continue
        head = ordered[0]
        body = ordered[1:]
        if alias_family(head.alias) != "cv_head":
            if len(ordered) < 3:
                continue
            if not all(
                _is_pure_vowel_onset_sequence_body_record(record, is_last=(index == len(ordered) - 1))
                for index, record in enumerate(ordered)
            ):
                continue
            restore_keys.update(record.key for record in ordered)
            continue
        if any(alias_family(record.alias) == "terminal_v_dash" for record in ordered):
            continue
        if not body:
            continue
        if not all(
            _is_pure_vowel_onset_sequence_body_record(record, is_last=(index == len(body) - 1))
            for index, record in enumerate(body)
        ):
            continue
        restore_keys.add(head.key)
        restore_keys.update(record.key for record in body)
    return restore_keys


def _should_restore_middle_dot_transition(
    pre_record: _OtoLineRecord,
    post_record: _OtoLineRecord,
) -> bool:
    if post_record.wav != pre_record.wav or post_record.alias != pre_record.alias:
        return False
    wav_name = Path(post_record.wav.replace("\\", "/")).name
    if not wav_name.startswith("_") or "\u30fb" not in wav_name:
        return False
    if _alias_has_attached_pitch_suffix(post_record.alias):
        return False
    parts = [part.strip() for part in str(post_record.alias or "").split() if part.strip()]
    if len(parts) < 2 or parts[-1] != "\u30fb":
        return False
    return alias_token_is_vowel(parts[0])


def _is_pure_vowel_onset_sequence_body_record(record: _OtoLineRecord, *, is_last: bool) -> bool:
    if is_last and _is_trailing_r_alias(record.alias):
        return _is_pure_vowel_onset_transition_profile(record)
    family = alias_family(record.alias)
    if family == "v":
        if _vowel_token_key(record.alias) is None:
            return False
        return _is_pure_vowel_onset_v_profile(record)
    if not _is_pure_vowel_transition_alias(record.alias):
        return False
    return _is_pure_vowel_onset_transition_profile(record)


def _is_pure_vowel_onset_transition_profile(record: _OtoLineRecord) -> bool:
    return (
        abs(record.consonant - _PURE_VOWEL_ONSET_SEQUENCE_CONSONANT_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and record.cutoff < 0.0
        and abs(abs(record.cutoff) - _PURE_VOWEL_ONSET_SEQUENCE_CUTOFF_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.preutterance - _PURE_VOWEL_ONSET_SEQUENCE_PREUTTERANCE_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.overlap - _PURE_VOWEL_ONSET_SEQUENCE_OVERLAP_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
    )


def _is_pure_vowel_onset_v_profile(record: _OtoLineRecord) -> bool:
    return (
        abs(record.consonant - _PURE_VOWEL_ONSET_SEQUENCE_V_CONSONANT_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and record.cutoff < 0.0
        and abs(abs(record.cutoff) - _PURE_VOWEL_ONSET_SEQUENCE_V_CUTOFF_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.preutterance - _PURE_VOWEL_ONSET_SEQUENCE_V_PREUTTERANCE_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.overlap - _PURE_VOWEL_ONSET_SEQUENCE_V_OVERLAP_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
    )


def _headless_vc_onset_sequence_restore_keys(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[tuple[str, str, int]]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    for record in records.values():
        grouped.setdefault(record.wav.lower(), []).append(record)

    restore_keys: set[tuple[str, str, int]] = set()
    for wav, wav_records in grouped.items():
        ordered = sorted(wav_records, key=lambda record: record.line_index)
        if len(ordered) < 2:
            continue
        wav_name = Path(wav.replace("\\", "/")).name
        if not wav_name.startswith("_"):
            continue
        if not all(_is_headless_vc_onset_sequence_record(record) for record in ordered):
            continue
        if not any(_alias_right_token_has_true_consonant(record.alias) for record in ordered):
            continue
        restore_keys.update(record.key for record in ordered)
    return restore_keys


def _is_headless_vc_onset_sequence_record(record: _OtoLineRecord) -> bool:
    family = alias_family(record.alias)
    if family not in {"vc", "vcv", "vv"}:
        return False
    tokens = _split_alias_tokens(record.alias)
    if len(tokens) < 2 or not _is_pure_sequence_vowel_token(tokens[0]):
        return False
    return _is_headless_vc_onset_profile(record)


def _is_headless_vc_onset_profile(record: _OtoLineRecord) -> bool:
    return (
        abs(record.consonant - _HEADLESS_VC_ONSET_SEQUENCE_CONSONANT_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and record.cutoff < 0.0
        and abs(abs(record.cutoff) - _HEADLESS_VC_ONSET_SEQUENCE_CUTOFF_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.preutterance - _HEADLESS_VC_ONSET_SEQUENCE_PREUTTERANCE_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.overlap - _HEADLESS_VC_ONSET_SEQUENCE_OVERLAP_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
    )


def _alias_right_token_has_true_consonant(alias: str) -> bool:
    tokens = _split_alias_tokens(alias)
    return len(tokens) >= 2 and alias_token_is_consonant_like(tokens[-1])


def _regular_pair_onset_sequence_restore_keys(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[tuple[str, str, int]]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    for record in records.values():
        grouped.setdefault(record.wav.lower(), []).append(record)

    restore_keys: set[tuple[str, str, int]] = set()
    for wav, wav_records in grouped.items():
        ordered = sorted(wav_records, key=lambda record: record.line_index)
        if len(ordered) < 4 or len(ordered) % 2 != 0:
            continue
        wav_name = Path(wav.replace("\\", "/")).name
        if not wav_name.startswith("_"):
            continue
        pairs: list[tuple[_OtoLineRecord, _OtoLineRecord, float]] = []
        for index in range(0, len(ordered), 2):
            vc_record = ordered[index]
            cv_record = ordered[index + 1]
            if alias_family(vc_record.alias) != "vc" or alias_family(cv_record.alias) != "cv":
                pairs = []
                break
            if abs((cv_record.offset - vc_record.offset) - _REGULAR_PAIR_ONSET_PAIR_SPAN_MS) > _REGULAR_PAIR_ONSET_TOLERANCE_MS:
                break
            pairs.append((vc_record, cv_record, (vc_record.offset + cv_record.offset) * 0.5))
        if len(pairs) < 2:
            continue
        previous_center: float | None = None
        for vc_record, cv_record, center in pairs:
            if previous_center is not None and abs((center - previous_center) - _REGULAR_PAIR_ONSET_STEP_MS) > _REGULAR_PAIR_ONSET_TOLERANCE_MS:
                break
            restore_keys.add(vc_record.key)
            restore_keys.add(cv_record.key)
            previous_center = center
    return restore_keys


def _terminal_release_r_full_restore_keys(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[tuple[str, str, int]]:
    restore_keys: set[tuple[str, str, int]] = set()
    for record in records.values():
        wav_name = Path(record.wav.replace("\\", "/")).name
        if not wav_name.startswith("_"):
            continue
        if not _is_trailing_r_alias(record.alias):
            continue
        if _is_pure_vowel_onset_transition_profile(record):
            restore_keys.add(record.key)
    return restore_keys


def _terminal_release_r_terminal_copy_restore_keys(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[tuple[str, str, int]]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    for record in records.values():
        grouped.setdefault(record.wav.lower(), []).append(record)

    restore_keys: set[tuple[str, str, int]] = set()
    for _wav, wav_records in grouped.items():
        terminal_by_left: dict[str, _OtoLineRecord] = {}
        for record in wav_records:
            if alias_family(record.alias) != "terminal_v_dash":
                continue
            tokens = _split_alias_tokens(record.alias)
            if len(tokens) < 2 or tokens[-1] != "-":
                continue
            left_key = _vowel_token_key(tokens[0])
            if left_key is not None:
                terminal_by_left[left_key] = record
        if not terminal_by_left:
            continue
        for record in wav_records:
            if not _is_trailing_r_alias(record.alias):
                continue
            tokens = _split_alias_tokens(record.alias)
            left_key = _vowel_token_key(tokens[0]) if tokens else None
            terminal = terminal_by_left.get(left_key or "")
            if terminal is None:
                continue
            if _records_have_same_timing(record, terminal):
                restore_keys.add(record.key)
    return restore_keys


def _standalone_release_r_offset_restore_keys(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[tuple[str, str, int]]:
    restore_keys: set[tuple[str, str, int]] = set()
    for record in records.values():
        wav_name = Path(record.wav.replace("\\", "/")).name.lower()
        if not re.fullmatch(r"[aeioun]r\.wav", wav_name):
            continue
        if not _is_trailing_r_alias(record.alias):
            continue
        if record.offset < _STANDALONE_RELEASE_R_MIN_OFFSET_MS:
            continue
        restore_keys.add(record.key)
    return restore_keys


def _should_cap_standalone_release_r_cutoff(record: _OtoLineRecord) -> bool:
    wav_name = Path(record.wav.replace("\\", "/")).name.lower()
    if not re.fullmatch(r"[aeioun]r\.wav", wav_name):
        return False
    if not _is_trailing_r_alias(record.alias):
        return False
    tokens = _split_alias_tokens(record.alias)
    left_key = _vowel_token_key(tokens[0]) if tokens else None
    if left_key not in _STANDALONE_RELEASE_R_NON_OPEN_LEFT_TOKENS:
        return False
    return float(record.cutoff) < -float(_STANDALONE_RELEASE_R_NON_OPEN_CUTOFF_CAP_MS)


def _split_alias_tokens(alias: str) -> list[str]:
    return [part.strip() for part in str(alias or "").split() if part.strip()]


def _is_trailing_r_alias(alias: str) -> bool:
    tokens = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(tokens) < 2:
        return False
    left = tokens[0]
    right = _strip_attached_pitch_suffix_preserve_case(tokens[-1])
    return _is_pure_sequence_vowel_token(left) and right == "R"


def _is_pure_vowel_sequence_middle(record: _OtoLineRecord) -> bool:
    if not _is_pure_vowel_transition_alias(record.alias):
        return False
    return (
        abs(record.consonant - _PURE_VOWEL_SEQUENCE_CONSONANT_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and record.cutoff < 0.0
        and abs(abs(record.cutoff) - _PURE_VOWEL_SEQUENCE_CUTOFF_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.preutterance - _PURE_VOWEL_SEQUENCE_PREUTTERANCE_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.overlap - _PURE_VOWEL_SEQUENCE_OVERLAP_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
    )


def _records_have_same_timing(left: _OtoLineRecord, right: _OtoLineRecord) -> bool:
    return (
        abs(left.offset - right.offset) <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(left.consonant - right.consonant) <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(left.cutoff - right.cutoff) <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(left.preutterance - right.preutterance) <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(left.overlap - right.overlap) <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
    )


def _terminal_standalone_vowel_restore_keys(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[tuple[str, str, int]]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    for record in records.values():
        grouped.setdefault(record.wav.lower(), []).append(record)

    restore_keys: set[tuple[str, str, int]] = set()
    for wav, wav_records in grouped.items():
        wav_name = Path(wav.replace("\\", "/")).name
        if not wav_name.startswith("_"):
            continue
        terminal = next((record for record in wav_records if alias_family(record.alias) == "terminal_v_dash"), None)
        if terminal is None:
            continue
        terminal_tokens = [part.strip() for part in terminal.alias.split() if part.strip()]
        if len(terminal_tokens) < 2 or terminal_tokens[-1] != "-":
            continue
        terminal_phone = terminal_tokens[0]
        for record in wav_records:
            if alias_family(record.alias) != "v":
                continue
            if not _terminal_standalone_vowel_alias_matches(record.alias, terminal_phone):
                continue
            if _is_terminal_standalone_vowel_profile(record, terminal):
                restore_keys.add(record.key)
    return restore_keys


def _terminal_standalone_vowel_alias_matches(alias: str, terminal_phone: str) -> bool:
    return _vowel_token_key(alias) == _vowel_token_key(terminal_phone)


def _vowel_token_key(token: str) -> str | None:
    value = str(token or "").strip()
    if not value:
        return None
    value = _strip_attached_pitch_suffix(value)
    return _VOWEL_TOKEN_KEYS.get(value.lower()) or _VOWEL_TOKEN_KEYS.get(value)


def _is_terminal_standalone_vowel_profile(record: _OtoLineRecord, terminal: _OtoLineRecord) -> bool:
    return (
        abs(
            record.offset
            - (
                terminal.offset
                + float(_TERMINAL_STANDALONE_VOWEL_OFFSET_AFTER_TERMINAL_MS)
            )
        )
        <= _TERMINAL_STANDALONE_VOWEL_TOLERANCE_MS
        and abs(record.consonant - _TERMINAL_STANDALONE_VOWEL_CONSONANT_MS)
        <= _TERMINAL_STANDALONE_VOWEL_TOLERANCE_MS
        and record.cutoff < 0.0
        and abs(abs(record.cutoff) - _TERMINAL_STANDALONE_VOWEL_CUTOFF_MS)
        <= _TERMINAL_STANDALONE_VOWEL_TOLERANCE_MS
        and abs(record.preutterance - _TERMINAL_STANDALONE_VOWEL_PREUTTERANCE_MS)
        <= _TERMINAL_STANDALONE_VOWEL_TOLERANCE_MS
        and abs(record.overlap - _TERMINAL_STANDALONE_VOWEL_OVERLAP_MS)
        <= _TERMINAL_STANDALONE_VOWEL_TOLERANCE_MS
    )


def _retime_terminal_v_dash_from_standalone_vowel_companions(post_lines: list[str]) -> set[int]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for line_index, line in enumerate(post_lines):
        parsed = parse_oto_line(line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        record = _OtoLineRecord(
            key=(wav, alias, occurrence),
            line_index=line_index,
            line=line,
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
        grouped.setdefault(wav.lower(), []).append(record)

    changed_indices: set[int] = set()
    for _wav, records in grouped.items():
        standalone_vowels = [record for record in records if alias_family(record.alias) == "v"]
        if not standalone_vowels:
            continue
        for terminal in records:
            if alias_family(terminal.alias) != "terminal_v_dash":
                continue
            tokens = [part.strip() for part in terminal.alias.split() if part.strip()]
            if len(tokens) < 2 or tokens[-1] != "-":
                continue
            terminal_phone = tokens[0]
            companion = next(
                (
                    record
                    for record in standalone_vowels
                    if _terminal_standalone_vowel_alias_matches(record.alias, terminal_phone)
                ),
                None,
            )
            if companion is None:
                continue
            if (
                terminal.offset
                >= companion.offset - float(_TERMINAL_STANDALONE_VOWEL_TERMINAL_BEFORE_COMPANION_TRIGGER_MS)
            ):
                continue
            target_offset = companion.offset + float(_TERMINAL_STANDALONE_VOWEL_TERMINAL_AFTER_V_PAD_MS)
            adjusted_line = _replace_oto_offset(post_lines[terminal.line_index], offset=target_offset)
            if adjusted_line != post_lines[terminal.line_index]:
                post_lines[terminal.line_index] = adjusted_line
                changed_indices.add(terminal.line_index)
    return changed_indices


def _regularize_romaji_following_cv_block_offsets(post_lines: list[str]) -> set[int]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for line_index, line in enumerate(post_lines):
        parsed = parse_oto_line(line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        record = _OtoLineRecord(
            key=(wav, alias, occurrence),
            line_index=line_index,
            line=line,
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
        grouped.setdefault(wav.lower(), []).append(record)

    changed_indices: set[int] = set()
    for _wav, records in grouped.items():
        ordered = sorted(records, key=lambda record: record.line_index)
        if not ordered:
            continue
        if Path(ordered[0].wav).name.startswith("_"):
            continue
        block_start: int | None = None
        seen_transition = False
        for index, record in enumerate(ordered):
            family = alias_family(record.alias)
            if family == "vc":
                seen_transition = True
                continue
            if seen_transition and family == "cv":
                block_start = index
                break
        if block_start is None:
            continue

        block_end = block_start
        while block_end < len(ordered) and alias_family(ordered[block_end].alias) == "cv":
            block_end += 1
        block = ordered[block_start:block_end]
        if len(block) < int(_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_MIN_ROWS):
            continue
        if any(_alias_has_attached_pitch_suffix(record.alias) for record in block):
            continue

        offsets = [float(record.offset) for record in block]
        gaps = [offsets[index + 1] - offsets[index] for index in range(len(offsets) - 1)]
        if not gaps:
            continue
        if min(gaps) <= 0.0:
            continue
        if max(gaps) < float(_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_TRIGGER_GAP_MS):
            continue

        step = float(_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_STEP_MS)
        if gaps[0] > float(_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_TRIGGER_GAP_MS):
            grid_start = offsets[1] - step
        else:
            grid_start = offsets[0]
        targets = [grid_start + (step * index) for index in range(len(block))]
        if min(targets) < 0.0:
            continue
        max_shift = max(abs(target - current) for target, current in zip(targets, offsets))
        if max_shift > float(_JAPANESE_CVVC_FOLLOWING_CV_BLOCK_GRID_MAX_SHIFT_MS):
            continue

        for record, target_offset in zip(block, targets):
            adjusted_line = _replace_oto_offset(post_lines[record.line_index], offset=target_offset)
            if adjusted_line != post_lines[record.line_index]:
                post_lines[record.line_index] = adjusted_line
                changed_indices.add(record.line_index)
    return changed_indices


def _retime_romaji_vc_before_following_cv_offsets(post_lines: list[str]) -> set[int]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for line_index, line in enumerate(post_lines):
        parsed = parse_oto_line(line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        record = _OtoLineRecord(
            key=(wav, alias, occurrence),
            line_index=line_index,
            line=line,
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
        grouped.setdefault(wav.lower(), []).append(record)

    changed_indices: set[int] = set()
    for _wav, records in grouped.items():
        ordered = sorted(records, key=lambda record: record.line_index)
        if not ordered:
            continue
        if Path(ordered[0].wav.replace("\\", "/")).name.startswith("_"):
            continue
        if any(_alias_has_attached_pitch_suffix(record.alias) for record in ordered):
            continue
        first_cv_index = next(
            (
                index
                for index, record in enumerate(ordered)
                if alias_family(record.alias) == "cv"
            ),
            None,
        )
        if first_cv_index is None or first_cv_index <= 0:
            continue

        vc_block = [
            record
            for record in ordered[:first_cv_index]
            if alias_family(record.alias) == "vc"
        ]
        cv_block: list[_OtoLineRecord] = []
        cv_index = first_cv_index
        while cv_index < len(ordered) and alias_family(ordered[cv_index].alias) == "cv":
            cv_block.append(ordered[cv_index])
            cv_index += 1
        if len(vc_block) < 2 or len(cv_block) < 2:
            continue

        pair_count = min(len(vc_block), len(cv_block))
        for vc_record, cv_record in zip(vc_block[:pair_count], cv_block[:pair_count]):
            current_vc_offset = float(_line_offset(post_lines[vc_record.line_index]) or vc_record.offset)
            current_cv_offset = float(_line_offset(post_lines[cv_record.line_index]) or cv_record.offset)
            gap = current_cv_offset - current_vc_offset
            if gap <= float(_JAPANESE_CVVC_ROMAJI_VC_FOLLOWING_CV_GAP_TRIGGER_MS):
                continue
            target_offset = current_cv_offset - float(
                _JAPANESE_CVVC_ROMAJI_VC_FOLLOWING_CV_TARGET_GAP_MS
            )
            shift = target_offset - current_vc_offset
            if shift <= 0.0:
                continue
            if shift > float(_JAPANESE_CVVC_ROMAJI_VC_FOLLOWING_CV_MAX_SHIFT_MS):
                continue
            adjusted_line = _replace_oto_offset(
                post_lines[vc_record.line_index],
                offset=target_offset,
            )
            if adjusted_line != post_lines[vc_record.line_index]:
                post_lines[vc_record.line_index] = adjusted_line
                changed_indices.add(vc_record.line_index)
    return changed_indices


def _shift_pitch_missing_cv_slot_offsets(post_lines: list[str]) -> set[int]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for line_index, line in enumerate(post_lines):
        parsed = parse_oto_line(line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        record = _OtoLineRecord(
            key=(wav, alias, occurrence),
            line_index=line_index,
            line=line,
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
        grouped.setdefault(wav.lower(), []).append(record)

    changed_indices: set[int] = set()
    for _wav, records in grouped.items():
        ordered = sorted(records, key=lambda record: record.line_index)
        if not ordered:
            continue
        if not Path(ordered[0].wav).name.startswith("_"):
            continue
        if not any(_alias_has_attached_pitch_suffix(record.alias) for record in ordered):
            continue
        slots = parse_kana_slots(Path(ordered[0].wav).stem)
        if len(slots) < 3:
            continue

        annotated: list[tuple[_OtoLineRecord, int | None]] = []
        for record in ordered:
            family = alias_family(record.alias)
            slot_index: int | None = None
            if family == "cv":
                slot_index = _pitch_cv_alias_slot_index(record.alias, slots)
            elif family == "vc":
                slot_index = _pitch_vc_alias_right_slot_index(record.alias, slots)
            annotated.append((record, slot_index))

        cv_rows = [
            (record, slot_index)
            for record, slot_index in annotated
            if slot_index is not None and alias_family(record.alias) == "cv"
        ]
        if len(cv_rows) < 2:
            continue

        shifts: list[tuple[int, int]] = []
        cumulative_steps = 0
        previous_record, previous_slot = cv_rows[0]
        for record, slot_index in cv_rows[1:]:
            slot_gap = int(slot_index) - int(previous_slot)
            offset_gap = float(record.offset) - float(previous_record.offset)
            if (
                slot_gap > 1
                and offset_gap < float(_JAPANESE_CVVC_PITCH_MISSING_CV_SLOT_COMPRESSED_GAP_MS)
            ):
                cumulative_steps += slot_gap - 1
                shifts.append((int(slot_index), cumulative_steps))
            previous_record, previous_slot = record, slot_index
        if not shifts:
            continue

        for record, slot_index in annotated:
            if slot_index is None:
                continue
            shift_steps = 0
            for shifted_slot, cumulative in shifts:
                if int(slot_index) >= shifted_slot:
                    shift_steps = cumulative
            if shift_steps <= 0:
                continue
            if shift_steps > int(_JAPANESE_CVVC_PITCH_MISSING_CV_SLOT_MAX_STEPS):
                continue
            target_offset = float(record.offset) + (
                float(_JAPANESE_CVVC_PITCH_MISSING_CV_SLOT_SHIFT_MS) * float(shift_steps)
            )
            adjusted_line = _replace_oto_offset(post_lines[record.line_index], offset=target_offset)
            if adjusted_line != post_lines[record.line_index]:
                post_lines[record.line_index] = adjusted_line
                changed_indices.add(record.line_index)
    return changed_indices


def _shift_pitch_headless_initial_slot_offsets(post_lines: list[str]) -> set[int]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for line_index, line in enumerate(post_lines):
        parsed = parse_oto_line(line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        record = _OtoLineRecord(
            key=(wav, alias, occurrence),
            line_index=line_index,
            line=line,
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
        grouped.setdefault(wav.lower(), []).append(record)

    changed_indices: set[int] = set()
    for _wav, records in grouped.items():
        ordered = sorted(records, key=lambda record: record.line_index)
        if not ordered:
            continue
        wav_name = Path(ordered[0].wav.replace("\\", "/")).name
        if not wav_name.startswith("_"):
            continue
        if not any(_alias_has_attached_pitch_suffix(record.alias) for record in ordered):
            continue
        if any(alias_family(record.alias) == "cv_head" for record in ordered):
            continue
        if len(ordered) > int(_JAPANESE_CVVC_PITCH_HEADLESS_INITIAL_SLOT_MAX_ROWS):
            continue
        slots = parse_kana_slots(Path(wav_name).stem)
        if len(slots) < 3:
            continue
        if len(tuple(slots[0])) != 1 or not _is_pure_sequence_vowel_token(tuple(slots[0])[0]):
            continue

        annotated = _ordered_kana_slot_annotations(ordered, slots)
        mapped = [(record, slot_index) for record, slot_index in annotated if slot_index is not None]
        if len(mapped) != len(ordered):
            continue
        if min(int(slot_index) for _record, slot_index in mapped) != 1:
            continue
        if not any(int(slot_index) > 1 for _record, slot_index in mapped):
            continue
        slot1_cv_rows = [
            record
            for record, slot_index in mapped
            if int(slot_index) == 1 and alias_family(record.alias) == "cv"
        ]
        if not slot1_cv_rows:
            continue
        slot1_cv_offset = min(float(record.offset) for record in slot1_cv_rows)
        if slot1_cv_offset > float(_JAPANESE_CVVC_PITCH_HEADLESS_INITIAL_SLOT_CV_MAX_MS):
            continue

        for record, _slot_index in mapped:
            target_offset = float(record.offset) + float(
                _JAPANESE_CVVC_PITCH_HEADLESS_INITIAL_SLOT_SHIFT_MS
            )
            adjusted_line = _replace_oto_offset(post_lines[record.line_index], offset=target_offset)
            if adjusted_line != post_lines[record.line_index]:
                post_lines[record.line_index] = adjusted_line
                changed_indices.add(record.line_index)
    return changed_indices


def _shift_pitch_katakana_n_grid_offsets(post_lines: list[str]) -> set[int]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for line_index, line in enumerate(post_lines):
        parsed = parse_oto_line(line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        record = _OtoLineRecord(
            key=(wav, alias, occurrence),
            line_index=line_index,
            line=line,
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
        grouped.setdefault(wav.lower(), []).append(record)

    changed_indices: set[int] = set()
    for _wav, records in grouped.items():
        ordered = sorted(records, key=lambda record: record.line_index)
        if not ordered:
            continue
        wav_name = Path(ordered[0].wav.replace("\\", "/")).name
        if not wav_name.startswith("_") or "\u30f3" not in wav_name:
            continue
        if not all(_alias_has_attached_pitch_suffix(record.alias) for record in ordered):
            continue
        if len(ordered) < int(_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_MIN_ROWS):
            continue
        if len(ordered) > int(_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_MAX_ROWS):
            continue
        if any(alias_family(record.alias) not in {"vc", "vv"} for record in ordered):
            continue

        offsets = [float(record.offset) for record in ordered]
        gaps = [offsets[index + 1] - offsets[index] for index in range(len(offsets) - 1)]
        if not gaps:
            continue
        if offsets[0] < float(_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_FIRST_MIN_MS):
            continue
        if min(gaps) < float(_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_GAP_MIN_MS):
            continue
        if max(gaps) > float(_JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_GAP_MAX_MS):
            continue

        for record in ordered:
            target_offset = float(record.offset) + float(
                _JAPANESE_CVVC_PITCH_KATAKANA_N_GRID_SHIFT_MS
            )
            adjusted_line = _replace_oto_offset(post_lines[record.line_index], offset=target_offset)
            if adjusted_line != post_lines[record.line_index]:
                post_lines[record.line_index] = adjusted_line
                changed_indices.add(record.line_index)
    return changed_indices


def _shift_underscore_kana_collapsed_slot1_offsets(post_lines: list[str]) -> set[int]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for line_index, line in enumerate(post_lines):
        parsed = parse_oto_line(line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        record = _OtoLineRecord(
            key=(wav, alias, occurrence),
            line_index=line_index,
            line=line,
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
        grouped.setdefault(wav.lower(), []).append(record)

    changed_indices: set[int] = set()
    for _wav, records in grouped.items():
        ordered = sorted(records, key=lambda record: record.line_index)
        if not ordered:
            continue
        if not Path(ordered[0].wav).name.startswith("_"):
            continue
        if any(_alias_has_attached_pitch_suffix(record.alias) for record in ordered):
            continue
        slots = parse_kana_slots(Path(ordered[0].wav).stem)
        if len(slots) < 3:
            continue

        annotated = _ordered_kana_slot_annotations(ordered, slots)
        head_rows = [
            record
            for record, slot_index in annotated
            if slot_index == 0 and alias_family(record.alias) == "cv_head"
        ]
        slot1_rows = [
            record
            for record, slot_index in annotated
            if slot_index == 1 and alias_family(record.alias) in {"cv", "vc"}
        ]
        slot1_cv_rows = [
            record
            for record in slot1_rows
            if alias_family(record.alias) == "cv"
        ]
        slot2_cv_rows = [
            record
            for record, slot_index in annotated
            if slot_index == 2 and alias_family(record.alias) == "cv"
        ]
        if not head_rows or not slot1_rows or not slot1_cv_rows or not slot2_cv_rows:
            continue

        head_offset = min(float(record.offset) for record in head_rows)
        slot1_cv_offset = min(float(record.offset) for record in slot1_cv_rows)
        slot2_cv_offset = min(float(record.offset) for record in slot2_cv_rows)
        if head_offset > float(_JAPANESE_CVVC_UNDERSCORE_KANA_SLOT1_HEAD_MAX_MS):
            continue
        if (
            slot1_cv_offset - head_offset
            >= float(_JAPANESE_CVVC_UNDERSCORE_KANA_SLOT1_HEAD_GAP_MAX_MS)
        ):
            continue
        if (
            slot2_cv_offset - slot1_cv_offset
            <= float(_JAPANESE_CVVC_UNDERSCORE_KANA_SLOT1_NEXT_GAP_MIN_MS)
        ):
            continue

        for record in slot1_rows:
            target_offset = float(record.offset) + float(_JAPANESE_CVVC_UNDERSCORE_KANA_SLOT1_SHIFT_MS)
            adjusted_line = _replace_oto_offset(post_lines[record.line_index], offset=target_offset)
            if adjusted_line != post_lines[record.line_index]:
                post_lines[record.line_index] = adjusted_line
                changed_indices.add(record.line_index)
    return changed_indices


def _shift_underscore_kana_delayed_slot_offsets(
    post_lines: list[str],
    pre_records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> set[int]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for line_index, line in enumerate(post_lines):
        parsed = parse_oto_line(line)
        if not isinstance(parsed, Mapping):
            continue
        wav = str(parsed["wav"])
        alias = str(parsed["alias"])
        base_key = (wav, alias)
        occurrence = occurrences.get(base_key, 0)
        occurrences[base_key] = occurrence + 1
        record = _OtoLineRecord(
            key=(wav, alias, occurrence),
            line_index=line_index,
            line=line,
            wav=wav,
            alias=alias,
            offset=float(parsed["offset"]),
            consonant=float(parsed["cons"]),
            cutoff=float(parsed["cutoff"]),
            preutterance=float(parsed["pre"]),
            overlap=float(parsed["ovl"]),
        )
        grouped.setdefault(wav.lower(), []).append(record)

    changed_indices: set[int] = set()
    for _wav, records in grouped.items():
        ordered = sorted(records, key=lambda record: record.line_index)
        if not ordered:
            continue
        if not Path(ordered[0].wav).name.startswith("_"):
            continue
        if any(_alias_has_attached_pitch_suffix(record.alias) for record in ordered):
            continue
        if not any(alias_family(record.alias) == "cv_head" for record in ordered):
            continue
        slots = parse_kana_slots(Path(ordered[0].wav).stem)
        if len(slots) < 4:
            continue

        first_n_slot = next(
            (index for index, slot in enumerate(slots) if tuple(slot) == ("n",)),
            len(slots),
        )
        annotated = _ordered_kana_slot_annotations(ordered, slots)
        by_slot: dict[int, list[_OtoLineRecord]] = {}
        for record, slot_index in annotated:
            if slot_index is None:
                continue
            by_slot.setdefault(int(slot_index), []).append(record)

        previous_cv_offset: float | None = None
        for slot_index in range(0, min(first_n_slot, len(slots))):
            slot_rows = by_slot.get(slot_index, [])
            cv_rows = [
                record
                for record in slot_rows
                if alias_family(record.alias) in {"cv", "v"}
            ]
            if not cv_rows:
                continue
            if previous_cv_offset is None:
                previous_cv_offset = min(float(record.offset) for record in cv_rows)
                continue

            transition_rows = [
                record
                for record in slot_rows
                if alias_family(record.alias) in {"vc", "vv"}
            ]
            if transition_rows:
                transition_offset = min(
                    float(_line_offset(post_lines[record.line_index]) or record.offset)
                    for record in transition_rows
                )
                if (
                    transition_offset - previous_cv_offset
                    > float(_JAPANESE_CVVC_UNDERSCORE_KANA_DELAYED_SLOT_GAP_MIN_MS)
                ):
                    for record in [*transition_rows, *cv_rows]:
                        pre_record = pre_records.get(record.key)
                        if pre_record is None:
                            continue
                        current_offset = float(
                            _line_offset(post_lines[record.line_index]) or record.offset
                        )
                        if (
                            current_offset - float(pre_record.offset)
                            < float(_JAPANESE_CVVC_UNDERSCORE_KANA_DELAYED_SLOT_MIN_ML_LATE_MS)
                        ):
                            continue
                        target_offset = current_offset + float(
                            _JAPANESE_CVVC_UNDERSCORE_KANA_DELAYED_SLOT_SHIFT_MS
                        )
                        target_offset = max(target_offset, float(pre_record.offset))
                        adjusted_line = _replace_oto_offset(
                            post_lines[record.line_index],
                            offset=target_offset,
                        )
                        if adjusted_line != post_lines[record.line_index]:
                            post_lines[record.line_index] = adjusted_line
                            changed_indices.add(record.line_index)

            previous_cv_offset = min(
                float(_line_offset(post_lines[record.line_index]) or record.offset)
                for record in cv_rows
            )
    return changed_indices


def _ordered_kana_slot_annotations(
    ordered: list[_OtoLineRecord],
    slots: list[tuple[str, ...]],
) -> list[tuple[_OtoLineRecord, int | None]]:
    annotated: list[tuple[_OtoLineRecord, int | None]] = []
    cv_cursor = 0
    for record in ordered:
        family = alias_family(record.alias)
        slot_index: int | None = None
        if family == "cv_head":
            slot_index = 0
        elif family == "cv":
            phones = tuple(parse_kana_text(_strip_attached_pitch_suffix_preserve_case(record.alias)))
            for index in range(cv_cursor + 1, len(slots)):
                if tuple(slots[index]) == phones:
                    slot_index = index
                    cv_cursor = index
                    break
            if slot_index is None:
                for index, slot in enumerate(slots):
                    if tuple(slot) == phones:
                        slot_index = index
                        break
        elif family == "vc":
            slot_index = _pitch_vc_alias_right_slot_index(record.alias, slots)
        annotated.append((record, slot_index))
    return annotated


def _line_offset(line: str) -> float | None:
    parsed = parse_oto_line(line)
    if not isinstance(parsed, Mapping):
        return None
    try:
        return float(parsed["offset"])
    except Exception:
        return None


def _pitch_cv_alias_slot_index(alias: str, slots: list[tuple[str, ...]]) -> int | None:
    phones = tuple(parse_kana_text(_strip_attached_pitch_suffix_preserve_case(alias)))
    if not phones:
        return None
    for index, slot in enumerate(slots):
        if tuple(slot) == phones:
            return index
    return None


def _pitch_vc_alias_right_slot_index(alias: str, slots: list[tuple[str, ...]]) -> int | None:
    parts = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(parts) < 2:
        return None
    left = _strip_attached_pitch_suffix_preserve_case(parts[0]).strip().lower()
    right = _strip_attached_pitch_suffix_preserve_case(parts[-1]).strip().lower()
    if not left or not right:
        return None
    for index, slot in enumerate(slots):
        if index <= 0 or len(slot) < 2:
            continue
        onset = tuple(str(phone or "").strip().lower() for phone in slot[:-1] if str(phone or "").strip())
        if not onset:
            continue
        if "".join(onset) != right and onset != (right,):
            continue
        previous_vowel = str(slots[index - 1][-1] if slots[index - 1] else "").strip().lower()
        if previous_vowel == left:
            return index
    return None


def _should_restore_initial_standalone_vowel_profile(
    pre_record: _OtoLineRecord,
    post_record: _OtoLineRecord,
) -> bool:
    if alias_family(pre_record.alias) != "v" or alias_family(post_record.alias) != "v":
        return False
    if not _is_initial_standalone_vowel_profile(pre_record):
        return False
    return (
        post_record.preutterance > _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        or post_record.overlap > _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        or (
            float(post_record.consonant) - float(pre_record.consonant)
            >= _CV_FIXED_EXPANSION_RESTORE_MIN_CONSONANT_MS
        )
        or (
            abs(float(post_record.cutoff)) - abs(float(pre_record.cutoff))
            >= _CV_FIXED_EXPANSION_RESTORE_MIN_CUTOFF_MS
        )
    )


def _is_initial_standalone_vowel_profile(record: _OtoLineRecord) -> bool:
    return (
        abs(record.consonant - _PURE_VOWEL_ONSET_SEQUENCE_V_CONSONANT_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and record.cutoff < 0.0
        and abs(abs(record.cutoff) - _PURE_VOWEL_ONSET_SEQUENCE_V_CUTOFF_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.preutterance - _PURE_VOWEL_ONSET_SEQUENCE_V_PREUTTERANCE_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
        and abs(record.overlap - _PURE_VOWEL_ONSET_SEQUENCE_V_OVERLAP_MS)
        <= _PURE_VOWEL_SEQUENCE_TOLERANCE_MS
    )


def _is_pure_vowel_transition_alias(alias: str) -> bool:
    tokens = [part.strip() for part in str(alias or "").split() if part.strip()]
    return len(tokens) >= 2 and all(_is_pure_sequence_vowel_token(token) for token in tokens)


def _is_pure_sequence_vowel_token(token: str) -> bool:
    value = str(token or "").strip()
    if not value:
        return False
    value = _strip_attached_pitch_suffix(value)
    return alias_token_is_vowel(value) or value.lower() == "n" or value in {"\u3093", "\u30f3"}


def _strip_attached_pitch_suffix(token: str) -> str:
    value = str(token or "").strip()
    match = _ALIAS_ATTACHED_PITCH_SUFFIX_RE.search(value.lower())
    if match is None or match.start() <= 0:
        return value
    stripped = value[: match.start()].rstrip("_- ").strip()
    return stripped or value


def _strip_attached_pitch_suffix_preserve_case(token: str) -> str:
    value = str(token or "").strip()
    match = _ALIAS_ATTACHED_PITCH_SUFFIX_RE.search(value)
    if match is None or match.start() <= 0:
        return value
    stripped = value[: match.start()].rstrip("_- ").strip()
    return stripped or value


def _alias_has_attached_pitch_suffix(alias: str) -> bool:
    value = str(alias or "").strip()
    return bool(value and _ALIAS_ATTACHED_PITCH_SUFFIX_RE.search(value))


def _is_terminal_vc_alias(alias: str) -> bool:
    parts = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(parts) < 2:
        return False
    right = _strip_attached_pitch_suffix_preserve_case(parts[-1]).strip()
    return right.startswith("-")


def _cv_next_transition_records(
    records: Mapping[tuple[str, str, int], _OtoLineRecord],
) -> dict[tuple[str, str, int], _OtoLineRecord]:
    grouped: dict[str, list[_OtoLineRecord]] = {}
    for record in records.values():
        grouped.setdefault(record.wav.lower(), []).append(record)

    out: dict[tuple[str, str, int], _OtoLineRecord] = {}
    for _wav, wav_records in grouped.items():
        ordered = sorted(wav_records, key=lambda record: record.line_index)
        for index, record in enumerate(ordered):
            if alias_family(record.alias) != "cv":
                continue
            for next_record in ordered[index + 1 :]:
                next_family = alias_family(next_record.alias)
                if next_family == "vcv" and _is_same_timing_vcv_companion_record(record, next_record):
                    continue
                if next_family in {"vc", "vv", "vcv"}:
                    out[record.key] = next_record
                    break
                if next_family == "cv_head":
                    break
    return out


def _is_same_timing_vcv_companion_record(source: _OtoLineRecord, candidate: _OtoLineRecord) -> bool:
    if alias_family(candidate.alias) != "vcv":
        return False
    offset_gap = abs(float(candidate.offset) - float(source.offset))
    if not math.isfinite(offset_gap):
        return False
    timing_close = (
        abs(float(candidate.consonant) - float(source.consonant)) <= 1e-3
        and abs(float(candidate.cutoff) - float(source.cutoff)) <= 1e-3
        and abs(float(candidate.preutterance) - float(source.preutterance)) <= 1e-3
        and abs(float(candidate.overlap) - float(source.overlap)) <= 1e-3
    )
    return timing_close or offset_gap <= 160.0


def _should_restore_cv_fixed_cutoff(pre_record: _OtoLineRecord, post_record: _OtoLineRecord) -> bool:
    consonant_expansion = float(post_record.consonant) - float(pre_record.consonant)
    cutoff_expansion = abs(float(post_record.cutoff)) - abs(float(pre_record.cutoff))
    return (
        consonant_expansion >= _CV_FIXED_EXPANSION_RESTORE_MIN_CONSONANT_MS
        and cutoff_expansion >= _CV_FIXED_EXPANSION_RESTORE_MIN_CUTOFF_MS
    )


def _replace_oto_fixed_and_cutoff(line: str, *, consonant: float, cutoff: float) -> str:
    if "=" not in line:
        return line
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line
    parts[2] = _format_oto_number(consonant)
    parts[3] = _format_oto_number(cutoff)
    return f"{wav_name}={','.join(parts)}"


def _replace_oto_offset_and_cutoff(line: str, *, offset: float, cutoff: float) -> str:
    if "=" not in line:
        return line
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line
    parts[1] = _format_oto_number(offset)
    parts[3] = _format_oto_number(cutoff)
    return f"{wav_name}={','.join(parts)}"


def _replace_oto_offset(line: str, *, offset: float) -> str:
    if "=" not in line:
        return line
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line
    parts[1] = _format_oto_number(offset)
    return f"{wav_name}={','.join(parts)}"


def _replace_oto_consonant(line: str, *, consonant: float) -> str:
    if "=" not in line:
        return line
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line
    parts[2] = _format_oto_number(consonant)
    return f"{wav_name}={','.join(parts)}"


def _replace_oto_overlap(line: str, overlap: float) -> str:
    if "=" not in line:
        return line
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line
    parts[5] = _format_oto_number(overlap)
    return f"{wav_name}={','.join(parts)}"


def _replace_oto_cutoff(line: str, cutoff: float) -> str:
    if "=" not in line:
        return line
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line
    parts[3] = _format_oto_number(cutoff)
    return f"{wav_name}={','.join(parts)}"


def _repair_vc_cutoff_order_line(line: str) -> tuple[str, bool]:
    parsed = parse_oto_line(line)
    if not isinstance(parsed, Mapping):
        return line, False
    alias = str(parsed["alias"])
    if alias_family(alias) != "vc":
        return line, False
    cutoff = float(parsed["cutoff"])
    consonant = float(parsed["cons"])
    if cutoff >= 0.0:
        return line, False
    current_tail = abs(cutoff) - consonant
    if current_tail > float(_JAPANESE_CVVC_VC_CUTOFF_ORDER_TRIGGER_TAIL_MS):
        return line, False
    repaired_cutoff = -(consonant + float(_JAPANESE_CVVC_VC_CUTOFF_ORDER_MIN_TAIL_MS))
    adjusted = _replace_oto_cutoff(line, repaired_cutoff)
    return adjusted, adjusted != line


def _repair_oto_parameter_order_line(line: str) -> tuple[str, bool]:
    parsed = parse_oto_line(line)
    if not isinstance(parsed, Mapping):
        return line, False
    if alias_family(str(parsed.get("alias", "") or "")) != "cv_head":
        return line, False
    try:
        consonant = max(0.0, float(parsed["cons"]))
        cutoff = float(parsed["cutoff"])
        preutterance = max(0.0, float(parsed["pre"]))
        overlap = max(0.0, float(parsed["ovl"]))
    except Exception:
        return line, False
    values = (consonant, cutoff, preutterance, overlap)
    if not all(math.isfinite(value) for value in values):
        return line, False

    preutterance = min(preutterance, consonant)
    overlap = min(overlap, preutterance)
    cutoff_abs = max(abs(cutoff), consonant + 8.0)
    repaired_cutoff = -cutoff_abs

    if "=" not in line:
        return line, False
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line, False
    parts[2] = _format_oto_number(consonant)
    parts[3] = _format_oto_number(repaired_cutoff)
    parts[4] = _format_oto_number(preutterance)
    parts[5] = _format_oto_number(overlap)
    repaired_line = f"{wav_name}={','.join(parts)}"
    return repaired_line, repaired_line != line


def _should_restore_underscore_vc_positive_offset_shift(
    pre_record: _OtoLineRecord,
    post_record: _OtoLineRecord,
) -> bool:
    if alias_family(pre_record.alias) != "vc" or alias_family(post_record.alias) != "vc":
        return False
    wav_name = Path(str(post_record.wav).replace("\\", "/")).name
    if not wav_name.startswith("_"):
        return False
    if _alias_has_attached_pitch_suffix(post_record.alias):
        return False
    offset_shift = float(post_record.offset) - float(pre_record.offset)
    return (
        offset_shift >= _JAPANESE_CVVC_UNDERSCORE_VC_POSITIVE_OFFSET_SHIFT_MIN_MS
        and offset_shift <= _JAPANESE_CVVC_UNDERSCORE_VC_POSITIVE_OFFSET_SHIFT_MAX_MS
    )


def _cap_cv_next_transition_cutoff(line: str, next_transition: _OtoLineRecord | None) -> str:
    if next_transition is None:
        return line
    parsed = parse_oto_line(line)
    if not isinstance(parsed, Mapping):
        return line
    alias = str(parsed["alias"])
    if alias_family(alias) != "cv":
        return line
    offset = float(parsed["offset"])
    consonant = float(parsed["cons"])
    cutoff = float(parsed["cutoff"])
    if cutoff >= 0.0:
        return line
    cutoff_abs = abs(cutoff)
    if _alias_has_attached_pitch_suffix(alias):
        pad_ms = float(_JAPANESE_CVVC_PITCHED_CV_NEXT_TRANSITION_CUTOFF_PAD_MS)
        min_excess_ms = float(_JAPANESE_CVVC_PITCHED_CV_NEXT_TRANSITION_CUTOFF_MIN_EXCESS_MS)
    else:
        pad_ms = float(_JAPANESE_CVVC_CV_NEXT_TRANSITION_CUTOFF_PAD_MS)
        min_excess_ms = float(_JAPANESE_CVVC_CV_NEXT_TRANSITION_CUTOFF_MIN_EXCESS_MS)
    target_end = float(next_transition.offset) + pad_ms
    current_end = offset + cutoff_abs
    values = (offset, consonant, cutoff_abs, target_end, current_end)
    if not all(math.isfinite(value) for value in values):
        return line
    if current_end <= target_end + min_excess_ms:
        return line
    min_cutoff_abs = max(0.0, consonant + 8.0)
    capped_abs = max(min_cutoff_abs, target_end - offset)
    if capped_abs >= cutoff_abs - 1e-6:
        return line
    return _replace_oto_cutoff(line, -capped_abs)


def _clamp_pitch_regular_pair_cv_cutoff(line: str) -> str:
    parsed = parse_oto_line(line)
    if not isinstance(parsed, Mapping):
        return line
    alias = str(parsed["alias"])
    if alias_family(alias) != "cv" or not _alias_has_attached_pitch_suffix(alias):
        return line
    cutoff = float(parsed["cutoff"])
    if cutoff >= 0.0:
        return line
    cutoff_abs = abs(cutoff)
    clamped_abs = min(
        max(cutoff_abs, float(_JAPANESE_CVVC_PITCHED_REGULAR_PAIR_CV_CUTOFF_MIN_MS)),
        float(_JAPANESE_CVVC_PITCHED_REGULAR_PAIR_CV_CUTOFF_MAX_MS),
    )
    if abs(clamped_abs - cutoff_abs) <= 1e-6:
        return line
    return _replace_oto_cutoff(line, -clamped_abs)


def _cap_cv_head_vowel_preutterance_and_overlap(line: str) -> str:
    parsed = parse_oto_line(line)
    if not isinstance(parsed, Mapping):
        return line
    alias = str(parsed["alias"])
    if alias_family(alias) != "cv_head":
        return line
    tokens = [part.strip() for part in alias.split() if part.strip()]
    if len(tokens) < 2 or tokens[0] != "-" or not alias_token_is_vowel(tokens[-1]):
        return line
    preutterance = float(parsed["pre"])
    overlap = float(parsed["ovl"])
    capped_preutterance = min(
        preutterance,
        float(_JAPANESE_CVVC_CV_HEAD_VOWEL_PREUTTERANCE_CAP_MS),
    )
    capped_overlap = min(
        overlap,
        float(_JAPANESE_CVVC_CV_HEAD_VOWEL_OVERLAP_CAP_MS),
        capped_preutterance,
    )
    if (
        abs(capped_preutterance - preutterance) <= 1e-6
        and abs(capped_overlap - overlap) <= 1e-6
    ):
        return line
    if "=" not in line:
        return line
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line
    parts[4] = _format_oto_number(capped_preutterance)
    parts[5] = _format_oto_number(capped_overlap)
    return f"{wav_name}={','.join(parts)}"


def _cap_cv_preutterance_and_overlap(line: str, record: _OtoLineRecord) -> str:
    capped_preutterance = min(
        float(record.preutterance),
        float(_JAPANESE_CVVC_CV_PREUTTERANCE_CAP_MS),
    )
    capped_overlap = min(
        float(record.overlap),
        float(_JAPANESE_CVVC_CV_OVERLAP_CAP_MS),
    )
    if (
        abs(capped_preutterance - float(record.preutterance)) <= 1e-6
        and abs(capped_overlap - float(record.overlap)) <= 1e-6
    ):
        return line
    if "=" not in line:
        return line
    wav_name, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return line
    parts[4] = _format_oto_number(capped_preutterance)
    parts[5] = _format_oto_number(capped_overlap)
    return f"{wav_name}={','.join(parts)}"


def _format_oto_number(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) <= 1e-6:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")
