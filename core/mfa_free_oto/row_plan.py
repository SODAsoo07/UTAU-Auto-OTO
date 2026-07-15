from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from core.model_context.filename import filename_syllable_order_tokens

from .kana import parse_kana_slots, phones_to_kana
from .manifest_audit import infer_filename_phone_sequence
from .oto_adapter import OtoTemplateRow, OtoTiming
from .types import is_vowel_phone


ROW_PLAN_SCHEMA_VERSION = 1

_IGNORE_TOKENS = {"", "sil", "pau", "sp", "ap", "br", "bre", "breath", "endbr"}
# Standalone sung coda segments in Korean VCV reclists (N'bui'L'bui'...).
_KOREAN_STANDALONE_CODA_TOKENS = {"n", "m", "l", "ng"}
_CONSONANT_CLUSTERS = (
    "ng",
    "sh",
    "ch",
    "ts",
    "ky",
    "gy",
    "ny",
    "hy",
    "by",
    "py",
    "my",
    "ry",
    "jy",
    "ty",
    "dy",
    "kw",
    "gw",
)
_KOREAN_VOWELS = (
    "yeo",
    "yae",
    "wae",
    "weo",
    "eo",
    "eu",
    "ae",
    "ya",
    "ye",
    "yo",
    "yu",
    "wa",
    "wo",
    "we",
    "wi",
    "eui",
    "ui",
    "a",
    "i",
    "u",
    "e",
    "o",
)
_JA_VOWELS = ("a", "i", "u", "e", "o", "n")


@dataclass(frozen=True)
class FilenameSlot:
    wav: str
    slot_index: int
    token: str
    onset: str
    vowel: str
    onset_phones: tuple[str, ...]
    vowel_phone: str
    phone_start_index: int
    vowel_phone_index: int
    coda_phones: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: int = ROW_PLAN_SCHEMA_VERSION

    @property
    def phones(self) -> tuple[str, ...]:
        return (*self.onset_phones, self.vowel_phone, *self.coda_phones)

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["onset_phones"] = list(self.onset_phones)
        data["warnings"] = list(self.warnings)
        return data


@dataclass(frozen=True)
class RowPlanRecord:
    wav: str
    alias: str
    role_family: str
    row_index: int
    slot_index: int
    left_slot_index: int
    right_slot_index: int
    expected_tokens: tuple[str, ...]
    expected_phone_indices: tuple[int, ...]
    source_row_index: int = -1
    source_timing_trusted: bool = False
    warnings: tuple[str, ...] = ()
    schema_version: int = ROW_PLAN_SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["expected_tokens"] = list(self.expected_tokens)
        data["expected_phone_indices"] = list(self.expected_phone_indices)
        data["warnings"] = list(self.warnings)
        return data


def build_filename_slots(
    wav_name: str,
    *,
    language: str = "",
    format_type: str = "",
) -> tuple[FilenameSlot, ...]:
    wav = os.path.basename(str(wav_name or ""))
    if _is_japanese_language(language):
        stem = os.path.splitext(wav)[0]
        kana_phone_slots = parse_kana_slots(stem)
        if kana_phone_slots:
            return _slots_from_kana_phone_slots(wav, kana_phone_slots)
    if _is_korean_language(language):
        # Korean VCV banks commonly concatenate romanized syllables without
        # apostrophes (``gagageo...``) and mix those chains with ``+`` groups.
        # The generic order tokenizer sees each chain as one oversized token,
        # which collapses an 8-syllable row plan into one slot and moves OTO
        # anchors by several seconds. Reuse the alignment filename segmenter so
        # lab generation and MFA-free timing agree on the syllable sequence.
        from core.alignment.lab_generator import split_korean_filename_tokens

        generic_tokens = filename_syllable_order_tokens(wav, language=language)
        segmented_tokens = split_korean_filename_tokens(wav, normalize_roman_case=True)
        # Keep the established tokenizer for already-delimited filenames and
        # test/runtime overrides.  The Korean segmenter is only authoritative
        # when it actually recovers additional syllable slots.
        tokens = segmented_tokens if len(segmented_tokens) > len(generic_tokens) else generic_tokens
    else:
        tokens = filename_syllable_order_tokens(wav, language=language)
    if not tokens:
        return _slots_from_phone_sequence(wav, infer_filename_phone_sequence(wav), language=language)

    slots: list[FilenameSlot] = []
    phone_index = 0
    for raw in tokens:
        token = _normalize_token(raw)
        if token in _IGNORE_TOKENS:
            continue
        parsed = _split_syllable_token(token, language=language, format_type=format_type)
        if parsed is None and token in _KOREAN_STANDALONE_CODA_TOKENS and str(language or "").strip().lower().startswith("ko"):
            # DiKOR-style reclists record standalone coda sonorants (N'bui'L'...)
            # as their own sung segments. Dropping them (no vowel -> no slot)
            # leaves the decoder with half the real segment count and shifts
            # every boundary late. Model them as syllabic-sonorant slots.
            slots.append(
                FilenameSlot(
                    wav=wav,
                    slot_index=len(slots),
                    token=token,
                    onset="",
                    vowel=token,
                    onset_phones=(),
                    vowel_phone=token,
                    coda_phones=(),
                    phone_start_index=phone_index,
                    vowel_phone_index=phone_index,
                    warnings=("standalone_coda_slot",),
                )
            )
            phone_index += 1
            continue
        if parsed is None:
            fallback = infer_filename_phone_sequence(token)
            if fallback:
                base_slot_index = len(slots)
                fallback_slots = _slots_from_phone_sequence(wav, fallback, language=language, start_phone_index=phone_index)
                slots.extend(
                    FilenameSlot(
                        wav=item.wav,
                        slot_index=base_slot_index + idx,
                        token=item.token,
                        onset=item.onset,
                        vowel=item.vowel,
                        onset_phones=item.onset_phones,
                        vowel_phone=item.vowel_phone,
                        coda_phones=item.coda_phones,
                        phone_start_index=item.phone_start_index,
                        vowel_phone_index=item.vowel_phone_index,
                        warnings=tuple(dict.fromkeys((*item.warnings, f"token_fallback:{token}"))),
                    )
                    for idx, item in enumerate(fallback_slots)
                )
                phone_index += sum(len(item.phones) for item in fallback_slots)
                continue
            continue
        onset, vowel, coda, warnings = parsed
        onset_phones = tuple(_split_onset_phones(onset))
        coda_phones = tuple(_split_onset_phones(coda))
        token_surface = f"{onset}{vowel}" if coda_phones else token
        slot = FilenameSlot(
            wav=wav,
            slot_index=len(slots),
            token=token_surface,
            onset=onset,
            vowel=vowel,
            onset_phones=onset_phones,
            vowel_phone=vowel,
            coda_phones=coda_phones,
            phone_start_index=phone_index,
            vowel_phone_index=phone_index + len(onset_phones),
            warnings=tuple(warnings),
        )
        slots.append(slot)
        phone_index += len(slot.phones)
    return tuple(slots)


def _slots_from_kana_phone_slots(wav: str, phone_slots: Sequence[Sequence[str]]) -> tuple[FilenameSlot, ...]:
    slots: list[FilenameSlot] = []
    phone_index = 0
    for raw_slot in phone_slots:
        phones = tuple(str(phone or "").strip().lower() for phone in raw_slot if str(phone or "").strip())
        if not phones:
            continue
        onset_phones = phones[:-1]
        vowel = phones[-1]
        onset = "".join(onset_phones)
        token = f"{onset}{vowel}" if onset else vowel
        slot = FilenameSlot(
            wav=wav,
            slot_index=len(slots),
            token=token,
            onset=onset,
            vowel=vowel,
            onset_phones=onset_phones,
            vowel_phone=vowel,
            coda_phones=(),
            phone_start_index=phone_index,
            vowel_phone_index=phone_index + len(onset_phones),
            warnings=("kana_filename_slot",),
        )
        slots.append(slot)
        phone_index += len(phones)
    return tuple(slots)


def filename_phone_sequence_from_slots(slots: Sequence[FilenameSlot]) -> tuple[str, ...]:
    phones: list[str] = []
    for slot in slots:
        phones.extend(slot.phones)
    return tuple(phones)


def build_filename_row_plan(
    wav_name: str,
    *,
    language: str = "",
    format_type: str = "cvvc",
    include_transitions: bool | None = None,
) -> tuple[RowPlanRecord, ...]:
    slots = build_filename_slots(wav_name, language=language, format_type=format_type)
    if not slots:
        return ()
    fmt = str(format_type or "").strip().lower()
    if fmt == "vcv":
        return _build_vcv_filename_row_plan(slots)
    if _is_japanese_language(language) and fmt in {"cvvc", "cv-vc"}:
        return _build_ja_cvvc_filename_row_plan(slots)
    if _is_korean_language(language) and fmt in {"cvc", "cvvc", "cv-vc"}:
        return _build_ko_cvc_filename_row_plan(slots)
    transitions = _format_supports_transitions(fmt) if include_transitions is None else bool(include_transitions)
    records: list[RowPlanRecord] = []
    for position, slot in enumerate(slots):
        if slot.onset:
            _append_record(
                records,
                wav=slot.wav,
                alias=slot.token,
                role_family="cv",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=slot.slot_index,
                expected_tokens=(slot.onset, slot.vowel),
                expected_phone_indices=tuple(range(slot.phone_start_index, slot.vowel_phone_index + 1)),
                warnings=slot.warnings,
            )
        else:
            _append_record(
                records,
                wav=slot.wav,
                alias=slot.vowel,
                role_family="v",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=slot.slot_index,
                expected_tokens=(slot.vowel,),
                expected_phone_indices=(slot.vowel_phone_index,),
                warnings=slot.warnings,
            )
        if not transitions:
            continue
        if slot.coda_phones:
            coda_alias = "".join(slot.coda_phones)
            _append_record(
                records,
                wav=slot.wav,
                alias=f"{slot.vowel}{coda_alias}",
                role_family="vc",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=slot.slot_index,
                expected_tokens=(slot.vowel, coda_alias),
                expected_phone_indices=tuple(
                    range(slot.vowel_phone_index, slot.vowel_phone_index + 1 + len(slot.coda_phones))
                ),
                warnings=slot.warnings,
            )
        next_slot = slots[position + 1] if position + 1 < len(slots) else None
        if next_slot is None:
            continue
        if next_slot.onset:
            _append_record(
                records,
                wav=slot.wav,
                alias=f"{slot.vowel} {next_slot.onset}",
                role_family="vc",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=next_slot.slot_index,
                expected_tokens=(slot.vowel, next_slot.onset),
                expected_phone_indices=(slot.vowel_phone_index, *range(next_slot.phone_start_index, next_slot.vowel_phone_index)),
                warnings=tuple(dict.fromkeys((*slot.warnings, *next_slot.warnings))),
            )
        else:
            _append_record(
                records,
                wav=slot.wav,
                alias=f"{slot.vowel} {next_slot.vowel}",
                role_family="vv",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=next_slot.slot_index,
                expected_tokens=(slot.vowel, next_slot.vowel),
                expected_phone_indices=(slot.vowel_phone_index, next_slot.vowel_phone_index),
                warnings=tuple(dict.fromkeys((*slot.warnings, *next_slot.warnings))),
            )
    return tuple(records)


def _build_ja_cvvc_filename_row_plan(slots: Sequence[FilenameSlot]) -> tuple[RowPlanRecord, ...]:
    """Japanese CVVC row plan: HEAD + (VC, CV)* + TAIL pattern."""
    records: list[RowPlanRecord] = []
    first = slots[0]

    # HEAD: "- {first_vowel}" or "- {onset}{vowel}" for CV-initial
    head_alias = f"- {first.vowel}" if not first.onset else f"- {_kana_alias_for_slot(first)}"
    _append_record(
        records,
        wav=first.wav,
        alias=head_alias,
        role_family="cv_head",
        slot_index=first.slot_index,
        left_slot_index=first.slot_index,
        right_slot_index=first.slot_index,
        expected_tokens=("-", *(first.onset_phones if first.onset else ()), first.vowel),
        expected_phone_indices=tuple(range(first.phone_start_index, first.vowel_phone_index + 1)),
        warnings=first.warnings,
    )

    # CV for the first slot (hiragana)
    _append_record(
        records,
        wav=first.wav,
        alias=_kana_alias_for_slot(first),
        role_family="cv" if first.onset else "v",
        slot_index=first.slot_index,
        left_slot_index=first.slot_index,
        right_slot_index=first.slot_index,
        expected_tokens=(*(first.onset_phones if first.onset else ()), first.vowel),
        expected_phone_indices=tuple(range(first.phone_start_index, first.vowel_phone_index + 1)),
        warnings=first.warnings,
    )

    for position in range(len(slots) - 1):
        slot = slots[position]
        next_slot = slots[position + 1]

        # VC: "{prev_vowel} {next_onset}" or VV: "{prev_vowel} {next_vowel}"
        if next_slot.onset:
            vc_onset = _ja_vc_onset(next_slot.onset, next_slot.onset_phones)
            _append_record(
                records,
                wav=slot.wav,
                alias=f"{slot.vowel} {vc_onset}",
                role_family="vc",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=next_slot.slot_index,
                expected_tokens=(slot.vowel, next_slot.onset),
                expected_phone_indices=(
                    slot.vowel_phone_index,
                    *range(next_slot.phone_start_index, next_slot.vowel_phone_index),
                ),
                warnings=tuple(dict.fromkeys((*slot.warnings, *next_slot.warnings))),
            )
        else:
            _append_record(
                records,
                wav=slot.wav,
                alias=f"{slot.vowel} {next_slot.vowel}",
                role_family="vv",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=next_slot.slot_index,
                expected_tokens=(slot.vowel, next_slot.vowel),
                expected_phone_indices=(slot.vowel_phone_index, next_slot.vowel_phone_index),
                warnings=tuple(dict.fromkeys((*slot.warnings, *next_slot.warnings))),
            )

        # CV for next slot (hiragana)
        _append_record(
            records,
            wav=next_slot.wav,
            alias=_kana_alias_for_slot(next_slot),
            role_family="cv" if next_slot.onset else "v",
            slot_index=next_slot.slot_index,
            left_slot_index=next_slot.slot_index,
            right_slot_index=next_slot.slot_index,
            expected_tokens=(*(next_slot.onset_phones if next_slot.onset else ()), next_slot.vowel),
            expected_phone_indices=tuple(range(next_slot.phone_start_index, next_slot.vowel_phone_index + 1)),
            warnings=next_slot.warnings,
        )

    # TAIL: "{last_vowel} R"
    last = slots[-1]
    _append_record(
        records,
        wav=last.wav,
        alias=f"{last.vowel} R",
        role_family="vc_tail",
        slot_index=last.slot_index,
        left_slot_index=last.slot_index,
        right_slot_index=last.slot_index,
        expected_tokens=(last.vowel, "R"),
        expected_phone_indices=(last.vowel_phone_index,),
        warnings=last.warnings,
    )

    return tuple(records)


def _kana_alias_for_slot(slot: FilenameSlot) -> str:
    """Get hiragana alias for a slot, falling back to romaji token."""
    all_phones = (*slot.onset_phones, slot.vowel_phone)
    kana = phones_to_kana(all_phones)
    if kana and kana != "".join(all_phones):
        return kana
    return slot.token


_JA_VC_ONSET_CONSONANTS = {
    "k", "g", "s", "sh", "z", "j", "t", "ch", "ts", "d",
    "n", "h", "f", "b", "p", "m", "y", "r", "w", "v",
    "ky", "gy", "ny", "hy", "by", "py", "my", "ry", "dy", "ty",
    "kw", "gw", "ng",
}


def _ja_vc_onset(onset: str, onset_phones: tuple[str, ...]) -> str:
    """Get the VC-transition onset consonant for Japanese CVVC.

    For geminates (っか = k,k,a), the VC uses a single consonant (k not kk).
    For palatalized via y-glide (chy, shy), the standard cluster is used (ch, sh).
    """
    if len(onset_phones) >= 2 and onset_phones[0] == onset_phones[1]:
        return onset_phones[0]
    if onset in _JA_VC_ONSET_CONSONANTS:
        return onset
    if onset.endswith("y") and onset[:-1] in _JA_VC_ONSET_CONSONANTS:
        return onset[:-1]
    return onset


def _build_ko_cvc_filename_row_plan(slots: Sequence[FilenameSlot]) -> tuple[RowPlanRecord, ...]:
    """Korean CVC/CVVC row plan: HEAD + (VC-space, VC-compact, CV)* + TAIL."""
    records: list[RowPlanRecord] = []
    first = slots[0]
    seen_vc: set[str] = set()

    first_token = f"{first.onset}{first.vowel}" if first.onset else first.vowel

    # HEAD: "- {CV}" and "-{CV}"
    for head_alias in (f"- {first_token}", f"-{first_token}"):
        _append_record(
            records,
            wav=first.wav,
            alias=head_alias,
            role_family="cv_head",
            slot_index=first.slot_index,
            left_slot_index=first.slot_index,
            right_slot_index=first.slot_index,
            expected_tokens=("-", *(first.onset_phones if first.onset else ()), first.vowel),
            expected_phone_indices=tuple(range(first.phone_start_index, first.vowel_phone_index + 1)),
            warnings=first.warnings,
        )

    if first.onset:
        _append_record(
            records,
            wav=first.wav,
            alias=first_token,
            role_family="cv",
            slot_index=first.slot_index,
            left_slot_index=first.slot_index,
            right_slot_index=first.slot_index,
            expected_tokens=(*first.onset_phones, first.vowel),
            expected_phone_indices=tuple(range(first.phone_start_index, first.vowel_phone_index + 1)),
            warnings=first.warnings,
        )

    def _emit_vc_pair(alias_space: str, alias_compact: str, **kwargs: object) -> None:
        for alias in (alias_space, alias_compact):
            if alias in seen_vc:
                continue
            seen_vc.add(alias)
            _append_record(records, alias=alias, **kwargs)

    for position in range(len(slots) - 1):
        slot = slots[position]
        next_slot = slots[position + 1]
        out_vowel = _ko_vc_vowel(slot.vowel)

        # Coda VC from current slot
        if slot.coda_phones:
            coda_str = "".join(slot.coda_phones)
            _emit_vc_pair(
                f"{out_vowel} {coda_str}",
                f"{out_vowel}{coda_str}",
                wav=slot.wav,
                role_family="vc",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=slot.slot_index,
                expected_tokens=(slot.vowel, *slot.coda_phones),
                expected_phone_indices=tuple(
                    range(slot.vowel_phone_index, slot.vowel_phone_index + 1 + len(slot.coda_phones))
                ),
                warnings=slot.warnings,
            )

        # VC transition to next slot
        full_onset = _ko_full_vc_onset(next_slot.onset, next_slot.vowel)
        if full_onset:
            _emit_vc_pair(
                f"{out_vowel} {full_onset}",
                f"{out_vowel}{full_onset}",
                wav=slot.wav,
                role_family="vc",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=next_slot.slot_index,
                expected_tokens=(slot.vowel, next_slot.onset),
                expected_phone_indices=(
                    slot.vowel_phone_index,
                    *range(next_slot.phone_start_index, next_slot.vowel_phone_index),
                ),
                warnings=tuple(dict.fromkeys((*slot.warnings, *next_slot.warnings))),
            )
            # CV for next slot
            next_token = f"{next_slot.onset}{next_slot.vowel}"
            _append_record(
                records,
                wav=next_slot.wav,
                alias=next_token,
                role_family="cv",
                slot_index=next_slot.slot_index,
                left_slot_index=next_slot.slot_index,
                right_slot_index=next_slot.slot_index,
                expected_tokens=(*next_slot.onset_phones, next_slot.vowel),
                expected_phone_indices=tuple(range(next_slot.phone_start_index, next_slot.vowel_phone_index + 1)),
                warnings=next_slot.warnings,
            )
        elif not next_slot.onset and not next_slot.coda_phones:
            _append_record(
                records,
                wav=slot.wav,
                alias=f"{out_vowel} {next_slot.vowel}",
                role_family="vv",
                slot_index=slot.slot_index,
                left_slot_index=slot.slot_index,
                right_slot_index=next_slot.slot_index,
                expected_tokens=(slot.vowel, next_slot.vowel),
                expected_phone_indices=(slot.vowel_phone_index, next_slot.vowel_phone_index),
                warnings=tuple(dict.fromkeys((*slot.warnings, *next_slot.warnings))),
            )

    # Last slot coda
    last = slots[-1]
    if last.coda_phones:
        out_vowel = _ko_vc_vowel(last.vowel)
        coda_str = "".join(last.coda_phones)
        _emit_vc_pair(
            f"{out_vowel} {coda_str}",
            f"{out_vowel}{coda_str}",
            wav=last.wav,
            role_family="vc",
            slot_index=last.slot_index,
            left_slot_index=last.slot_index,
            right_slot_index=last.slot_index,
            expected_tokens=(last.vowel, *last.coda_phones),
            expected_phone_indices=tuple(
                range(last.vowel_phone_index, last.vowel_phone_index + 1 + len(last.coda_phones))
            ),
            warnings=last.warnings,
        )

    # TAIL
    tail_vowel = _ko_vc_vowel(last.vowel)
    _append_record(
        records,
        wav=last.wav,
        alias=f"{tail_vowel} R",
        role_family="vc_tail",
        slot_index=last.slot_index,
        left_slot_index=last.slot_index,
        right_slot_index=last.slot_index,
        expected_tokens=(last.vowel, "R"),
        expected_phone_indices=(last.vowel_phone_index,),
        warnings=last.warnings,
    )

    return tuple(records)


_KO_FORTIS_TO_BASE = {"bb": "b", "dd": "d", "gg": "g", "ss": "s", "jj": "j"}
_KO_GLIDE_VOWELS = {
    "ya": ("y", "a"), "yae": ("y", "ae"), "ye": ("y", "e"),
    "yo": ("y", "o"), "yu": ("y", "u"), "yeo": ("y", "eo"),
    "wa": ("w", "a"), "wae": ("w", "ae"), "we": ("w", "e"),
    "wi": ("w", "i"), "wo": ("w", "o"), "weo": ("w", "eo"),
}


def _ko_vc_onset(onset: str) -> str:
    """Normalize Korean onset for VC transition alias.

    Fortis doubled consonants are reduced to single (bb→b, dd→d, gg→g, ss→s, jj→j).
    Aspirated nasals (mh, nh, ngh) use the plain base.
    """
    if onset in _KO_FORTIS_TO_BASE:
        return _KO_FORTIS_TO_BASE[onset]
    if onset.endswith("h") and onset[:-1] in ("m", "n", "ng"):
        return onset[:-1]
    return onset


def _ko_vc_vowel(vowel: str) -> str:
    """Extract the nucleus from a Korean compound vowel (strip y/w glide)."""
    if vowel in _KO_GLIDE_VOWELS:
        return _KO_GLIDE_VOWELS[vowel][1]
    return vowel


def _ko_full_vc_onset(onset: str, next_vowel: str) -> str:
    """Build the full VC onset including glide from next slot's compound vowel.

    e.g., onset='b' + vowel='ya' → 'by'; onset='' + vowel='wae' → 'w'
    Normalization is applied to the consonant BEFORE adding the glide.
    """
    glide = _KO_GLIDE_VOWELS.get(next_vowel, ("", ""))[0]
    normalized = _ko_vc_onset(onset) if onset else ""
    raw = f"{normalized}{glide}" if normalized or glide else ""
    return raw


def _build_vcv_filename_row_plan(slots: Sequence[FilenameSlot]) -> tuple[RowPlanRecord, ...]:
    records: list[RowPlanRecord] = []
    first = slots[0]
    _append_record(
        records,
        wav=first.wav,
        alias=f"- {first.token}",
        role_family="cv_head",
        slot_index=first.slot_index,
        left_slot_index=first.slot_index,
        right_slot_index=first.slot_index,
        expected_tokens=("-", first.token),
        expected_phone_indices=tuple(range(first.phone_start_index, first.vowel_phone_index + 1)),
        warnings=first.warnings,
    )
    for position, slot in enumerate(slots[:-1]):
        next_slot = slots[position + 1]
        role_family = "vcv" if next_slot.onset_phones else "vv"
        _append_record(
            records,
            wav=slot.wav,
            alias=f"{slot.vowel} {next_slot.token}",
            role_family=role_family,
            slot_index=slot.slot_index,
            left_slot_index=slot.slot_index,
            right_slot_index=next_slot.slot_index,
            expected_tokens=(slot.vowel, next_slot.token),
            expected_phone_indices=(
                slot.vowel_phone_index,
                *range(next_slot.phone_start_index, next_slot.vowel_phone_index + 1),
            ),
            warnings=tuple(dict.fromkeys((*slot.warnings, *next_slot.warnings))),
        )
    return tuple(records)


def row_plan_to_template_rows(records: Sequence[RowPlanRecord]) -> list[OtoTemplateRow]:
    zero = OtoTiming(offset=0.0, consonant=0.0, cutoff=0.0, preutterance=0.0, overlap=0.0)
    return [
        OtoTemplateRow(
            wav=record.wav,
            alias=record.alias,
            timing=zero,
            raw_line=f"{record.wav}={record.alias},0,0,0,0,0",
            expected_phone_indices=record.expected_phone_indices or None,
        )
        for record in records
    ]


def build_filename_template_rows(
    wav_name: str,
    *,
    language: str = "",
    format_type: str = "cvvc",
    include_transitions: bool | None = None,
) -> tuple[list[OtoTemplateRow], tuple[str, ...], tuple[RowPlanRecord, ...]]:
    records = build_filename_row_plan(
        wav_name,
        language=language,
        format_type=format_type,
        include_transitions=include_transitions,
    )
    slots = build_filename_slots(wav_name, language=language, format_type=format_type)
    return row_plan_to_template_rows(records), filename_phone_sequence_from_slots(slots), records


def write_row_plan_jsonl(path: str | Path, records: Sequence[RowPlanRecord]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return len(records)


def _append_record(records: list[RowPlanRecord], **kwargs: object) -> None:
    records.append(RowPlanRecord(row_index=len(records), source_timing_trusted=False, **kwargs))


def _format_supports_transitions(format_type: str) -> bool:
    fmt = str(format_type or "").strip().lower()
    return fmt in {"cvvc", "cvc", "vcv", "vccv", "cv-vc", "full"} or "vc" in fmt


def _slots_from_phone_sequence(
    wav: str,
    phones: Sequence[str],
    *,
    language: str,
    start_phone_index: int = 0,
) -> tuple[FilenameSlot, ...]:
    normalized = [str(phone or "").strip().lower() for phone in phones if str(phone or "").strip()]
    slots: list[FilenameSlot] = []
    pending_onset: list[str] = []
    for local_idx, phone in enumerate(normalized):
        phone_index = start_phone_index + local_idx
        if _is_phone_sequence_slot_vowel(normalized, local_idx, language=language):
            onset = "".join(pending_onset)
            token = f"{onset}{phone}" if onset else phone
            slots.append(
                FilenameSlot(
                    wav=wav,
                    slot_index=len(slots),
                    token=token,
                    onset=onset,
                    vowel=phone,
                    onset_phones=tuple(pending_onset),
                    vowel_phone=phone,
                    coda_phones=(),
                    phone_start_index=phone_index - len(pending_onset),
                    vowel_phone_index=phone_index,
                    warnings=("phone_sequence_fallback",),
                )
            )
            pending_onset = []
        else:
            pending_onset.append(phone)
    return tuple(slots)


def _is_phone_sequence_slot_vowel(phones: Sequence[str], index: int, *, language: str) -> bool:
    phone = str(phones[index] or "").strip().lower()
    if not is_vowel_phone(phone, language):
        return False
    if _is_japanese_language(language) and phone == "n" and index + 1 < len(phones):
        next_phone = str(phones[index + 1] or "").strip().lower()
        if next_phone == "y":
            return False
    return True


def _split_syllable_token(
    token: str,
    *,
    language: str,
    format_type: str,
) -> tuple[str, str, str, tuple[str, ...]] | None:
    text = _normalize_token(token)
    if not text:
        return None
    vowels = _vowel_inventory(language=language, format_type=format_type)
    if text in vowels and is_vowel_phone(text, language):
        return "", text, "", ()
    if _is_korean_language(language) or str(format_type or "").strip().lower() in {"cvc", "cvvc", "cv-vc"}:
        for vowel in sorted(vowels, key=len, reverse=True):
            idx = text.find(vowel)
            if idx < 0:
                continue
            onset = text[:idx]
            coda = text[idx + len(vowel):]
            if not onset and not coda and not is_vowel_phone(vowel, language):
                continue
            warnings: list[str] = []
            if onset and not any(char.isalpha() for char in onset):
                warnings.append(f"non_alpha_onset:{onset}")
            if coda and not any(char.isalpha() for char in coda):
                warnings.append(f"non_alpha_coda:{coda}")
            return onset, vowel, coda, tuple(warnings)
    for vowel in sorted(vowels, key=len, reverse=True):
        if not text.endswith(vowel):
            continue
        onset = text[: -len(vowel)]
        if not onset and not is_vowel_phone(vowel, language):
            continue
        warnings: list[str] = []
        if onset and not any(char.isalpha() for char in onset):
            warnings.append(f"non_alpha_onset:{onset}")
        return onset, vowel, "", tuple(warnings)
    if is_vowel_phone(text, language):
        return "", text, "", ()
    return None


def _vowel_inventory(*, language: str, format_type: str) -> tuple[str, ...]:
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    if _is_japanese_language(language):
        return _JA_VOWELS
    values: list[str] = []
    if _is_korean_language(language) or fmt in {"cvc", "cvvc", "cv-vc"}:
        values.extend(_KOREAN_VOWELS)
    if not _is_korean_language(language):
        values.extend(_JA_VOWELS)
    seen: list[str] = []
    for item in values:
        if item not in seen:
            seen.append(item)
    return tuple(seen)


def _is_japanese_language(language: str) -> bool:
    return str(language or "").strip().lower() in {"ja", "japanese", "jp"}


def _is_korean_language(language: str) -> bool:
    return str(language or "").strip().lower() in {"ko", "korean", "kr"}


def _split_onset_phones(onset: str) -> list[str]:
    text = _normalize_token(onset)
    if not text:
        return []
    out: list[str] = []
    pos = 0
    while pos < len(text):
        cluster = next((item for item in _CONSONANT_CLUSTERS if text.startswith(item, pos)), None)
        if cluster:
            if cluster in {"ky", "gy", "ny", "hy", "by", "py", "my", "ry", "jy", "ty", "dy"}:
                out.extend([cluster[0], "y"])
            else:
                out.append(cluster)
            pos += len(cluster)
            continue
        char = text[pos]
        if char.isalpha():
            out.append(char)
        pos += 1
    return out


def _normalize_token(token: object) -> str:
    text = str(token or "").strip().lower().strip("_-*")
    return "".join(ch for ch in text if ch.isalnum())


__all__ = [
    "FilenameSlot",
    "ROW_PLAN_SCHEMA_VERSION",
    "RowPlanRecord",
    "build_filename_row_plan",
    "build_filename_slots",
    "build_filename_template_rows",
    "filename_phone_sequence_from_slots",
    "row_plan_to_template_rows",
    "write_row_plan_jsonl",
]
