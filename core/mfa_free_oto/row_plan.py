from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from core.model_context.filename import filename_syllable_order_tokens

from .kana import parse_kana_slots
from .manifest_audit import infer_filename_phone_sequence
from .oto_adapter import OtoTemplateRow, OtoTiming
from .types import is_vowel_phone


ROW_PLAN_SCHEMA_VERSION = 1

_IGNORE_TOKENS = {"", "sil", "pau", "sp", "ap", "br", "bre", "breath", "endbr"}
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
