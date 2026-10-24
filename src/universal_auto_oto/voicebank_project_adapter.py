from __future__ import annotations

import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .domain import AliasPlan, PhoneDefinition, PhoneStateDefinition, RecordingSpec, SourceOtoRow
from .generic_alias_binding import bind_pack_aliases
from .kana import parse_kana_slots
from .japanese_cvvc import alias_part_key, special_slots
from .language_pack import LanguagePack, find_language_pack, pack_formats
from .pack_style import NO_STYLE, PackStyle
from .oto_io import read_text_auto, repair_kana_mojibake
from .planning import _kana_surface_slots, _surface_phone_key, recording_tokens
from .project import (
    AlignmentConfig,
    AudioConfig,
    CalibrationConfig,
    GenerationConfig,
    ProjectSpec,
)
from .usability_review import release_vowel_end_overrides, vc_consonant_end_overrides


SUPPORTED_FORMATS = frozenset({"cv", "cvc", "cvvc", "vcv"})
_VOWELS = frozenset(
    {
        "a",
        "ae",
        "e",
        "eo",
        "eu",
        "eui",
        "i",
        "o",
        "oe",
        "u",
        "ui",
        "wa",
        "wae",
        "we",
        "weo",
        "wi",
        "wo",
        "ya",
        "yae",
        "ye",
        "yeo",
        "yo",
        "yu",
    }
)
_VOWEL_NUCLEI = tuple(sorted(_VOWELS, key=lambda value: (-len(value), value)))
_STOP_PHONES = frozenset(
    {
        "b",
        "bb",
        "c",
        "ch",
        "d",
        "dd",
        "g",
        "gg",
        "j",
        "jj",
        "k",
        "kk",
        "p",
        "pp",
        "q",
        "t",
        "tt",
        "ts",
    }
)
_FRICATIVE_PHONES = frozenset({"f", "h", "s", "sh", "ss", "th", "v", "x", "z", "zh"})
_NASAL_PHONES = frozenset({"m", "n", "ng", "ny"})
_LIQUID_PHONES = frozenset({"l", "r", "rr"})
_BREATH_PHONES = frozenset({"br", "breath", "exh", "vf"})
_SILENCE_PHONES = frozenset({"pau", "sil", "sp"})
_PITCH_SUFFIX_RE = re.compile(r"(?:[A-Ga-g](?:#|b)?\d+)$")
_ROMAN_TOKEN_RE = re.compile(r"[A-Za-z]+")
_SPECIAL_VARIANT_RE = re.compile(r"^(br|breath|exh|vf|pau|sil|sp)[_-]?\d+$", re.IGNORECASE)
_TAGGED_PHONE_RE = re.compile(r"^(?P<phone>.+?)(?:#|@)(?P<occurrence>\d{1,6})$")
_KOREAN_CVVC_TAGGED_VARIANT_RE = re.compile(r"^(?P<base>[A-Za-z]+)\((?P<tag>vf|cl)\)$", re.IGNORECASE)
_KOREAN_VCV_TERMINAL_CODAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NGH", ("ng", "h")),
    ("K", ("k",)),
    ("N", ("n",)),
    ("L", ("l",)),
)
_KOREAN_CVVC_LEADING_CODA_TOKENS = frozenset({"n"})
_STANDALONE_SONORANTS = frozenset({"n", "m", "ng", "l"})
_KOREAN_CODAS = frozenset({"k", "t", "p", "n", "m", "ng", "l"})
_KOREAN_CVVC_VOCAL_FRY_CODAS = frozenset({"l", "m", "n", "ng"})


class VoicebankReviewRequired(ValueError):
    """Raised when routed generation cannot establish a trusted phone contract."""

    status = "review_required"

    def __init__(self, reason: str, *, wav: str | None = None, detail: str | None = None):
        self.reason = reason
        self.wav = wav
        self.detail = detail
        message = f"review_required:{reason}"
        if wav:
            message += f":{wav}"
        if detail:
            message += f":{detail}"
        super().__init__(message)

    def to_dict(self) -> dict[str, str | None]:
        return {"status": self.status, "reason": self.reason, "wav": self.wav, "detail": self.detail}


@dataclass(frozen=True)
class OccurrencePhoneToken:
    phone_id: str
    occurrence: int
    occurrence_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "phone_id": self.phone_id,
            "occurrence": self.occurrence,
            "occurrence_id": self.occurrence_id,
        }


@dataclass(frozen=True)
class RoutedProjectBuild:
    project: ProjectSpec
    manifest: Mapping[str, object]
    input_hashes: Mapping[str, object]
    route_id: str
    # Rows left out under unmapped_alias_policy="exclude_row"; never silent.
    excluded_rows: tuple[Mapping[str, object], ...] = ()


UNMAPPED_ALIAS_POLICIES = frozenset({"fail", "exclude_row"})
# "consonant": VC/VC-semivowel rows end inside the next consonant
# (usability_review.vc_consonant_end_overrides); "template" keeps the template.
VC_END_POLICIES = frozenset({"template", "consonant"})
# "vowel_end": '-' releases put the preutterance at the vowel's end
# (usability_review.release_vowel_end_overrides); "template" keeps the template.
RELEASE_PRE_POLICIES = frozenset({"template", "vowel_end"})


@dataclass(frozen=True)
class _PhoneSlot:
    phones: tuple[str, ...]
    start: int
    vowel: int
    end: int


def tag_phone_occurrences(phones: Sequence[str]) -> tuple[OccurrencePhoneToken, ...]:
    """Assign stable, one-based occurrence IDs without changing inventory IDs."""

    counts: Counter[str] = Counter()
    tagged: list[OccurrencePhoneToken] = []
    for raw_phone in phones:
        phone = str(raw_phone).strip().casefold()
        if not phone:
            raise VoicebankReviewRequired("empty_phone_token")
        counts[phone] += 1
        occurrence = counts[phone]
        tagged.append(
            OccurrencePhoneToken(
                phone_id=phone,
                occurrence=occurrence,
                occurrence_id=f"{_safe_token(phone)}#{occurrence:04d}",
            )
        )
    return tuple(tagged)


def parse_filename_phone_slots(
    source: str,
    *,
    format_hint: str | None,
    language_hint: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Parse only explicitly supported filename/reclist formats.

    The parser is deliberately conservative.  Routed generation must stop for
    review instead of silently substituting the legacy slot heuristic.
    """

    format_type = _require_format(format_hint)
    repaired = repair_kana_mojibake(Path(str(source)).stem)
    if repaired:  # a kana file name extracted with the Korean code page ("궇.wav" is "あ.wav")
        source = Path(str(source)).with_name(repaired + Path(str(source)).suffix).as_posix()
    stem = Path(str(source)).stem.strip().strip("_")
    if format_type == 'cvvc' and _is_japanese(language_hint):
        special = special_slots(source)
        if special:
            return tuple(phones for phones, _ in special)
    kana_slots = parse_kana_slots(stem)
    if kana_slots:
        dotted = _kana_slots_with_fry_dots(stem)
        if dotted:
            return dotted
        return tuple(tuple(phone.casefold() for phone in slot) for slot in kana_slots)

    raw_tokens = list(recording_tokens(stem))
    if raw_tokens and _is_pitch_token(raw_tokens[-1]):
        raw_tokens.pop()
    slots: list[tuple[str, ...]] = []
    pending_leading_coda: tuple[str, ...] = ()
    for token_index, raw_token in enumerate(raw_tokens):
        token = _PITCH_SUFFIX_RE.sub("", raw_token.strip()).strip("_'- ").casefold()
        if not token:
            continue
        # Japanese romaji lists write a final breath glued to the vowel (aR, nR).
        if (_is_japanese(language_hint) and len(raw_token) > 1 and raw_token.endswith("R")
                and token_index == len(raw_tokens) - 1):
            before = _parse_roman_slot(token[:-1], format_type=format_type, language_hint=language_hint)
            if before:
                slots.append(before)
                slots.append(("exh",))
                continue
        korean_cvc = format_type == "cvc" and (language_hint or "").casefold() in {"ko", "kr", "korean"}
        surface = _PITCH_SUFFIX_RE.sub("", raw_token.strip()).strip("_'- ")
        if korean_cvc and surface.endswith("Long") and len(surface) > 4:
            # aLong: one held vowel recorded for V, "V V" and "V R"; two slots of the same vowel.
            held = _parse_roman_slot(surface[:-4].casefold(), format_type=format_type, language_hint=language_hint)
            if len(held) == 1 and _is_vowel_id(held[0]):
                slots.extend((held, held))
                continue
        if korean_cvc and len(surface) > 1 and surface.endswith("R"):
            # Korean CVC lists glue the closing breath to each syllable (aR_eR, eunR_eumR).
            before = _parse_roman_slot(surface[:-1].casefold(), format_type=format_type, language_hint=language_hint)
            if before:
                slots.append((*before, "exh"))
                continue
        # Uppercase R denotes ending exhalation, not the consonant r.
        if raw_token == "R" and token_index == len(raw_tokens) - 1:
            if pending_leading_coda:
                slots.append(pending_leading_coda)
                pending_leading_coda = ()
            slots.append(("exh",))
            continue
        if (
            format_type == "cvvc"
            and (language_hint or "").casefold() in {"ko", "kr", "korean"}
            and token in _KOREAN_CVVC_LEADING_CODA_TOKENS
        ):
            if pending_leading_coda:
                slots.append(pending_leading_coda)
            pending_leading_coda = (token,)
            continue
        compact_duplicate_slots = _parse_korean_cvvc_compact_duplicate_slots(
            token,
            format_type=format_type,
            language_hint=language_hint,
        )
        if compact_duplicate_slots:
            if pending_leading_coda:
                compact_duplicate_slots = (
                    (*pending_leading_coda, *compact_duplicate_slots[0]),
                    *compact_duplicate_slots[1:],
                )
                pending_leading_coda = ()
            slots.extend(compact_duplicate_slots)
            continue
        parsed = _parse_korean_vcv_terminal_coda(
            raw_token,
            is_terminal=token_index == len(raw_tokens) - 1,
            format_type=format_type,
            language_hint=language_hint,
        )
        if not parsed:
            parsed = _parse_korean_cvvc_tagged_variant(
                raw_token,
                format_type=format_type,
                language_hint=language_hint,
            )
        if not parsed:
            # Uppercase coda L distinguishes the sung syllable siL from sil.
            if raw_token == "siL" and format_type == "cvvc" and (language_hint or "").casefold() in {"ko", "kr", "korean"}:
                parsed = ("s", "i", "l")
        if not parsed:
            parsed = _parse_roman_slot(token, format_type=format_type, language_hint=language_hint)
        if not parsed:
            raise VoicebankReviewRequired(
                "unparsed_filename_token",
                wav=str(source),
                detail=raw_token,
            )
        if pending_leading_coda:
            parsed = (*pending_leading_coda, *parsed)
            pending_leading_coda = ()
        if (format_type == "cvvc" and (language_hint or "").casefold() in {"ko", "kr", "korean"}
                and raw_token.endswith("H") and parsed[-1] in {"nh", "mh", "ngh", "lh"}):
            parsed = (*parsed[:-1], parsed[-1][:-1], "exh")
        slots.append(parsed)
    if pending_leading_coda:
        slots.append(pending_leading_coda)
    if not slots:
        raise VoicebankReviewRequired("empty_filename_transcript", wav=str(source))
    # In this tagged reclist, apostrophes delimit glottal-stop transitions.
    # Ordinary apostrophe-delimited CVVC filenames remain simple separators.
    if (format_type == "cvvc" and (language_hint or "").casefold() in {"ko", "kr", "korean"}
            and "(cl)" in Path(source).stem.casefold()):
        # The initial vowel also carries the glottal onset ('eu), otherwise
        # its alias incorrectly jumps to the final q+eu occurrence.
        slots = [("q", *slot) for slot in slots]
    return tuple(slots)


def _kana_slots_with_fry_dots(stem: str) -> tuple[tuple[str, ...], ...]:
    """Kana lists write a vocal fry or glottal stop as a middle dot (あ・あ).

    parse_kana_slots treats the dot as a separator; keep it as a ``vf`` slot
    (onset + fry states cover a short glottal closure too).
    """
    normalized = unicodedata.normalize("NFKC", stem)
    if "\u30fb" not in normalized:
        return ()
    slots: list[tuple[str, ...]] = []
    for index, part in enumerate(normalized.split("\u30fb")):
        if index:
            slots.append(("vf",))
        parsed = parse_kana_slots(part) if part.strip("_-' ") else []
        if part.strip("_-' ") and not parsed:
            return ()
        slots.extend(tuple(phone.casefold() for phone in slot) for slot in parsed)
    return tuple(slots)


def _parse_korean_cvvc_compact_duplicate_slots(
    token: str,
    *,
    format_type: str,
    language_hint: str | None,
) -> tuple[tuple[str, ...], ...]:
    """Expand a documented Korean CVVC ``CVCV`` compact token safely.

    A token such as ``ddadda`` is two occurrences of ``dda``. Treating its
    second half as a coda (``dd | a | dda``) silently invents a phone that
    contains another vowel and makes HSMM timing drift by one occurrence per
    token. Keep both CV occurrences so the alias planner can bind CV and VC
    rows to their correct occurrences.

    This is limited to an exact duplicated-CV surface. Other multi-vowel
    compact forms remain on the conservative parser path and require a
    manifest/reclist when ambiguous.
    """

    if format_type != "cvvc" or (language_hint or "").casefold() not in {"ko", "kr", "korean"}:
        return ()
    if len(token) < 2 or len(token) % 2:
        return ()
    half = token[: len(token) // 2]
    if token[len(token) // 2 :] != half:
        return ()
    parsed = _parse_roman_slot(half, format_type=format_type, language_hint=language_hint)
    # A genuine coda must never be discarded merely because a surface repeats.
    if not parsed or len(parsed) > 2 or any(_is_vowel_id(phone) for phone in parsed[2:]):
        return ()
    return (parsed, parsed)


def build_routed_voicebank_project(
    *,
    bank: Path,
    wavs: Sequence[Path],
    base_rows_by_wav: Mapping[str, Sequence[SourceOtoRow]],
    format_hint: str | None,
    language_hint: str | None,
    recursive: bool,
    output_encoding: str,
    evidence_provider: str = "cuda",
    recordings_manifest: Path | None = None,
    unmapped_alias_policy: str = "fail",
    vc_end_policy: str = "template",
    release_pre_policy: str = "template",
    style_profile: PackStyle | str | None = None,
) -> RoutedProjectBuild:
    if unmapped_alias_policy not in UNMAPPED_ALIAS_POLICIES:
        raise ValueError(f"unsupported unmapped_alias_policy: {unmapped_alias_policy}")
    if vc_end_policy not in VC_END_POLICIES:
        raise ValueError(f"unsupported vc_end_policy: {vc_end_policy}")
    if release_pre_policy not in RELEASE_PRE_POLICIES:
        raise ValueError(f"unsupported release_pre_policy: {release_pre_policy}")
    excluded_rows: list[dict[str, object]] = []
    skipped_wavs: list[dict[str, object]] = []
    bank = bank.resolve()
    format_type = _require_format(format_hint)
    language = (language_hint or "mixed").strip().casefold() or "mixed"
    # A language pack (language_packs/*.yaml) replaces the hand-written
    # Korean/Japanese filename and alias rules for its language and format.
    pack = find_language_pack(language, format_type)
    if pack is None and format_type not in SUPPORTED_FORMATS:
        raise VoicebankReviewRequired("no_language_pack", detail=f"{language}/{format_type}")
    # None: the pack's default style; "none": templates only; or a fitted profile.
    style = (pack.style if style_profile is None else None if style_profile == NO_STYLE
             else style_profile) if pack is not None else None
    if isinstance(style, PackStyle) and style.pack_id != pack.pack_id:
        raise VoicebankReviewRequired("style_profile_pack_mismatch", detail=f"{style.pack_id} != {pack.pack_id}")
    artifacts = _discover_contract_artifacts(bank)
    if recordings_manifest is not None:
        manifest_path = recordings_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        artifacts["recordings"] = manifest_path
    explicit_inventory = _load_inventory(artifacts.get("inventory"))
    manifest_records = _load_recording_manifest(artifacts.get("recordings"))
    reclist_entries = _load_reclist(artifacts.get("reclist"))

    inventory: dict[str, PhoneDefinition] = dict(explicit_inventory)
    recordings: list[RecordingSpec] = []
    manifest_recordings: list[dict[str, object]] = []
    template_ids: set[str] = set()
    wav_hashes: dict[str, str] = {}

    wi_as_y_glide = (language in {"ko", "kr", "korean"} and format_type == "cvvc"
                     and _bank_writes_wi_onset_as_y(base_rows_by_wav.values()))
    for wav in wavs:
        resolved_wav = wav.resolve()
        try:
            relative = resolved_wav.relative_to(bank).as_posix()
        except ValueError as exc:
            raise VoicebankReviewRequired("wav_outside_voicebank", wav=str(wav)) from exc
        wav_hashes[relative] = _file_sha256(resolved_wav)
        raw_record = _match_manifest_record(manifest_records, relative)
        source_rows = tuple(base_rows_by_wav.get(relative, ()))
        try:
            phones, slots, transcript_source, transcript_artifact = (_pack_phone_contract(
                pack,
                wav=resolved_wav,
                raw_record=raw_record,
                reclist_entries=reclist_entries,
                explicit_inventory=inventory,
            ) if pack is not None else _recording_phone_contract(
                wav=resolved_wav,
                relative=relative,
                raw_record=raw_record,
                reclist_entries=reclist_entries,
                format_type=format_type,
                language_hint=language,
                explicit_inventory=inventory,
                alias_hints=tuple(row.alias for row in source_rows),
            ))
        except VoicebankReviewRequired as exc:
            if unmapped_alias_policy != "exclude_row":
                raise
            if not source_rows:
                # A stray WAV with an unreadable name and no OTO rows (sample.wav): nothing to generate.
                skipped_wavs.append({"wav": relative, "reason": exc.reason, "detail": exc.detail})
                continue
            excluded_rows.extend(_excluded_row(relative, row, exc.reason, exc.detail) for row in source_rows)
            continue
        inferred_exhalations: tuple[int, ...] = ()
        if (language in {"ko", "kr", "korean"} and format_type == "cvvc"
                and transcript_source == "original_filename_only"):
            phones, slots, inferred_exhalations = _alias_exhalation_slots(slots, source_rows)
        tagged = tag_phone_occurrences(phones)
        for token in tagged:
            if token.phone_id not in inventory:
                if explicit_inventory:
                    raise VoicebankReviewRequired(
                        "inventory_phone_missing",
                        wav=relative,
                        detail=token.phone_id,
                    )
                inventory[token.phone_id] = (pack.phone_definition(token.phone_id) if pack is not None
                                             else _inferred_phone_definition(token.phone_id))

        if raw_record is not None and _manifest_aliases(raw_record):
            aliases = _parse_manifest_aliases(raw_record, len(phones), relative)
            alias_source = "recording_manifest"
            alias_binding_policy = "explicit_manifest_phone_indices"
            base_lines = []
        elif source_rows and pack is not None:
            unmapped = []
            aliases = bind_pack_aliases(pack, source_rows, phones, wav.name,
                                        unmapped=unmapped if unmapped_alias_policy == "exclude_row" else None,
                                        style=style)
            excluded_rows.extend(_excluded_row(relative, row, "unmapped_alias", detail) for row, detail in unmapped)
            if not aliases and unmapped:
                continue
            unmapped_lines = {row.source_index for row, _ in unmapped}
            base_lines = [row.source_index for row in source_rows if row.source_index not in unmapped_lines]
            if len(base_lines) != len(aliases):
                base_lines = []
            alias_source = "base_oto_alias_only"
            alias_binding_policy = f"language_pack:{pack.pack_id}"
        elif pack is not None:
            if unmapped_alias_policy == "exclude_row" and base_rows_by_wav:
                # The bank's oto.ini names no alias for this take; nothing to write.
                continue
            raise VoicebankReviewRequired("language_pack_needs_alias_list", wav=relative, detail=pack.pack_id)
        elif source_rows:
            unmapped: list[tuple[SourceOtoRow, str]] = []
            # Kana saved as cp932 and read as cp949 ("- 궭" = "- ち") binds as the kana; the row is
            # written twice below, with the kana and with the original text.
            garbled = {row.source_index: row.alias for row in source_rows if repair_kana_mojibake(row.alias)}
            binding_rows = tuple(replace(row, alias=repair_kana_mojibake(row.alias)) if row.source_index in garbled
                                 else row for row in source_rows)
            aliases = _planned_aliases(
                binding_rows,
                slots,
                format_type,
                wav.name,
                language_hint=language,
                unmapped=unmapped if unmapped_alias_policy == "exclude_row" else None,
                wi_as_y_glide=wi_as_y_glide,
            )
            excluded_rows.extend(
                _excluded_row(relative, replace(row, alias=garbled.get(row.source_index, row.alias)), "unmapped_alias", detail)
                for row, detail in unmapped)
            if not aliases and unmapped:
                continue
            unmapped_lines = {row.source_index for row, _ in unmapped}
            mapped_lines = [row.source_index for row in binding_rows if row.source_index not in unmapped_lines]
            if len(mapped_lines) == len(aliases):
                expanded, base_lines = [], []
                for line, plan in zip(mapped_lines, aliases):
                    expanded.append(plan)
                    base_lines.append(line)
                    if line in garbled:
                        expanded.append(replace(plan, alias=garbled[line]))
                        base_lines.append(line)
                aliases = tuple(expanded)
            else:
                base_lines = []
            alias_source = "base_oto_alias_only"
            alias_binding_policy = "filename_semantics_v4_coda_exhalation"
        else:
            # Generated aliases already have a known filename slot identity.
            # Re-matching their text would make repeated filename tokens ambiguous.
            surfaces = _kana_surface_slots(wav.stem) or recording_tokens(wav.name)
            if surfaces and _is_pitch_token(surfaces[-1]):
                surfaces = surfaces[:-1]
            generated = []
            for slot_index, slot in enumerate(slots):
                indices, template, role = _alias_mapping("cv", slot_index, slots)
                surface = surfaces[slot_index] if len(surfaces) == len(slots) else "".join(phones[i] for i in indices)
                generated.append(AliasPlan(surface, tuple(indices), template, role))
            aliases = tuple(generated)
            alias_source = "filename_fallback"
            alias_binding_policy = "filename_slot_identity_v1"
            base_lines = []
        if not aliases:
            raise VoicebankReviewRequired("empty_alias_plan", wav=relative)
        if vc_end_policy == "consonant":
            aliases = tuple(_with_overrides(plan, vc_consonant_end_overrides(plan)) for plan in aliases)
        if release_pre_policy == "vowel_end":
            aliases = tuple(_with_overrides(plan, release_vowel_end_overrides(plan)) for plan in aliases)
        template_ids.update(plan.template for plan in aliases)
        review_metadata = _recording_review_metadata(raw_record, bank)
        metadata = {
            "format_hint": format_type,
            "language_hint": language,
            **({"language_pack": pack.pack_id} if pack is not None else {}),
            "phone_tokens": [token.to_dict() for token in tagged],
            "parser_source": transcript_source,
            "parser_artifact": transcript_artifact,
            "alias_source": alias_source,
            "alias_binding_policy": alias_binding_policy,
            "alias_inferred_exhalation_indices": list(inferred_exhalations),
            "phone_slots": [{"start": slot.start, "vowel": slot.vowel, "end": slot.end} for slot in slots],
            "base_oto_timing_reused": False,
            # Line of the base oto.ini each alias plan came from (row_report: recording_metadata).
            "alias_base_lines": list(base_lines),
            **review_metadata,
        }
        recording = RecordingSpec(
            wav=relative,
            phones=tuple(token.phone_id for token in tagged),
            aliases=aliases,
            transcript_source=transcript_source,
            label_source="none",
            metadata=metadata,
        )
        recordings.append(recording)
        manifest_recordings.append(
            {
                "wav": relative,
                "wav_sha256": wav_hashes[relative],
                "transcript_source": transcript_source,
                "transcript_artifact": transcript_artifact,
                "alias_source": alias_source,
                "alias_binding_policy": alias_binding_policy,
                "alias_inferred_exhalation_indices": list(inferred_exhalations),
                "phone_tokens": [token.to_dict() for token in tagged],
                "aliases": [
                    {
                        "alias": plan.alias,
                        "phones": list(plan.phone_indices),
                        "phone_occurrence_ids": [tagged[i].occurrence_id for i in plan.phone_indices],
                        "template": plan.template,
                        "role": plan.role,
                        **({"group_starts": list(plan.group_starts)} if plan.group_starts else {}),
                        **({"anchor_shifts": dict(plan.anchor_shifts)} if plan.anchor_shifts else {}),
                    }
                    for plan in aliases
                ],
            }
        )

    if not recordings:
        raise VoicebankReviewRequired("empty_recording_manifest")
    alignment_backend, feature_backend, route_id = _route_alignment(
        language_hint=language,
        format_hint=format_type,
        recordings=recordings,
    )
    audio = AudioConfig(
        recursive=recursive,
        convert_to_mono=True,
        internal_sample_rate=44100,
        resampler="soxr_hq",
    )
    alignment = AlignmentConfig(
        backend=alignment_backend,
        device="cpu",
        model=None,
        feature_backend=feature_backend,
        evidence_provider=evidence_provider,
        use_ctc_prior=False,
        use_hsmm=True,
        use_prototypes=False,
    )
    generation = GenerationConfig(
        templates=tuple(sorted(template_ids)),
        output="generated/oto.ini",
        output_layout="preserve_source_directories" if recursive else "single_root",
        output_encoding=output_encoding,
        atomic_write=True,
        backup_existing=True,
        write_debug_timeline=True,
        write_textgrid=False,
        # CV banks get one oto.ini: no risk model separates their rows (2026-10-24).
        split_review=format_type != "cv",
    )
    calibration = CalibrationConfig(enabled=False, profile=None)
    inventory_sha256 = _semantic_sha256(_inventory_semantics(inventory))
    recordings_sha256 = _semantic_sha256(_recording_semantics(recordings))
    project_sha256 = _semantic_sha256(
        {
            "format_version": 1,
            "name": f"Routed voicebank {bank.name}",
            "language_hint": language,
            "audio": audio.__dict__,
            "alignment": alignment.__dict__,
            "generation": generation.__dict__,
            "calibration": calibration.__dict__,
            "inventory_sha256": inventory_sha256,
            "recordings_sha256": recordings_sha256,
        }
    )
    project = ProjectSpec(
        format_version=1,
        path=bank / ".auto_oto_routed_project.yaml",
        root=bank,
        name=f"Routed voicebank {bank.name}",
        language_hint=language,
        inventory_file=_artifact_label(artifacts.get("inventory"), bank, "derived_inventory"),
        recordings_file=_artifact_label(artifacts.get("recordings"), bank, "derived_recordings"),
        inventory=inventory,
        recordings=tuple(recordings),
        audio=audio,
        alignment=alignment,
        generation=generation,
        calibration=calibration,
        project_sha256=project_sha256,
        inventory_sha256=inventory_sha256,
        recordings_sha256=recordings_sha256,
    )
    artifact_hashes = {
        name: {"path": str(path), "sha256": _file_sha256(path)}
        for name, path in sorted(artifacts.items())
        if path is not None
    }
    input_hashes: dict[str, object] = {
        "wavs": dict(sorted(wav_hashes.items())),
        "contract_artifacts": artifact_hashes,
        "project_sha256": project_sha256,
        "inventory_sha256": inventory_sha256,
        "recordings_sha256": recordings_sha256,
    }
    manifest = {
        "schema_version": 2,
        "kind": "routed_voicebank_project",
        "bank": str(bank),
        "format_hint": format_type,
        "language_hint": language,
        **({"language_pack": {"id": pack.pack_id, "sha256": pack.sha256,
                              "style": {"source": style.source, "sha256": style.sha256} if style else None}}
           if pack is not None else {}),
        "route_id": route_id,
        "alignment_backend": alignment_backend,
        "feature_backend": feature_backend,
        "alignment_model": None,
        "alignment_model_sha256": None,
        "project_sha256": project_sha256,
        "inventory_sha256": inventory_sha256,
        "recordings_sha256": recordings_sha256,
        "input_hashes": input_hashes,
        "base_oto_timing_reused": False,
        "recordings": manifest_recordings,
    }
    if vc_end_policy != "template":
        manifest["vc_end_policy"] = vc_end_policy
    if release_pre_policy != "template":
        manifest["release_pre_policy"] = release_pre_policy
    if excluded_rows:
        manifest["unmapped_alias_policy"] = unmapped_alias_policy
        manifest["excluded_rows"] = excluded_rows
        manifest["skipped_wavs_without_rows"] = skipped_wavs
    return RoutedProjectBuild(project=project, manifest=manifest, input_hashes=input_hashes, route_id=route_id,
                              excluded_rows=tuple(excluded_rows))


def _with_overrides(plan: AliasPlan, changes: Mapping[str, str]) -> AliasPlan:
    """Add policy expressions; anchors the plan already overrides keep priority."""
    return replace(plan, overrides={**changes, **plan.overrides}) if changes else plan


def _excluded_row(wav: str, row: SourceOtoRow, reason: str, detail: str | None) -> dict[str, object]:
    return {"wav": wav, "alias": row.alias, "source_index": row.source_index, "reason": reason,
            "detail": detail, "written_to_oto": False}


def _require_format(format_hint: str | None) -> str:
    value = (format_hint or "").strip().casefold()
    if value not in SUPPORTED_FORMATS | pack_formats():
        raise VoicebankReviewRequired(
            "unknown_parse_format",
            detail=value or "missing_format_hint",
        )
    return value


def _parse_roman_slot(token: str, *, format_type: str, language_hint: str | None) -> tuple[str, ...]:
    token = _normalize_special_variant(token)
    if token in _BREATH_PHONES or token in _SILENCE_PHONES:
        return (token,)
    if token in _STANDALONE_SONORANTS and (language_hint or "").casefold() in {"ko", "kr", "korean"}:
        return (token,)
    # Japanese romaji moraic nasal (N, n) is a syllable of its own.
    if token == "n" and _is_japanese(language_hint):
        return ("n",)
    if not _ROMAN_TOKEN_RE.fullmatch(token):
        return ()
    candidates: list[tuple[int, str]] = []
    for start in range(len(token)):
        for vowel in _VOWEL_NUCLEI:
            if token.startswith(vowel, start):
                candidates.append((start, vowel))
    if not candidates:
        return ()
    start, vowel = min(candidates, key=lambda item: (item[0], -len(item[1]), item[1]))
    onset = token[:start]
    coda = token[start + len(vowel) :]
    if (_is_japanese(language_hint) and not onset and len(vowel) == 2 and vowel[0] in "wy"
            and vowel[1] in "aiueo"):
        # Japanese wa/ya are glide + vowel like kana わ/や, so "a w" and "- y" have a consonant to bind.
        onset, vowel = vowel[0], vowel[1]
    standalone_korean_vc = (
        format_type == "cv" and not onset and coda in _KOREAN_CODAS
        and (language_hint or "").casefold() in {"ko", "kr", "korean"}
    )
    if format_type in {"cv", "vcv"} and coda and not standalone_korean_vc:
        return ()
    # A coda that still contains a vowel nucleus (kaki -> k+a+"ki") is a
    # second syllable, not a coda; leave it for review or a manifest.
    if any(nucleus in coda for nucleus in _VOWEL_NUCLEI):
        return ()
    phones = tuple(value for value in (onset, vowel, coda) if value)
    if not phones or vowel not in phones:
        return ()
    return phones


def _is_japanese(language_hint: str | None) -> bool:
    return (language_hint or "").casefold() in {"ja", "jp", "japanese"}


def _normalize_special_variant(token: str) -> str:
    """Map numbered breath/silence filename variants to canonical phones.

    Voicebanks commonly use names such as ``br1`` or ``sil2`` for multiple
    recordings of one non-vocal phone. The number identifies the sample, not a
    distinct phone in the routed project contract. Other unknown numeric
    filename tokens remain rejected by the conservative parser.
    """

    value = str(token).strip().casefold()
    match = _SPECIAL_VARIANT_RE.fullmatch(value)
    if match:
        return match.group(1).casefold()
    return value


def _parse_korean_vcv_terminal_coda(
    raw_token: str,
    *,
    is_terminal: bool,
    format_type: str,
    language_hint: str | None,
) -> tuple[str, ...]:
    """Parse documented DiKOR terminal-coda markers without broad guessing.

    These markers are case-sensitive filename notation, so parsing must happen
    before the normal case-folded roman-slot path.  The supported forms are
    intentionally restricted to forms observed in the release intake.
    """

    if not is_terminal or format_type != "vcv" or (language_hint or "").casefold() not in {"ko", "kr", "korean"}:
        return ()
    surface = _PITCH_SUFFIX_RE.sub("", raw_token.strip()).strip("_'- ")
    for suffix, coda_phones in _KOREAN_VCV_TERMINAL_CODAS:
        if not surface.endswith(suffix) or len(surface) == len(suffix):
            continue
        base = surface[: -len(suffix)]
        parsed_base = _parse_roman_slot(
            base.casefold(),
            format_type=format_type,
            language_hint=language_hint,
        )
        if parsed_base:
            return (*parsed_base, *coda_phones)
    return ()


def _parse_korean_cvvc_tagged_variant(
    raw_token: str,
    *,
    format_type: str,
    language_hint: str | None,
) -> tuple[str, ...]:
    """Parse the narrowly documented Korean CVVC ``(vf)``/``(cl)`` tags.

    The confirmed reclist exposes both fry onset and fry ending aliases.
    The cl tag is handled across slots by the filename parser. These are
    inferred filename contracts, never edits to a supplied phone manifest.
    """

    if format_type != "cvvc" or (language_hint or "").casefold() not in {"ko", "kr", "korean"}:
        return ()
    match = _KOREAN_CVVC_TAGGED_VARIANT_RE.fullmatch(raw_token.strip())
    if match is None:
        return ()
    base = match.group("base").casefold()
    tag = match.group("tag").casefold()
    parsed_base = _parse_roman_slot(base, format_type=format_type, language_hint=language_hint)
    if tag == "cl":
        return parsed_base
    if parsed_base:
        return ("vf", *parsed_base, "vf")
    if base in _KOREAN_CVVC_VOCAL_FRY_CODAS:
        return ("vf", base, "vf")
    return ()


def _is_pitch_token(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Ga-g](?:#|b)?\d+", value.strip()))


def _pack_phone_contract(
    pack: LanguagePack,
    *,
    wav: Path,
    raw_record: Mapping[str, object] | None,
    reclist_entries: Mapping[str, tuple[str, str]],
    explicit_inventory: Mapping[str, PhoneDefinition],
) -> tuple[tuple[str, ...], tuple[_PhoneSlot, ...], str, str | None]:
    """Phones and syllable slots from a manifest, reclist line or file name, read with the pack."""
    if raw_record is not None:
        raw_phones = raw_record.get("phones")
        if isinstance(raw_phones, Sequence) and not isinstance(raw_phones, (str, bytes)) and raw_phones:
            phones = tuple(_untag_phone(str(value), explicit_inventory) for value in raw_phones)
            slots = _manifest_slots(raw_record, phones, {**{p: pack.phone_definition(p) for p in phones
                                                            if p in pack.phones}, **explicit_inventory})
            return phones, slots, "voicebank_recording_manifest", "recordings"
    reclist = reclist_entries.get(_normalized_recording_key(wav.name))
    source, transcript_source, artifact = wav.name, "original_filename_only", None
    if reclist is not None:
        source, artifact = reclist
        transcript_source = "voicebank_reclist"
    chunks = pack.parse_filename(source)
    if chunks is None:
        raise VoicebankReviewRequired("unparsed_filename_token", wav=str(wav), detail=pack.pack_id)
    phones: list[str] = []
    slots: list[_PhoneSlot] = []
    for chunk in chunks:
        nuclei = [i for i, phone in enumerate(chunk) if pack.is_nucleus(phone)]
        if not nuclei:
            if not slots:
                raise VoicebankReviewRequired("ambiguous_slot_vowel", wav=str(wav), detail=" ".join(chunk))
            # A consonant-only chunk closes the previous syllable.
            last = slots.pop()
            slots.append(_PhoneSlot((*last.phones, *chunk), last.start, last.vowel, last.end + len(chunk)))
            phones.extend(chunk)
            continue
        base = len(phones)
        # Consonants before a nucleus are its onset; the last nucleus keeps the coda.
        for k, nucleus in enumerate(nuclei):
            start = nuclei[k - 1] + 1 if k else 0
            end = nucleus if k + 1 < len(nuclei) else len(chunk) - 1
            slots.append(_PhoneSlot(tuple(chunk[start:end + 1]), base + start, base + nucleus, base + end))
        phones.extend(chunk)
    return tuple(phones), tuple(slots), transcript_source, artifact


def _recording_phone_contract(
    *,
    wav: Path,
    relative: str,
    raw_record: Mapping[str, object] | None,
    reclist_entries: Mapping[str, tuple[str, str]],
    format_type: str,
    language_hint: str,
    explicit_inventory: Mapping[str, PhoneDefinition],
    alias_hints: Sequence[str] = (),
) -> tuple[tuple[str, ...], tuple[_PhoneSlot, ...], str, str | None]:
    if raw_record is not None:
        raw_phones = raw_record.get("phones")
        if isinstance(raw_phones, Sequence) and not isinstance(raw_phones, (str, bytes)) and raw_phones:
            phones = tuple(_untag_phone(str(value), explicit_inventory) for value in raw_phones)
            slots = _manifest_slots(raw_record, phones, explicit_inventory)
            return phones, slots, "voicebank_recording_manifest", "recordings"
    reclist = reclist_entries.get(_normalized_recording_key(wav.name))
    if reclist is not None:
        text, artifact = reclist
        slot_phones = parse_filename_phone_slots(
            text,
            format_hint=format_type,
            language_hint=language_hint,
        )
        phones, slots = _flatten_slots(slot_phones)
        return phones, slots, "voicebank_reclist", artifact
    try:
        slot_phones = parse_filename_phone_slots(
            wav.name,
            format_hint=format_type,
            language_hint=language_hint,
        )
    except VoicebankReviewRequired:
        compact = (_segment_korean_vcv_compact(wav.name, alias_hints)
                   if format_type == "vcv" and language_hint in {"ko", "kr", "korean"} and alias_hints else ())
        if not compact:
            raise
        phones, slots = _flatten_slots(compact)
        return phones, slots, "korean_vcv_compact_segmentation", None
    phones, slots = _flatten_slots(slot_phones)
    special = special_slots(wav.name) if format_type == 'cvvc' and _is_japanese(language_hint) else ()
    if special and special[0][1].endswith('L'):
        from .japanese_timing import long_phrase_layout
        try:
            layout = long_phrase_layout(wav)
        except (RuntimeError, OSError):
            layout = 'unresolved'
        if layout == 'continuous':
            phones, slots = _flatten_slots((slot_phones[0], slot_phones[-1]))
            return phones, slots, 'japanese_continuous_long_acoustic_layout_v1', None
    return phones, slots, "original_filename_only", None


_KOREAN_VCV_COMPACT_ONSETS = ("bb", "dd", "gg", "jj", "ss", "pp", "tt", "kk", "ch", "th", "sh", "b", "d", "g", "j", "s",
                              "p", "t", "k", "h", "m", "n", "r", "l", "f", "v", "z", "")
_KOREAN_VCV_COMPACT_VOWELS = ("yeo", "weo", "yae", "wae", "eui", "ae", "eo", "eu", "oe", "ui", "ya", "ye", "yo", "yu",
                              "wa", "we", "wi", "wo", "a", "e", "i", "o", "u")
# Upper-case filename codas; H/R are the closing breath.
_KOREAN_VCV_COMPACT_CODAS = ("NG", "K", "T", "P", "N", "M", "L", "H", "R")
_KOREAN_VCV_NUCLEUS = {"ya": "a", "yeo": "eo", "yo": "o", "yu": "u", "ye": "e", "yae": "ae", "wa": "a", "weo": "eo",
                       "we": "e", "wi": "i", "wo": "o", "wae": "ae", "ui": "i", "eui": "i", "oe": "e"}
_KOREAN_TENSE_RESPELLING = {"pp": "bb", "tt": "dd", "kk": "gg"}


def _segment_korean_vcv_compact(source: str, alias_hints: Sequence[str] = ()) -> tuple[tuple[str, ...], ...]:
    """Segment run-together Korean VCV names (``_gagageogage…K``).

    Units are ``[onset] vowel`` or an upper-case coda/breath; separators (' _ + - space) only
    bound units. With the bank's aliases, the lattice keeps the segmentation whose adjacent-unit
    VCV aliases (``a geo``, ``- ga``, ``a k``, ``l ba``) explain most of them, then the fewest
    units, and returns () when none is explained. Without aliases (alias generation), the list
    structure decides: VCV lists repeat one base syllable (ga ga geo ga ge …), so the
    segmentation whose most common unit occurs most often wins, then the fewest distinct units,
    then the fewest units.
    """

    stem = _PITCH_SUFFIX_RE.sub("", Path(str(source)).stem.strip())
    tokens = [token for token in re.split(r"[_'+\- ]+", stem) if token]
    if not tokens:
        return ()
    hints = {_PITCH_SUFFIX_RE.sub("", " ".join(str(alias).split())).casefold() for alias in alias_hints}
    units = _korean_vcv_units_by_aliases(tokens, hints) if hints else _korean_vcv_units_by_structure(tokens)
    return _korean_vcv_units_to_slots(units) if units else ()


def _korean_vcv_token_units(token: str, start: int) -> list[tuple[int, tuple[str, str]]]:
    out = [(start + len(coda), ("", coda)) for coda in _KOREAN_VCV_COMPACT_CODAS if token.startswith(coda, start)]
    for onset in _KOREAN_VCV_COMPACT_ONSETS:
        if not token.startswith(onset, start):
            continue
        j = start + len(onset)
        out += [(j + len(vowel), (onset, vowel)) for vowel in _KOREAN_VCV_COMPACT_VOWELS if token.startswith(vowel, j)]
    return out


def _korean_vcv_units_by_aliases(tokens: Sequence[str], hints: set[str]) -> tuple[tuple[str, str], ...]:
    def left(unit: tuple[str, str]) -> str:
        onset, vowel = unit
        return vowel.casefold() if not onset and vowel.isupper() else _KOREAN_VCV_NUCLEUS.get(vowel, vowel)

    def right(unit: tuple[str, str]) -> str:
        return "".join(unit).casefold()

    # Lattice over the whole file: state = last unit; score = explained aliases, then fewer units.
    best: dict[tuple[str, str] | None, tuple[float, tuple[tuple[str, str], ...]]] = {None: (0.0, ())}
    for token in tokens:
        frontier = {(0, key): value for key, value in best.items()}
        finished: dict[tuple[str, str] | None, tuple[float, tuple[tuple[str, str], ...]]] = {}
        while frontier:
            (pos, last), (score, path) = min(frontier.items(), key=lambda item: item[0][0])
            del frontier[(pos, last)]
            if pos == len(token):
                if last not in finished or score > finished[last][0]:
                    finished[last] = (score, path)
                continue
            for end, unit in _korean_vcv_token_units(token, pos):
                if last is None:
                    gain = float(f"- {right(unit)}" in hints) if unit[1].islower() else 0.0
                else:
                    gain = float(f"{left(last)} {right(unit)}" in hints)
                candidate = (score + gain - 0.001, (*path, unit))
                key = (end, unit)
                if key not in frontier or candidate[0] > frontier[key][0]:
                    frontier[key] = candidate
        if not finished:
            return ()
        best = finished
    score, units = max(best.values(), key=lambda item: item[0])
    return units if score > 0 else ()


def _korean_vcv_units_by_structure(tokens: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Alias-free segmentation: VCV lists repeat their first (base) syllable.

    For each candidate base unit, an exact DP picks the segmentation with the most units equal
    to it, minus a penalty for each bare vowel glued to a preceding syllable, then the fewest
    units; the best first unit wins. ``bbeobbeobbe…`` stays bbeo bbeo bbe rather than
    bbe o bbe o, ``aaeoae…`` stays a a eo a e (5 a's, fewer split vowels).
    """
    # Base candidates: the sung unit opening any token (L'aL'eo… opens with a coda mark).
    firsts = {unit for token in tokens for _end, unit in _korean_vcv_token_units(token, 0) if unit[1].islower()}
    best: tuple[float, tuple[tuple[str, str], ...]] | None = None
    for first in sorted(firsts):
        state: tuple[float, tuple[tuple[str, str], ...]] | None = (0.0, ())
        for token in tokens:
            table: dict[int, tuple[float, tuple[tuple[str, str], ...]]] = {0: state}
            for pos in range(len(token)):
                if pos not in table:
                    continue
                score, path = table[pos]
                for end, unit in _korean_vcv_token_units(token, pos):
                    # A bare vowel glued to a consonant-initial syllable is almost always a longer vowel
                    # split apart (bbe+o for bbeo, bbe+u for bbeu); separators (+o) are exempt.
                    split_vowel = pos > 0 and not unit[0] and unit[1].islower() and bool(path and path[-1][0])
                    candidate = (score + (1.0 if unit == first else 0.0) - (2.0 if split_vowel else 0.0) - 0.001,
                                 (*path, unit))
                    if end not in table or candidate[0] > table[end][0]:
                        table[end] = candidate
            state = table.get(len(token))
            if state is None:
                break
        if state is not None and state[1] and (best is None or state[0] > best[0]):
            best = state
    return best[1] if best is not None else ()


def _korean_vcv_units_to_slots(units: Sequence[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    # A trailing '-' only marks the closing release; it has no phone of its own.
    slots: list[tuple[str, ...]] = []
    for index, (onset, vowel) in enumerate(units):
        if vowel.islower():
            slots.append(tuple(value for value in (onset, vowel) if value))
            continue
        coda = vowel.casefold()
        terminal = all(not unit[1].islower() for unit in units[index:])
        if coda == "r" and index == len(units) - 1:
            slots.append(("exh",))
        elif slots and (terminal or coda not in _STANDALONE_SONORANTS):
            # Final codas (gaK, aNGH) and mid-file obstruents close the preceding syllable.
            slots[-1] = (*slots[-1], coda)
        else:
            # Mid-file sonorant codas (weoN'bbeo, L'ba) are syllabic slots of their own.
            slots.append((coda,))
    return tuple(slots)


def _untag_phone(value: str, inventory: Mapping[str, PhoneDefinition]) -> str:
    phone = value.strip().casefold()
    match = _TAGGED_PHONE_RE.fullmatch(phone)
    if match and match.group("phone") in inventory:
        return match.group("phone")
    return phone


def _flatten_slots(slot_phones: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], tuple[_PhoneSlot, ...]]:
    phones: list[str] = []
    slots: list[_PhoneSlot] = []
    for raw_slot in slot_phones:
        normalized = tuple(str(phone).strip().casefold() for phone in raw_slot if str(phone).strip())
        if not normalized:
            continue
        vowel_positions = [index for index, phone in enumerate(normalized) if _is_vowel_id(phone)]
        if len(vowel_positions) != 1:
            if len(normalized) == 1 and normalized[0] in (_BREATH_PHONES | _SILENCE_PHONES | _STANDALONE_SONORANTS | {'s', 'sh'}):
                # This is a slot endpoint, not a change to the phone's acoustic class.
                vowel_position = 0
            elif len(normalized) == 3 and normalized[0] == normalized[2] == "vf" and normalized[1] in _KOREAN_CVVC_VOCAL_FRY_CODAS:
                # The coda is the central sustained segment in this tagged
                # fry-onset/coda/fry-ending recording, despite not being a vowel.
                vowel_position = 1
            elif len(normalized) == 2 and normalized[1] == "vf" and normalized[0] in _KOREAN_CVVC_VOCAL_FRY_CODAS:
                vowel_position = 1
            elif len(normalized) == 2 and normalized[1] == "exh" and normalized[0] in _STANDALONE_SONORANTS:
                vowel_position = 0
            elif len(normalized) == 2 and normalized[0] == 'n' and normalized[1] in {'n', 'm', 'ng'}:
                vowel_position = 0
            else:
                raise VoicebankReviewRequired("ambiguous_slot_vowel", detail=" ".join(normalized))
        else:
            vowel_position = vowel_positions[0]
        start = len(phones)
        phones.extend(normalized)
        slots.append(
            _PhoneSlot(
                phones=normalized,
                start=start,
                vowel=start + vowel_position,
                end=len(phones) - 1,
            )
        )
    if not phones or not slots:
        raise VoicebankReviewRequired("empty_phone_sequence")
    return tuple(phones), tuple(slots)


def _manifest_slots(
    record: Mapping[str, object],
    phones: Sequence[str],
    inventory: Mapping[str, PhoneDefinition],
) -> tuple[_PhoneSlot, ...]:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        raw_slots = metadata.get("phone_slots")
        if isinstance(raw_slots, Sequence) and not isinstance(raw_slots, (str, bytes)) and raw_slots:
            slots: list[_PhoneSlot] = []
            for raw_slot in raw_slots:
                if not isinstance(raw_slot, Mapping):
                    raise VoicebankReviewRequired("invalid_manifest_phone_slots")
                try:
                    start = int(raw_slot["start"])
                    vowel = int(raw_slot["vowel"])
                    end = int(raw_slot.get("end", vowel))
                except (KeyError, TypeError, ValueError) as exc:
                    raise VoicebankReviewRequired("invalid_manifest_phone_slots") from exc
                if not (0 <= start <= vowel <= end < len(phones)):
                    raise VoicebankReviewRequired("invalid_manifest_phone_slots")
                slots.append(_PhoneSlot(tuple(phones[start : end + 1]), start, vowel, end))
            return tuple(slots)
    vowels = [
        index
        for index, phone in enumerate(phones)
        if _is_vowel_definition(inventory.get(phone)) or _is_vowel_id(phone)
    ]
    if not vowels:
        raise VoicebankReviewRequired("manifest_vowel_not_found")
    slots = []
    previous_vowel = -1
    for vowel in vowels:
        start = previous_vowel + 1
        slots.append(_PhoneSlot(tuple(phones[start : vowel + 1]), start, vowel, vowel))
        previous_vowel = vowel
    return tuple(slots)


def _recording_review_metadata(raw_record: Mapping[str, object] | None, bank: Path) -> dict[str, str]:
    raw_metadata: Mapping[str, object] = {}
    if raw_record is not None:
        candidate = raw_record.get("metadata")
        if isinstance(candidate, Mapping):
            raw_metadata = candidate
    raw_split = raw_metadata.get("evaluation_split") or raw_metadata.get("split")
    evaluation_split = str(raw_split).strip().casefold() if raw_split is not None else "train"
    if evaluation_split not in {"train", "validation", "locked_test"}:
        raise VoicebankReviewRequired("invalid_evaluation_split", detail=str(raw_split))
    raw_bank_id = raw_metadata.get("bank_id")
    bank_id = str(raw_bank_id).strip() if raw_bank_id is not None else bank.name.strip()
    if not bank_id:
        raise VoicebankReviewRequired("empty_bank_id", detail=str(bank))
    raw_root_speaker_id = raw_metadata.get("root_speaker_id")
    root_speaker_id = str(raw_root_speaker_id).strip() if raw_root_speaker_id is not None else ""
    return {
        "split": evaluation_split,
        "bank_id": bank_id,
        "root_speaker_id": root_speaker_id,
    }


def _alias_surface(alias: str, *, preserve_coda_case: bool = False, japanese: bool = False) -> tuple[tuple[str, ...], int | None]:
    """Separate style/pitch from explicit occurrence without swallowing ``ka2``.

    Pitch suffixes must use uppercase note names; a bare trailing integer is
    an occurrence ordinal. #N/@N are also accepted and are unambiguous.
    """
    value = alias.strip()
    # A kanji glued to the end names the take's variant ("a R夜", "a あ昼"); 息/吸 are breath marks.
    value = re.sub(r"(?<=\S)[一-鿿]+$", lambda m: m[0] if set(m[0]) & {"息", "吸"} else "", value)
    while True:
        stripped = re.sub(r"(?:_[A-Za-z]+|_?[A-G](?:#|b)?\d+[A-Za-z]*)$", "", value).strip()
        if stripped == value:
            break
        value = stripped
    if len(value) > 1 and value.endswith("V"):
        value = value[:-1].rstrip()
    match = re.search(r"(?:[#@]?)(\d+)$", value)
    occurrence = int(match[1]) if match else None
    if match:
        value = value[:match.start()].rstrip()
    if value.startswith("-"):
        value = "- " + value[1:].lstrip()
    value = re.sub(r"([*'\u30fb])", r" \1 ", unicodedata.normalize("NFKC", value))
    def part_key(part: str) -> str:
        if part in {"-", "R", "*", "'", "\u30fb"} or (preserve_coda_case and part in {"K", "T", "P", "N", "M", "NG", "L"}):
            return part
        if japanese:
            return alias_part_key(part) or ('H' if part == 'H' else _surface_phone_key(part))
        return _surface_phone_key(part)

    parts = tuple(part_key(part) for part in value.split())
    return parts, occurrence


def _alias_exhalation_slots(slots: Sequence[_PhoneSlot], rows: Sequence[SourceOtoRow]):
    """Use confirmed R notation to include breath absent from filename spelling.

    Only the filename-derived Korean CVVC builder calls this. Explicit phone
    manifests and reviewed phone sequences are never changed by this inference.
    """
    targets = set()
    phones = {s.start+i: p for s in slots for i, p in enumerate(s.phones)}
    for row in rows:
        key, ordinal = _alias_surface(row.alias)
        if len(key) != 2 or key[-1] != "R":
            continue
        matches = [i for i, p in phones.items() if p == key[0]]
        index = (ordinal-1) if ordinal is not None else len(matches)-1
        if 0 <= index < len(matches):
            target = matches[index]
            if phones.get(target+1) != "exh":
                targets.add(target)
    expanded, inserted = [], []
    count = 0
    for slot in slots:
        values = []
        for i, phone in enumerate(slot.phones):
            values.append(phone)
            count += 1
            if slot.start+i in targets:
                values.append("exh")
                inserted.append(count)
                count += 1
        expanded.append(tuple(values))
    result_phones, result_slots = _flatten_slots(expanded)
    return result_phones, result_slots, tuple(inserted)


def _planned_aliases(
    rows: Sequence[SourceOtoRow],
    slots: Sequence[_PhoneSlot],
    format_type: str,
    wav_name: str,
    *,
    language_hint: str | None = None,
    unmapped: list | None = None,
    wi_as_y_glide: bool = False,
    lowercase_vc_prefers_onset: bool = True,
) -> tuple[AliasPlan, ...]:
    """Bind alias content to filename phone occurrences, never OTO row order.

    Several aliases may share an occurrence. Explicit ordinals take priority;
    unnumbered R endings select the last event, other aliases the first event.
    Missing bindings still require review.
    """
    candidates: dict[tuple[str, ...], list[AliasPlan]] = {}
    phones = {slot.start + i: phone for slot in slots for i, phone in enumerate(slot.phones)}
    japanese = (language_hint or "").casefold() in {"ja", "jp", "japanese"}
    korean_cvvc = format_type == "cvvc" and (language_hint or "").casefold() in {"ko", "kr", "korean"}
    korean_cvc = format_type == "cvc" and (language_hint or "").casefold() in {"ko", "kr", "korean"}
    korean_vcv = format_type == "vcv" and (language_hint or "").casefold() in {"ko", "kr", "korean"}
    pause_heads = _pause_token_slot_indices(wav_name, len(slots))

    def add(key: tuple[str, ...], indices: Sequence[int], role: str,
            template: str | None = None) -> None:
        plan = AliasPlan("", tuple(indices), template or f"builtin:{role}", role)
        bucket = candidates.setdefault(key, [])
        if plan not in bucket:
            bucket.append(plan)

    # This existing Korean convention is justified by duplicated filename
    # chunks themselves, not by adjacency or counts of rows in the base OTO.
    compact = []
    for token in recording_tokens(wav_name):
        pair = _parse_korean_cvvc_compact_duplicate_slots(
            token.casefold(), format_type=format_type, language_hint=language_hint)
        if not pair:
            compact = []
            break
        compact.extend(pair)
    compact_pairs = bool(compact) and tuple(compact) == tuple(s.phones for s in slots)
    coda_spelling = {"b": "p", "bb": "p", "p": "p", "pp": "p",
                     "d": "t", "dd": "t", "t": "t", "tt": "t",
                     "g": "k", "gg": "k", "k": "k", "kk": "k", "r": "l"}

    isolated_sonorants = {s.phones[0] for s in slots
                         if len(s.phones) == 1 and s.phones[0] in _STANDALONE_SONORANTS}
    for phone_index, phone in phones.items():
        if phone in isolated_sonorants:
            add((phone,), (phone_index,), "special", "builtin:special_generic")
        if phone_index == 0 and phone in _STANDALONE_SONORANTS:
            add(("-", phone), (phone_index,), "special", "builtin:special_generic")

    for index, slot in enumerate(slots):
        vowel = phones[slot.vowel]
        cv_indices, cv_template, cv_role = _alias_mapping("cv", index, slots)
        cv_key = "".join(phones[i] for i in cv_indices)
        add((cv_key,), cv_indices, cv_role, cv_template)
        if _is_vowel_id(vowel):
            add((vowel,), (slot.vowel,), "v")
        head_eligible = (index == 0 or index in pause_heads
                         or slots[index-1].phones[-1] in (_BREATH_PHONES | _SILENCE_PHONES))
        if head_eligible:
            head_role = "cv_head" if cv_role == "cv" else cv_role
            add(("-", cv_key), cv_indices, head_role,
                "builtin:cv_head" if head_role == "cv_head" else cv_template)
            if japanese and slot.vowel > slot.start and not _is_vowel_id(phones[slot.start]):
                # Japanese CVVC heads name the onset alone ("- ch", "- s") or with its glide ("- ky").
                onset = phones[slot.start]
                add(("-", onset), (slot.start,), "special", "builtin:special_generic")
                if slot.vowel - slot.start == 2 and phones[slot.start + 1] in {"y", "w"}:
                    add(("-", onset + phones[slot.start + 1]), (slot.start, slot.start + 1), "special",
                        "builtin:consonant_semivowel")
                elif len(vowel) == 2 and vowel[0] in "yw" and vowel[1] in "aiueo":  # romaji kya -> k + ya
                    add(("-", onset + vowel[0]), (slot.start, slot.vowel), "special", "builtin:consonant_semivowel")
        if len(slots) == 1 and slot.phones[0] in _BREATH_PHONES:
            add((), (slot.start,), "breath")
        if japanese and slot.vowel - slot.start == 2 and phones[slot.start + 1] in {"y", "w"}:
            # Kana yoon/glide onsets (きゃ, くぁ) keep the glide as its own
            # phone; bind the whole C+glide+V surface, not only glide+V.
            cluster = tuple(range(slot.start, slot.vowel + 1))
            cluster_key = "".join(phones[i] for i in cluster)
            add((cluster_key,), cluster, "cv", "builtin:cv_cluster")
            if head_eligible:
                add(("-", cluster_key), cluster, "cv_head", "builtin:cv_cluster")
            left_index = slots[index-1].end if index > 0 else None
            if left_index is not None and phones[left_index] not in (_BREATH_PHONES | _SILENCE_PHONES):
                add((phones[left_index], cluster_key), (left_index, *cluster), "vcv", "builtin:vcv_cluster")
                add((phones[left_index], phones[slot.start] + phones[slot.start + 1]),
                    (left_index, slot.start, slot.start + 1), "vc", "builtin:vc_semivowel")
        if _is_vowel_id(vowel) or vowel in _STANDALONE_SONORANTS:
            release_indices = (slot.vowel,)
            release_template = "builtin:release"
            if index+1 < len(slots) and slots[index+1].phones == ("exh",):
                release_indices = (slot.vowel, slots[index+1].start)
                release_template = "builtin:ending_exhalation"
            if phones.get(slot.vowel+1) == "exh":
                release_indices = (slot.vowel, slot.vowel+1)
                release_template = "builtin:ending_exhalation"
            for marker in ("R", "-"):
                if korean_cvvc and marker == "-":
                    add((vowel, marker), (slot.vowel,), "release", "builtin:release")
                elif not korean_cvvc or marker != "R" or release_template == "builtin:ending_exhalation":
                    add((vowel, marker), release_indices, "release", release_template)
        if index > 0:
            previous = slots[index-1]
            left = phones[previous.end]
            if left not in (_BREATH_PHONES | _SILENCE_PHONES) and vowel not in (_BREATH_PHONES | _SILENCE_PHONES):
                # Korean VCV aliases name a diphthong by its nucleus on the left (bya'bye: "a bye").
                lefts = dict.fromkeys((left, _KOREAN_VCV_NUCLEUS.get(left, left) if korean_vcv else left))
                if cv_role == "cv" and cv_indices[0] == slot.start:
                    for spelling in lefts:
                        add((spelling, cv_key), (previous.end, *cv_indices), "vcv")
                elif cv_role in {"v", "special"}:
                    # Japanese N mora transitions use the existing VV contract.
                    role = "vv" if japanese or _is_vowel_id(left) == _is_vowel_id(vowel) else ("cv" if _is_vowel_id(vowel) else "vc")
                    for spelling in lefts:
                        add((spelling, cv_key), (previous.end, slot.vowel), role)
            if compact_pairs and index % 2 == 1:
                onset = phones[slot.start]
                indices = tuple(dict.fromkeys((previous.vowel, slot.start)))
                role = "vc" if slot.start != slot.vowel else "vv"
                add((phones[previous.vowel], onset), indices, role)
                if slot.start != slot.vowel:
                    add((phones[previous.vowel] + coda_spelling.get(onset, onset),), indices, role)

    # Adjacent phone pairs cover VC, explicit CV, codas and consonant clusters.
    # Do not invent a missing trailing consonant or skip across intervening phones.
    for left in sorted(phones):
        if left+1 not in phones:
            continue
        a, b = phones[left], phones[left+1]
        if a in (_BREATH_PHONES | _SILENCE_PHONES) or b in (_BREATH_PHONES | _SILENCE_PHONES):
            continue
        key = (a, b)
        if any(p.phone_indices == (left, left+1) for p in candidates.get(key, ())):
            continue  # preserve the role of this pair, but keep other occurrences
        role = "cv" if _is_vowel_id(b) and not _is_vowel_id(a) else "vv" if _is_vowel_id(b) else "vc"
        add(key, (left, left+1), role)

    if japanese:
        special = special_slots(wav_name) if format_type == 'cvvc' else ()
        if special and special[0][1].endswith('L') and len(slots) == 2:
            special = (special[0], special[-1])
        if special and len(special) == len(slots):
            nasal_candidates: dict[tuple[str, ...], list[AliasPlan]] = {}
            for index, (slot, (raw_phones, surface)) in enumerate(zip(slots, special)):
                key = alias_part_key(surface)
                if key and surface.endswith('L'):
                    add((key,), (slot.vowel,), 'v', 'builtin:long_vowel')
                elif surface == 'H' and index:
                    add((phones[slots[index-1].vowel], 'H'), (slots[index-1].vowel, slot.start),
                        'release', 'builtin:ending_exhalation')
                elif surface.startswith('ん') and key and len(surface) > 1:
                    # A single filename mora contains n -> m/ng/n; keep both
                    # occurrences so n んm is a real internal transition.
                    nasal_candidates.setdefault(('n', key), []).append(AliasPlan('', (slot.start, slot.end),
                        'builtin:vc_soft_voiced', 'vc'))
                    if index:
                        previous = slots[index-1]
                        nasal_candidates.setdefault((phones[previous.end], key), []).append(AliasPlan('',
                            (previous.end, slot.start, slot.end), 'builtin:vc_nasal_variant', 'vc'))
                elif surface.startswith(('l', 'x')) and key:
                    indices, template, role = _alias_mapping('cv', index, slots)
                    if raw_phones[0] == 'trill':
                        indices, template = [slot.start, slot.start + 1, slot.vowel], 'builtin:cv_trill'
                    add((key,), indices, role, template)
                    add(('-', key), indices, 'cv_head', template)
                    if index:
                        previous = slots[index - 1]
                        if raw_phones[0] == 'trill':
                            add((phones[previous.end], 'x'), (previous.end, slot.start, slot.start + 1),
                                'vc', 'builtin:vc_trill')
                        else:
                            add((phones[previous.end], 'l'), (previous.end, slot.start), 'vc')
                # Fricative-to-CV aliases retain the frication as the offset,
                # with preutterance at the following vowel (same as base OTO).
                if index and slots[index - 1].phones in {('s',), ('sh',)} and slot.start < slot.vowel:
                    previous = slots[index - 1]
                    add((phones[previous.start], ''.join(slot.phones)),
                        (previous.start, *range(slot.start, slot.vowel + 1)), 'vcv', 'builtin:vcv')
            candidates.update(nasal_candidates)
        for i in sorted(phones):
            if phones[i] != "vf":
                continue
            if _is_vowel_id(phones.get(i+1, "")) or phones.get(i+1) == "n":
                add(("\u30fb", phones[i+1]), (i, i+1), "cv")
            if _is_vowel_id(phones.get(i-1, "")) or phones.get(i-1) == "n":
                add((phones[i-1], "\u30fb"), (i-1, i), "vc")

    if korean_cvc:
        # Korean CVC lists spell a coda as one syllable: a + standalone NG/M/N/L
        # token (a-NG -> ang) or a + tense stop closure (ba-bba -> ap). The
        # spaced form "a b" stays the ordinary VC into a lenis onset.
        standalone = {s.start for s in slots if len(s.phones) == 1 and s.phones[0] in _STANDALONE_SONORANTS}
        tense = {"bb": "p", "dd": "t", "gg": "k"}
        for left in sorted(phones):
            a, b = phones[left], phones.get(left+1)
            if b is not None and _is_vowel_id(a) and (left+1 in standalone or b in tense):
                # Same templates as Korean CVVC coda notation: held sonorant or stop closure.
                add((a + tense.get(b, b),), (left, left+1), "vc",
                    "builtin:vc_soft_voiced" if left+1 in standalone else "builtin:vc_hard")
        # In-syllable codas are also written glued: "ang"/"am"/"al" (V->coda) and
        # "nga"/"ngi" (coda -> next vowel-initial syllable); "n R" closes with breath.
        in_slot_codas = {s.start + i for s in slots for i, p in enumerate(s.phones)
                         if p in _KOREAN_CODAS and s.start + i > s.vowel}
        for left in sorted(phones):
            a, b = phones[left], phones.get(left+1)
            if b is None:
                continue
            if _is_vowel_id(a) and left+1 in in_slot_codas:
                template = "builtin:vc_soft_voiced" if b in _STANDALONE_SONORANTS else "builtin:vc_hard"
                for spelling in dict.fromkeys((a, _KOREAN_VCV_NUCLEUS.get(a, a))):
                    add((spelling + b,), (left, left+1), "vc", template)
            if left in in_slot_codas and a in _STANDALONE_SONORANTS and _is_vowel_id(b):
                add((a + b,), (left, left+1), "cv")
            if left in in_slot_codas and a in _STANDALONE_SONORANTS and b == "exh":
                add((a, "R"), (left, left+1), "release", "builtin:ending_exhalation")

    if korean_cvvc:
        nuclei = {"wa": "a", "wi": "i", "wo": "o", "we": "e", "weo": "eo",
                  "ya": "a", "ye": "e", "yeo": "eo", "yo": "o", "yu": "u"}
        fortis = {"tt": "t", "kk": "k", "pp": "p", "ss": "s", "jj": "j"}
        def glide_of(vowel: str) -> str:
            return "w" if vowel in {"wa", "wi", "wo", "we", "weo", "o", "u"} else "y" if vowel in {"ya", "ye", "yeo", "yo", "yu", "i"} else ""

        for i in sorted(phones):
            a, b = phones[i], phones.get(i+1)
            if a in _STANDALONE_SONORANTS:
                for marker in ("R", "-"):
                    if marker == "R":
                        if b == "exh":
                            add((a, marker), (i, i+1), "release", "builtin:ending_exhalation")
                    else:
                        add((a, marker), (i,), "release")
            if b is None:
                continue
            if i == 0 and not _is_vowel_id(a) and _is_vowel_id(b):
                add(("-", a), (i,), "special", "builtin:special_generic")
                if glide_of(b):
                    add(("-", a+glide_of(b)), (i, i+1), "special", "builtin:consonant_semivowel")
            if a == "vf":
                template = ("builtin:fry_to_sustain" if b in _STANDALONE_SONORANTS
                    and len(phones) == 3 and phones.get(i+2) == "vf" else "builtin:cv")
                add(("*", b), (i, i+1), "cv", template)
            if b == "vf":
                add((a, "*"), (i, i+1), "vc", "builtin:ending_exhalation")
            if b == "exh" and a in _STANDALONE_SONORANTS:
                add((a, "h"), (i, i+1), "release", "builtin:ending_exhalation")
            if a == "q":
                add(("'", b), (i, i+1), "cv")
            if b == "q":
                add((a, "'"), (i, i+1), "vc")
            if _is_vowel_id(b) and a in (_STANDALONE_SONORANTS | {"r"}):
                add((a+b,), (i, i+1), "cv")
                if a == "r":
                    add(("l"+b,), (i, i+1), "cv")
            if _is_vowel_id(a) and _is_vowel_id(b) and glide_of(b):
                add((nuclei.get(a, a), glide_of(b)), (i, i+1), "vc", "builtin:v_semivowel")
            if not _is_vowel_id(b) and b not in {"vf", "q"}:
                spellings = {b, fortis.get(b, b)} | ({"l"} if b == "r" else set())
                following = phones.get(i+2, "")
                for spelling in sorted(spellings):
                    add((nuclei.get(a, a), spelling), (i, i+1), "vc")
                    glide = "y" if wi_as_y_glide and following == "wi" else glide_of(following)
                    if glide:
                        add((nuclei.get(a, a), spelling+glide),
                            (i, i+1, i+2), "vc", "builtin:vc_semivowel")

    if korean_cvvc:
        # Uppercase codas are distinct from generic VC consonants. Keep the
        # same lexical phone when applicable, but stop before its release for
        # K/T/P; sonorant codas retain their voiced interval without spillover.
        coda_indices = {s.start+i for s in slots for i, p in enumerate(s.phones)
                        if p in _KOREAN_CODAS and (s.start+i > s.vowel or
                        len(s.phones) == 1 or (s.phones[0] == s.phones[-1] == "vf"))}
        for key, plans in list(candidates.items()):
            choices = [(part, part.upper()) if part in {"k", "t", "p", "n", "m", "ng", "l"} else (part,) for part in key]
            for variant in itertools.product(*choices):
                if variant == key:
                    continue
                for plan in plans:
                    capital_positions = [i for i, part in enumerate(variant) if part in {"K", "T", "P", "N", "M", "NG", "L"}]
                    if any((plan.phone_indices[0] if i == 0 else plan.phone_indices[-1]) not in coda_indices
                           and not (i == len(variant)-1 and variant[i] in {"K", "T", "P"} and plan.role == "vc")
                           for i in capital_positions):
                        continue
                    template = plan.template
                    if len(variant) == 2 and variant[-1] in {"K", "T", "P", "N", "M", "NG", "L"} and plan.role == "vc":
                        template = "builtin:vc_hard" if variant[-1] in {"K", "T", "P"} else "builtin:vc_soft_voiced"
                    add(variant, plan.phone_indices, plan.role, template)

    vcv_coda_indices = {s.start + i for s in slots for i, p in enumerate(s.phones)
                        if korean_vcv and p in _KOREAN_CODAS and s.start + i > s.vowel}
    result = []
    for row in rows:
        key, occurrence = _alias_surface(row.alias, preserve_coda_case=korean_cvvc, japanese=japanese)
        # br2.wav/br2 is a named breath variant, not the second event in a
        # one-phone file. Only accept an exact filename/alias variant match.
        variant = row.alias.strip().casefold()
        if (len(phones) == 1 and _SPECIAL_VARIANT_RE.fullmatch(variant)
                and variant == Path(wav_name).stem.strip("_").casefold()):
            key, occurrence = (_normalize_special_variant(variant),), None
        if japanese and key not in candidates and "wo" in key:
            key = tuple("o" if part == "wo" else part for part in key)
        if (japanese and key not in candidates and len(key) == 2 and len(key[1]) > 1
                and key[1].endswith("y") and not _is_vowel_id(key[1][:-1])):
            # Fallback only: CVVC lists write the consonant before i as Cy (i ky -> き).
            before_i = [plan for plan in candidates.get((key[0], key[1][:-1]), ())
                        if phones.get(plan.phone_indices[-1] + 1) == "i"]
            if before_i:
                candidates[key] = before_i
        if key not in candidates and "*" in key and "vf" not in phones.values():
            # A trailing * tags a vocal-fry take of the same syllables (seen in a Korean VCV bank).
            stripped = tuple(part for part in key if part != "*")
            if stripped in candidates:
                key = stripped
        if (key not in candidates and len(phones) == 1 and next(iter(phones.values())) in _BREATH_PHONES
                and len(key) == 1 and key[0] in {"br", "breath", "exh"}):
            # 息2.wav=br2: the number names a breath sample, not an occurrence.
            key, occurrence = (next(iter(phones.values())),), None
        if japanese and key not in candidates:
            key = _unique_transliterated_key(key, candidates)
        if korean_cvvc and key not in candidates:
            key = _unique_korean_respelling(key, candidates, phones)
        if korean_cvc and key not in candidates:
            # ㄹ onsets are filed as r but aliased as l (ral'ril: "la", rwal: "lwa").
            respelled = tuple(("r" + part[1:]) if part.startswith("l") and len(part) > 1 and part[1] in "aeiouwy"
                              else part for part in key)
            key = respelled if respelled in candidates else key
        if korean_vcv and key not in candidates:
            # Tense onsets are romanized either way (ㅃ: bb/pp); files and aliases may differ.
            respelled = tuple(re.sub(r"^(-?)(pp|tt|kk)", lambda m: m.group(1) + _KOREAN_TENSE_RESPELLING[m.group(2)], part)
                              for part in key)
            key = respelled if respelled in candidates else key
        if key not in candidates and len(key) >= 2 and key[0] == "-":
            # Heads are normally limited to file start, breath/silence or a pause
            # token; by user policy a head may also name a mid-file syllable,
            # bound like any unnumbered alias to its first occurrence.
            inner = key[1:]
            if "*" in inner and "vf" not in phones.values():
                inner = tuple(part for part in inner if part != "*")
            if japanese and inner not in candidates:
                inner = _unique_transliterated_key(inner, candidates)
            if korean_cvvc and inner not in candidates:
                inner = _unique_korean_respelling(inner, candidates, phones)
            if inner in candidates:
                candidates[key] = [_as_head_plan(plan) for plan in candidates[inner]]
        matches = sorted(candidates.get(key, ()), key=lambda p: p.phone_indices)
        all_matches = matches
        if korean_cvvc and occurrence is not None and len(key) == 2 and key[-1] == "R":
            targets = [i for i, p in phones.items() if p == key[0].casefold()]
            target = targets[occurrence-1] if 1 <= occurrence <= len(targets) else None
            matches = [p for p in matches if p.phone_indices[0] == target]
        elif occurrence is not None:
            matches = matches[occurrence-1:occurrence] if 1 <= occurrence <= len(matches) else []
        elif korean_vcv and len(key) == 2 and _is_vowel_id(key[0]) and key[1] in _KOREAN_CODAS and any(
                p.phone_indices[-1] in vcv_coda_indices for p in matches):
            # VCV lists have no VC-into-onset rows: "a n" in nana…naN is the final N coda.
            matches = [p for p in matches if p.phone_indices[-1] in vcv_coda_indices][-1:]
        elif len(key) == 2 and (key[-1] == "R" or ((korean_cvvc or korean_vcv) and key[-1] == "-")):
            # R endings take the last event. Korean CVVC lists also mean the
            # phrase end by a trailing '-' (user-confirmed 2026-10-09); other
            # formats keep the first event, as their banks are not consistent.
            matches = matches[-1:]
        elif (japanese and format_type == 'cvvc' and special_slots(wav_name) and len(key) == 1
              and any(alias_part_key(surface) == key[0] and surface.startswith(('l', 'x'))
                      for _, surface in special_slots(wav_name))):
            # Heads use the first take. A plain repeated CV uses the first
            # phrase-internal take; explicit #N/@N retains its original meaning.
            internal = [p for p in matches if min(p.phone_indices) > slots[0].end]
            matches = (internal or matches)[:1]
        else:
            if (korean_cvvc and lowercase_vc_prefers_onset and len(key) == 2 and _is_vowel_id(key[0])
                    and key[1] in {"k", "t", "p", "n", "m", "l"}):
                # Codas are written in uppercase ("i M"); a lowercase VC names the next
                # syllable's onset (miM'..mi'meu: "i m" = mi -> meu, not the M coda).
                onsets = [p for p in matches if p.phone_indices[-1] not in coda_indices]
                matches = onsets or matches
            if wi_as_y_glide and len(key) == 2 and key[1].endswith("y"):
                # Banks without any Cw spelling write the consonant before wi as Cy, and their
                # lists record it there (gi'gi'gwi: "i gy" = i -> gwi, "i g" = i -> gi).
                before_wi = [p for p in matches if len(p.phone_indices) == 3 and phones.get(p.phone_indices[-1]) == "wi"]
                matches = before_wi or matches
            matches = matches[:1]
        if not matches:
            detail = json.dumps({
                "alias": row.alias, "occurrence": occurrence,
                "candidate_phone_indices": [list(p.phone_indices) for p in all_matches],
                "resolution": ("choose an in-range occurrence (#N/@N or trailing N)"
                               if all_matches else "check alias notation or provide recordings manifest phone indices"),
            }, ensure_ascii=False)
            if unmapped is None:
                raise VoicebankReviewRequired("unmapped_alias", wav=wav_name, detail=detail)
            unmapped.append((row, detail))
            continue
        selected = matches[0]
        overrides = dict(selected.overrides)
        if (japanese and format_type == 'cvvc' and special_slots(wav_name)
                and special_slots(wav_name)[0][1].endswith('L')
                and selected.template == 'builtin:ending_exhalation'):
            # Short releases cannot fit a 25 ms fixed tail plus a 15 ms trim.
            # Keep both anchors inside the measured release instead of relying
            # on the renderer's generic repair/clamping pass.
            overrides.update(fixed_abs='min(max(pre_abs + 25, phone[1].end - 70), phone[1].end - 15)',
                             cutoff_abs='phone[1].end - 5')
        result.append(AliasPlan(row.alias, selected.phone_indices, selected.template, selected.role, overrides))
    return tuple(result)


def _bank_writes_wi_onset_as_y(rows_by_wav) -> bool:
    """True when a Korean CVVC base OTO has Cy VC aliases but never spells a consonant with w
    (e.g. "i gy" recorded before gwi). Banks that write Cw (e.g. "i gw") keep wi as w."""
    has_cy = False
    for rows in rows_by_wav:
        for row in rows:
            key, _ = _alias_surface(row.alias, preserve_coda_case=True)
            for part in key:
                if len(part) > 1 and part.endswith("w") and not _is_vowel_id(part):
                    return False
            has_cy = has_cy or (len(key) == 2 and len(key[1]) > 1 and key[1].endswith("y") and not _is_vowel_id(key[1]))
    return has_cy


def _pause_token_slot_indices(wav_name: str, slot_count: int) -> frozenset[int]:
    """Slots that follow an explicit pause token such as ``ba'-'bya``.

    A lone '-' between tokens is only a separator (``ka-ki``). A separator run
    that holds '-' together with another separator marks a sung pause, so
    the next syllable may bind a head alias. Requires one token per slot.
    """
    parts = re.split(r"([_'\-]+)", Path(wav_name).stem)
    heads, count, pending = set(), 0, False
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[_'\-]+", part):
            pending = pending or ("-" in part and len(part) > 1 and count > 0)
            continue
        if pending:
            heads.add(count)
        pending = False
        count += 1
    return frozenset(heads) if count == slot_count else frozenset()


_KOREAN_VOWEL_SPELLINGS = ("yeo", "weo", "yae", "wae", "eui", "ae", "eo", "eu", "oe", "ui", "ya", "ye",
                           "yo", "yu", "wa", "we", "wi", "wo", "a", "e", "i", "o", "u")
_KOREAN_TO_KANA_VOWEL = {"yeo": "yo", "weo": "wo", "eu": "u", "eo": "o", "ae": "e"}
_KOREAN_TO_KANA_ONSET = {"jj": ("ts",), "j": ("z", "ts"), "dd": ("t",), "tt": ("t",), "kk": ("k",),
                         "pp": ("p",), "ss": ("s",), "hw": ("f",), "bw": ("v",), "dw": ("d",)}


def _korean_spelling_variants(part: str) -> set[str]:
    """Japanese spellings a Korean-romanized alias part can stand for.

    Japanese banks add Korean-support aliases over the same kana recordings
    (eo ka = お→か, ng sa = ん→さ, - hwa = ふぁ).
    """
    if part in {"-", "R", "*", "'", "\u30fb"}:
        return {part}
    if part == "ng":
        return {part, "n"}
    variants = {part}
    # hwa may be hw+a (ふぁ) as well as h+wa, so try every vowel split.
    for vowel in [v for v in _KOREAN_VOWEL_SPELLINGS if part.endswith(v)] or [""]:
        onset = part[:len(part) - len(vowel)]
        onsets = {onset, *_KOREAN_TO_KANA_ONSET.get(onset, ())}
        if onset == "s" and vowel.startswith("i"):
            onsets.add("sh")
        vowels = {vowel, _KOREAN_TO_KANA_VOWEL.get(vowel, vowel)}
        variants |= {o + v for o in onsets for v in vowels}
        variants |= {o + "y" + v for o in onsets & {"ch", "sh", "j"} for v in vowels if v == "e"}
    return variants


_KOREAN_GLIDE_NUCLEI = {"wa": "a", "wi": "i", "wo": "o", "we": "e", "weo": "eo",
                        "ya": "a", "ye": "e", "yeo": "eo", "yo": "o", "yu": "u"}


def _unique_korean_respelling(key: tuple[str, ...], candidates, phones: Mapping[int, str]) -> tuple[str, ...]:
    """Fallback only for Korean CVVC lists with their own spelling habits.

    VV aliases name the previous syllable by its nucleus (ya ya -> "a ya"),
    wo may stand for weo, and a lenis onset may stand for the fortis one
    recorded (kkwa -> "- kwa"). A respelling is used only when unique; the
    nucleus pairs are added to ``candidates`` only for that key.
    """
    def variants(part: str) -> set[str]:
        out = {part, "weo" if part == "wo" else part}
        for lenis in ("k", "t", "p", "s", "j"):
            if part.startswith(lenis) and not part.startswith(lenis * 2) and len(part) > 1:
                out.add(lenis + part)
        return out

    pairs: dict[tuple[str, ...], list[AliasPlan]] = {}
    for i in sorted(phones):
        a, b = phones[i], phones.get(i + 1)
        if b is not None and _is_vowel_id(a) and _is_vowel_id(b) and a in _KOREAN_GLIDE_NUCLEI:
            pairs.setdefault((_KOREAN_GLIDE_NUCLEI[a], b), []).append(AliasPlan("", (i, i + 1), "builtin:vv", "vv"))
    found = {variant for variant in itertools.product(*(variants(part) for part in key))
             if (variant != key and variant in candidates) or variant in pairs}
    if len(found) != 1:
        return key
    chosen = found.pop()
    if chosen not in candidates:
        candidates[chosen] = pairs[chosen]
    return chosen


def _as_head_plan(plan: AliasPlan) -> AliasPlan:
    if plan.role in {"cv", "cv_head"}:
        template = "builtin:cv_head" if plan.template == "builtin:cv" else plan.template
        return AliasPlan(plan.alias, plan.phone_indices, template, "cv_head", plan.overrides)
    if plan.role == "v":
        return AliasPlan(plan.alias, plan.phone_indices, "builtin:v_head", "v_head", plan.overrides)
    return plan


def _unique_transliterated_key(key: tuple[str, ...], candidates) -> tuple[str, ...]:
    """Fallback only: accept a Korean-to-kana respelling when exactly one exists."""
    found = {variant for variant in itertools.product(*(_korean_spelling_variants(part) for part in key))
             if variant != key and variant in candidates}
    return found.pop() if len(found) == 1 else key


def _alias_mapping(role: str, slot_index: int, slots: Sequence[_PhoneSlot]) -> tuple[list[int], str, str]:
    if not slots:
        return [], "builtin:cv", "cv"
    index = min(max(0, int(slot_index)), len(slots) - 1)
    slot = slots[index]
    normalized = str(role or "cv").casefold()
    if len(slot.phones) == 1 and slot.phones[0] in _BREATH_PHONES and (len(slots) == 1 or normalized in {"cv", "cv_head", "v", "breath"}):
        return [slot.start], "builtin:breath", "breath"
    if len(slot.phones) == 1 and slot.phones[0] in _STANDALONE_SONORANTS and normalized in {"cv", "cv_head", "v"}:
        return [slot.start], "builtin:special_generic", "special"
    if normalized == "breath":
        return [slot.start], "builtin:breath", "breath"
    if normalized in {"release", "vc_tail"}:
        return [slot.vowel], "builtin:release", "release"
    if normalized in {"vc", "cvc_vc"}:
        if index + 1 < len(slots):
            next_slot = slots[index + 1]
            consonant = next_slot.start if next_slot.start < next_slot.vowel else next_slot.vowel
            indices = list(dict.fromkeys((slot.vowel, consonant)))
            template = "builtin:vc" if len(indices) == 2 else "builtin:vv"
            return indices, template, "vc" if len(indices) == 2 else "vv"
        return [slot.vowel], "builtin:release", "release"
    if normalized in {"vcv", "vv"} and index > 0:
        previous_vowel = slots[index - 1].vowel
        if slot.start < slot.vowel and normalized == "vcv":
            return [previous_vowel, slot.vowel - 1, slot.vowel], "builtin:vcv", "vcv"
        return [previous_vowel, slot.vowel], "builtin:vv", "vv"
    cv_indices = [slot.vowel - 1, slot.vowel] if slot.start < slot.vowel else [slot.vowel]
    if slot.start == slot.vowel:
        return cv_indices, "builtin:v", "v"
    template = "builtin:cv_head" if normalized == "cv_head" else "builtin:cv"
    return cv_indices, template, "cv_head" if normalized == "cv_head" else "cv"


def _manifest_aliases(record: Mapping[str, object]) -> Sequence[object]:
    aliases = record.get("aliases")
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
        return aliases
    return ()


def _parse_manifest_aliases(record: Mapping[str, object], phone_count: int, wav: str) -> tuple[AliasPlan, ...]:
    result: list[AliasPlan] = []
    for raw_alias in _manifest_aliases(record):
        if not isinstance(raw_alias, Mapping):
            raise VoicebankReviewRequired("invalid_manifest_alias", wav=wav)
        # Empty OTO aliases are valid (the host uses the WAV filename). An
        # explicit empty value must round-trip, while a missing field is invalid.
        raw_name = raw_alias.get("alias")
        if not isinstance(raw_name, str):
            raise VoicebankReviewRequired("invalid_manifest_alias", wav=wav)
        alias = raw_name.strip()
        template = str(raw_alias.get("template", "")).strip()
        raw_indices = raw_alias.get("phones")
        if not template or not isinstance(raw_indices, Sequence) or isinstance(raw_indices, (str, bytes)):
            raise VoicebankReviewRequired("invalid_manifest_alias", wav=wav, detail=alias or None)
        try:
            indices = tuple(int(value) for value in raw_indices)
        except (TypeError, ValueError) as exc:
            raise VoicebankReviewRequired("invalid_manifest_alias", wav=wav, detail=alias) from exc
        if not indices or any(index < 0 or index >= phone_count for index in indices):
            raise VoicebankReviewRequired("invalid_manifest_alias", wav=wav, detail=alias)
        role = str(raw_alias.get("role") or template.rsplit(":", 1)[-1]).strip().casefold()
        overrides = raw_alias.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise VoicebankReviewRequired("invalid_manifest_alias", wav=wav, detail=alias)
        result.append(AliasPlan(alias, indices, template, role, dict(overrides)))
    return tuple(result)


def _route_alignment(
    *,
    language_hint: str,
    format_hint: str,
    recordings: Sequence[RecordingSpec],
) -> tuple[str, str, str]:
    phones = {phone for recording in recordings for phone in recording.phones}
    korean = language_hint in {"ko", "kr", "korean"}
    if korean and format_hint == "vcv" and phones.intersection({"bb", "dd", "gg", "ss", "jj"}):
        return "hsmm", "numpy_spectral_hsmm_v1", "korean_fortis_vcv_measured_v1_route"
    return "legacy_hsmm_compat", "numpy_multiband_pitch_v2", "default_occurrence_guard_route"


def _inferred_phone_definition(phone: str) -> PhoneDefinition:
    if _is_vowel_id(phone):
        features = {"syllabic": True, "voiced": True, "vowel_like": True}
        states = (
            PhoneStateDefinition("transition", "vowel_transition", 15.0, 220.0, 55.0),
            PhoneStateDefinition("stable", "vowel_stable", 45.0, 1400.0, 360.0),
        )
    elif phone in _SILENCE_PHONES:
        features = {"manner": "silence", "voiced": False, "continuant": False}
        states = (PhoneStateDefinition("silence", "silence", 20.0, 1600.0, 120.0),)
    elif phone == 'trill':
        features = {"manner": "liquid", "voiced": True, "sonorant": True, "continuant": True}
        states = (PhoneStateDefinition('irregular', 'aperiodic', 10.0, 400.0, 70.0),)
    elif phone == "vf":
        features = {"manner": "vocal_fry", "voiced": True, "continuant": True}
        states = (
            PhoneStateDefinition("onset", "irregular_voicing_onset", 8.0, 120.0, 30.0),
            PhoneStateDefinition("fry", "aperiodic", 8.0, 700.0, 70.0),
        )
    elif phone in _BREATH_PHONES:
        features = {"manner": "breath", "voiced": False, "continuant": True}
        states = (PhoneStateDefinition("breath", "breath", 20.0, 1000.0, 120.0),)
    elif phone in _STOP_PHONES:
        features = {"manner": "stop", "voiced": phone[:1] in {"b", "d", "g", "j", "z"}, "continuant": False}
        states = (
            PhoneStateDefinition("closure", "closure", 8.0, 220.0, 45.0),
            PhoneStateDefinition("release", "burst", 5.0, 80.0, 14.0),
            PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
        )
    elif phone in _FRICATIVE_PHONES:
        features = {"manner": "fricative", "voiced": phone[:1] in {"v", "z"}, "continuant": True}
        states = (
            PhoneStateDefinition("frication", "frication", 12.0, 300.0, 75.0),
            PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
        )
    elif phone in _NASAL_PHONES:
        features = {"manner": "nasal", "voiced": True, "sonorant": True, "continuant": True}
        states = (
            PhoneStateDefinition("consonant", "consonant", 10.0, 260.0, 55.0),
            PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
        )
    elif phone in _LIQUID_PHONES:
        features = {"manner": "liquid", "voiced": True, "sonorant": True, "continuant": True}
        states = (
            PhoneStateDefinition("consonant", "consonant", 10.0, 260.0, 55.0),
            PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
        )
    else:
        # Unlisted symbols must remain reviewable instead of being silently
        # promoted to a voiced sonorant class.
        features = {"manner": "unknown", "voiced": False, "continuant": True}
        states = (
            PhoneStateDefinition("consonant", "consonant", 10.0, 260.0, 55.0),
            PhoneStateDefinition("transition", "transition", 8.0, 160.0, 35.0),
        )
    return PhoneDefinition(id=phone, display_name=phone, features=features, states=states)


def _is_vowel_id(phone: str) -> bool:
    return phone.casefold() in _VOWELS


def _is_vowel_definition(definition: PhoneDefinition | None) -> bool:
    return bool(definition and (definition.features.get("syllabic") or definition.features.get("vowel_like")))


def _discover_contract_artifacts(bank: Path) -> dict[str, Path]:
    project_path = bank / "project.yaml"
    inventory_path: Path | None = None
    recordings_path: Path | None = None
    result: dict[str, Path] = {}
    if project_path.is_file():
        payload = _read_yaml(project_path, "project")
        inventory_path = _safe_bank_relative(bank, payload.get("inventory"), "project.inventory")
        recordings_path = _safe_bank_relative(bank, payload.get("recordings"), "project.recordings")
        result["project"] = project_path
    else:
        if (bank / "phones.yaml").is_file():
            inventory_path = bank / "phones.yaml"
        if (bank / "recordings.yaml").is_file():
            recordings_path = bank / "recordings.yaml"
    if inventory_path is not None and inventory_path.is_file():
        result["inventory"] = inventory_path
    if recordings_path is not None and recordings_path.is_file():
        result["recordings"] = recordings_path
    for name in ("_reclist.txt", "reclist.txt"):
        candidate = bank / name
        if candidate.is_file():
            result["reclist"] = candidate
            break
    return result


def _safe_bank_relative(bank: Path, value: object, label: str) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = (bank / value).resolve()
    try:
        candidate.relative_to(bank)
    except ValueError as exc:
        raise VoicebankReviewRequired("contract_path_outside_voicebank", detail=label) from exc
    return candidate


def _load_inventory(path: Path | None) -> dict[str, PhoneDefinition]:
    if path is None:
        return {}
    payload = _read_yaml(path, "inventory")
    raw_phones = payload.get("phones")
    if not isinstance(raw_phones, Mapping) or not raw_phones:
        raise VoicebankReviewRequired("invalid_inventory", detail=str(path))
    result: dict[str, PhoneDefinition] = {}
    for raw_id, raw_definition in raw_phones.items():
        phone_id = str(raw_id).strip().casefold()
        if not phone_id or not isinstance(raw_definition, Mapping):
            raise VoicebankReviewRequired("invalid_inventory_phone", detail=str(raw_id))
        raw_states = raw_definition.get("states")
        if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)) or not raw_states:
            raise VoicebankReviewRequired("invalid_inventory_states", detail=phone_id)
        states: list[PhoneStateDefinition] = []
        for raw_state in raw_states:
            if not isinstance(raw_state, Mapping):
                raise VoicebankReviewRequired("invalid_inventory_states", detail=phone_id)
            try:
                state = PhoneStateDefinition(
                    id=str(raw_state["id"]).strip(),
                    acoustic_role=str(raw_state["acoustic_role"]).strip(),
                    min_ms=float(raw_state["min_ms"]),
                    max_ms=float(raw_state["max_ms"]),
                    mode_ms=None if raw_state.get("mode_ms") is None else float(raw_state["mode_ms"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise VoicebankReviewRequired("invalid_inventory_states", detail=phone_id) from exc
            if not state.id or not state.acoustic_role or state.min_ms < 0.0 or state.max_ms <= 0.0 or state.min_ms > state.max_ms:
                raise VoicebankReviewRequired("invalid_inventory_states", detail=phone_id)
            states.append(state)
        features = raw_definition.get("features", {})
        if not isinstance(features, Mapping):
            raise VoicebankReviewRequired("invalid_inventory_features", detail=phone_id)
        aliases = raw_definition.get("aliases", ())
        prototype_ids = raw_definition.get("prototype_ids", ())
        result[phone_id] = PhoneDefinition(
            id=phone_id,
            display_name=str(raw_definition.get("display_name") or phone_id),
            features=dict(features),
            states=tuple(states),
            aliases=tuple(str(value) for value in aliases) if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)) else (),
            prototype_ids=tuple(str(value) for value in prototype_ids) if isinstance(prototype_ids, Sequence) and not isinstance(prototype_ids, (str, bytes)) else (),
        )
    return result


def _load_recording_manifest(path: Path | None) -> tuple[Mapping[str, object], ...]:
    if path is None:
        return ()
    payload = _read_yaml(path, "recordings")
    raw_records = payload.get("recordings")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise VoicebankReviewRequired("invalid_recording_manifest", detail=str(path))
    result = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping) or not isinstance(raw_record.get("wav"), str):
            raise VoicebankReviewRequired("invalid_recording_manifest", detail=str(path))
        result.append(raw_record)
    return tuple(result)


def _match_manifest_record(
    records: Sequence[Mapping[str, object]],
    relative: str,
) -> Mapping[str, object] | None:
    exact = [record for record in records if str(record.get("wav", "")).replace("\\", "/").casefold() == relative.casefold()]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise VoicebankReviewRequired("duplicate_manifest_recording", wav=relative)
    key = _normalized_recording_key(relative)
    matches = [record for record in records if _normalized_recording_key(str(record.get("wav", ""))) == key]
    if len(matches) > 1:
        raise VoicebankReviewRequired("ambiguous_manifest_recording", wav=relative)
    return matches[0] if matches else None


def _load_reclist(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    text, _encoding, _newline = read_text_auto(path)
    result: dict[str, tuple[str, str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        source = line.split("=", 1)[0].strip().strip("_")
        key = _normalized_recording_key(source)
        if not key:
            continue
        if key in result:
            raise VoicebankReviewRequired("duplicate_reclist_entry", detail=source)
        result[key] = (source, str(path))
    return result


def _normalized_recording_key(value: str) -> str:
    stem = Path(value.replace("\\", "/")).stem.casefold()
    return "".join(character for character in stem if character.isalnum())


def _read_yaml(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise VoicebankReviewRequired(f"invalid_{label}_yaml", detail=str(path)) from exc
    if not isinstance(payload, Mapping):
        raise VoicebankReviewRequired(f"invalid_{label}_yaml", detail=str(path))
    return payload


def _artifact_label(path: Path | None, bank: Path, fallback: str) -> str:
    if path is None:
        return fallback
    try:
        return path.resolve().relative_to(bank).as_posix()
    except ValueError:
        return str(path.resolve())


def _inventory_semantics(inventory: Mapping[str, PhoneDefinition]) -> dict[str, object]:
    return {
        phone_id: {
            "display_name": phone.display_name,
            "features": dict(phone.features),
            "aliases": list(phone.aliases),
            "prototype_ids": list(phone.prototype_ids),
            "states": [state.__dict__ for state in phone.states],
        }
        for phone_id, phone in sorted(inventory.items())
    }


def _recording_semantics(recordings: Sequence[RecordingSpec]) -> list[object]:
    return [
        {
            "wav": recording.wav,
            "phones": list(recording.phones),
            "aliases": [
                {
                    "alias": alias.alias,
                    "phone_indices": list(alias.phone_indices),
                    "template": alias.template,
                    "role": alias.role,
                    "overrides": dict(alias.overrides),
                }
                for alias in recording.aliases
            ],
            "transcript_source": recording.transcript_source,
            "label_source": recording.label_source,
            "metadata": dict(recording.metadata),
        }
        for recording in recordings
    ]


def _semantic_sha256(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_token(value: str) -> str:
    encoded: list[str] = []
    for character in value:
        if character.isalnum() or character in {"_", "-"}:
            encoded.append(character)
        else:
            encoded.append(f"x{ord(character):04x}")
    return "".join(encoded) or "phone"


__all__ = [
    "OccurrencePhoneToken",
    "RoutedProjectBuild",
    "SUPPORTED_FORMATS",
    "VoicebankReviewRequired",
    "build_routed_voicebank_project",
    "parse_filename_phone_slots",
    "tag_phone_occurrences",
]
