from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .domain import PlannedAlias, SourceOtoRow
from .kana import parse_kana_slots


_SEPARATOR_RE = re.compile(r"[_'\-]+")


def normalize_wav_key(value: str) -> str:
    stem = Path(value).stem.casefold()
    return "".join(character for character in stem if character.isalnum())


def recording_tokens(wav_name: str) -> tuple[str, ...]:
    stem = Path(wav_name).stem
    tokens = tuple(token for token in _SEPARATOR_RE.split(stem) if token)
    return tokens or (stem,)


def group_source_rows(rows: list[SourceOtoRow]) -> dict[str, list[SourceOtoRow]]:
    grouped: dict[str, list[SourceOtoRow]] = defaultdict(list)
    for row in rows:
        grouped[normalize_wav_key(row.source_wav)].append(row)
    return dict(grouped)


def infer_slot_count(
    wav_name: str,
    source_rows: list[SourceOtoRow] | None,
    format_hint: str | None = None,
) -> int:
    tokens = recording_tokens(wav_name)
    stem = Path(wav_name).stem.casefold()
    if "long" in stem:
        return 1
    normalized_format = (format_hint or "").strip().casefold()
    if normalized_format == "cvvc" and len(tokens) == 1:
        kana_slots = _kana_surface_slots(Path(wav_name).stem)
        if len(kana_slots) > 1:
            return len(kana_slots)
    if (
        normalized_format == "cv"
        and len(tokens) == 1
        and source_rows
        and len(source_rows) > 1
        and re.search(r"\d+$", stem)
    ):
        # Numbered compact reclists (for example a36.wav) contain up to four
        # sequential utterances plus synonym aliases in the same WAV.
        return min(4, len(source_rows))
    if source_rows is not None and len(source_rows) == 1:
        return 1
    if normalized_format == "vcv" and source_rows:
        if len(tokens) > 1:
            return len(tokens)
        tail_release = 1 if _is_tail_release_alias(source_rows[-1].alias) else 0
        return max(1, len(source_rows) - tail_release)
    return max(1, len(tokens))


def fallback_source_rows(wav_name: str) -> list[SourceOtoRow]:
    tokens = recording_tokens(wav_name)
    return [
        SourceOtoRow(source_wav=wav_name, alias=token, source_index=index)
        for index, token in enumerate(tokens)
    ]


def plan_aliases(
    rows: list[SourceOtoRow],
    slot_count: int,
    format_hint: str | None = None,
    wav_name: str | None = None,
) -> tuple[PlannedAlias, ...]:
    count = len(rows)
    if count == 0:
        return ()
    normalized_format = (format_hint or "").strip().casefold()
    def finish(plan: tuple[PlannedAlias, ...]) -> tuple[PlannedAlias, ...]:
        return _refine_plan_with_recording_semantics(plan, wav_name, normalized_format)

    if normalized_format == "vcv":
        return finish(_plan_vcv_aliases(rows, slot_count))
    role_block_plan = _plan_cv_then_vc_role_blocks(rows, slot_count)
    if role_block_plan:
        return finish(role_block_plan)
    head_first_cvc_plan = _plan_cv_head_then_interleaved_cvc(rows, slot_count)
    if head_first_cvc_plan:
        return finish(head_first_cvc_plan)
    hybrid_plan = _plan_interleaved_cvc_with_cv_head(rows, slot_count)
    if hybrid_plan:
        return finish(hybrid_plan)
    trailing_vowel_plan = _plan_interleaved_cvc_with_trailing_vowels(rows, slot_count)
    if trailing_vowel_plan:
        return finish(trailing_vowel_plan)
    verified_pair_plan = _plan_verified_interleaved_cv_coda_pairs(
        rows,
        slot_count,
        normalized_format,
        wav_name,
    )
    if verified_pair_plan:
        return finish(verified_pair_plan)
    if normalized_format == "cvvc":
        semantic_plan = _plan_transition_format_aliases(rows, slot_count, wav_name)
        if semantic_plan:
            return finish(semantic_plan)
    grouped_variants = _detect_grouped_variant_layout(rows, slot_count)
    variants_per_slot = 1
    if not grouped_variants and slot_count > 1 and count % (slot_count - 1) == 0:
        candidate = count // (slot_count - 1)
        if candidate in {2, 3}:
            variants_per_slot = candidate
    planned: list[PlannedAlias] = []
    for index, row in enumerate(rows):
        if grouped_variants:
            slot_index = index % slot_count
            variant_index = index // slot_count
            role = _infer_role(row.alias, variant_index, grouped_variants, slot_count)
            planned.append(
                PlannedAlias(
                    source=row,
                    slot_index=slot_index,
                    variant_index=variant_index,
                    variants_per_slot=grouped_variants,
                    role=role,
                )
            )
            continue
        if slot_count <= 1:
            slot_index = 0
            variant_index = index
        elif normalized_format == "cv" and count == slot_count - 1:
            slot_index = min(slot_count - 1, index + 1)
            variant_index = 0
        elif variants_per_slot > 1:
            slot_index = min(slot_count - 1, index // variants_per_slot)
            variant_index = index % variants_per_slot
        elif count == slot_count:
            slot_index = index
            variant_index = 0
        else:
            slot_index = min(slot_count - 1, int(index * slot_count / max(1, count)))
            variant_index = 0
        role = _infer_role(row.alias, variant_index, variants_per_slot, slot_count)
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=slot_index,
                variant_index=variant_index,
                variants_per_slot=variants_per_slot,
                role=role,
            )
        )
    return finish(tuple(planned))


def _refine_plan_with_recording_semantics(
    plan: tuple[PlannedAlias, ...],
    wav_name: str | None,
    format_type: str,
) -> tuple[PlannedAlias, ...]:
    """Refine slots from filename/alias meaning without using source timing."""
    if not wav_name or format_type not in {"cv", "cvc", "cvvc", "vcv"} or len(plan) <= 1:
        return plan
    if format_type == "vcv":
        return _refine_vowel_initial_vcv_terminal_coda_plan(plan, wav_name)
    # Continuous-kana CVVC banks currently couple row conventions to the local
    # renderer differently from separator-based banks. The measured holdout gate
    # is worse when semantic slot replacement is enabled for that layout.
    if format_type == "cvvc" and len(parse_kana_slots(Path(wav_name).stem)) > 1:
        return plan
    slots = _recording_slot_descriptors(wav_name, format_type)
    if len(slots) <= 1:
        return plan
    if format_type == "cv":
        return _refine_cv_plan(plan, slots)
    repeated_cv_overrides = _repeated_cv_slot_overrides(plan, slots)
    refined: list[PlannedAlias] = []
    for index, item in enumerate(plan):
        repeated_slot = repeated_cv_overrides.get(index)
        if repeated_slot is not None:
            refined.append(replace(item, slot_index=repeated_slot, role="cv"))
            continue
        match = _match_alias_to_recording_slot(item.source.alias, slots, format_type)
        if match is None:
            refined.append(item)
            continue
        slot_index, matched_role = match
        role = item.role if item.role in {"breath", "release"} else matched_role
        refined.append(replace(item, slot_index=slot_index, role=role))
    return tuple(refined)


def _repeated_cv_slot_overrides(
    plan: tuple[PlannedAlias, ...],
    slots: tuple[tuple[str, bool], ...],
) -> dict[int, int]:
    """Map numbered aliases such as ``weo2`` to repeated filename occurrences.

    A plain repeated alias takes the earliest remaining occurrence, regardless
    of whether the numbered alias was written first in the source OTO.  This
    uses only alias spelling and filename topology, never source timing.
    """

    slot_indices: dict[str, list[int]] = defaultdict(list)
    for slot_index, (slot_key, _repeated) in enumerate(slots):
        slot_indices[slot_key].append(slot_index)
    candidates: dict[str, list[tuple[int, int | None]]] = defaultdict(list)
    for plan_index, item in enumerate(plan):
        if item.role != "cv":
            continue
        parsed = _cv_alias_surface_and_occurrence(item.source.alias)
        if parsed is None:
            continue
        surface, occurrence = parsed
        if len(slot_indices.get(surface, ())) > 1:
            candidates[surface].append((plan_index, occurrence))

    overrides: dict[int, int] = {}
    for surface, aliases in candidates.items():
        available = list(slot_indices[surface])
        numbered = [(index, occurrence) for index, occurrence in aliases if occurrence is not None]
        if len({occurrence for _index, occurrence in numbered}) != len(numbered):
            continue
        if any(occurrence is None or occurrence < 1 or occurrence > len(available) for _index, occurrence in numbered):
            continue
        for plan_index, occurrence in numbered:
            assert occurrence is not None
            slot_index = available[occurrence - 1]
            overrides[plan_index] = slot_index
            available.remove(slot_index)
        plain = [plan_index for plan_index, occurrence in aliases if occurrence is None]
        if len(plain) != len(available):
            for plan_index, _occurrence in numbered:
                overrides.pop(plan_index, None)
            continue
        for plan_index, slot_index in zip(plain, available):
            overrides[plan_index] = slot_index
    return overrides


def _cv_alias_surface_and_occurrence(alias: str) -> tuple[str, int | None] | None:
    value = _strip_matching_suffix(alias).strip()
    if not value or len(value.split()) != 1:
        return None
    match = re.fullmatch(r"(.+?)(\d+)?", value)
    if match is None:
        return None
    surface = _surface_phone_key(match.group(1))
    if not surface:
        return None
    occurrence_text = match.group(2)
    return surface, int(occurrence_text) if occurrence_text else None


def _refine_vowel_initial_vcv_terminal_coda_plan(
    plan: tuple[PlannedAlias, ...],
    wav_name: str,
) -> tuple[PlannedAlias, ...]:
    """Map vowel-initial DiKOR VCV transitions before a marked final coda.

    This path is deliberately limited to the documented uppercase terminal
    markers.  Consonant-initial VCV banks retain their existing planner.
    """

    if plan[0].source.alias.strip().startswith("-"):
        return plan
    slots = _recording_slot_descriptors(wav_name, "vcv")
    if len(slots) <= 1:
        return plan
    terminal_owner = _dikor_terminal_coda_owner(recording_tokens(wav_name)[-1])
    if not terminal_owner:
        return plan
    slots = (*slots[:-1], (terminal_owner, slots[-1][1]))
    refined: list[PlannedAlias] = []
    for item in plan:
        if _infer_semantic_role(item.source.alias) != "vv":
            refined.append(item)
            continue
        match = _match_alias_to_recording_slot(item.source.alias, slots, "vcv")
        if match is None or match[1] != "vv":
            refined.append(item)
            continue
        slot_index, role = match
        refined.append(replace(item, slot_index=slot_index, role=role))
    return tuple(refined)


def _dikor_terminal_coda_owner(raw_token: str) -> str:
    """Return the CV owner for a case-sensitive DiKOR terminal coda marker."""

    token = raw_token.strip().strip("_'- ")
    for suffix in ("NGH", "K", "N", "L"):
        if token.endswith(suffix) and len(token) > len(suffix):
            owner = token[: -len(suffix)].casefold()
            if _phone_vowel(owner):
                return owner
    return ""


def _refine_cv_plan(
    plan: tuple[PlannedAlias, ...],
    slots: tuple[tuple[str, bool], ...],
) -> tuple[PlannedAlias, ...]:
    """Map sparse CV aliases to filename slots in written order.

    Most CV reclists use the first filename token only as a guide, so the
    established count-based plan starts at slot 1. A measured edge case is a
    repeated leading token (``a-a-i-...``) whose first alias intentionally owns
    the first occurrence. Resolve only aliases with an explicit surface match;
    ambiguous non-leading rows retain the established fallback.
    """

    refined: list[PlannedAlias] = []
    previous_slot = -1
    repeated_leading = (
        len(plan) == len(slots) - 1
        and len(slots) >= 2
        and slots[0][0] == slots[1][0]
    )
    for position, item in enumerate(plan):
        key = _surface_phone_key(_strip_matching_suffix(item.source.alias))
        matches = [index for index, (slot_key, _repeated) in enumerate(slots) if key and slot_key == key]
        selected: int | None = None
        if position == 0 and repeated_leading and matches[:2] == [0, 1]:
            selected = 0
        else:
            forward = [index for index in matches if index > previous_slot]
            if len(forward) == 1 or (forward and previous_slot >= 0):
                selected = forward[0]
            elif len(matches) == 1:
                selected = matches[0]
        if selected is None:
            refined.append(item)
            previous_slot = max(previous_slot, item.slot_index)
            continue
        refined.append(replace(item, slot_index=selected, role="cv"))
        previous_slot = selected
    return tuple(refined)


def _recording_slot_descriptors(wav_name: str, format_type: str | None = None) -> tuple[tuple[str, bool], ...]:
    stem = Path(wav_name).stem
    kana_slots = parse_kana_slots(stem)
    if len(kana_slots) > 1:
        return tuple(("".join(phones), False) for phones in kana_slots)
    descriptors: list[tuple[str, bool]] = []
    apostrophe_groups = [group.strip("'-") for group in stem.split("_") if group.strip("'-")]
    # Three-or-more repeated chunks are the measured Korean CVVC layout where
    # each chunk owns a CV-head/transition pair. Two-chunk files occur in banks
    # with incompatible alias ordering, so keep the established fallback there.
    if len(apostrophe_groups) >= 3 and any("'" in group for group in apostrophe_groups):
        for group in apostrophe_groups:
            group_tokens = [token for token in re.split(r"['-]+", group) if token]
            keys = ["".join(character for character in token.casefold() if character.isalpha()) for token in group_tokens]
            if not keys or any(not key or not key.isascii() for key in keys):
                return ()
            repeated_pair = len(keys) == 2 and keys[0] == keys[1]
            descriptors.extend((key, repeated_pair and index == 0) for index, key in enumerate(keys))
        return tuple(descriptors)
    tokens = list(recording_tokens(wav_name))
    if len(tokens) > 1 and re.fullmatch(r"[A-Ga-g](?:#|b)?\d+[A-Za-z]*", tokens[-1]):
        tokens.pop()
    # Keep planning topology consistent with the routed Korean-CVVC parser:
    # a leading standalone ``n`` is a coda attached to the following CV, not
    # its own timing slot.  Leaving it as a slot shifts every following alias.
    if (format_type or "").casefold() == "cvvc" and len(tokens) > 1 and tokens[0].casefold() == "n":
        tokens.pop(0)
    for token in tokens:
        key = "".join(character for character in token.casefold() if character.isalpha())
        if not key or not key.isascii():
            return ()
        repeated = len(key) % 2 == 0 and key[: len(key) // 2] == key[len(key) // 2 :]
        if repeated:
            key = key[: len(key) // 2]
        descriptors.append((key, repeated))
    return tuple(descriptors)


def _match_alias_to_recording_slot(
    alias: str,
    slots: tuple[tuple[str, bool], ...],
    format_type: str,
) -> tuple[int, str] | None:
    value = _strip_matching_suffix(alias).strip()
    semantic_role = _infer_semantic_role(alias)
    if semantic_role == "release":
        # Some CVVC lists encode the release as its own final filename token
        # (for example ``_NG'R.wav``) and provide case/style synonyms for that
        # one event.  The owner-like alias text must not pull those rows back
        # to the preceding coda slot.
        if _release_slot_owner_key(slots[-1][0]) in {"r", "release"}:
            return len(slots) - 1, "release"
        # Multi-vowel ending recordings often expose one release row per
        # vowel (for example ``a R``, ``i R``, ...).  Sending every release
        # to the final slot creates multi-second occurrence shifts.  Use the
        # left vowel only when it identifies exactly one filename slot; keep
        # the established final-slot fallback for ambiguous/repeated vowels.
        parts = value.split()
        if len(parts) >= 2:
            left_key = _surface_phone_key(parts[-2])
            exact_matches = [
                index
                for index, (slot_key, _repeated) in enumerate(slots)
                if left_key and _release_slot_owner_key(slot_key) == left_key
            ]
            if len(exact_matches) == 1:
                return exact_matches[0], "release"
            left_vowel = _phone_vowel(left_key)
            matches = [
                index
                for index, (slot_key, _repeated) in enumerate(slots)
                if left_vowel and _release_slot_vowel(slot_key) == left_vowel
            ]
            if len(matches) == 1:
                return matches[0], "release"
        return len(slots) - 1, "release"
    if value.startswith("-"):
        key = _surface_phone_key(value.lstrip("-").strip())
        matches = [index for index, (slot_key, _repeated) in enumerate(slots) if slot_key == key]
        repeated_starts = [index for index in matches if slots[index][1]]
        if len(repeated_starts) == 1:
            return repeated_starts[0], "cv_head"
        return (matches[0], "cv_head") if len(matches) == 1 else None
    parts = value.split()
    if len(parts) == 1:
        key = _surface_phone_key(parts[0])
        matches = [index for index, (slot_key, _repeated) in enumerate(slots) if slot_key == key]
        repeated_starts = [index for index in matches if slots[index][1]]
        if len(repeated_starts) == 1:
            return repeated_starts[0], "cv"
        if not matches:
            key_onset = _phone_onset(key)
            key_vowel = _phone_vowel(key)
            matches = [
                index
                for index, (slot_key, _repeated) in enumerate(slots)
                if (
                    _phone_onset(slot_key) == key_onset
                    and {key_vowel, _phone_vowel(slot_key)} <= {"o", "eo"}
                )
                or (key.endswith("wo") and slot_key == f"{key[:-2]}weo")
            ]
        return (matches[0], "cv") if key and len(matches) == 1 else None
    if len(parts) != 2:
        return None
    left = _surface_phone_key(parts[0])
    right = _surface_phone_key(parts[1])
    if not left or not right:
        return None
    candidates: list[tuple[int, bool]] = []
    if semantic_role == "vv":
        for index, (slot_key, repeated) in enumerate(slots):
            if repeated and _phone_vowel(slot_key) == right and left in _ROMAN_VOWEL_TOKENS:
                candidates.append((index, True))
            if index + 1 < len(slots):
                next_key = slots[index + 1][0]
                if (
                    not (repeated and next_key == slot_key)
                    and _phone_vowel(slot_key) == left
                    and _phone_vowel(next_key) == right
                ):
                    candidates.append((index, False))
    else:
        for index, (slot_key, repeated) in enumerate(slots):
            if repeated and _phone_vowel(slot_key) == left and _phone_onset(slot_key) == right:
                candidates.append((index, True))
            if index + 1 < len(slots):
                next_key = slots[index + 1][0]
                if _phone_vowel(slot_key) == left and _phone_onset(next_key) == right:
                    candidates.append((index, False))
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        return None
    slot_index, repeated_transition = unique[0]
    if semantic_role == "vv":
        role = "vv"
        if not repeated_transition:
            slot_index = min(len(slots) - 1, slot_index + 1)
    elif format_type == "cvc" and repeated_transition:
        role = "cvc_vc"
    else:
        role = "vc"
    return slot_index, role


def _strip_matching_suffix(value: str) -> str:
    stripped = _strip_alias_style_suffix(value.strip())
    if len(stripped) > 1 and stripped.endswith("V"):
        stripped = stripped[:-1].rstrip()
    return stripped


def _surface_phone_key(value: str) -> str:
    kana_slots = parse_kana_slots(value)
    if len(kana_slots) == 1:
        return "".join(kana_slots[0])
    key = "".join(character for character in value.casefold() if character.isalpha())
    return key if key.isascii() else ""


_PHONE_VOWELS = ("yeo", "yae", "wae", "weo", "ae", "eo", "eu", "oe", "ui", "ya", "ye", "yo", "yu", "wa", "we", "wi", "wo", "a", "i", "u", "e", "o", "n")


def _phone_vowel(key: str) -> str:
    return next((vowel for vowel in _PHONE_VOWELS if key.endswith(vowel)), "")


def _phone_onset(key: str) -> str:
    vowel = _phone_vowel(key)
    return key[: -len(vowel)] if vowel else key


def _release_slot_vowel(key: str) -> str:
    """Return the vowel for release-list tokens such as ``aH``/``eoH``."""
    direct = _phone_vowel(key)
    if direct:
        return direct
    if key.endswith("h"):
        return _phone_vowel(key[:-1])
    return ""


def _release_slot_owner_key(key: str) -> str:
    """Strip the explicit trailing H marker used by Korean release lists."""
    return key[:-1] if key.endswith("h") and len(key) > 1 else key


def _plan_cv_then_vc_role_blocks(
    rows: list[SourceOtoRow],
    slot_count: int,
) -> tuple[PlannedAlias, ...]:
    if slot_count <= 1 or len(rows) < slot_count * 2:
        return ()
    first = rows[:slot_count]
    second = rows[slot_count : slot_count * 2]
    trailing = rows[slot_count * 2 :]
    if not all(_infer_semantic_role(row.alias) == "cv" for row in first):
        return ()
    if not all(_infer_semantic_role(row.alias) == "vc" for row in second):
        return ()
    if trailing and not all(_infer_semantic_role(row.alias) == "cv" for row in trailing):
        return ()
    planned: list[PlannedAlias] = []
    for slot_index, row in enumerate(first):
        planned.append(PlannedAlias(row, slot_index, 0, 3, "cv"))
    for slot_index, row in enumerate(second):
        planned.append(PlannedAlias(row, slot_index, 1, 3, "cvc_vc"))
    trailing_count = len(trailing)
    for occurrence, row in enumerate(trailing):
        if trailing_count <= 1:
            slot_index = 0
        else:
            slot_index = int(round(occurrence * (slot_count - 1) / (trailing_count - 1)))
        planned.append(PlannedAlias(row, slot_index, 2, 3, "cv"))
    return tuple(planned)


def _plan_verified_interleaved_cv_coda_pairs(
    rows: list[SourceOtoRow],
    slot_count: int,
    format_type: str,
    wav_name: str | None,
) -> tuple[PlannedAlias, ...]:
    """Recognize ``CV, coda, CV, coda`` only with filename confirmation.

    Korean CVC/CVVC banks commonly store compact aliases such as ``bba, ap,
    bbi, ip``.  A generic CVVC role counter sends the compact coda rows to
    progressively later slots.  Requiring every even row to match the
    corresponding filename slot and every odd row to begin with that slot's
    vowel keeps this correction deterministic and avoids guessing from row
    order alone.
    """

    # The same layout exists in some legacy CVC banks, but its offset p90 or
    # >200 ms rate regressed in 6/10 measured CVC items despite lower mean
    # error.  Keep this automatic correction CVVC-only until landmark Gold can
    # distinguish those CVC conventions.
    if format_type != "cvvc" or not wav_name or slot_count <= 1:
        return ()
    if len(rows) != slot_count * 2:
        return ()
    slots = _recording_slot_descriptors(wav_name, format_type)
    if len(slots) != slot_count:
        return ()
    planned: list[PlannedAlias] = []
    for slot_index, (slot_key, _repeated) in enumerate(slots):
        cv_row = rows[slot_index * 2]
        coda_row = rows[slot_index * 2 + 1]
        cv_key = _surface_phone_key(_strip_matching_suffix(cv_row.alias))
        if cv_key != slot_key or _infer_semantic_role(cv_row.alias) != "cv":
            return ()
        coda_value = _strip_matching_suffix(coda_row.alias).strip()
        coda_parts = coda_value.split()
        slot_vowel = _phone_vowel(slot_key)
        if not slot_vowel or len(coda_parts) not in {1, 2}:
            return ()
        left_key = _surface_phone_key(coda_parts[0])
        if not left_key.startswith(slot_vowel):
            return ()
        if len(coda_parts) == 1:
            compact_coda = left_key[len(slot_vowel) :]
            if not compact_coda or compact_coda in _ROMAN_VOWEL_TOKENS:
                return ()
        if len(coda_parts) == 2 and _infer_semantic_role(coda_row.alias) != "vc":
            return ()
        planned.append(PlannedAlias(cv_row, slot_index, 0, 2, "cv"))
        # The legacy fallback already places the final compact coda on the
        # final slot.  Changing only its role selects a different timing
        # template without fixing a slot error, and measured holdout rows get
        # worse.  Preserve that established role until the VC template is
        # calibrated on real landmark Gold.
        if slot_index == slot_count - 1:
            coda_role = "cv"
        else:
            coda_role = "vc"
        planned.append(PlannedAlias(coda_row, slot_index, 1, 2, coda_role))
    return tuple(planned)


def _plan_cv_head_then_interleaved_cvc(
    rows: list[SourceOtoRow],
    slot_count: int,
) -> tuple[PlannedAlias, ...]:
    if slot_count <= 1 or len(rows) != slot_count * 3:
        return ()
    head_rows = rows[:slot_count]
    interleaved = rows[slot_count:]
    if not all(row.alias.strip().startswith("-") for row in head_rows):
        return ()
    if not all(
        not interleaved[index].alias.strip().startswith("-")
        and not interleaved[index + 1].alias.strip().startswith("-")
        and len(interleaved[index].alias.strip().split()) == 1
        and len(interleaved[index + 1].alias.strip().split()) == 1
        for index in range(0, len(interleaved), 2)
    ):
        return ()
    planned: list[PlannedAlias] = []
    for slot_index, row in enumerate(head_rows):
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=slot_index,
                variant_index=0,
                variants_per_slot=3,
                role="cv_head",
            )
        )
    for index, row in enumerate(interleaved):
        variant = index % 2
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=index // 2,
                variant_index=variant + 1,
                variants_per_slot=3,
                role="cv" if variant == 0 else "cvc",
            )
        )
    return tuple(planned)


def _plan_transition_format_aliases(
    rows: list[SourceOtoRow],
    slot_count: int,
    wav_name: str | None = None,
) -> tuple[PlannedAlias, ...]:
    """Map CVC/CVVC rows by alias semantics instead of source line order."""
    if slot_count <= 1:
        return ()
    roles = [_infer_semantic_role(row.alias) for row in rows]
    counts: dict[str, int] = defaultdict(int)
    for role in roles:
        counts[role] += 1
    reverse_vc_slots = _reverse_vc_slots_when_filename_confirms(rows, roles, slot_count, wav_name)
    occurrences: dict[str, int] = defaultdict(int)
    planned: list[PlannedAlias] = []
    for row, role in zip(rows, roles):
        occurrence = occurrences[role]
        occurrences[role] += 1
        if role == "cv_head":
            if counts[role] > 1:
                slot_index = min(slot_count - 1, occurrence)
            else:
                slot_index = 1 if counts.get("vc", 0) and _is_consonant_only_head(row.alias) else 0
        elif role == "vc":
            slot_index = (
                reverse_vc_slots[occurrence]
                if reverse_vc_slots is not None
                else min(max(0, slot_count - 2), occurrence)
            )
        elif role == "vv":
            slot_index = min(slot_count - 1, occurrence + 1)
        elif role == "cv":
            leading_skip = 1 if counts[role] < slot_count else 0
            slot_index = min(slot_count - 1, occurrence + leading_skip)
        elif role in {"release", "breath"}:
            slot_index = slot_count - 1
        else:
            slot_index = min(slot_count - 1, occurrence)
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=slot_index,
                variant_index=occurrence,
                variants_per_slot=1,
                role=role,
            )
        )
    return tuple(planned)


def _reverse_vc_slots_when_filename_confirms(
    rows: list[SourceOtoRow],
    roles: list[str],
    slot_count: int,
    wav_name: str | None,
) -> tuple[int, ...] | None:
    """Recognize a tail-to-head Korean CVVC VC block without using OTO timing."""

    if not wav_name or slot_count < 2:
        return None
    expected = _filename_transition_left_surfaces(wav_name)
    vc_rows = [row for row, role in zip(rows, roles) if role == "vc"]
    observed = tuple(_vc_left_surface(row.alias) for row in vc_rows)
    if not all(observed):
        return None
    if len(expected) == slot_count - 1 and len(vc_rows) == len(expected):
        if observed == tuple(reversed(expected)):
            return tuple(reversed(range(slot_count - 1)))
        return None
    # Korean CVVC ``n_CV...`` owns the initial n/b boundary inside its first
    # slot.  The remaining VC rows are still written tail-to-head, so retain a
    # slot-zero placeholder for the dedicated in-slot n-b mapper.
    if (
        len(expected) == slot_count
        and len(vc_rows) == len(expected)
        and expected[0] == "n"
        and observed == tuple(reversed(expected))
    ):
        return (*reversed(range(slot_count - 1)), 0)
    return None


def _filename_transition_left_surfaces(wav_name: str) -> tuple[str, ...]:
    tokens = recording_tokens(wav_name)
    if len(tokens) < 2:
        return ()
    surfaces = tuple(_filename_token_vowel_surface(token) for token in tokens)
    return surfaces[:-1] if all(surfaces) else ()


def _vc_left_surface(alias: str) -> str:
    parts = _strip_matching_suffix(alias).split()
    return _filename_token_vowel_surface(parts[0]) if len(parts) >= 2 else ""


def _filename_token_vowel_surface(value: str) -> str:
    token = "".join(character for character in value.casefold() if character.isalpha())
    if not token:
        return ""
    for suffix, surface in (
        ("eui", "eu"), ("yeo", "eo"), ("weo", "eo"),
        ("ya", "a"), ("ye", "e"), ("yo", "o"), ("yu", "u"),
        ("wa", "a"), ("we", "e"), ("wi", "i"),
        ("eo", "eo"), ("eu", "eu"), ("ae", "ae"),
        ("a", "a"), ("i", "i"), ("u", "u"), ("e", "e"), ("o", "o"),
    ):
        if token.endswith(suffix):
            return surface
    return token


def _plan_vcv_aliases(rows: list[SourceOtoRow], slot_count: int) -> tuple[PlannedAlias, ...]:
    leading = rows[0].alias.strip().startswith("-")
    trailing = _is_tail_release_alias(rows[-1].alias)
    two_part_suffixes = [row.alias.strip().casefold().split() for row in rows]
    suffixes = {parts[-1] for parts in two_part_suffixes if len(parts) >= 2}
    repeated_coda_series = not leading and len(suffixes) == 1 and all(len(parts) >= 2 for parts in two_part_suffixes)
    planned: list[PlannedAlias] = []
    for index, row in enumerate(rows):
        if leading and index == 0:
            slot_index = 0
            role = "cv_head"
        elif trailing and index == len(rows) - 1:
            slot_index = max(0, slot_count - 1)
            role = "release"
        elif leading:
            slot_index = min(slot_count - 1, max(0, index - 1))
            role = "vcv"
        else:
            slot_index = min(slot_count - 1, index)
            role = "vc_end" if repeated_coda_series else "vcv"
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=slot_index,
                variant_index=0,
                variants_per_slot=1,
                role=role,
            )
        )
    return tuple(planned)


def _detect_grouped_variant_layout(rows: list[SourceOtoRow], slot_count: int) -> int:
    if slot_count <= 1 or len(rows) % slot_count != 0:
        return 0
    block_count = len(rows) // slot_count
    if block_count not in {2, 3}:
        return 0
    blocks = [rows[index * slot_count : (index + 1) * slot_count] for index in range(block_count)]
    leading_flags = [all(row.alias.strip().startswith("-") for row in block) for block in blocks]
    if any(leading_flags) and not all(leading_flags):
        return block_count
    return 0


def _plan_interleaved_cvc_with_cv_head(
    rows: list[SourceOtoRow],
    slot_count: int,
) -> tuple[PlannedAlias, ...]:
    if slot_count <= 1 or len(rows) != slot_count * 3:
        return ()
    interleaved = rows[: slot_count * 2]
    head_rows = rows[slot_count * 2 :]
    if not all(row.alias.strip().startswith("-") for row in head_rows):
        return ()
    if not all(
        len(interleaved[index].alias.strip().split()) <= 1
        and not interleaved[index + 1].alias.strip().startswith("-")
        for index in range(0, len(interleaved), 2)
    ):
        return ()
    planned: list[PlannedAlias] = []
    for index, row in enumerate(interleaved):
        variant = index % 2
        transition_role = "cvc_vc" if len(row.alias.strip().split()) >= 2 else "cvc"
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=index // 2,
                variant_index=variant,
                variants_per_slot=3,
                role="cv" if variant == 0 else transition_role,
            )
        )
    for slot_index, row in enumerate(head_rows):
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=slot_index,
                variant_index=2,
                variants_per_slot=3,
                role="cv_head",
            )
        )
    return tuple(planned)


def _plan_interleaved_cvc_with_trailing_vowels(
    rows: list[SourceOtoRow],
    slot_count: int,
) -> tuple[PlannedAlias, ...]:
    transition_slots = slot_count - 1
    if transition_slots <= 0 or len(rows) != transition_slots * 3:
        return ()
    interleaved = rows[: transition_slots * 2]
    vowel_rows = rows[transition_slots * 2 :]
    if not all(
        len(interleaved[index].alias.strip().split()) == 1
        and len(interleaved[index + 1].alias.strip().split()) >= 2
        for index in range(0, len(interleaved), 2)
    ):
        return ()
    if not all(_is_vowel_like_token(row.alias) for row in vowel_rows):
        return ()
    planned: list[PlannedAlias] = []
    for index, row in enumerate(interleaved):
        variant = index % 2
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=index // 2,
                variant_index=variant,
                variants_per_slot=3,
                role="cv" if variant == 0 else "vc",
            )
        )
    for slot_index, row in enumerate(vowel_rows):
        planned.append(
            PlannedAlias(
                source=row,
                slot_index=slot_index,
                variant_index=2,
                variants_per_slot=3,
                role="cv",
            )
        )
    return tuple(planned)


def _is_tail_release_alias(alias: str) -> bool:
    parts = alias.strip().split()
    if not parts:
        return False
    tail = _strip_alias_style_suffix(parts[-1])
    return bool(re.fullmatch(r"R\d*", tail)) or bool(re.fullmatch(r"-\d*", tail))


def _infer_role(alias: str, variant_index: int, variants_per_slot: int, slot_count: int) -> str:
    semantic_role = _infer_semantic_role(alias)
    if semantic_role != "cv":
        return semantic_role
    value = alias.strip()
    folded = value.casefold()
    if variants_per_slot == 3 and variant_index == 1:
        return "cvc"
    if variants_per_slot >= 2 and variant_index == variants_per_slot - 1:
        return "vc"
    if slot_count == 1 and folded.endswith("r"):
        return "release"
    return "cv"


_ROMAN_VOWEL_TOKENS = {
    "a", "i", "u", "e", "o",
    "ae", "eo", "eu", "eui", "oe", "ui", "wa", "wae", "we", "weo", "wi", "wo",
    "ya", "yae", "ye", "yeo", "yo", "yu",
}
_KANA_VOWEL_TOKENS = set("あいうえおアイウエオ")
_ALIAS_STYLE_SUFFIX_RE = re.compile(r"(?:_[A-Za-z]+|[A-Ga-g](?:#|b)?\d+[A-Za-z]*)$")


def _strip_alias_style_suffix(value: str) -> str:
    previous = value
    while True:
        stripped = _ALIAS_STYLE_SUFFIX_RE.sub("", previous).strip()
        if stripped == previous:
            return stripped
        previous = stripped


def _is_vowel_like_token(value: str) -> bool:
    token = _strip_alias_style_suffix(value).casefold().strip("-_'")
    if token in _ROMAN_VOWEL_TOKENS:
        return True
    return len(token) == 1 and token in _KANA_VOWEL_TOKENS


def _infer_semantic_role(alias: str) -> str:
    value = alias.strip()
    folded = value.casefold()
    parts = value.split()
    if folded.startswith("br"):
        return "breath"
    if value.startswith("-"):
        return "cv_head"
    if len(parts) >= 2 and re.fullmatch(r"R\d*", _strip_alias_style_suffix(parts[-1])):
        return "release"
    if len(parts) == 2 and all(_is_vowel_like_token(part) for part in parts):
        return "vv"
    if len(parts) >= 2:
        return "vc"
    return "cv"


def _is_consonant_only_head(alias: str) -> bool:
    value = _strip_alias_style_suffix(alias.strip().lstrip("-").strip())
    if not value or any(character in _KANA_VOWEL_TOKENS for character in value):
        return False
    folded = value.casefold()
    return folded.isascii() and not any(vowel in folded for vowel in "aeiou")


_SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
_KANA_SEPARATORS = set("_-' ・ー\t\r\n")


def _kana_surface_slots(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value)
    slots: list[str] = []
    pending = ""
    seen_kana = False
    for character in normalized:
        codepoint = ord(character)
        is_kana = 0x3040 <= codepoint <= 0x30FF
        if character in _KANA_SEPARATORS:
            continue
        if character.isascii():
            if character.isdigit() or character.casefold() in set("abcdefg#b"):
                continue
            return ()
        if not is_kana:
            return ()
        seen_kana = True
        if character in {"っ", "ッ"}:
            pending += character
        elif character in _SMALL_KANA and slots:
            slots[-1] += character
        else:
            slots.append(pending + character)
            pending = ""
    return tuple(slots) if seen_kana else ()
