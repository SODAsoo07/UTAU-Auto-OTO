"""Build boundary-event training manifests from trusted manual/gold OTO files.

This is evaluation/training tooling. Runtime HSMM generation must not import it.

The output is meant to feed the phoneme-boundary model with OTO-derived anchor
events from manually checked OTO rows. It should not be used with generated OTO
files unless the caller deliberately marks the case as untrusted for diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.generation.common.oto_alias_family import alias_family
from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.phoneme_boundary.plan import build_phoneme_plan, phones_from_alias
from core.phoneme_boundary.phones import coarse_for_phone

TRUSTED_LABEL_SOURCES = {
    "manual_gold",
    "manual_oto_gold",
    "human_gold",
    "hand_checked_oto",
}
REJECTED_LABEL_SOURCES = {
    "generated",
    "auto",
    "auto_oto",
    "hsmm",
    "mfa",
    "source_oto",
    "pseudo_label",
}
VOWEL_HINT_TOKENS = {
    "a",
    "i",
    "u",
    "e",
    "o",
    "ae",
    "eo",
    "eu",
    "ya",
    "ye",
    "yo",
    "yu",
    "wa",
    "we",
    "wi",
    "wo",
}
CONSONANT_HINT_TOKENS = {
    "b",
    "bb",
    "ch",
    "d",
    "dd",
    "f",
    "g",
    "gg",
    "h",
    "j",
    "jj",
    "k",
    "kh",
    "l",
    "m",
    "n",
    "ng",
    "p",
    "ph",
    "r",
    "s",
    "sh",
    "ss",
    "t",
    "th",
    "ts",
    "v",
    "w",
    "y",
    "z",
}


def _read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _stable_id(*parts: object, length: int = 16) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_rel_or_abs(path: str, base: str = "") -> str:
    abs_path = os.path.abspath(str(path or ""))
    if not base:
        return abs_path
    try:
        return os.path.relpath(abs_path, os.path.abspath(base))
    except ValueError:
        return abs_path


def _wav_duration_ms(path: str) -> float:
    try:
        with wave.open(path, "rb") as handle:
            frames = int(handle.getnframes())
            rate = max(1, int(handle.getframerate()))
            return 1000.0 * float(frames) / float(rate)
    except Exception:
        return 0.0


def _case_list_from_payload(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("cases"), list):
            return [dict(item) for item in payload["cases"] if isinstance(item, dict)]
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    raise ValueError("cases JSON must be a dict, a list, or {'cases': [...]}")


def load_cases(*, cases_json: str = "", single_case: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if cases_json:
        cases.extend(_case_list_from_payload(_read_json(cases_json)))
    if single_case:
        gold = str(single_case.get("gold") or single_case.get("oto") or "").strip()
        if gold:
            cases.append(dict(single_case))

    normalized: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        gold = os.path.abspath(str(case.get("gold") or case.get("oto") or case.get("manual") or ""))
        if not gold or not os.path.isfile(gold):
            raise FileNotFoundError(f"gold oto.ini not found for case {idx}: {gold}")
        wav_dir_raw = str(case.get("wav_dir") or case.get("wav-dir") or "").strip()
        wav_dir = os.path.abspath(wav_dir_raw) if wav_dir_raw else os.path.dirname(gold)
        case_id = str(case.get("case_id") or case.get("id") or Path(gold).parent.name or f"case_{idx:03d}")
        label_source = str(case.get("label_source") or "manual_oto_gold").strip().lower()
        normalized.append(
            {
                "case_id": case_id,
                "voicebank_id": str(case.get("voicebank_id") or case_id),
                "language": str(case.get("language") or case.get("lang") or "").strip().lower(),
                "format_type": str(case.get("format_type") or case.get("format") or "").strip().lower(),
                "gold": gold,
                "wav_dir": wav_dir,
                "label_source": label_source,
                "trusted": bool(case.get("trusted", label_source in TRUSTED_LABEL_SOURCES)),
            }
        )
    if not normalized:
        raise ValueError("no cases were provided")
    return normalized


def _validate_case_trust(case: dict[str, Any], *, allow_untrusted: bool) -> None:
    label_source = str(case.get("label_source") or "").strip().lower()
    if label_source in REJECTED_LABEL_SOURCES and not allow_untrusted:
        raise ValueError(
            f"{case['case_id']}: label_source '{label_source}' is rejected for boundary training. "
            "Use a hand-checked manual gold OTO or pass --allow-untrusted for diagnostics only."
        )
    trusted = bool(case.get("trusted", label_source in TRUSTED_LABEL_SOURCES))
    if not trusted and not allow_untrusted:
        raise ValueError(
            f"{case['case_id']}: case is not marked trusted. "
            "Set label_source=manual_oto_gold/manual_gold or pass --allow-untrusted for diagnostics only."
        )


def _read_oto_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = read_text_with_fallback(path)
    for line_index, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        row = dict(parsed)
        row["line_index"] = int(line_index)
        row["alias_family"] = alias_family(str(row.get("alias") or ""))
        rows.append(row)
    return rows


def _cutoff_end_abs(offset_ms: float, cutoff: float, duration_ms: float, *, mode: str) -> tuple[float, str]:
    mode_norm = str(mode or "utau").strip().lower()
    if mode_norm == "relative":
        if cutoff < 0.0:
            return offset_ms + abs(cutoff), "negative_relative_to_offset"
        return offset_ms + cutoff, "positive_relative_to_offset"
    if mode_norm == "utau":
        if cutoff < 0.0 and duration_ms > 0.0:
            return max(offset_ms, duration_ms + cutoff), "negative_right_blank_from_end"
        if cutoff < 0.0:
            return offset_ms + abs(cutoff), "negative_relative_to_offset_fallback"
        return offset_ms + cutoff, "positive_relative_to_offset"
    raise ValueError(f"unknown cutoff end mode: {mode}")


def _clamp(value: float, lo: float, hi: float) -> float:
    if hi < lo:
        return float(lo)
    return max(float(lo), min(float(hi), float(value)))


def _tokens_from_alias(alias: str, language: str) -> list[str]:
    return [str(item).strip() for item in phones_from_alias(alias, language) if str(item).strip()]


def _is_vowel_token(token: str, language: str) -> bool:
    value = str(token or "").strip().lower()
    return value in VOWEL_HINT_TOKENS or coarse_for_phone(token, language=language) == "V"


def _is_consonant_token(token: str, language: str) -> bool:
    value = str(token or "").strip().lower()
    return value in CONSONANT_HINT_TOKENS or coarse_for_phone(token, language=language).startswith("C_")


def _alias_phone_hints(alias: str, language: str) -> dict[str, str]:
    tokens = _tokens_from_alias(alias, language)
    parts = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(parts) >= 2:
        left = parts[0]
        right = parts[-1]
        return {
            "left": "" if left == "-" else left,
            "right": "" if right == "-" else right,
            "vowel": right if _is_vowel_token(right, language) else left if _is_vowel_token(left, language) else "",
            "consonant": right if _is_consonant_token(right, language) else left if _is_consonant_token(left, language) else "",
        }
    vowels = [token for token in tokens if _is_vowel_token(token, language)]
    consonants = [token for token in tokens if _is_consonant_token(token, language)]
    return {
        "left": tokens[0] if tokens else "",
        "right": tokens[-1] if tokens else "",
        "vowel": vowels[-1] if vowels else "",
        "consonant": consonants[0] if consonants else "",
    }


def _boundary_alias_family(alias: str, language: str) -> str:
    family = alias_family(alias)
    parts = [part.strip() for part in str(alias or "").split() if part.strip()]
    if len(parts) >= 2:
        left = parts[0]
        right = parts[-1]
        if right == "-":
            return "terminal_v_dash"
        if left == "-":
            return "cv_head"
        if _is_vowel_token(left, language) and _is_consonant_token(right, language):
            return "vc"
        if _is_vowel_token(left, language) and _is_vowel_token(right, language):
            return "vv"
    return family


def _make_event(
    label: str,
    time_ms: float,
    *,
    alias: str,
    alias_index: int,
    alias_family_name: str,
    phone: str = "",
    confidence: float = 1.0,
    source_anchor: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "time_ms": round(max(0.0, float(time_ms)), 3),
        "alias": alias,
        "alias_index": int(alias_index),
        "alias_family": alias_family_name,
        "phone": phone,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
        "source": "manual_oto_profile",
        "source_anchor": source_anchor,
    }


def _events_from_oto_row(
    row: dict[str, Any],
    *,
    alias_index: int,
    duration_ms: float,
    language: str,
    cutoff_end_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alias = str(row.get("alias") or "")
    family = _boundary_alias_family(alias, language)
    offset_ms = max(0.0, _float(row.get("offset")))
    pre_ms = max(0.0, _float(row.get("pre")))
    cons_ms = max(0.0, _float(row.get("cons")))
    cutoff = _float(row.get("cutoff"))
    cutoff_abs, cutoff_mode = _cutoff_end_abs(offset_ms, cutoff, duration_ms, mode=cutoff_end_mode)
    if duration_ms > 0.0:
        cutoff_abs = _clamp(cutoff_abs, offset_ms + 1.0, duration_ms)
    else:
        cutoff_abs = max(offset_ms + 1.0, cutoff_abs)
    pre_abs = _clamp(offset_ms + pre_ms, offset_ms + 1.0, cutoff_abs - 1.0)
    cons_abs = _clamp(offset_ms + cons_ms, offset_ms + 1.0, cutoff_abs - 1.0)
    active_len = max(1.0, cutoff_abs - offset_ms)
    hints = _alias_phone_hints(alias, language)
    vowel_phone = hints.get("vowel", "")
    consonant_phone = hints.get("consonant", "")
    nucleus = _clamp(pre_abs + 0.45 * max(1.0, cutoff_abs - pre_abs), pre_abs, cutoff_abs)

    events: list[dict[str, Any]] = []
    events.append(
        _make_event(
            "phone_start",
            offset_ms,
            alias=alias,
            alias_index=alias_index,
            alias_family_name=family,
            phone=consonant_phone or vowel_phone,
            confidence=0.72,
            source_anchor="offset",
        )
    )
    events.append(
        _make_event(
            "phone_end",
            cutoff_abs,
            alias=alias,
            alias_index=alias_index,
            alias_family_name=family,
            phone=vowel_phone or consonant_phone,
            confidence=0.68,
            source_anchor="cutoff_end",
        )
    )

    if family in {"cv", "cv_head", "vcv", "spaced"}:
        if consonant_phone and pre_abs - offset_ms >= 6.0:
            events.append(
                _make_event(
                    "consonant_onset",
                    offset_ms,
                    alias=alias,
                    alias_index=alias_index,
                    alias_family_name=family,
                    phone=consonant_phone,
                    confidence=0.82,
                    source_anchor="offset",
                )
            )
        events.append(
            _make_event(
                "vowel_onset",
                pre_abs,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone=vowel_phone,
                confidence=0.92,
                source_anchor="offset_plus_preutterance",
            )
        )
        events.append(
            _make_event(
                "vowel_nucleus",
                nucleus,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone=vowel_phone,
                confidence=0.70,
                source_anchor="pre_to_cutoff_midpoint",
            )
        )
        events.append(
            _make_event(
                "vowel_end",
                cutoff_abs,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone=vowel_phone,
                confidence=0.68,
                source_anchor="cutoff_end",
            )
        )
    elif family in {"v", "vv"}:
        onset = pre_abs if pre_ms > 1.0 else offset_ms
        events.append(
            _make_event(
                "vowel_onset",
                onset,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone=vowel_phone,
                confidence=0.78,
                source_anchor="offset_plus_preutterance",
            )
        )
        events.append(
            _make_event(
                "vowel_nucleus",
                nucleus,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone=vowel_phone,
                confidence=0.82,
                source_anchor="pre_to_cutoff_midpoint",
            )
        )
        events.append(
            _make_event(
                "vowel_end",
                cutoff_abs,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone=vowel_phone,
                confidence=0.68,
                source_anchor="cutoff_end",
            )
        )
    elif family == "vc":
        transition_abs = pre_abs
        events.append(
            _make_event(
                "vowel_end",
                transition_abs,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone=hints.get("left", "") or vowel_phone,
                confidence=0.82,
                source_anchor="offset_plus_preutterance",
            )
        )
        if consonant_phone:
            events.append(
                _make_event(
                    "consonant_onset",
                    transition_abs,
                    alias=alias,
                    alias_index=alias_index,
                    alias_family_name=family,
                    phone=consonant_phone,
                    confidence=0.78,
                    source_anchor="offset_plus_preutterance",
                )
            )
    elif family == "terminal_v_dash":
        events.append(
            _make_event(
                "vowel_end",
                pre_abs,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone=vowel_phone or hints.get("left", ""),
                confidence=0.74,
                source_anchor="offset_plus_preutterance",
            )
        )
        events.append(
            _make_event(
                "silence_boundary",
                cutoff_abs,
                alias=alias,
                alias_index=alias_index,
                alias_family_name=family,
                phone="",
                confidence=0.72,
                source_anchor="cutoff_end",
            )
        )

    anchors = {
        "offset_abs_ms": round(offset_ms, 3),
        "pre_abs_ms": round(pre_abs, 3),
        "cons_abs_ms": round(cons_abs, 3),
        "cutoff_abs_ms": round(cutoff_abs, 3),
        "cutoff_mode": cutoff_mode,
        "active_len_ms": round(active_len, 3),
    }
    return _dedupe_events(events), anchors


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int, int], dict[str, Any]] = {}
    for event in events:
        key = (
            str(event.get("label") or ""),
            int(event.get("alias_index") or 0),
            int(round(_float(event.get("time_ms")) / 2.0)),
        )
        previous = best.get(key)
        if previous is None or _float(event.get("confidence")) > _float(previous.get("confidence")):
            best[key] = event
    return sorted(best.values(), key=lambda item: (_float(item.get("time_ms")), str(item.get("label") or "")))


def _expected_phones_for_wav(wav_path: str, aliases: list[str], *, language: str, duration_ms: float) -> list[dict[str, Any]]:
    plan = build_phoneme_plan(
        wav_path=wav_path,
        aliases=aliases,
        language=language,
        duration_ms=duration_ms,
    )
    phones: list[dict[str, Any]] = []
    for alias_plan in plan.aliases:
        for phone in alias_plan.phones:
            phones.append(
                {
                    "phone": phone.text,
                    "phone_index": int(phone.index),
                    "alias": alias_plan.alias,
                    "alias_index": int(alias_plan.alias_index),
                    "coarse": phone.coarse,
                }
            )
    return phones


def build_boundary_event_rows(
    cases: list[dict[str, Any]],
    *,
    cutoff_end_mode: str = "utau",
    relative_to: str = "",
    allow_untrusted: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        _validate_case_trust(case, allow_untrusted=allow_untrusted)
        oto_rows = _read_oto_rows(str(case["gold"]))
        for oto_row in oto_rows:
            oto_row["alias_family"] = _boundary_alias_family(str(oto_row.get("alias") or ""), str(case["language"]))
        by_wav: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for oto_row in oto_rows:
            by_wav[str(oto_row.get("wav") or "")].append(oto_row)

        event_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        skipped_missing_wav = 0
        for wav_name, wav_rows in sorted(by_wav.items()):
            wav_path = os.path.abspath(os.path.join(str(case["wav_dir"]), wav_name))
            if not os.path.isfile(wav_path):
                skipped_missing_wav += 1
            duration_ms = _wav_duration_ms(wav_path)
            aliases = [str(item.get("alias") or "") for item in wav_rows]
            events: list[dict[str, Any]] = []
            oto_anchor_rows: list[dict[str, Any]] = []
            for alias_index, oto_row in enumerate(wav_rows):
                family = str(oto_row.get("alias_family") or "")
                family_counts.update([family])
                row_events, anchors = _events_from_oto_row(
                    oto_row,
                    alias_index=alias_index,
                    duration_ms=duration_ms,
                    language=str(case["language"]),
                    cutoff_end_mode=cutoff_end_mode,
                )
                events.extend(row_events)
                event_counts.update(str(event.get("label") or "") for event in row_events)
                oto_anchor_rows.append(
                    {
                        "alias": str(oto_row.get("alias") or ""),
                        "alias_index": int(alias_index),
                        "alias_family": family,
                        "line_index": int(oto_row.get("line_index") or 0),
                        **anchors,
                    }
                )
            row_id = _stable_id(case["case_id"], wav_name, str(case["gold"]), length=20)
            rows.append(
                {
                    "schema_version": 1,
                    "row_id": row_id,
                    "case_id": str(case["case_id"]),
                    "voicebank_id": str(case["voicebank_id"]),
                    "group_id": f"{case['case_id']}::{wav_name}",
                    "wav_name": wav_name,
                    "wav_path": _safe_rel_or_abs(wav_path, relative_to),
                    "duration_ms": round(duration_ms, 3),
                    "language": str(case["language"]),
                    "format_type": str(case["format_type"]),
                    "label_source": "manual_oto_gold" if bool(case.get("trusted")) else str(case["label_source"]),
                    "label_format": "oto_ini_boundary_events",
                    "label_policy": {
                        "manual_checked_oto_required": not allow_untrusted,
                        "generated_oto_disallowed": not allow_untrusted,
                        "cutoff_end_mode": cutoff_end_mode,
                        "cv_vowel_onset": "offset_plus_preutterance",
                        "vc_vowel_end_and_consonant_onset": "offset_plus_preutterance",
                        "vowel_nucleus": "pre_to_cutoff_midpoint_pseudo_gold",
                    },
                    "aliases": aliases,
                    "expected_phones": _expected_phones_for_wav(
                        wav_path,
                        aliases,
                        language=str(case["language"]),
                        duration_ms=duration_ms,
                    ),
                    "events": sorted(events, key=lambda item: (_float(item.get("time_ms")), int(item.get("alias_index") or 0))),
                    "oto_anchor_rows": oto_anchor_rows,
                    "needs_manual_labels": False,
                    "trainable": bool(events and duration_ms > 0.0),
                }
            )
        case_summaries.append(
            {
                "case_id": str(case["case_id"]),
                "voicebank_id": str(case["voicebank_id"]),
                "language": str(case["language"]),
                "format_type": str(case["format_type"]),
                "gold": os.path.abspath(str(case["gold"])),
                "wav_dir": os.path.abspath(str(case["wav_dir"])),
                "oto_rows": int(len(oto_rows)),
                "wav_groups": int(len(by_wav)),
                "missing_wav_groups": int(skipped_missing_wav),
                "event_counts": dict(sorted(event_counts.items())),
                "alias_family_counts": dict(sorted(family_counts.items())),
            }
        )
    summary = {
        "rows_total": int(len(rows)),
        "trainable_rows": int(sum(1 for row in rows if bool(row.get("trainable")))),
        "cutoff_end_mode": cutoff_end_mode,
        "cases": case_summaries,
        "event_counts": dict(sorted(Counter(event["label"] for row in rows for event in row.get("events", [])).items())),
        "alias_family_counts": dict(
            sorted(Counter(item["alias_family"] for row in rows for item in row.get("oto_anchor_rows", [])).items())
        ),
        "label_source_policy": {
            "trusted_sources": sorted(TRUSTED_LABEL_SOURCES),
            "rejected_sources": sorted(REJECTED_LABEL_SOURCES),
            "allow_untrusted": bool(allow_untrusted),
        },
    }
    return rows, summary


def split_rows(
    rows: list[dict[str, Any]],
    *,
    heldout_ratio: float = 0.2,
    seed: int = 42,
    group_column: str = "group_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in rows if bool(row.get("trainable", True))]
    excluded = len(rows) - len(eligible)
    if not eligible:
        return [], [], {
            "group_column": group_column,
            "groups_total": 0,
            "train_rows": 0,
            "heldout_rows": 0,
            "excluded_non_trainable_rows": int(excluded),
        }
    groups = sorted({str(row.get(group_column) or row.get("group_id") or row.get("row_id")) for row in eligible})
    shuffled = list(groups)
    random.Random(int(seed)).shuffle(shuffled)
    if len(shuffled) <= 1:
        heldout_groups: set[str] = set()
    else:
        heldout_n = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(heldout_ratio)))))
        heldout_groups = set(shuffled[:heldout_n])
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for row in eligible:
        group = str(row.get(group_column) or row.get("group_id") or row.get("row_id"))
        out = dict(row)
        out["split"] = "heldout" if group in heldout_groups else "train"
        if group in heldout_groups:
            heldout.append(out)
        else:
            train.append(out)
    return train, heldout, {
        "group_column": group_column,
        "heldout_ratio": float(heldout_ratio),
        "seed": int(seed),
        "groups_total": int(len(groups)),
        "train_groups": int(len(set(groups) - heldout_groups)),
        "heldout_groups": int(len(heldout_groups)),
        "train_rows": int(len(train)),
        "heldout_rows": int(len(heldout)),
        "excluded_non_trainable_rows": int(excluded),
    }


def write_outputs(
    out_dir: str,
    rows: list[dict[str, Any]],
    train: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, str]:
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    paths = {
        "dataset_jsonl": os.path.join(out_dir_abs, "boundary_events.jsonl"),
        "train_jsonl": os.path.join(out_dir_abs, "train.jsonl"),
        "heldout_jsonl": os.path.join(out_dir_abs, "heldout.jsonl"),
        "summary": os.path.join(out_dir_abs, "summary.json"),
    }
    _write_jsonl(paths["dataset_jsonl"], rows)
    _write_jsonl(paths["train_jsonl"], train)
    _write_jsonl(paths["heldout_jsonl"], heldout)
    summary_out = dict(summary)
    summary_out["outputs"] = paths
    _write_json(paths["summary"], summary_out)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", default="", help="JSON list/dict with case_id, language, format_type, gold, wav_dir")
    parser.add_argument("--case-id", default="", help="Single-case id when not using --cases-json")
    parser.add_argument("--voicebank-id", default="", help="Single-case voicebank id")
    parser.add_argument("--language", default="", help="Single-case language")
    parser.add_argument("--format-type", default="", help="Single-case format type")
    parser.add_argument("--gold", default="", help="Single-case trusted manual/gold oto.ini")
    parser.add_argument("--wav-dir", default="", help="Single-case WAV directory")
    parser.add_argument("--label-source", default="manual_oto_gold")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--relative-to", default="", help="Store wav paths relative to this directory.")
    parser.add_argument("--cutoff-end-mode", choices=("utau", "relative"), default="utau")
    parser.add_argument("--heldout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument(
        "--allow-untrusted",
        action="store_true",
        help="Allow generated/untrusted OTO labels for diagnostics only. Do not train release models from these rows.",
    )
    args = parser.parse_args()

    cases = load_cases(
        cases_json=args.cases_json,
        single_case={
            "case_id": args.case_id,
            "voicebank_id": args.voicebank_id,
            "language": args.language,
            "format_type": args.format_type,
            "gold": args.gold,
            "wav_dir": args.wav_dir,
            "label_source": args.label_source,
        },
    )
    rows, summary = build_boundary_event_rows(
        cases,
        cutoff_end_mode=str(args.cutoff_end_mode),
        relative_to=str(args.relative_to or ""),
        allow_untrusted=bool(args.allow_untrusted),
    )
    train, heldout, split_summary = split_rows(
        rows,
        heldout_ratio=float(args.heldout_ratio),
        seed=int(args.seed),
        group_column=str(args.group_column or "group_id"),
    )
    summary["split"] = split_summary
    paths = write_outputs(str(args.out_dir), rows, train, heldout, summary)
    print(
        json.dumps(
            {
                "rows_total": len(rows),
                "train_rows": len(train),
                "heldout_rows": len(heldout),
                "out_dir": os.path.abspath(str(args.out_dir)),
                "summary": paths["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
