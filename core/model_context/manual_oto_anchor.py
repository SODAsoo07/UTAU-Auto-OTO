from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

VOWELS = frozenset("aeiou")
SILENCE_ALIASES = frozenset({"", "sil", "pau", "sp", "ap", "rest", "r"})
BREATH_ALIASES = frozenset({"br", "bre", "breath", "endbr", "exh", "axh", "息", "息吸", "吸"})
_PITCH_SUFFIX_RE = re.compile(r"(?:[a-g](?:#|b)?-?\d+p?)$", re.IGNORECASE)


def _roman_alias_stem_is_phone_like(stem: str) -> bool:
    text = str(stem or "").strip().lower()
    if not text or not re.fullmatch(r"[a-z]{1,6}", text):
        return False
    vowels = {
        "eui",
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
        "ui",
        "a",
        "i",
        "u",
        "e",
        "o",
        "n",
    }
    consonants = {
        "b",
        "ch",
        "d",
        "dh",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "m",
        "ng",
        "ny",
        "p",
        "r",
        "s",
        "sh",
        "ss",
        "t",
        "ts",
        "v",
        "w",
        "y",
        "z",
    }
    yoon = {"ky", "gy", "ny", "hy", "by", "py", "my", "ry", "jy", "dy", "ty"}
    if not any(ch in VOWELS for ch in text) and text not in (vowels | consonants | yoon):
        return False
    units = tuple(sorted(vowels | consonants | yoon, key=len, reverse=True))
    pos = 0
    while pos < len(text):
        unit = next((item for item in units if text.startswith(item, pos)), None)
        if unit is None:
            return False
        pos += len(unit)
    return True


def _strip_roman_attached_pitch_suffix_token(token: str) -> str:
    raw = str(token or "").strip()
    if not raw or not all(char.isascii() for char in raw):
        return ""
    suffix_re = re.compile(r"[A-Za-z]{0,4}[A-Ga-g](?:#|b)?[0-8][A-Za-z]{0,4}")
    for split in range(len(raw) - 1, 0, -1):
        stem = raw[:split].rstrip("_- ").strip()
        suffix = raw[split:]
        if not stem or not suffix:
            continue
        if not suffix_re.fullmatch(suffix):
            continue
        if _roman_alias_stem_is_phone_like(stem):
            return stem
    return ""


def _strip_roman_attached_pitch_suffix_alias(raw: str) -> str:
    parts = [part for part in str(raw or "").strip().split() if part]
    if not parts:
        return ""
    stripped = _strip_roman_attached_pitch_suffix_token(parts[-1])
    if not stripped:
        return ""
    return " ".join([*parts[:-1], stripped]).strip()


def _breath_alias_attached_pitch_style_suffix_stem(token: str) -> str:
    raw = str(token or "").strip()
    if not raw or any(char.isspace() for char in raw) or not raw.isascii():
        return ""
    lowered = raw.lower()
    for stem in ("breath", "bre", "br"):
        if not lowered.startswith(stem):
            continue
        suffix = raw[len(stem) :]
        if suffix and re.fullmatch(r"[A-Ga-g](?:#|b)?[0-8][A-Za-z]{0,4}", suffix):
            return stem
    return ""


@dataclass(frozen=True)
class ManualOtoAnchorRecord:
    raw_params: dict[str, float]
    anchors_abs_ms: dict[str, float]
    repaired_anchors_abs_ms: dict[str, float]
    warnings: tuple[str, ...]
    usable_as_target: bool
    label_quality: str


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clean_alias(alias: object) -> str:
    return str(alias or "").strip()


def _alias_core(alias: object) -> str:
    raw = _clean_alias(alias)
    text = raw.lower()
    # Auto_OTO often appends voicebank suffixes with `_suffix`; role
    # classification should stay tied to the spoken alias body.
    text = text.split("_", 1)[0].strip()
    breath_stem = _breath_alias_attached_pitch_style_suffix_stem(raw.split("_", 1)[0].strip())
    if breath_stem:
        return breath_stem
    kana_positions = [
        idx for idx, char in enumerate(raw)
        if "\u3040" <= char <= "\u30ff"
    ]
    if kana_positions:
        suffix_start = kana_positions[-1] + 1
        suffix = raw[suffix_start:]
        if suffix and re.fullmatch(r"[A-Za-z]{0,4}[A-Ga-g](?:#|b)?[0-8][A-Za-z]{0,4}", suffix):
            return raw[:suffix_start].strip().lower()
    roman_stripped = _strip_roman_attached_pitch_suffix_alias(raw)
    if roman_stripped:
        return roman_stripped.lower()
    match = _PITCH_SUFFIX_RE.search(raw)
    if match is not None and match.start() > 0:
        start = match.start()
        if (
            start > 0
            and raw[start - 1].isupper()
            and re.fullmatch(r"[A-Z]", raw[start - 1])
            and (raw[: start - 1] or "") in {"R", "L"}
        ):
            start -= 1
        stripped = raw[:start].rstrip("_- ").strip().lower()
        if stripped:
            return stripped
    return _PITCH_SUFFIX_RE.sub("", text).strip()


def _split_alias_parts(alias: object) -> list[str]:
    return [part for part in _alias_core(alias).replace("-", " - ").split() if part]


def _alias_part_order_terms(part: object, *, language: str = "") -> list[str]:
    text = str(part or "").strip().lower().strip("*_")
    if not text or text == "-":
        return []
    text = re.sub(r"[^\w\u3041-\u3096\u30A1-\u30FA\u30FC]+", "", text)
    if not text:
        return []
    lang = str(language or "").strip().lower()
    if lang in {"ja", "jp", "japanese"}:
        try:
            from core.ja_lab_generator import kana_to_romaji_syllables, split_romaji_token_to_syllables

            if re.search(r"[\u3041-\u3096\u30A1-\u30FA\u30FC]", text):
                terms = kana_to_romaji_syllables(text)
            elif re.fullmatch(r"[a-z]+", text):
                if not any(ch in VOWELS for ch in text) and _roman_alias_stem_is_phone_like(text):
                    return [text]
                terms = split_romaji_token_to_syllables(text)
            else:
                terms = []
        except Exception:
            terms = []
        if terms:
            return [str(term).strip().lower() for term in terms if str(term).strip()]
    return [text]


def manual_oto_alias_order_terms(
    alias: object,
    *,
    language: str = "",
    format_type: str = "",
    alias_role: str = "",
) -> list[str]:
    parts = [part for part in _split_alias_parts(alias) if part != "-"]
    if not parts:
        return []
    role = str(alias_role or "").strip().lower() or classify_manual_oto_alias_role(
        alias,
        language=language,
        format_type=format_type,
    )
    fmt = str(format_type or "").strip().lower()
    if fmt == "cvvc" and role in {"vc", "vv"} and len(parts) >= 2:
        ordered_parts = [parts[0], parts[-1], *parts[1:-1]]
    elif role in {"vc", "vcv", "vv"} and len(parts) >= 2:
        ordered_parts = [parts[-1], parts[0], *parts[1:-1]]
    elif role == "v" and parts:
        ordered_parts = [parts[-1]]
    else:
        ordered_parts = parts

    out: list[str] = []
    for part in ordered_parts:
        for term in _alias_part_order_terms(part, language=language):
            if term and term not in out:
                out.append(term)
    return out


def _is_vowel_like(token: str) -> bool:
    text = str(token or "").strip().lower()
    return bool(text) and all(ch in VOWELS or ch in {"n", "ん", "ン"} for ch in text)


def _has_vowel(token: str) -> bool:
    return any(ch in VOWELS for ch in str(token or "").strip().lower())


def classify_manual_oto_alias_role(
    alias: object,
    *,
    language: str = "",
    format_type: str = "",
) -> str:
    text = _alias_core(alias)
    if not text:
        return "invalid"
    if text in SILENCE_ALIASES:
        return "sil"
    if text in BREATH_ALIASES:
        return "br"

    parts = _split_alias_parts(text)
    if not parts:
        return "invalid"

    if parts[0] == "-" and len(parts) >= 2:
        return "cv"
    if parts[-1] == "-":
        return "v"

    spoken = [part for part in parts if part != "-"]
    if len(spoken) >= 2:
        left = spoken[0]
        right = spoken[-1]
        fmt = str(format_type or "").strip().lower()
        if right == "-":
            return "v"
        if fmt == "cvvc":
            return "vv" if _is_vowel_like(left) and _is_vowel_like(right) else "vc"
        if fmt == "vcv":
            return "vv" if _is_vowel_like(left) and _is_vowel_like(right) else "vcv"
        if _is_vowel_like(left) and _is_vowel_like(right):
            return "vv"
        if _is_vowel_like(left) and _has_vowel(right):
            return "vcv"
        # CVVC transition rows are intentionally collapsed into VC.
        return "vc"

    token = spoken[0]
    if _is_vowel_like(token):
        return "v"
    if _has_vowel(token):
        return "cv"
    if str(format_type or "").strip().lower() in {"cv", "cvvc", "vcv"}:
        return "cv"
    if token in SILENCE_ALIASES:
        return "sil"
    if token in BREATH_ALIASES:
        return "br"
    return "other"


def normalize_manual_oto_params(params: Mapping[str, object]) -> dict[str, float]:
    return {
        "offset": _as_float(params.get("offset", params.get("off", 0.0))),
        "consonant": _as_float(params.get("consonant", params.get("cons", 0.0))),
        "cutoff": _as_float(params.get("cutoff", params.get("cut", 0.0))),
        "preutterance": _as_float(params.get("preutterance", params.get("pre", 0.0))),
        "overlap": _as_float(params.get("overlap", params.get("ovl", 0.0))),
    }


def manual_oto_params_to_anchor_record(
    params: Mapping[str, object],
    *,
    duration_ms: float,
) -> ManualOtoAnchorRecord:
    raw = normalize_manual_oto_params(params)
    duration = max(0.0, _as_float(duration_ms))
    offset_abs = raw["offset"]
    overlap_abs = offset_abs + raw["overlap"]
    pre_abs = offset_abs + raw["preutterance"]
    fixed_end_abs = offset_abs + raw["consonant"]
    if raw["cutoff"] < 0.0 and duration > 0.0:
        cutoff_abs = duration + raw["cutoff"]
    else:
        cutoff_abs = offset_abs + raw["cutoff"]

    anchors = {
        "offset": offset_abs,
        "overlap": overlap_abs,
        "preutterance": pre_abs,
        "fixed_end": fixed_end_abs,
        "cutoff": cutoff_abs,
    }

    warnings: list[str] = []
    if duration <= 1.0:
        warnings.append("low_duration")
    if all(abs(raw[key]) < 1e-6 for key in raw):
        warnings.append("all_zero_params")
    if raw["cutoff"] >= 0.0:
        warnings.append("positive_cutoff")
    if offset_abs < 0.0 or (duration > 0.0 and offset_abs >= duration):
        warnings.append("offset_out_of_range")
    if overlap_abs < 0.0 or (duration > 0.0 and overlap_abs > duration):
        warnings.append("overlap_out_of_range")
    if pre_abs < 0.0 or (duration > 0.0 and pre_abs > duration):
        warnings.append("preutterance_out_of_range")
    if fixed_end_abs < 0.0 or (duration > 0.0 and fixed_end_abs > duration):
        warnings.append("fixed_end_out_of_range")
    if cutoff_abs < 0.0 or (duration > 0.0 and cutoff_abs > duration):
        warnings.append("cutoff_out_of_range")
    if overlap_abs > pre_abs:
        warnings.append("overlap_after_preutterance")
    if pre_abs > fixed_end_abs:
        warnings.append("preutterance_after_fixed_end")
    if fixed_end_abs >= cutoff_abs:
        warnings.append("fixed_end_after_cutoff")

    repaired_offset = min(max(0.0, offset_abs), max(0.0, duration - 1.0)) if duration > 0.0 else max(0.0, offset_abs)
    repaired_overlap = max(repaired_offset, overlap_abs)
    repaired_pre = max(repaired_overlap, pre_abs)
    repaired_fixed = max(repaired_pre, fixed_end_abs)
    if duration > 0.0:
        repaired_cutoff = min(duration, max(repaired_fixed + 1.0, cutoff_abs))
    else:
        repaired_cutoff = max(repaired_fixed + 1.0, cutoff_abs)
    repaired = {
        "offset": repaired_offset,
        "overlap": repaired_overlap,
        "preutterance": repaired_pre,
        "fixed_end": repaired_fixed,
        "cutoff": repaired_cutoff,
    }

    hard_warnings = {
        "low_duration",
        "all_zero_params",
        "offset_out_of_range",
        "preutterance_out_of_range",
        "fixed_end_after_cutoff",
    }
    usable = not any(flag in hard_warnings for flag in warnings)
    if usable and warnings:
        quality = "flagged_manual"
    elif usable:
        quality = "trusted_manual"
    else:
        quality = "unusable_manual"
    return ManualOtoAnchorRecord(
        raw_params=raw,
        anchors_abs_ms=anchors,
        repaired_anchors_abs_ms=repaired,
        warnings=tuple(warnings),
        usable_as_target=usable,
        label_quality=quality,
    )


__all__ = [
    "ManualOtoAnchorRecord",
    "classify_manual_oto_alias_role",
    "manual_oto_alias_order_terms",
    "manual_oto_params_to_anchor_record",
    "normalize_manual_oto_params",
]
