from __future__ import annotations

import re
from dataclasses import dataclass, replace

from core.format_type_utils import normalize_format_type


_ROMAN_VOWELS = {"a", "e", "i", "o", "u"}
_ROMAN_STOP = {
    "k", "g", "t", "d", "p", "b", "kk", "gg", "tt", "dd", "pp", "bb", "c", "q",
}
_ROMAN_AFFRICATE = {"ch", "ts", "j", "dz", "tsh", "dzh", "jj"}
_ROMAN_FRICATIVE = {"s", "sh", "z", "f", "h", "v", "x"}
_ROMAN_SONORANT = {
    "m", "n", "ng", "r", "l", "y", "w",
    "ny", "my", "ry", "hy", "by", "py", "ly",
}
_ROMAN_GLIDE = {"y", "w", "j"}

_JA_STOP_CHARS = set("かきくけこがぎぐげごたちつてとだぢづでどぱぴぷぺぽばびぶべぼ")
_JA_AFFRICATE_CHARS = set("ちつじぢ")
_JA_FRICATIVE_CHARS = set("さしすせそざじずぜぞはひふへほふ")
_JA_SONORANT_CHARS = set("なにぬねのまみむめもらりるれろやゆよわゐゑをん")


@dataclass(frozen=True)
class ContinuityTuning:
    backtrack_max: float
    min_gap: float
    step_max: float
    cutoff_delta_max: float
    min_cut_gap: float


def resolve_alias_bucket(alias_type: str) -> str:
    t = str(alias_type or "").strip().lower()
    if t in {"cv", "cv_head", "mono", "vcv"}:
        return "cv"
    if t == "vc":
        return "vc"
    if t == "vv":
        return "vv"
    return ""


def detect_edge_role(alias_text: str) -> str:
    alias = str(alias_text or "").strip()
    if not alias:
        return "mid"
    if alias.startswith("-"):
        return "head"
    if alias.endswith("-") or " -" in alias:
        return "tail"
    return "mid"


def _tokenize_alias(alias_text: str) -> list[str]:
    text = str(alias_text or "")
    return [tok for tok in re.findall(r"[A-Za-z]+|[가-힣]+|[ぁ-ゖァ-ヺー]+", text) if tok]


def _is_vowel_like_token(token: str) -> bool:
    t = str(token or "").strip().lower()
    if not t:
        return False
    if t in _ROMAN_VOWELS:
        return True
    if len(t) == 1 and t in {"a", "e", "i", "o", "u"}:
        return True
    return False


def _pick_consonant_token(alias_text: str, bucket: str) -> str:
    tokens = _tokenize_alias(alias_text)
    if not tokens:
        return ""
    if bucket == "vc":
        for tok in reversed(tokens):
            if not _is_vowel_like_token(tok):
                return tok
    elif bucket == "cv":
        for tok in tokens:
            if not _is_vowel_like_token(tok):
                return tok
    return ""


def _classify_hangul_onset(token: str) -> str:
    if not token:
        return "other"
    ch = str(token)[0]
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return "other"
    onset_idx = (code - 0xAC00) // 588
    # ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ
    if onset_idx in {0, 1, 3, 4, 7, 8, 12, 13, 14, 15, 16, 17}:
        return "stop"
    if onset_idx in {2, 5, 6, 11}:
        return "sonorant"
    if onset_idx in {9, 10, 18}:
        return "fricative"
    return "other"


def _classify_japanese_kana(token: str) -> str:
    if not token:
        return "other"
    ch = str(token)[0]
    if ch in _JA_STOP_CHARS:
        return "stop"
    if ch in _JA_AFFRICATE_CHARS:
        return "affricate"
    if ch in _JA_FRICATIVE_CHARS:
        return "fricative"
    if ch in _JA_SONORANT_CHARS:
        return "sonorant"
    return "other"


def classify_consonant_type(token: str, language: str) -> str:
    t = str(token or "").strip().lower()
    if not t:
        return "none"
    if re.fullmatch(r"[a-z]+", t):
        if t in _ROMAN_STOP:
            return "stop"
        if t in _ROMAN_AFFRICATE:
            return "affricate"
        if t in _ROMAN_FRICATIVE:
            return "fricative"
        if t in _ROMAN_GLIDE:
            return "glide"
        if t in _ROMAN_SONORANT:
            return "sonorant"
        if len(t) <= 2 and t in {"m", "n", "r", "l", "y", "w"}:
            return "sonorant"
        return "other"
    lang = str(language or "").strip().lower()
    if re.search(r"[가-힣]", t):
        return _classify_hangul_onset(t)
    if re.search(r"[ぁ-ゖァ-ヺー]", t):
        if lang == "japanese":
            return _classify_japanese_kana(t)
    return "other"


def _default_tuning(bucket: str) -> ContinuityTuning:
    if bucket == "cv":
        return ContinuityTuning(
            backtrack_max=8.0,
            min_gap=2.0,
            step_max=900.0,
            cutoff_delta_max=320.0,
            min_cut_gap=8.0,
        )
    if bucket == "vc":
        return ContinuityTuning(
            backtrack_max=10.0,
            min_gap=2.0,
            step_max=1000.0,
            cutoff_delta_max=420.0,
            min_cut_gap=8.0,
        )
    if bucket == "vv":
        return ContinuityTuning(
            backtrack_max=12.0,
            min_gap=1.5,
            step_max=1200.0,
            cutoff_delta_max=520.0,
            min_cut_gap=8.0,
        )
    return ContinuityTuning(8.0, 2.0, 900.0, 320.0, 8.0)


def _apply_edge_and_phoneme_mods(
    tuning: ContinuityTuning,
    *,
    edge_role: str,
    consonant_type: str,
) -> ContinuityTuning:
    t = tuning
    if edge_role == "head":
        t = replace(
            t,
            backtrack_max=max(2.0, t.backtrack_max - 2.0),
            step_max=max(200.0, t.step_max - 120.0),
            cutoff_delta_max=max(80.0, t.cutoff_delta_max - 60.0),
        )
    elif edge_role == "tail":
        t = replace(
            t,
            backtrack_max=max(2.0, t.backtrack_max - 1.0),
            step_max=max(200.0, t.step_max - 60.0),
            cutoff_delta_max=max(80.0, t.cutoff_delta_max - 30.0),
        )

    c = str(consonant_type or "").strip().lower()
    if c in {"stop", "affricate"}:
        t = replace(
            t,
            min_gap=t.min_gap + 0.5,
            cutoff_delta_max=max(80.0, t.cutoff_delta_max - 80.0),
        )
    elif c in {"sonorant", "glide"}:
        t = replace(
            t,
            backtrack_max=max(2.0, t.backtrack_max - 2.0),
            step_max=max(200.0, t.step_max - 140.0),
            cutoff_delta_max=max(80.0, t.cutoff_delta_max - 70.0),
        )
    elif c == "fricative":
        t = replace(
            t,
            cutoff_delta_max=max(80.0, t.cutoff_delta_max - 30.0),
        )
    return t


def _apply_format_specific_tuning(
    tuning: ContinuityTuning,
    *,
    language: str,
    format_type: str,
    bucket: str,
    edge_role: str,
    consonant_type: str,
) -> ContinuityTuning:
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower()
    t = tuning

    if fmt in {"cvvc", "cvc"} and bucket == "cv":
        t = replace(
            t,
            min_gap=t.min_gap + 0.5,
            cutoff_delta_max=max(80.0, t.cutoff_delta_max - 30.0),
        )
    if fmt in {"cvvc", "cvc"} and bucket in {"vc", "vv"} and str(consonant_type) in {"sonorant", "glide"}:
        t = replace(
            t,
            backtrack_max=max(2.0, t.backtrack_max - 1.0),
            step_max=max(200.0, t.step_max - 100.0),
            cutoff_delta_max=max(80.0, t.cutoff_delta_max - 60.0),
        )
    if lang == "japanese" and fmt == "cvvc" and edge_role == "head":
        t = replace(
            t,
            min_gap=t.min_gap + 0.5,
            backtrack_max=max(2.0, t.backtrack_max - 1.0),
        )
    if lang == "korean" and fmt in {"cvvc", "cvc"} and bucket == "vc":
        t = replace(
            t,
            cutoff_delta_max=max(80.0, t.cutoff_delta_max - 30.0),
        )
    return t


def resolve_continuity_tuning(
    *,
    language: str,
    format_type: str,
    alias_type: str,
    alias_text: str,
) -> tuple[str, str, str, ContinuityTuning]:
    bucket = resolve_alias_bucket(alias_type)
    edge_role = detect_edge_role(alias_text)
    consonant_token = _pick_consonant_token(alias_text, bucket)
    consonant_type = classify_consonant_type(consonant_token, language)
    tuning = _default_tuning(bucket)
    tuning = _apply_edge_and_phoneme_mods(
        tuning,
        edge_role=edge_role,
        consonant_type=consonant_type,
    )
    tuning = _apply_format_specific_tuning(
        tuning,
        language=language,
        format_type=format_type,
        bucket=bucket,
        edge_role=edge_role,
        consonant_type=consonant_type,
    )
    tuning = replace(
        tuning,
        backtrack_max=max(0.0, float(tuning.backtrack_max)),
        min_gap=max(0.0, float(tuning.min_gap)),
        step_max=max(40.0, float(tuning.step_max)),
        cutoff_delta_max=max(20.0, float(tuning.cutoff_delta_max)),
        min_cut_gap=max(2.0, float(tuning.min_cut_gap)),
    )
    return bucket, edge_role, consonant_type, tuning


__all__ = [
    "ContinuityTuning",
    "classify_consonant_type",
    "detect_edge_role",
    "resolve_alias_bucket",
    "resolve_continuity_tuning",
]
