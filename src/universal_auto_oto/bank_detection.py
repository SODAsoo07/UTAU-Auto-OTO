"""Voicebank format detection and language sanity checks for the product UI.

The language is always chosen by the user. The format (CV/CVC/CVVC/VCV) is
detected from the alias patterns of an existing oto.ini when one is given;
without one, file names only separate single-syllable CV banks from list
recordings, and a format named in the folder name is used as a weaker hint.
Japanese CVVC and VCV lists share the same file names (``_ああいあうえあ``), so
file names alone never decide between them.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .i18n import t
from .kana import parse_kana_slots
from .planning import recording_tokens


FORMATS = ("cv", "cvc", "cvvc", "vcv")

_KANA_RE = re.compile(r"[ぁ-ゖァ-ヺ]")
_HANGUL_RE = re.compile(r"[가-힣ᄀ-ᇿㄱ-ㆎ]")
_ROMAN_VOWEL_RE = re.compile(r"[aeiou]")
# Korean romanization spellings that Japanese romaji never uses.
_KOREAN_ROMAN_RE = re.compile(r"(eo|eu|ae|oe|ui)(?![a-z])|(ng|ss|jj|kk|tt|pp|dd|gg|bb)")
_ROMAN_CODAS = ("ng", "n", "m", "l", "k", "t", "p")
_SPECIAL_TOKENS = {"-", "R", "br", "息", "吸", "・", "*", "sil", "pau", "breath"}
_FOLDER_FORMAT_RE = re.compile(r"(vcv|cvvc|cv-vc|cvc|cv)(?![a-z])", re.IGNORECASE)
_PITCH_TOKEN_RE = re.compile(r"^[A-Ga-g][#b]?\d$")


@dataclass(frozen=True)
class FormatDetection:
    format: str | None
    confidence: str  # "high", "medium", "low" or "none"
    source: str  # "base_oto", "folder_name", "filenames" or "none"
    evidence: tuple[str, ...] = ()
    counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": list(self.evidence),
            "counts": dict(self.counts),
        }


@dataclass(frozen=True)
class LanguageCheck:
    language: str
    warning: str | None
    kana_ratio: float
    hangul_ratio: float
    korean_roman_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "warning": self.warning,
            "kana_ratio": round(self.kana_ratio, 3),
            "hangul_ratio": round(self.hangul_ratio, 3),
            "korean_roman_ratio": round(self.korean_roman_ratio, 3),
        }


def strip_alias_suffix(aliases: Sequence[str]) -> tuple[list[str], str]:
    """Remove a bank-wide pitch/expression suffix (``C4``, ``_D4``, `` pop``) and duplicate numbers."""

    values = [str(alias).strip() for alias in aliases if str(alias).strip()]
    suffix = ""
    if values:
        for length in range(8, 0, -1):
            counts = Counter(value[-length:] for value in values if len(value) > length)
            if not counts:
                continue
            candidate, count = counts.most_common(1)[0]
            if count >= 0.6 * len(values) and (candidate[0] in " _" or any(ch.isdigit() for ch in candidate)):
                suffix = candidate
                break
    stripped = []
    for value in values:
        if suffix and value.endswith(suffix) and len(value) > len(suffix):
            value = value[: -len(suffix)]
        stripped.append(re.sub(r"(?<=\D)\d+$", "", value).strip())
    return stripped, suffix


def classify_alias(alias: str) -> str:
    """Classify one suffix-free alias as cv, v, vc_syllable, vc, vcv, vv, head, tail or other."""

    parts = alias.split()
    if not parts:
        return "other"
    if len(parts) == 1:
        token = parts[0]
        if _is_special(token):
            return "other"
        shape = _token_shape(token)
        if shape is None:
            return "other"
        has_vowel, starts_with_vowel, ends_with_coda = shape
        if not has_vowel:
            return "other"
        if not starts_with_vowel:
            return "cv"
        return "vc_syllable" if ends_with_coda else "v"
    if len(parts) == 3:
        # Phonetic spelling of a VCV alias ("a k a", "n g a").
        shapes = [_token_shape(part) for part in parts]
        if all(shape is not None for shape in shapes) and shapes[0][0] and not shapes[1][0] and shapes[2][0]:
            return "vcv"
        return "other"
    if len(parts) != 2:
        return "other"
    left, right = parts
    if left == "-":
        return "head"
    if right == "-" or right.startswith("R") or _is_special(right):
        return "tail"
    if _is_special(left):
        return "other"
    shape = _token_shape(right)
    if shape is None:
        return "other"
    has_vowel, starts_with_vowel, _ = shape
    left_shape = _token_shape(left)
    if left_shape is not None and not left_shape[0]:
        # Spaced CV ("g a") or consonant cluster; neither VCV nor VC.
        return "cv" if has_vowel else "other"
    if not has_vowel:
        return "vc"
    return "vv" if starts_with_vowel else "vcv"


def detect_format_from_aliases(aliases: Sequence[str], *, language: str | None = None) -> FormatDetection:
    stripped, suffix = strip_alias_suffix(aliases)
    counts = Counter(classify_alias(alias) for alias in stripped)
    cv, vcv, vc, vc_syllable = counts["cv"], counts["vcv"], counts["vc"], counts["vc_syllable"]
    core = cv + vcv + vc + vc_syllable
    evidence = [
        t("별칭 {count}개 분석", count=len(stripped))
        + (t(" (접미사 '{suffix}' 제외)", suffix=suffix.strip()) if suffix.strip() else ""),
        t("CV {cv} · VCV {vcv} · VC {vc} · 받침 음절(예: ang) {vc_syllable} · 머리(- CV) {head}", cv=cv, vcv=vcv,
          vc=vc, vc_syllable=vc_syllable, head=counts["head"]),
    ]
    if core < 8:
        return FormatDetection(None, "none", "base_oto", (*evidence, t("판정에 쓸 별칭이 너무 적습니다.")), dict(counts))
    korean = (language or "").casefold() in {"korean", "ko", "kr"}
    if vcv >= 0.3 * core and vcv > cv:
        confidence = "high" if vcv >= 0.6 * core else "medium"
        return FormatDetection("vcv", confidence, "base_oto", (*evidence, t("모음+CV 별칭(a ka)이 대부분입니다.")), dict(counts))
    if vc >= 0.08 * core:
        if korean and vc_syllable >= 0.05 * core:
            return FormatDetection(
                "cvc", "medium", "base_oto",
                (*evidence, t("VC 별칭과 받침 음절 별칭(ang, ot)이 함께 있습니다.")), dict(counts),
            )
        confidence = "high" if cv >= 0.25 * core else "medium"
        return FormatDetection("cvvc", confidence, "base_oto", (*evidence, t("CV와 VC(a k) 별칭이 함께 있습니다.")), dict(counts))
    if cv >= 0.5 * core:
        return FormatDetection("cv", "high" if vc == 0 else "medium", "base_oto",
                               (*evidence, t("단독 CV 별칭이 대부분입니다.")), dict(counts))
    return FormatDetection(None, "low", "base_oto", (*evidence, t("별칭 패턴이 어느 형식과도 뚜렷하게 맞지 않습니다.")), dict(counts))


def detect_format_from_language_pack(
    wav_names: Sequence[str], aliases: Sequence[str], *, language: str | None
) -> FormatDetection | None:
    """Languages served by a language pack: the pack whose symbols read the names best."""

    from .generic_alias_binding import parse_pack_alias
    from .language_pack import builtin_language_packs

    packs = [pack for pack in builtin_language_packs() if (language or "").casefold() in pack.languages]
    if not packs:
        return None
    scored = []
    for pack in packs:
        files = sum(1 for name in wav_names if pack.parse_filename(name) is not None)
        read = sum(1 for alias in aliases if parse_pack_alias(pack, alias) is not None)
        share = (files + read) / max(1, len(wav_names) + len(aliases))
        scored.append((share, files, read, pack))
    share, files, read, pack = max(scored, key=lambda item: item[0])
    format_type = sorted(pack.formats)[0]
    evidence = (f"언어팩 {pack.display_name}: 파일 이름 {files}/{len(wav_names)}개, 별칭 {read}/{len(aliases)}개를 읽었습니다.",)
    if share < 0.5:
        return FormatDetection(None, "low", "language_pack", (*evidence, "언어팩 표기와 맞지 않습니다. 형식을 직접 선택하세요."))
    return FormatDetection(format_type, "high" if share >= 0.9 else "medium", "language_pack", evidence)


def detect_format_from_folder_name(bank: Path) -> FormatDetection | None:
    for name in (bank.name, bank.parent.name):
        normalized = unicodedata.normalize("NFKC", name)
        found = {
            match.group(1).casefold().replace("cv-vc", "cvvc")
            for match in _FOLDER_FORMAT_RE.finditer(normalized)
            if _starts_word(normalized, match.start())
        }
        if len(found) == 1:
            value = found.pop()
            return FormatDetection(value, "medium", "folder_name", (t("폴더 이름 '{name}'에 {format}가 들어 있습니다.", name=name, format=value.upper()),))
    return None


def slot_counts_from_filenames(wav_names: Iterable[str]) -> list[int]:
    counts = []
    for name in wav_names:
        stem = Path(name).stem
        kana = parse_kana_slots(stem)
        if kana:
            counts.append(len(kana))
            continue
        tokens = [token for token in recording_tokens(stem) if not _PITCH_TOKEN_RE.match(token)]
        counts.append(len(tokens))
    return counts


def detect_format_from_filenames(wav_names: Sequence[str]) -> FormatDetection | None:
    counts = slot_counts_from_filenames(wav_names)
    if len(counts) < 5:
        return None
    single = sum(1 for count in counts if count <= 1) / len(counts)
    median = statistics.median(counts)
    if single >= 0.8:
        return FormatDetection(
            "cv", "medium", "filenames",
            (t("WAV {count}개 중 {share:.0%}가 한 음절 파일입니다.", count=len(counts), share=single),),
            {"single_syllable_files": sum(1 for count in counts if count <= 1)},
        )
    return FormatDetection(
        None, "none", "filenames",
        (t("여러 음절 목록 파일입니다(중앙값 {median:g}음절). 파일 이름만으로는 CVVC/VCV/CVC를 구분할 수 없습니다.",
    median=median),),
    )


def detect_format(
    bank: Path,
    wav_names: Sequence[str],
    *,
    base_aliases: Sequence[str] | None = None,
    language: str | None = None,
) -> FormatDetection:
    """Detect the format; the existing oto.ini wins, then single-syllable files, then the folder name."""

    from_pack = detect_format_from_language_pack(wav_names, base_aliases or (), language=language)
    if from_pack is not None:
        return from_pack
    if base_aliases:
        from_oto = detect_format_from_aliases(base_aliases, language=language)
        if from_oto.format is not None:
            folder = detect_format_from_folder_name(bank)
            if folder is not None and folder.format != from_oto.format and _alias_share(from_oto.counts, folder.format) >= (
                0.05 if folder.format == "cvc" else 0.2
            ):
                return FormatDetection(
                    folder.format, "medium", "base_oto+folder_name",
                    (*from_oto.evidence, *folder.evidence,
                     t("여러 형식 별칭이 섞인 OTO라 폴더 이름의 {format}를 따릅니다.", format=folder.format.upper())),
                    from_oto.counts,
                )
            if folder is not None and folder.format != from_oto.format:
                return FormatDetection(
                    from_oto.format, "low", from_oto.source,
                    (*from_oto.evidence, t("폴더 이름은 {format}를 가리켜 결과가 엇갈립니다. 형식을 확인하세요.", format=folder.format.upper())),
                    from_oto.counts,
                )
            return from_oto
    from_files = detect_format_from_filenames(wav_names)
    if from_files is not None and from_files.format is not None:
        return from_files
    from_folder = detect_format_from_folder_name(bank)
    if from_folder is not None:
        notes = from_files.evidence if from_files is not None else ()
        return FormatDetection(from_folder.format, from_folder.confidence, from_folder.source,
                               (*notes, *from_folder.evidence))
    notes: tuple[str, ...] = ()
    if base_aliases:
        notes += detect_format_from_aliases(base_aliases, language=language).evidence
    if from_files is not None:
        notes += from_files.evidence
    return FormatDetection(None, "none", "none", (*notes, t("형식을 직접 선택하세요.")))


def check_language(
    language: str,
    wav_names: Sequence[str],
    aliases: Sequence[str] = (),
) -> LanguageCheck:
    """Warn when the chosen language does not match the file names/aliases."""

    texts = [Path(name).stem for name in wav_names] + [str(alias) for alias in aliases]
    total = max(1, len(texts))
    kana = sum(1 for text in texts if _KANA_RE.search(text)) / total
    hangul = sum(1 for text in texts if _HANGUL_RE.search(text)) / total
    roman_texts = [text.casefold() for text in texts if not _KANA_RE.search(text) and not _HANGUL_RE.search(text)]
    korean_roman = (sum(1 for text in roman_texts if _KOREAN_ROMAN_RE.search(text)) / len(roman_texts)
                    if roman_texts else 0.0)
    normalized = language.casefold()
    warning = None
    if normalized in {"korean", "ko", "kr"} and kana >= 0.2:
        warning = t("언어를 한국어로 골랐지만 파일 이름/별칭의 {share:.0%}에 일본어 가나가 있습니다. "
                    "일본어 보이스뱅크가 아닌지 확인하세요.", share=kana)
    elif normalized in {"japanese", "ja", "jp"} and hangul >= 0.1:
        warning = t("언어를 일본어로 골랐지만 파일 이름/별칭의 {share:.0%}에 한글이 있습니다. "
                    "한국어 보이스뱅크가 아닌지 확인하세요.", share=hangul)
    elif normalized in {"japanese", "ja", "jp"} and kana < 0.05 and len(roman_texts) >= 5 and korean_roman >= 0.25:
        warning = t("언어를 일본어로 골랐지만 로마자 이름의 {share:.0%}에 한국어 표기(eo, eu, ng, ss 등)가 있습니다. "
                    "한국어 보이스뱅크가 아닌지 확인하세요.", share=korean_roman)
    return LanguageCheck(language, warning, kana, hangul, korean_roman)


def _alias_share(counts: dict[str, int], format_type: str) -> float:
    core = sum(counts.get(key, 0) for key in ("cv", "vcv", "vc", "vc_syllable"))
    key = {"cv": "cv", "vcv": "vcv", "cvvc": "vc", "cvc": "vc_syllable"}[format_type]
    return counts.get(key, 0) / core if core else 0.0


def _starts_word(text: str, index: int) -> bool:
    """True at a word start; a case change counts (``DiKORvcv``)."""

    if index == 0 or not text[index - 1].isascii() or not text[index - 1].isalpha():
        return True
    return text[index - 1].isupper() != text[index].isupper()


def _is_special(token: str) -> bool:
    if token in _SPECIAL_TOKENS or token.startswith("br") or token.startswith("*") or token.startswith("・"):
        return True
    return any(mark in token for mark in ("息", "吸", "・"))


def _token_shape(token: str) -> tuple[bool, bool, bool] | None:
    """Return (has_vowel, starts_with_vowel, ends_with_coda) for a kana or roman token."""

    if _KANA_RE.search(token):
        slots = parse_kana_slots(token)
        if not slots:
            return None
        phones = [phone for slot in slots for phone in slot]
        vowels = {"a", "i", "u", "e", "o"}
        has_vowel = any(phone in vowels for phone in phones)
        return has_vowel, phones[0] in vowels, phones[-1] == "n" and len(phones) > 1
    if _HANGUL_RE.search(token):
        return None
    if not re.fullmatch(r"[A-Za-z']+", token):
        return None
    lowered = token.casefold().strip("'")
    if not lowered:
        return None
    has_vowel = bool(_ROMAN_VOWEL_RE.search(lowered))
    starts_with_vowel = lowered[0] in "aeiou"
    tail = re.split(r"[aeiou]", lowered)[-1]
    ends_with_coda = has_vowel and tail in _ROMAN_CODAS
    return has_vowel, starts_with_vowel, ends_with_coda


__all__ = [
    "FORMATS",
    "FormatDetection",
    "LanguageCheck",
    "check_language",
    "classify_alias",
    "detect_format",
    "detect_format_from_aliases",
    "detect_format_from_filenames",
    "detect_format_from_folder_name",
    "slot_counts_from_filenames",
    "strip_alias_suffix",
]
