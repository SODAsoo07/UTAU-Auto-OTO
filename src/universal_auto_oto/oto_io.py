from __future__ import annotations

import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from .domain import SourceOtoRow


def read_text_auto(path: Path) -> tuple[str, str, str]:
    data = path.read_bytes()
    candidates: list[tuple[str, str]] = []
    if data.startswith(b"\xef\xbb\xbf"):
        candidates.append(("utf-8-sig", "utf-8-sig"))
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates.append(("utf-16", "utf-16"))
    candidates.append(("utf-8", "utf-8"))
    for codec, label in candidates:
        try:
            text = data.decode(codec)
        except UnicodeDecodeError:
            continue
        newline = "\r\n" if b"\r\n" in data else "\n"
        return text, label, newline
    legacy_decodings = (("cp932", "cp932"), ("shift_jis", "shift_jis"), ("cp949", "cp949"), ("euc_kr", "euc-kr"))
    decoded: list[tuple[float, str, str]] = []
    for codec, label in legacy_decodings:
        try:
            text = data.decode(codec)
        except UnicodeDecodeError:
            continue
        decoded.append((_legacy_text_score(text), text, label))
    if decoded:
        _score, text, label = max(decoded, key=lambda item: item[0])
        newline = "\r\n" if b"\r\n" in data else "\n"
        return text, label, newline
    raise UnicodeError(f"지원되는 인코딩으로 읽을 수 없습니다: {path}")


def _legacy_text_score(text: str) -> float:
    score = 0.0
    for character in text:
        codepoint = ord(character)
        if character in "\r\n\t" or 0x20 <= codepoint <= 0x7E:
            score += 0.05
        elif 0xAC00 <= codepoint <= 0xD7AF or 0x1100 <= codepoint <= 0x11FF:
            score += 3.0
        elif 0x3040 <= codepoint <= 0x30FF:
            # CP932 kana bytes are often also decodable as rare CP949 Hangul.
            # A small kana preference prevents Japanese alias blocks from being
            # selected as Korean mojibake while leaving unambiguous CP949 intact.
            score += 3.2
        elif 0xFF66 <= codepoint <= 0xFF9F:
            score += 0.25
        elif 0x4E00 <= codepoint <= 0x9FFF:
            score += 1.0
        elif unicodedata.category(character).startswith(("L", "S")):
            score -= 2.5
        elif unicodedata.category(character).startswith("C"):
            score -= 8.0
    return score


def parse_oto(path: Path) -> tuple[list[SourceOtoRow], str, str]:
    text, encoding, newline = read_text_auto(path)
    rows: list[SourceOtoRow] = []
    for index, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        wav, payload = line.split("=", 1)
        alias = payload.split(",", 1)[0]
        if not wav.strip():
            continue
        rows.append(SourceOtoRow(source_wav=wav.strip(), alias=alias.strip(), source_index=index))
    if not rows:
        raise ValueError(f"OTO 행이 없습니다: {path}")
    return rows, encoding, newline


def resolve_output_encoding(requested: str, source_encoding: str | None) -> str:
    value = requested.strip().lower().replace("-", "_")
    if value == "preserve":
        value = (source_encoding or "utf-8").lower().replace("-", "_")
    if value in {"utf8", "utf_8", "utf_8_sig"}:
        return "utf-8"
    if value in {"cp932", "ms932"}:
        return "cp932"
    if value in {"shift_jis", "sjis"}:
        return "shift_jis"
    if value in {"cp949", "ms949", "uhc"}:
        return "cp949"
    if value in {"euc_kr", "euckr"}:
        return "euc_kr"
    raise ValueError(f"지원하지 않는 출력 인코딩: {requested}")


def write_lines_atomic(
    path: Path,
    lines: Iterable[str],
    *,
    encoding: str,
    newline: str = "\r\n",
    backup_existing: bool = True,
) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = newline.join(str(line) for line in lines) + newline
    payload.encode(encoding, errors="strict")
    backup: Path | None = None
    if path.exists() and backup_existing:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = path.with_name(f"{path.name}.{stamp}.bak")
        shutil.copy2(path, backup)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return backup


_HANGUL_SYLLABLES = re.compile(r"[가-힣]")
# Kana and kanji (夜, 息, 吸 in take names), ASCII and separators; at least one kana or kanji.
_REPAIRED_KANA = re.compile(r"[ぁ-ゖァ-ヺー・々一-鿿 A-Za-z0-9_#\-']*[ぁ-ゖァ-ヺ一-鿿][ぁ-ゖァ-ヺー・々一-鿿 A-Za-z0-9_#\-']*")


def _is_ks_x_1001_hangul(syllable: str) -> bool:
    """True for the 2,350 everyday syllables (cp949 lead B0-C8, trail A1-FE); the rest are UHC extensions."""
    lead, trail = syllable.encode("cp949")
    return 0xB0 <= lead <= 0xC8 and 0xA1 <= trail <= 0xFE


def repair_kana_mojibake(alias: str) -> str | None:
    """Kana saved as cp932 and read back as cp949 shows up as rare Hangul ("- 궭" is "- ち", "뽭" is "夜").

    Returns the repaired text when every Hangul syllable is a rare UHC extension syllable and the
    bytes read as cp932 give kana/kanji text, else None. Real Korean aliases use everyday syllables
    (KS X 1001: 가, 카, 쌍 ...), so they are left alone.
    """

    syllables = _HANGUL_SYLLABLES.findall(alias)
    if not syllables or any(_is_ks_x_1001_hangul(syllable) for syllable in syllables):
        return None
    try:
        repaired = alias.encode("cp949").decode("cp932")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return repaired if _REPAIRED_KANA.fullmatch(repaired) else None


def normalize_alias_suffix(suffix: str | None) -> str:
    """User suffix text -> the part after '_' ("C4" and "_C4" both give "C4"); blank means none."""

    text = str(suffix or "").strip()
    text = text[1:].strip() if text.startswith("_") else text
    if any(character in text for character in ",=\r\n"):
        raise ValueError(f"alias suffix cannot contain ',', '=' or a line break: {suffix!r}")
    return text


def with_alias_suffix(alias: str, suffix: str | None) -> str:
    """alias + "_" + suffix, as the previous AutoOTO did; an alias that already has it is kept."""

    normalized = normalize_alias_suffix(suffix)
    if not normalized or not alias or alias.endswith(f"_{normalized}"):
        return alias
    return f"{alias}_{normalized}"


def apply_alias_suffix_to_oto(path: Path, suffix: str | None, *, backup_existing: bool = True) -> tuple[int, int]:
    """Add the suffix to every alias of an existing oto file, keeping its encoding and line ends.

    Returns (changed aliases, aliases). Running it again changes nothing.
    """

    text, encoding, newline = read_text_auto(path)
    lines, changed, total = [], 0, 0
    for raw in text.splitlines():
        if "=" not in raw or raw.lstrip().startswith("#"):
            lines.append(raw)
            continue
        wav, payload = raw.split("=", 1)
        alias, separator, rest = payload.partition(",")
        if wav.strip() and alias.strip():
            total += 1
            suffixed = with_alias_suffix(alias.strip(), suffix)
            if suffixed != alias.strip():
                changed += 1
                payload = f"{suffixed}{separator}{rest}"
        lines.append(f"{wav}={payload}")
    if changed:
        write_lines_atomic(path, lines, encoding=encoding, newline=newline, backup_existing=backup_existing)
    return changed, total


def validate_oto_line_values(
    *,
    duration_ms: float,
    offset: float,
    consonant: float,
    cutoff: float,
    preutterance: float,
    overlap: float,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    offset_abs = offset
    overlap_abs = offset + overlap
    pre_abs = offset + preutterance
    fixed_abs = offset + consonant
    cutoff_abs = oto_cutoff_end_abs(
        duration_ms=duration_ms,
        offset=offset,
        cutoff=cutoff,
    )
    if offset_abs < 0 or offset_abs >= duration_ms:
        reasons.append("offset_out_of_range")
    if not (offset_abs <= overlap_abs <= pre_abs <= fixed_abs <= cutoff_abs <= duration_ms + 1e-3):
        reasons.append("anchor_order_invalid")
    if cutoff >= -0.001:
        reasons.append("cutoff_not_negative")
    return not reasons, tuple(reasons)


def oto_cutoff_end_abs(*, duration_ms: float, offset: float, cutoff: float) -> float:
    """Convert UTAU cutoff to an absolute usable-region end position.

    Negative cutoff (minus blank) is a length measured from offset. Positive
    cutoff is a conventional right-blank length measured backward from WAV end.
    """
    if cutoff <= 0:
        return offset + abs(cutoff)
    return duration_ms - cutoff
