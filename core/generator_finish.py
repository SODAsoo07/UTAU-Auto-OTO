from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class GeneratorFinishContext:
    log_fn: Callable[[str], None]
    anchor_stats: Mapping[str, object]
    anchor_log_path: str
    anchor_log_dir: str
    cleanup_timing_jsonl: bool
    timing_jsonl_prefix: str


def _normalize_text_encoding(value: str | None, default: str = "utf-8") -> str:
    enc = str(value or "").strip().lower().replace("-", "_")
    if not enc:
        return default
    if enc in {"utf8", "utf_8"}:
        return "utf-8"
    if enc in {"utf8_sig", "utf_8_sig"}:
        return "utf-8-sig"
    if enc in {"shift_jis", "shiftjis", "sjis", "shift_js"}:
        return "shift_jis"
    if enc in {"cp932", "ms932", "windows_31j"}:
        return "cp932"
    return str(value).strip()


def _detect_text_encoding(path: str, default: str = "utf-8") -> str:
    if not path or not os.path.exists(path):
        return _normalize_text_encoding(default, default=default)
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return _normalize_text_encoding(default, default=default)

    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    def _safe_decode(enc: str) -> str | None:
        try:
            return raw.decode(enc)
        except Exception:
            return None

    def _looks_east_asian(ch: str) -> bool:
        o = ord(ch)
        return (
            0x3040 <= o <= 0x30FF  # Hiragana/Katakana
            or 0x4E00 <= o <= 0x9FFF  # CJK
            or 0xAC00 <= o <= 0xD7A3  # Hangul syllables
            or 0x1100 <= o <= 0x11FF  # Hangul jamo
        )

    def _text_quality_score(text: str) -> float:
        score = 0.0
        east_asian = 0
        suspicious = 0
        for ch in text:
            if ch in "\r\n\t":
                continue
            o = ord(ch)
            if ch == "\ufffd":
                score -= 8.0
                suspicious += 1
                continue
            cat = unicodedata.category(ch)
            if cat.startswith("C"):
                score -= 5.0
                suspicious += 1
                continue
            if _looks_east_asian(ch):
                east_asian += 1
                score += 2.2
                continue
            if 0x80 <= o <= 0x024F:
                suspicious += 1
                score -= 0.8

        line_bonus = 0.0
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if "=" in s and s.count(",") >= 5:
                line_bonus += 1.5
            elif "=" in s:
                line_bonus += 0.6
        score += line_bonus
        score += min(east_asian, 20) * 0.1
        score -= min(suspicious, 40) * 0.05
        return score

    utf8_text = _safe_decode("utf-8")
    cp932_text = _safe_decode("cp932")
    if utf8_text is not None and cp932_text is not None and any(b >= 0x80 for b in raw):
        utf8_score = _text_quality_score(utf8_text)
        cp932_score = _text_quality_score(cp932_text)
        if cp932_score >= utf8_score + 2.0:
            return "cp932"
        return "utf-8"

    for enc in ("utf-8", "cp932", "shift_jis", "cp949", "euc-kr"):
        if _safe_decode(enc) is not None:
            return _normalize_text_encoding(enc, default=default)
    return _normalize_text_encoding(default, default=default)


def write_oto_lines(out_path: str, lines: Sequence[str], encoding: str | None = None) -> str:
    target_enc = _normalize_text_encoding(encoding) if encoding else _detect_text_encoding(out_path, default="utf-8")
    try:
        with open(out_path, "w", encoding=target_enc) as f:
            for line in lines:
                f.write(str(line) + "\n")
        return target_enc
    except UnicodeEncodeError:
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(str(line) + "\n")
        return "utf-8"


def cleanup_timing_anchor_jsonl_files(log_dir: str, prefix: str) -> tuple[int, int]:
    if not log_dir or not os.path.isdir(log_dir):
        return 0, 0
    removed = 0
    failed = 0
    try:
        names = os.listdir(log_dir)
    except Exception:
        return 0, 1
    for name in names:
        if not name.startswith(prefix) or not name.endswith(".jsonl"):
            continue
        path = os.path.join(log_dir, name)
        try:
            os.remove(path)
            removed += 1
        except Exception:
            failed += 1
    return removed, failed


def write_jsonl_records(path: str, rows: Sequence[object]) -> int:
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def finalize_generator_finish(context: GeneratorFinishContext) -> None:
    anchor_locked = int(context.anchor_stats.get("anchor_locked_count", 0) or 0)
    if anchor_locked > 0:
        context.log_fn(
            "[AnchorLock] summary: "
            f"anchor_locked_count={anchor_locked}, "
            f"cutoff_clamped_count={int(context.anchor_stats.get('cutoff_clamped_count', 0) or 0)}, "
            f"vc_cutoff_leak_guard_count={int(context.anchor_stats.get('vc_cutoff_leak_guard_count', 0) or 0)}"
        )
        context.log_fn(f"[AnchorLock] detail log: {context.anchor_log_path}")

    if not context.cleanup_timing_jsonl:
        return

    removed_count, failed_count = cleanup_timing_anchor_jsonl_files(
        context.anchor_log_dir,
        context.timing_jsonl_prefix,
    )
    if removed_count > 0:
        context.log_fn(f"[AnchorLock] timing jsonl cleanup: {removed_count} files")
    if failed_count > 0:
        context.log_fn(f"[AnchorLock] timing jsonl cleanup failed: {failed_count} files")


__all__ = [
    "GeneratorFinishContext",
    "cleanup_timing_anchor_jsonl_files",
    "finalize_generator_finish",
    "write_jsonl_records",
    "write_oto_lines",
]
