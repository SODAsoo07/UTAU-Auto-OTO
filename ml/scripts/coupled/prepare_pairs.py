from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml.oto_ml_batch_prepare import prepare_staged_auto_pairs, write_prepare_report
from core.runtime.runtime_encoding import bootstrap_utf8_runtime


def _escape_unusual_chars(text: str) -> str:
    safe_ranges = [
        (0x20, 0x7E),  # ASCII printable
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0xAC00, 0xD7A3),  # Hangul syllables
        (0xFF01, 0xFF60),  # Fullwidth ASCII variants
        (0xFF65, 0xFF9F),  # Halfwidth Katakana
    ]

    def is_safe(ch: str) -> bool:
        code = ord(ch)
        if code in (0x09, 0x0A, 0x0D):
            return True
        return any(start <= code <= end for start, end in safe_ranges)

    out = []
    for ch in text:
        if is_safe(ch):
            out.append(ch)
        else:
            code = ord(ch)
            if code <= 0xFFFF:
                out.append(f"\\u{code:04X}")
            else:
                out.append(f"\\U{code:08X}")
    return "".join(out)


def _safe_print(message: str) -> None:
    text = _escape_unusual_chars(str(message))
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        data = text.encode(enc, errors="replace")
        sys.stdout.buffer.write(data + b"\n")
        sys.stdout.flush()


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Generate lab/dict/TextGrid/auto-oto for staged dataset copies.")
    ap.add_argument(
        "--dataset-root",
        default=os.path.join(ROOT, "dataset"),
        help="Staged dataset root",
    )
    ap.add_argument("--dry-run", action="store_true", help="Discover planned jobs without running MFA/OTO generation")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of work items")
    ap.add_argument("--workers", type=int, default=0, help="Parallel worker count (0=sequential)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse prepared_auto_pairs.json as a checkpoint and skip items already processed",
    )
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="When resuming, retry items previously recorded as skip/failed",
    )
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    report_path = os.path.join(dataset_root, "_manifest", "prepared_auto_pairs.json")
    result = prepare_staged_auto_pairs(
        dataset_root,
        dry_run=args.dry_run,
        limit=args.limit,
        progress_callback=_safe_print,
        report_path=report_path,
        resume=args.resume,
        retry_failed=args.retry_failed,
        workers=args.workers,
    )
    write_prepare_report(report_path, result)
    print(json.dumps({
        "summary": result.get("summary", {}),
        "report_path": os.path.abspath(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
