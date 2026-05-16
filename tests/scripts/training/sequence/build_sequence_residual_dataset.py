from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def _log(message: str) -> None:
    print(str(message), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build file-locked sequence-anchor residual training rows from a manual oto.ini."
    )
    parser.add_argument("--voicebank", required=True, help="Voicebank directory containing wav files.")
    parser.add_argument("--oto-path", required=True, help="Manual/reference oto.ini path.")
    parser.add_argument("--out-dir", required=True, help="Output directory for JSONL and summary.")
    parser.add_argument("--language", default="korean", help="Language hint. Default: korean.")
    parser.add_argument("--format", dest="format_type", default="", help="Format hint such as cvc, cvvc, vcv.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional file limit for smoke tests.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file progress logs.")
    args = parser.parse_args()
    from core.sequence_residual_preprocessor import write_sequence_residual_dataset

    result = write_sequence_residual_dataset(
        voicebank_dir=args.voicebank,
        manual_oto_path=args.oto_path,
        output_dir=args.out_dir,
        language=args.language,
        format_type=args.format_type,
        max_files=max(0, int(args.max_files or 0)),
        callback=None if args.quiet else _log,
    )
    summary = result.get("summary", {})
    print(json.dumps({"rows_path": result.get("rows_path"), "summary_path": result.get("summary_path"), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
