from __future__ import annotations

import argparse
import json

from core.mfa_free_oto.manifest import build_goldset_scaffold, write_manifest_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a manual-gold MFA-free OTO manifest scaffold from wav files."
    )
    parser.add_argument("--wav-dir", required=True, help="Directory containing candidate wav files.")
    parser.add_argument("--out", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--language", required=True, help="Language tag, e.g. korean or japanese.")
    parser.add_argument("--format-type", required=True, help="Bank format tag, e.g. CV, CVC, CVVC, VCV.")
    parser.add_argument("--aliases", help="Optional alias sidecar: jsonl/csv/tsv/plain text.")
    parser.add_argument(
        "--relative-to",
        help="Store wav paths relative to this directory. Defaults to absolute paths.",
    )
    args = parser.parse_args()

    rows, summary = build_goldset_scaffold(
        args.wav_dir,
        language=args.language,
        format_type=args.format_type,
        aliases_path=args.aliases,
        relative_to=args.relative_to,
    )
    write_manifest_jsonl(args.out, rows)
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
