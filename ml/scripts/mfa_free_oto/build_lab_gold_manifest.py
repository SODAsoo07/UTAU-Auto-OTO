from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.mfa_free_oto.htk_lab import build_gold_manifest_from_htk_lab_dirs, write_summary_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert paired HTK .lab/.wav directories into MFA-free OTO manual_gold manifests."
    )
    parser.add_argument(
        "--lab-root",
        action="append",
        required=True,
        help="Directory containing paired .lab/.wav files. Can be passed more than once.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--language", required=True, help="Language tag, e.g. japanese.")
    parser.add_argument("--format-type", required=True, help="Bank format tag, e.g. CV or VCV.")
    parser.add_argument("--dataset-id", help="Dataset id stored in each row.")
    parser.add_argument(
        "--relative-to",
        help="Store wav/lab paths relative to this directory. Defaults to absolute paths.",
    )
    parser.add_argument("--summary-out", help="Optional summary JSON path.")
    args = parser.parse_args()

    _rows, summary = build_gold_manifest_from_htk_lab_dirs(
        args.lab_root,
        out_path=args.out,
        language=args.language,
        format_type=args.format_type,
        dataset_id=args.dataset_id,
        relative_to=args.relative_to,
    )
    if args.summary_out:
        write_summary_json(args.summary_out, summary)
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
