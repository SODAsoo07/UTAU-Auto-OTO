from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_autofree import build_rows_for_training
from core.oto_ml.autofree.schema import write_dataset_csv
from core.runtime_encoding import bootstrap_utf8_runtime


def main() -> None:
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Build Auto-Free dataset CSV (absolute targets).")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--manual", required=True, help="Manual gold oto.ini")
    ap.add_argument("--wav-dir", required=True, help="WAV directory")
    ap.add_argument("--tg-dir", default="", help="TextGrid directory (optional for audio_only)")
    ap.add_argument("--source-mode", default="auto", choices=["auto", "textgrid", "audio_only"])
    ap.add_argument("--token-map", default="", help="External token map path (.json/.txt)")
    ap.add_argument("--format-override", default="", help="Optional fixed format_type")
    ap.add_argument("--voicebank-id", default="", help="Optional explicit voicebank id")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    rows, stats = build_rows_for_training(
        language=args.lang,
        manual_oto_path=args.manual,
        wav_dir=args.wav_dir,
        tg_dir=args.tg_dir,
        source_mode=args.source_mode,
        token_map_path=args.token_map,
        format_type_override=args.format_override,
        voicebank_id=args.voicebank_id,
    )
    write_dataset_csv(args.out, rows, append=bool(args.append))
    stats = dict(stats or {})
    stats["saved_rows"] = int(len(rows))
    stats["out_csv"] = os.path.abspath(args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
