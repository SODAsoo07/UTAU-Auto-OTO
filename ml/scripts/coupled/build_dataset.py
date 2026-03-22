from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_coupled import build_and_save_coupled_dataset
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Build mel+oto coupled training dataset CSV.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--auto", default="", help="Auto-generated oto.ini path (optional with auto-oto policy)")
    ap.add_argument("--manual", required=True, help="Manually corrected oto.ini path")
    ap.add_argument("--tg-dir", required=True, help="TextGrid directory")
    ap.add_argument("--wav-dir", required=True, help="WAV directory")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--voicebank-id", default="", help="Optional explicit voicebank id")
    ap.add_argument("--custom-phonemes", default="", help="Optional custom phoneme map")
    ap.add_argument("--format-override", default="", help="Optional forced format_type for all rows")
    ap.add_argument("--append", action="store_true", help="Append to an existing CSV")
    ap.add_argument(
        "--auto-oto-policy",
        default="require",
        choices=["require", "generate-temp", "generate-persist"],
        help="Auto OTO behavior when missing (require|generate-temp|generate-persist)",
    )
    args = ap.parse_args()

    policy = str(args.auto_oto_policy or "").strip().lower()
    if policy in {"", "require", "required"} and not str(args.auto or "").strip():
        raise SystemExit("--auto is required when --auto-oto-policy=require")

    stats = build_and_save_coupled_dataset(
        language=args.lang,
        auto_oto_path=args.auto,
        manual_oto_path=args.manual,
        tg_dir=args.tg_dir,
        wav_dir=args.wav_dir,
        out_csv=args.out,
        custom_phonemes_path=args.custom_phonemes,
        voicebank_id=args.voicebank_id,
        append=args.append,
        format_type_override=args.format_override,
        auto_oto_policy=args.auto_oto_policy,
    )
    print(f"saved_rows={stats.get('saved_rows', 0)}")
    print(f"matched_rows={stats.get('matched_rows', 0)}")
    print(f"skipped_rows={stats.get('skipped_rows', 0)}")
    print(f"out_csv={stats.get('out_csv', '')}")


if __name__ == "__main__":
    main()
