from __future__ import annotations

import argparse
import json

from core.coarse_crnn.boundary_generator import generate_oto_with_boundary_decoder


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate oto.ini using boundary scorer + wav-level decoder.")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--source-oto", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="korean")
    ap.add_argument("--format-type", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--alias-suffix", default="")
    ap.add_argument("--special-aliases", default="", help="Comma-separated special alias list")
    args = ap.parse_args()

    special = {item.strip() for item in str(args.special_aliases or "").split(",") if item.strip()}
    processed, total, errors = generate_oto_with_boundary_decoder(
        wav_dir=args.wav_dir,
        out_path=args.out,
        source_oto_path=args.source_oto,
        language=args.language,
        format_type=args.format_type,
        model_path=args.model,
        device=args.device,
        alias_suffix=args.alias_suffix,
        special_aliases=special,
    )
    print(json.dumps({"processed": processed, "total": total, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if (processed > 0 and not errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())

