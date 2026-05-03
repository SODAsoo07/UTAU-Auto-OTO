from __future__ import annotations

import argparse
import os

from core.coarse_crnn.oto_inference import format_oto_line, predict_oto, write_prediction_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one oto.ini row with the OTO anchor CRNN.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--prev-alias", default="")
    parser.add_argument("--next-alias", default="")
    parser.add_argument("--language", required=True)
    parser.add_argument("--format-type", default="other")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--file-row-count", type=int, default=1)
    parser.add_argument("--model", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_anchor_crnn.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    prediction = predict_oto(
        wav_path=args.audio,
        model_path=args.model,
        language=args.language,
        format_type=args.format_type,
        alias=args.alias,
        prev_alias=args.prev_alias,
        next_alias=args.next_alias,
        row_index_in_wav=int(args.row_index),
        file_row_count=int(args.file_row_count),
        device=args.device,
    )
    line = format_oto_line(os.path.basename(args.audio), args.alias, prediction.params)
    print(line)
    print(f"confidence={prediction.confidence:.4f}")
    if args.out_json:
        write_prediction_json(
            args.out_json,
            prediction,
            meta={
                "audio": os.path.abspath(args.audio),
                "alias": args.alias,
                "language": args.language,
                "format_type": args.format_type,
                "model": os.path.abspath(args.model),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
