from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_coupled import train_coupled_bundle
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Train coupled mel+oto OTO ML bundle.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--dataset", required=True, help="Input dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output model directory")
    ap.add_argument("--group-column", default="wav_norm")
    ap.add_argument("--alias-types", default="", help="Comma-separated alias_type filter")
    ap.add_argument("--min-mapping-confidence", type=float, default=0.0)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--min-confidence", type=float, default=0.55)
    args = ap.parse_args()

    alias_types = [v.strip() for v in str(args.alias_types).split(",") if v.strip()]
    meta = train_coupled_bundle(
        language=args.lang,
        format_type=args.format,
        dataset_csv=args.dataset,
        out_dir=args.out_dir,
        group_column=args.group_column,
        alias_types=alias_types,
        min_mapping_confidence=float(args.min_mapping_confidence),
        device=args.device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        min_confidence=float(args.min_confidence),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
