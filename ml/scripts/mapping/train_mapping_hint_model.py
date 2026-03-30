from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.mapping_supervised_training import train_mapping_hint_model
from core.runtime_encoding import bootstrap_utf8_runtime


def main() -> int:
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Train mapping-hint LightGBM model.")
    ap.add_argument("--dataset", required=True, help="Input CSV built by build_mapping_hint_dataset.py")
    ap.add_argument("--out-dir", required=True, help="Output model directory")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv)")
    ap.add_argument("--num-boost-round", type=int, default=700)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    args = ap.parse_args()

    meta = train_mapping_hint_model(
        dataset_csv=args.dataset,
        out_dir=args.out_dir,
        language=args.lang,
        format_type=args.format,
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

