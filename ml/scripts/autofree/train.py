from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_autofree import train_autofree_bundle
from core.runtime_encoding import bootstrap_utf8_runtime


def main() -> None:
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Train Auto-Free absolute OTO model bundle.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--dataset", required=True, help="Input Auto-Free dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output model directory")
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument("--num-boost-round", type=int, default=500)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    args = ap.parse_args()

    meta = train_autofree_bundle(
        language=args.lang,
        format_type=args.format,
        dataset_csv=args.dataset,
        out_dir=args.out_dir,
        group_column=args.group_column,
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
