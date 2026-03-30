from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import train_lightgbm_selector_bundle
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Train LightGBM selector bundle.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--selector-dataset", required=True, help="Input selector dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output selector model directory")
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument("--objective", default="auto", choices=["auto", "pointwise", "ranking", "pairwise"])
    ap.add_argument("--num-boost-round", type=int, default=400)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--alias-family", default="", help="Alias family filter (cv/vc/vcv/vowel)")
    args = ap.parse_args()

    meta = train_lightgbm_selector_bundle(
        language=str(args.lang),
        format_type=str(args.format),
        selector_dataset_csv=str(args.selector_dataset),
        out_dir=str(args.out_dir),
        group_column=str(args.group_column),
        objective=str(args.objective),
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
        alias_family=str(args.alias_family or ""),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
