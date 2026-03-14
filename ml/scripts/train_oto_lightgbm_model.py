from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import train_lightgbm_bundle
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Train LightGBM OTO ML bundle.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--dataset", required=True, help="Input dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output model directory")
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument("--alias-types", default="", help="Comma-separated alias_type filter")
    ap.add_argument("--alias-family", default="", help="Alias family filter (cv/vc/vcv/vowel)")
    ap.add_argument("--min-mapping-confidence", type=float, default=0.0)
    ap.add_argument("--num-boost-round", type=int, default=500)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--require-train-keep", action="store_true")
    ap.add_argument("--exclude-nuclei-fallback", action="store_true")
    args = ap.parse_args()

    alias_types = [v.strip() for v in str(args.alias_types).split(",") if v.strip()]
    alias_family = str(args.alias_family or "").strip()

    meta = train_lightgbm_bundle(
        language=args.lang,
        format_type=args.format,
        dataset_csv=args.dataset,
        out_dir=args.out_dir,
        group_column=args.group_column,
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
        alias_types=alias_types,
        alias_family=alias_family,
        min_mapping_confidence=float(args.min_mapping_confidence),
        require_train_keep=bool(args.require_train_keep),
        exclude_nuclei_fallback=bool(args.exclude_nuclei_fallback),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
