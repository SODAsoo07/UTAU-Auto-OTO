from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import train_lightgbm_bundle


def main():
    ap = argparse.ArgumentParser(description="Train LightGBM OTO ML bundle.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--dataset", required=True, help="Input dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output model directory")
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument("--alias-types", default="", help="Comma-separated alias_type filter")
    ap.add_argument("--alias-groups", default="", help="Comma-separated alias_group filter")
    ap.add_argument("--require-train-keep", action="store_true", help="Keep only rows marked train_keep_default=1")
    args = ap.parse_args()

    alias_types = [v.strip() for v in str(args.alias_types).split(",") if v.strip()]
    alias_groups = [v.strip() for v in str(args.alias_groups).split(",") if v.strip()]

    meta = train_lightgbm_bundle(
        language=args.lang,
        format_type=args.format,
        dataset_csv=args.dataset,
        out_dir=args.out_dir,
        group_column=args.group_column,
        alias_types=alias_types,
        alias_groups=alias_groups,
        require_train_keep=args.require_train_keep,
    )
    print(f"backend={meta.get('backend')}")
    print(f"language={meta.get('language')}")
    print(f"format_type={meta.get('format_type')}")
    print(f"train_rows={meta.get('train_rows')}")


if __name__ == "__main__":
    main()
