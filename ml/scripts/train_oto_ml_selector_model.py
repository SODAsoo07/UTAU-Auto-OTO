from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import train_lightgbm_selector_bundle
from core.oto_ml_policy import normalize_alias_family


def main():
    ap = argparse.ArgumentParser(description="Train LightGBM selector bundle for OTO ML candidate routing.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--alias-family", default="all", choices=["all", "cv", "vowel", "bridge"])
    ap.add_argument("--dataset", required=True, help="Selector dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output model directory")
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument("--objective", choices=["pointwise", "ranking"], default="pointwise")
    args = ap.parse_args()

    meta = train_lightgbm_selector_bundle(
        language=args.lang,
        format_type=args.format,
        selector_dataset_csv=args.dataset,
        out_dir=args.out_dir,
        group_column=args.group_column,
        objective=args.objective,
        alias_family=normalize_alias_family(args.alias_family),
    )
    print(f"language={meta.get('language')}")
    print(f"format_type={meta.get('format_type')}")
    print(f"selector_objective={meta.get('selector_objective')}")
    print(f"train_rows={meta.get('train_rows')}")
    print(f"metrics={meta.get('metrics')}")


if __name__ == "__main__":
    main()
