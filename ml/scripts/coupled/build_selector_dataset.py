from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_selector import build_selector_dataset_csv_from_delta_dataset
from core.runtime.runtime_encoding import bootstrap_utf8_runtime


def _split_csv_tokens(raw: str):
    return [v.strip() for v in str(raw or "").split(",") if v.strip()]


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Build selector ranking dataset from coupled delta dataset CSV.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--delta-dataset", required=True, help="Input coupled delta dataset CSV")
    ap.add_argument("--out", required=True, help="Output selector dataset CSV")
    ap.add_argument("--alias-types", default="", help="Comma-separated alias_type filter")
    ap.add_argument("--alias-groups", default="", help="Comma-separated alias_group filter")
    ap.add_argument("--min-mapping-confidence", type=float, default=0.0)
    ap.add_argument("--require-train-keep", action="store_true")
    ap.add_argument("--exclude-nuclei-fallback", action="store_true")
    args = ap.parse_args()

    summary = build_selector_dataset_csv_from_delta_dataset(
        delta_dataset_csv=str(args.delta_dataset),
        out_csv=str(args.out),
        language=str(args.lang),
        format_type=str(args.format),
        alias_types=_split_csv_tokens(args.alias_types),
        alias_groups=_split_csv_tokens(args.alias_groups),
        require_train_keep=bool(args.require_train_keep),
        min_mapping_confidence=float(args.min_mapping_confidence),
        exclude_nuclei_fallback=bool(args.exclude_nuclei_fallback),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
