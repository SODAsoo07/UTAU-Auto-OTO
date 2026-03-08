from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_selector import build_selector_dataset_csv_from_delta_dataset


def main():
    ap = argparse.ArgumentParser(description="Build selector training dataset from delta OTO ML dataset.")
    ap.add_argument("--dataset", required=True, help="Input delta dataset CSV")
    ap.add_argument("--out", required=True, help="Output selector dataset CSV")
    ap.add_argument("--lang", default="", choices=["", "korean", "japanese"])
    ap.add_argument("--format", default="", help="Format type filter (cv/cvc/cvvc/vcv/general)")
    args = ap.parse_args()

    summary = build_selector_dataset_csv_from_delta_dataset(
        args.dataset,
        args.out,
        language=args.lang,
        format_type=args.format,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
