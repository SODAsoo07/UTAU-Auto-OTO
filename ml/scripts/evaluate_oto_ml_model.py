from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import evaluate_lightgbm_bundle
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Evaluate LightGBM OTO ML bundle.")
    ap.add_argument("--lang", default="")
    ap.add_argument("--format", default="", help="Format type for filtering/evaluation (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    summary = evaluate_lightgbm_bundle(
        model_dir=args.model_dir,
        dataset_csv=args.dataset,
        language=args.lang,
        format_type=args.format,
    )
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
