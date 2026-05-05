from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import evaluate_lightgbm_selector_bundle
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Evaluate LightGBM selector bundle.")
    ap.add_argument("--model-dir", required=True, help="Selector model directory")
    ap.add_argument("--selector-dataset", required=True, help="Selector dataset CSV")
    ap.add_argument("--lang", default="", help="Optional language filter")
    ap.add_argument("--format", default="", help="Optional format filter")
    ap.add_argument("--out", default="", help="Optional output summary JSON path")
    args = ap.parse_args()

    summary = evaluate_lightgbm_selector_bundle(
        model_dir=str(args.model_dir),
        selector_dataset_csv=str(args.selector_dataset),
        language=str(args.lang or ""),
        format_type=str(args.format or ""),
    )
    out_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(out_text)
    out_path = os.path.abspath(str(args.out or "").strip())
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text)


if __name__ == "__main__":
    main()
