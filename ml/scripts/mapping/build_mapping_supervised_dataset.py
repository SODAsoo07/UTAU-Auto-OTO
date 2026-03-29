from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.mapping_supervised_training import build_mapping_supervised_dataset
from core.runtime_encoding import bootstrap_utf8_runtime


def main() -> int:
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Build mapping-supervised candidate dataset CSV.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--formats", default="", help="Comma-separated format filter (e.g. cv,cvc,cvvc,vcv)")
    ap.add_argument("--max-voicebanks", type=int, default=0, help="0 = no limit")
    ap.add_argument(
        "--monotonic-prior-weight",
        type=float,
        default=-1.0,
        help="Negative value uses format default. Higher = stronger monotonic label prior.",
    )
    ap.add_argument(
        "--auto-oto-policy",
        default="generate-temp",
        choices=["require", "generate-temp", "generate-persist"],
    )
    args = ap.parse_args()

    formats = [v.strip() for v in str(args.formats or "").split(",") if v.strip()]
    stats = build_mapping_supervised_dataset(
        dataset_root=args.dataset_root,
        out_csv=args.out_csv,
        language=args.lang,
        format_types=formats,
        max_voicebanks=int(args.max_voicebanks),
        auto_oto_policy=str(args.auto_oto_policy or "generate-temp"),
        monotonic_prior_weight=float(args.monotonic_prior_weight),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
