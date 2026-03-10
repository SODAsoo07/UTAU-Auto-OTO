from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_policy import alias_family_to_alias_types, default_training_filters, normalize_alias_family
from core.oto_ml_selector import build_selector_dataset_csv_from_delta_dataset
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Build selector training dataset from delta OTO ML dataset.")
    ap.add_argument("--dataset", required=True, help="Input delta dataset CSV")
    ap.add_argument("--out", required=True, help="Output selector dataset CSV")
    ap.add_argument("--lang", default="", choices=["", "korean", "japanese"])
    ap.add_argument("--format", default="", help="Format type filter (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--alias-family", default="all", choices=["all", "cv", "vowel", "bridge"])
    args = ap.parse_args()
    alias_family = normalize_alias_family(args.alias_family)
    defaults = default_training_filters(args.lang, args.format, alias_family=alias_family) if args.lang and args.format else {
        "require_train_keep": False,
        "min_mapping_confidence": 0.0,
        "exclude_nuclei_fallback": False,
        "use_pseudo_labels": False,
    }

    summary = build_selector_dataset_csv_from_delta_dataset(
        args.dataset,
        args.out,
        language=args.lang,
        format_type=args.format,
        alias_types=alias_family_to_alias_types(alias_family),
        require_train_keep=bool(defaults["require_train_keep"]),
        min_mapping_confidence=float(defaults["min_mapping_confidence"]),
        exclude_nuclei_fallback=bool(defaults["exclude_nuclei_fallback"]),
        use_pseudo_labels=bool(defaults["use_pseudo_labels"]),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
