import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import train_lightgbm_selector_bundle
from core.oto_ml_policy import default_training_filters, normalize_alias_family
from core.oto_ml_selector import build_selector_dataset_csv_from_delta_dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Train selector bundle from delta dataset.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--format", required=True, dest="format_type")
    parser.add_argument("--delta-dataset", required=True, dest="delta_dataset")
    parser.add_argument("--selector-dataset", default="", dest="selector_dataset")
    parser.add_argument("--out-dir", required=True, dest="out_dir")
    parser.add_argument("--objective", default="auto")
    parser.add_argument("--alias-family", default="")
    parser.add_argument("--group-column", default="voicebank_id")
    parser.add_argument("--min-mapping-confidence", type=float, default=None)
    parser.add_argument("--require-train-keep", action="store_true")
    parser.add_argument("--exclude-nuclei-fallback", action="store_true")
    parser.add_argument("--use-pseudo-labels", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    language = str(args.language or "").strip().lower()
    format_type = str(args.format_type or "").strip().lower()
    alias_family = normalize_alias_family(args.alias_family)

    filters = default_training_filters(language, format_type, alias_family=alias_family)
    if args.min_mapping_confidence is not None:
        filters["min_mapping_confidence"] = float(args.min_mapping_confidence)
    if args.require_train_keep:
        filters["require_train_keep"] = True
    if args.exclude_nuclei_fallback:
        filters["exclude_nuclei_fallback"] = True
    if args.use_pseudo_labels:
        filters["use_pseudo_labels"] = True

    selector_dataset = str(args.selector_dataset or "").strip()
    if not selector_dataset:
        selector_dataset = os.path.join(
            os.path.dirname(os.path.abspath(args.delta_dataset)),
            f"selector_dataset_{language}_{format_type}.csv",
        )
        build_selector_dataset_csv_from_delta_dataset(
            args.delta_dataset,
            selector_dataset,
            language=language,
            format_type=format_type,
            alias_types=None,
            alias_groups=None,
            require_train_keep=bool(filters.get("require_train_keep", False)),
            min_mapping_confidence=float(filters.get("min_mapping_confidence", 0.0)),
            exclude_nuclei_fallback=bool(filters.get("exclude_nuclei_fallback", False)),
            use_pseudo_labels=bool(filters.get("use_pseudo_labels", False)),
        )

    meta = train_lightgbm_selector_bundle(
        language=language,
        format_type=format_type,
        selector_dataset_csv=selector_dataset,
        out_dir=args.out_dir,
        group_column=args.group_column,
        objective=args.objective,
        alias_family=alias_family,
    )
    print("selector_meta.json written:", os.path.join(args.out_dir, "selector_meta.json"))
    print("objective:", meta.get("selector_objective", ""))
    print("train_rows:", meta.get("train_rows", 0))


if __name__ == "__main__":
    main()
