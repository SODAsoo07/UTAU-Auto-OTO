from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import train_lightgbm_bundle, train_lightgbm_selector_bundle
from core.oto_ml_policy import alias_family_to_alias_types, default_training_filters, normalize_alias_family
from core.oto_ml_selector import build_selector_dataset_csv_from_delta_dataset
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Train LightGBM OTO ML bundle.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--dataset", required=True, help="Input dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output model directory")
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument(
        "--alias-family",
        default="all",
        choices=["all", "cv", "vowel", "bridge"],
        help="Train a family-scoped model instead of mixing all alias families",
    )
    ap.add_argument("--alias-types", default="", help="Comma-separated alias_type filter")
    ap.add_argument("--alias-groups", default="", help="Comma-separated alias_group filter")
    ap.add_argument(
        "--require-train-keep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep only rows marked train_keep_default=1 (default: policy-controlled)",
    )
    ap.add_argument(
        "--min-mapping-confidence",
        type=float,
        default=-1.0,
        help="Minimum mapping_confidence. <0 means conservative auto default by language/format.",
    )
    ap.add_argument(
        "--exclude-nuclei-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Exclude rows that used nuclei fallback (default: policy-controlled)",
    )
    ap.add_argument(
        "--use-pseudo-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use pseudo-labeled rows with sample weights when present (default: policy-controlled)",
    )
    ap.add_argument(
        "--pseudo-weight-high",
        type=float,
        default=0.7,
        help="Sample weight for pseudo_high rows (default: 0.7)",
    )
    ap.add_argument(
        "--pseudo-weight-mid",
        type=float,
        default=0.4,
        help="Sample weight for pseudo_mid rows (default: 0.4)",
    )
    ap.add_argument(
        "--include-selector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also train candidate selector bundle in the same output directory",
    )
    ap.add_argument(
        "--selector-objective",
        choices=["pointwise", "ranking"],
        default="pointwise",
        help="Candidate selector objective (default: pointwise)",
    )
    ap.add_argument(
        "--selector-dataset-out",
        default="",
        help="Optional path to save expanded selector training dataset CSV",
    )
    args = ap.parse_args()

    alias_types = [v.strip() for v in str(args.alias_types).split(",") if v.strip()]
    alias_groups = [v.strip() for v in str(args.alias_groups).split(",") if v.strip()]
    alias_family = normalize_alias_family(args.alias_family)
    if alias_family and not alias_types:
        alias_types = alias_family_to_alias_types(alias_family)

    defaults = default_training_filters(args.lang, args.format, alias_family=alias_family)
    min_conf = float(args.min_mapping_confidence)
    if min_conf < 0.0:
        min_conf = float(defaults["min_mapping_confidence"])
    require_train_keep = defaults["require_train_keep"] if args.require_train_keep is None else bool(args.require_train_keep)
    exclude_nuclei_fallback = (
        defaults["exclude_nuclei_fallback"]
        if args.exclude_nuclei_fallback is None
        else bool(args.exclude_nuclei_fallback)
    )
    use_pseudo_labels = defaults["use_pseudo_labels"] if args.use_pseudo_labels is None else bool(args.use_pseudo_labels)

    meta = train_lightgbm_bundle(
        language=args.lang,
        format_type=args.format,
        dataset_csv=args.dataset,
        out_dir=args.out_dir,
        group_column=args.group_column,
        alias_types=alias_types,
        alias_groups=alias_groups,
        alias_family=alias_family,
        require_train_keep=bool(require_train_keep),
        min_mapping_confidence=min_conf,
        exclude_nuclei_fallback=bool(exclude_nuclei_fallback),
        use_pseudo_labels=bool(use_pseudo_labels),
        pseudo_weight_high=float(args.pseudo_weight_high),
        pseudo_weight_mid=float(args.pseudo_weight_mid),
    )
    print(f"backend={meta.get('backend')}")
    print(f"language={meta.get('language')}")
    print(f"format_type={meta.get('format_type')}")
    print(f"train_rows={meta.get('train_rows')}")
    print(f"filters={meta.get('filters')}")

    if args.include_selector:
        selector_dataset = args.selector_dataset_out.strip() or os.path.join(args.out_dir, "selector_dataset.csv")
        selector_build = build_selector_dataset_csv_from_delta_dataset(
            args.dataset,
            selector_dataset,
            language=args.lang,
            format_type=args.format,
            alias_types=alias_types,
            alias_groups=alias_groups,
            require_train_keep=bool(require_train_keep),
            min_mapping_confidence=min_conf,
            exclude_nuclei_fallback=bool(exclude_nuclei_fallback),
            use_pseudo_labels=bool(use_pseudo_labels),
        )
        selector_meta = train_lightgbm_selector_bundle(
            language=args.lang,
            format_type=args.format,
            selector_dataset_csv=selector_dataset,
            out_dir=args.out_dir,
            group_column=args.group_column,
            objective=args.selector_objective,
            alias_family=alias_family,
        )
        print(f"selector_dataset_rows={selector_build.get('rows')}")
        print(f"selector_dataset_groups={selector_build.get('groups')}")
        print(f"selector_objective={selector_meta.get('selector_objective')}")
        print(f"selector_metrics={selector_meta.get('metrics')}")


if __name__ == "__main__":
    main()
