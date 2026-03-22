import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_ensemble import train_ensemble_bundle


def _parse_args():
    parser = argparse.ArgumentParser(description="Train ensemble OTO ML bundle from delta dataset.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--format", required=True, dest="format_type")
    parser.add_argument("--dataset", required=True, dest="dataset_csv")
    parser.add_argument(
        "--rawmel-cache-dir",
        required=True,
        dest="rawmel_cache_dir",
        help="RawMel cache path (exact hash cache dir or parent root; auto-resolves by language/format).",
    )
    parser.add_argument("--out-dir", required=True, dest="out_dir")
    parser.add_argument("--group-column", default="voicebank_id")
    parser.add_argument("--alias-family", default="")
    parser.add_argument("--min-mapping-confidence", type=float, default=0.0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--lightgbm-num-boost-round", type=int, default=500)
    parser.add_argument("--lightgbm-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--coupled-epochs", type=int, default=70)
    parser.add_argument("--coupled-batch-size", type=int, default=192)
    parser.add_argument("--coupled-learning-rate", type=float, default=1e-3)
    parser.add_argument("--coupled-backend", default="coupled_nn_v2_rawmel", choices=["coupled_nn_v1", "coupled_nn_v2_rawmel"])
    parser.add_argument("--coupled-device", default="auto", help="auto/cpu/cuda")
    parser.add_argument(
        "--enforce-language-format",
        dest="enforce_language_format",
        action="store_true",
        help="When set, strictly filter dataset by --language and --format before training.",
    )
    parser.add_argument(
        "--no-enforce-language-format",
        dest="enforce_language_format",
        action="store_false",
        help="Skip strict language/format filtering (useful for global alias-family merged datasets).",
    )
    parser.set_defaults(enforce_language_format=True)
    parser.add_argument("--alias-type", action="append", dest="alias_types")
    return parser.parse_args()


def main():
    args = _parse_args()
    meta = train_ensemble_bundle(
        language=args.language,
        format_type=args.format_type,
        dataset_csv=args.dataset_csv,
        out_dir=args.out_dir,
        rawmel_cache_dir=args.rawmel_cache_dir,
        group_column=args.group_column,
        alias_types=args.alias_types,
        alias_family=str(args.alias_family or "").strip(),
        min_mapping_confidence=float(args.min_mapping_confidence),
        num_folds=int(args.num_folds),
        lightgbm_num_boost_round=int(args.lightgbm_num_boost_round),
        lightgbm_early_stopping_rounds=int(args.lightgbm_early_stopping_rounds),
        coupled_epochs=int(args.coupled_epochs),
        coupled_batch_size=int(args.coupled_batch_size),
        coupled_learning_rate=float(args.coupled_learning_rate),
        coupled_backend=str(args.coupled_backend or "").strip().lower(),
        coupled_device=str(args.coupled_device or "").strip(),
        enforce_language_format=bool(args.enforce_language_format),
    )
    print("model_meta.json written:", os.path.join(args.out_dir, "model_meta.json"))
    print("backend:", meta.get("backend", ""))
    print("oof_folds:", meta.get("oof_folds", 0))


if __name__ == "__main__":
    main()
