import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_ensemble import train_ensemble_bundle
from core.oto_ml.features.mel_patches import resolve_rawmel_cache_dir


def _parse_args():
    parser = argparse.ArgumentParser(description="Train ensemble OTO ML bundle from delta dataset.")
    parser.add_argument("--language", required=True)
    parser.add_argument("--format", required=True, dest="format_type")
    parser.add_argument("--dataset", required=True, dest="dataset_csv")
    parser.add_argument("--rawmel-cache-dir", required=True, dest="rawmel_cache_dir")
    parser.add_argument("--out-dir", required=True, dest="out_dir")
    parser.add_argument("--group-column", default="voicebank_id")
    parser.add_argument("--min-mapping-confidence", type=float, default=0.0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--lightgbm-num-boost-round", type=int, default=500)
    parser.add_argument("--lightgbm-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--coupled-epochs", type=int, default=70)
    parser.add_argument("--coupled-batch-size", type=int, default=192)
    parser.add_argument("--coupled-learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--alias-type", action="append", dest="alias_types")
    return parser.parse_args()


def _resolve_cache_leaf(language: str, format_type: str, cache_hint: str) -> str:
    hint = os.path.abspath(str(cache_hint or "").strip())
    if not hint:
        return ""
    manifest_here = os.path.join(hint, "manifest.json")
    if os.path.isfile(manifest_here):
        return hint

    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    direct_candidates = [
        os.path.join(hint, f"{lang}_{fmt}", "rawmel_cache"),
        os.path.join(hint, lang, fmt, "v1"),
        os.path.join(hint, lang, fmt),
    ]
    for candidate in direct_candidates:
        if os.path.isfile(os.path.join(candidate, "manifest.json")):
            return candidate

    resolved = resolve_rawmel_cache_dir(
        language=lang,
        format_type=fmt,
        root_hint=hint,
    )
    if resolved and os.path.isfile(os.path.join(resolved, "manifest.json")):
        return resolved
    return ""


def main():
    args = _parse_args()
    resolved_cache_dir = _resolve_cache_leaf(args.language, args.format_type, args.rawmel_cache_dir)
    if not resolved_cache_dir:
        raise FileNotFoundError(
            "rawmel cache could not be resolved. "
            f"hint={args.rawmel_cache_dir}, language={args.language}, format={args.format_type}"
        )
    if os.path.normcase(os.path.normpath(resolved_cache_dir)) != os.path.normcase(
        os.path.normpath(str(args.rawmel_cache_dir or ""))
    ):
        print(f"[INFO] rawmel cache resolved: {args.rawmel_cache_dir} -> {resolved_cache_dir}")
    meta = train_ensemble_bundle(
        language=args.language,
        format_type=args.format_type,
        dataset_csv=args.dataset_csv,
        out_dir=args.out_dir,
        rawmel_cache_dir=resolved_cache_dir,
        group_column=args.group_column,
        alias_types=args.alias_types,
        min_mapping_confidence=float(args.min_mapping_confidence),
        num_folds=int(args.num_folds),
        lightgbm_num_boost_round=int(args.lightgbm_num_boost_round),
        lightgbm_early_stopping_rounds=int(args.lightgbm_early_stopping_rounds),
        coupled_epochs=int(args.coupled_epochs),
        coupled_batch_size=int(args.coupled_batch_size),
        coupled_learning_rate=float(args.coupled_learning_rate),
        coupled_device=str(args.device),
    )
    print("model_meta.json written:", os.path.join(args.out_dir, "model_meta.json"))
    print("backend:", meta.get("backend", ""))
    print("oof_folds:", meta.get("oof_folds", 0))


if __name__ == "__main__":
    main()
