from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_coupled import train_coupled_bundle, train_coupled_bundle_rawmel
from core.oto_ml.features.mel_patches import resolve_rawmel_cache_dir
from core.runtime_encoding import bootstrap_utf8_runtime


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Train coupled mel+oto OTO ML bundle.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--format", required=True, help="Format type (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--dataset", required=True, help="Input dataset CSV")
    ap.add_argument("--out-dir", required=True, help="Output model directory")
    ap.add_argument("--group-column", default="wav_norm")
    ap.add_argument("--alias-types", default="", help="Comma-separated alias_type filter")
    ap.add_argument("--alias-family", default="", help="Alias family filter (cv/vc/vcv/vowel)")
    ap.add_argument("--min-mapping-confidence", type=float, default=0.0)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--min-confidence", type=float, default=0.50)
    ap.add_argument("--progress-every", type=int, default=100, help="Print progress every N batches (0 disables).")
    ap.add_argument("--backend", default="coupled_nn_v1", help="coupled_nn_v1 or coupled_nn_v2_rawmel")
    ap.add_argument("--rawmel-cache", default="", help="Raw mel patch cache directory for coupled_nn_v2_rawmel")
    ap.add_argument("--rawmel-prefetch", default="auto", help="Raw mel cache prefetch: auto|none|train|gpu")
    ap.add_argument("--rawmel-max-shard-cache", type=int, default=2, help="Max mel patch shards to keep in memory")
    ap.add_argument("--sampler", default="", help="Optional sampler override: group_balanced|shuffle")
    ap.add_argument("--hard-mining-strength", type=float, default=-1.0, help="Optional hard-example mining strength override")
    ap.add_argument("--pair-warmup-epochs", type=int, default=-1, help="Optional VC/CV pair loss warmup epochs override")
    args = ap.parse_args()

    alias_types = [v.strip() for v in str(args.alias_types).split(",") if v.strip()]
    alias_family = str(args.alias_family or "").strip()
    backend = str(args.backend or "coupled_nn_v1").strip().lower()
    env_prefix = "UTOA_ML_RAWMEL_" if backend == "coupled_nn_v2_rawmel" else "UTOA_ML_COUPLED_"
    if str(args.sampler or "").strip():
        os.environ[f"{env_prefix}SAMPLER"] = str(args.sampler).strip()
    if float(args.hard_mining_strength) >= 0.0:
        os.environ[f"{env_prefix}HARD_MINING_STRENGTH"] = str(float(args.hard_mining_strength))
    if int(args.pair_warmup_epochs) >= 0:
        os.environ[f"{env_prefix}PAIR_WARMUP_EPOCHS"] = str(int(args.pair_warmup_epochs))
    rawmel_cache_hint = str(args.rawmel_cache or "").strip()
    if backend == "coupled_nn_v2_rawmel":
        if rawmel_cache_hint and not os.path.isdir(rawmel_cache_hint):
            print(f"[TRAIN] rawmel_cache not found: {rawmel_cache_hint}")
            rawmel_cache_hint = ""
        rawmel_cache = resolve_rawmel_cache_dir(
            language=args.lang,
            format_type=args.format,
            root_hint=rawmel_cache_hint,
            extra_roots=[
                os.path.join(ROOT, "ml_workspace", "rawmel_cache_noml_auto"),
                os.path.join(ROOT, "ml_workspace", "rawmel_cache"),
            ],
        )
        if rawmel_cache:
            print(f"[TRAIN] rawmel_cache auto-selected: {rawmel_cache}")
        if not rawmel_cache:
            raise SystemExit("--rawmel-cache is required for coupled_nn_v2_rawmel (no auto cache found)")
        meta = train_coupled_bundle_rawmel(
            language=args.lang,
            format_type=args.format,
            dataset_csv=args.dataset,
            out_dir=args.out_dir,
            rawmel_cache_dir=rawmel_cache,
            group_column=args.group_column,
            alias_types=alias_types,
            alias_family=alias_family,
            min_mapping_confidence=float(args.min_mapping_confidence),
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            min_confidence=float(args.min_confidence),
            progress_every=int(args.progress_every),
            rawmel_prefetch=str(args.rawmel_prefetch),
            rawmel_max_shard_cache=int(args.rawmel_max_shard_cache),
        )
    else:
        meta = train_coupled_bundle(
            language=args.lang,
            format_type=args.format,
            dataset_csv=args.dataset,
            out_dir=args.out_dir,
            group_column=args.group_column,
            alias_types=alias_types,
            alias_family=alias_family,
            min_mapping_confidence=float(args.min_mapping_confidence),
            device=args.device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            min_confidence=float(args.min_confidence),
            progress_every=int(args.progress_every),
        )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
