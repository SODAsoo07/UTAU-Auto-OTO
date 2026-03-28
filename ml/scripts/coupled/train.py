from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.format_type_utils import normalize_format_type
from core.oto_ml_coupled import train_coupled_bundle, train_coupled_bundle_rawmel
from core.oto_ml.features.mel_patches import MEL_PATCH_CACHE_VERSION, default_patch_cache_root
from core.runtime_encoding import bootstrap_utf8_runtime


def _pick_rawmel_cache_candidate(version_dir: str) -> str:
    """
    Resolve the actual cache leaf directory that contains manifest.json.
    Supports both layouts:
      - <...>/<version>/<spec_hash>/manifest.json
      - <...>/<version>/manifest.json  (legacy/single-spec cache)
    """
    if not version_dir or not os.path.isdir(version_dir):
        return ""
    candidates = []
    manifest_here = os.path.join(version_dir, "manifest.json")
    if os.path.isfile(manifest_here):
        try:
            mtime = os.path.getmtime(manifest_here)
        except Exception:
            mtime = 0.0
        candidates.append((mtime, version_dir))
    for name in os.listdir(version_dir):
        path = os.path.join(version_dir, name)
        if not os.path.isdir(path):
            continue
        manifest = os.path.join(path, "manifest.json")
        if os.path.isfile(manifest):
            try:
                mtime = os.path.getmtime(manifest)
            except Exception:
                mtime = 0.0
            candidates.append((mtime, path))
    if not candidates:
        return ""
    candidates.sort(reverse=True, key=lambda v: v[0])
    return candidates[0][1]


def _auto_rawmel_cache_dir(language: str, format_type: str, root_hint: str = "") -> str:
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower()
    roots = []
    hinted = str(root_hint or "").strip()
    if hinted:
        roots.append(hinted)
    roots.append(default_patch_cache_root())
    roots.append(os.path.join(ROOT, "ml_workspace", "rawmel_cache_noml_auto"))
    roots.append(os.path.join(ROOT, "ml_workspace", "rawmel_cache"))

    seen = set()
    unique_roots = []
    for root in roots:
        path = os.path.normpath(root)
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(path)

    for root in unique_roots:
        if not os.path.isdir(root):
            continue
        # If a leaf cache path is supplied directly, respect it.
        if os.path.isfile(os.path.join(root, "manifest.json")):
            return root
        version_dirs = [
            os.path.join(root, lang, fmt, str(MEL_PATCH_CACHE_VERSION)),
            os.path.join(root, lang, fmt),
            os.path.join(root, str(MEL_PATCH_CACHE_VERSION)),
        ]
        for version_dir in version_dirs:
            picked = _pick_rawmel_cache_candidate(version_dir)
            if picked:
                return picked
    return ""


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
        rawmel_cache = _auto_rawmel_cache_dir(args.lang, args.format, root_hint=rawmel_cache_hint)
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
