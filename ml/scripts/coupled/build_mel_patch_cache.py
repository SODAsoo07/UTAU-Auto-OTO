from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Tuple

from core.format_type_utils import normalize_format_type, normalize_language_name
from core.oto_ml_collection_discovery import discover_training_candidates_from_dataset_root
from core.oto_ml.features.schema import FEATURE_VERSION
from core.oto_ml.features.mel_patches import (
    DEFAULT_FRAME_HOP_MS,
    DEFAULT_MEL_BINS,
    DEFAULT_ONSET_WINDOW_MS,
    DEFAULT_TAIL_WINDOW_MS,
    DEFAULT_WIN_MS,
    MelPatchCacheWriter,
    build_patch_spec,
    patch_cache_dir,
    make_mel_patch_key,
    resolve_mel_patch_anchors,
    build_wav_index,
    find_wav_path,
    extract_patches_for_row,
)
from core.runtime_encoding import bootstrap_utf8_runtime


def _parse_window(text: str, default: Tuple[float, float]) -> Tuple[float, float]:
    if not text:
        return default
    raw = str(text).strip().replace(" ", "")
    if "," not in raw:
        return default
    parts = raw.split(",", 1)
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return default


def _infer_unique(values) -> str:
    uniq = {str(v).strip().lower() for v in values if str(v).strip()}
    if len(uniq) == 1:
        return list(uniq)[0]
    return ""


def _build_candidate_map(dataset_root: str):
    candidates = discover_training_candidates_from_dataset_root(dataset_root)
    mapping = {}
    for c in candidates:
        key = (
            str(c.language).strip().lower(),
            normalize_format_type(c.language, c.format_type) or str(c.format_type).strip().lower(),
            str(c.voicebank_id).strip(),
        )
        mapping[key] = c.wav_dir
    return mapping


def main():
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Build raw mel patch cache for coupled v2.")
    ap.add_argument("--dataset", required=True, help="Input dataset CSV")
    ap.add_argument("--dataset-root", required=True, help="Dataset root (for voicebank wav_dir discovery)")
    ap.add_argument("--out-dir", default="", help="Cache output directory (optional)")
    ap.add_argument("--lang", default="", help="Language override (korean/japanese)")
    ap.add_argument("--format", default="", help="Format override (cv/cvc/cvvc/vcv/general)")
    ap.add_argument("--mel-bins", type=int, default=DEFAULT_MEL_BINS)
    ap.add_argument("--frame-hop-ms", type=float, default=DEFAULT_FRAME_HOP_MS)
    ap.add_argument("--win-ms", type=float, default=DEFAULT_WIN_MS)
    ap.add_argument("--onset-window-ms", default="", help="e.g. -80,160")
    ap.add_argument("--tail-window-ms", default="", help="e.g. -160,80")
    ap.add_argument("--max-rows-per-shard", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    dataset_csv = os.path.abspath(args.dataset)
    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isfile(dataset_csv):
        raise FileNotFoundError(dataset_csv)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(dataset_root)

    candidate_map = _build_candidate_map(dataset_root)
    wav_index_cache: Dict[str, Dict[str, str]] = {}

    rows = []
    with open(dataset_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError("Dataset CSV is empty")

    inferred_lang = _infer_unique([r.get("language", "") for r in rows])
    inferred_fmt = _infer_unique([r.get("format_type", "") for r in rows])
    lang = normalize_language_name(args.lang or inferred_lang or "korean")
    fmt = normalize_format_type(lang, args.format or inferred_fmt or "general") or "general"

    onset_window = _parse_window(args.onset_window_ms, DEFAULT_ONSET_WINDOW_MS)
    tail_window = _parse_window(args.tail_window_ms, DEFAULT_TAIL_WINDOW_MS)
    patch_spec = build_patch_spec(
        mel_bins=int(args.mel_bins),
        frame_hop_ms=float(args.frame_hop_ms),
        win_ms=float(args.win_ms),
        onset_window_ms=onset_window,
        tail_window_ms=tail_window,
    )
    patch_spec["source_feature_version"] = str(FEATURE_VERSION)
    out_dir = args.out_dir.strip() or patch_cache_dir(lang, fmt, patch_spec)
    os.makedirs(out_dir, exist_ok=True)

    writer = MelPatchCacheWriter(
        out_dir,
        patch_spec,
        max_rows_per_shard=int(args.max_rows_per_shard),
    )

    missing = []
    mel_cache = {}
    total = 0
    for row in rows:
        row_lang = normalize_language_name(row.get("language", "") or lang)
        row_fmt = normalize_format_type(row_lang, row.get("format_type", "") or fmt) or fmt
        if row_lang != lang or row_fmt != fmt:
            continue
        voicebank_id = str(row.get("voicebank_id", "")).strip()
        key = (row_lang, row_fmt, voicebank_id)
        wav_dir = candidate_map.get(key, "")
        if not wav_dir:
            missing.append({"reason": "wav_dir_missing", "voicebank_id": voicebank_id})
            if not args.allow_missing:
                continue
        wav_name = str(row.get("wav", "")).strip()
        if not wav_name:
            missing.append({"reason": "wav_name_missing", "voicebank_id": voicebank_id})
            if not args.allow_missing:
                continue
        if wav_dir not in wav_index_cache:
            wav_index_cache[wav_dir] = build_wav_index(wav_dir)
        wav_path = find_wav_path(wav_name, wav_dir, wav_index_cache.get(wav_dir))
        if not wav_path:
            missing.append({"reason": "wav_not_found", "voicebank_id": voicebank_id, "wav": wav_name})
            if not args.allow_missing:
                continue

        onset_anchor = float(row.get("mel_onset_anchor_ms", 0.0) or 0.0)
        tail_anchor = float(row.get("mel_tail_anchor_ms", 0.0) or 0.0)
        if onset_anchor <= 0.0 or tail_anchor <= 0.0:
            onset_anchor, tail_anchor, _src = resolve_mel_patch_anchors(row)

        try:
            onset_patch, tail_patch = extract_patches_for_row(
                wav_path=wav_path,
                onset_anchor_ms=onset_anchor,
                tail_anchor_ms=tail_anchor,
                patch_spec=patch_spec,
                mel_cache=mel_cache,
            )
        except Exception as exc:
            missing.append({"reason": "patch_extract_failed", "wav": wav_name, "err": str(exc)})
            if not args.allow_missing:
                continue

        patch_key = str(row.get("mel_patch_key", "") or "").strip()
        if not patch_key:
            patch_key = make_mel_patch_key(
                language=row_lang,
                format_type=row_fmt,
                voicebank_id=voicebank_id,
                wav_norm=str(row.get("wav_norm", "") or ""),
                alias_norm=str(row.get("alias_norm", "") or ""),
                occurrence_index=int(float(row.get("occurrence_index", 0) or 0)),
                row_index_in_wav=int(float(row.get("row_index_in_wav", 0) or 0)),
            )
        if not patch_key:
            missing.append({"reason": "mel_patch_key_missing", "wav": wav_name})
            if not args.allow_missing:
                continue

        writer.append(patch_key, onset_patch, tail_patch)
        total += 1

    if missing and not args.allow_missing:
        raise RuntimeError(f"Missing patches: {len(missing)} (first={missing[:3]})")

    manifest_path = writer.finalize(total_rows=total)
    print(manifest_path)


if __name__ == "__main__":
    main()
