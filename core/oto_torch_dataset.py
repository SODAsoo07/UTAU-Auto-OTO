from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.oto_ml_features import CATEGORICAL_FEATURES, TARGET_NAMES, get_delta_clip_limits, normalize_key
from core.oto_torch_features import (
    DEFAULT_MEL_BINS,
    DEFAULT_WINDOW_FRAMES,
    build_tabular_parts,
    extract_centered_mel_window,
    get_numeric_feature_names,
)


TORCH_TASKS: Dict[str, Dict[str, object]] = {
    "japanese_bridge_vcv": {
        "language": "japanese",
        "formats": {"vcv"},
        "alias_types": {"vc", "vv", "vcv", "cv_head"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "japanese_bridge_cvvc": {
        "language": "japanese",
        "formats": {"cvvc"},
        "alias_types": {"vc", "vv", "cv_head"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "japanese_bridge_vcv_vv": {
        "language": "japanese",
        "formats": {"vcv"},
        "alias_types": {"vv"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "japanese_bridge_vcv_vc": {
        "language": "japanese",
        "formats": {"vcv"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "japanese_bridge_vcv_vcv": {
        "language": "japanese",
        "formats": {"vcv"},
        "alias_types": {"vcv"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "japanese_bridge_vcv_cv_head": {
        "language": "japanese",
        "formats": {"vcv"},
        "alias_types": {"cv_head"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "japanese_bridge_cvvc_vv": {
        "language": "japanese",
        "formats": {"cvvc"},
        "alias_types": {"vv"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "japanese_bridge_cvvc_vc": {
        "language": "japanese",
        "formats": {"cvvc"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "japanese_bridge_cvvc_cv_head": {
        "language": "japanese",
        "formats": {"cvvc"},
        "alias_types": {"cv_head"},
        "min_mapping_confidence": 0.0,
        "require_train_keep": True,
    },
    "korean_bridge_cvvc": {
        "language": "korean",
        "formats": {"cvvc"},
        "alias_types": {"vc", "vv"},
        "min_mapping_confidence": 0.3,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "allowed_coda_types_for_vc": {"stop", "nasal", "liquid"},
    },
    "korean_bridge_cvvc_vv": {
        "language": "korean",
        "formats": {"cvvc"},
        "alias_types": {"vv"},
        "min_mapping_confidence": 0.45,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "require_alias_occurrence_mapping": True,
    },
    "korean_bridge_cvvc_vc": {
        "language": "korean",
        "formats": {"cvvc"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.45,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "allowed_coda_types_for_vc": {"stop", "nasal", "liquid"},
    },
    "korean_bridge_cvvc_vc_sonorant": {
        "language": "korean",
        "formats": {"cvvc"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.50,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "allowed_alias_groups": {"vc_sonorant"},
        "allowed_coda_types_for_vc": {"nasal", "liquid"},
        "max_abs_targets": {
            "delta_cons": 220.0,
            "delta_cutoff": 1600.0,
            "delta_pre": 180.0,
            "delta_ovl": 140.0,
        },
    },
    "korean_bridge_cvvc_vc_stop": {
        "language": "korean",
        "formats": {"cvvc"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.60,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "allowed_alias_groups": {"vc_stop"},
        "allowed_coda_types_for_vc": {"stop"},
        "max_abs_targets": {
            "delta_offset": 1800.0,
            "delta_cons": 220.0,
            "delta_cutoff": 1800.0,
            "delta_pre": 160.0,
            "delta_ovl": 90.0,
        },
    },
    "korean_bridge_cvc": {
        "language": "korean",
        "formats": {"cvc"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.3,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "allowed_coda_types_for_vc": {"stop", "nasal", "liquid"},
    },
    "korean_bridge_cvc_vc": {
        "language": "korean",
        "formats": {"cvc"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.45,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "allowed_coda_types_for_vc": {"stop", "nasal", "liquid"},
    },
    "korean_bridge_cvc_vc_sonorant": {
        "language": "korean",
        "formats": {"cvc"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.50,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "allowed_alias_groups": {"vc_sonorant"},
        "allowed_coda_types_for_vc": {"nasal", "liquid"},
        "max_abs_targets": {
            "delta_cons": 220.0,
            "delta_cutoff": 2200.0,
            "delta_pre": 180.0,
            "delta_ovl": 140.0,
        },
    },
    "korean_bridge_cvc_vc_stop": {
        "language": "korean",
        "formats": {"cvc"},
        "alias_types": {"vc"},
        "min_mapping_confidence": 0.60,
        "require_train_keep": False,
        "exclude_nuclei_fallback": True,
        "allowed_alias_groups": {"vc_stop"},
        "allowed_coda_types_for_vc": {"stop"},
        "max_abs_targets": {
            "delta_offset": 2400.0,
            "delta_cons": 220.0,
            "delta_cutoff": 2400.0,
            "delta_pre": 160.0,
            "delta_ovl": 90.0,
        },
    },
}

_ALL_WAV_DIRS_KEY = "__all_wav_dirs__"


@dataclass
class TorchDatasetSummary:
    task_name: str
    out_dir: str
    row_count: int
    shard_count: int
    voicebank_count: int
    skipped_rows: int
    index_path: str


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_build_manifest() -> str:
    return os.path.join(os.path.dirname(_repo_root()), "ml_workspace", "reports", "training_build_results.json")


def _default_dataset_root() -> str:
    return os.path.join(_repo_root(), "dataset")


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _task_spec(task_name: str) -> Dict[str, object]:
    if task_name not in TORCH_TASKS:
        raise KeyError(f"Unknown torch dataset task: {task_name}")
    return TORCH_TASKS[task_name]


def _to_float(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def _has_nonempty_value(row: Dict[str, object], key: str) -> bool:
    value = row.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _keep_row_for_task(row: Dict[str, str], spec: Dict[str, object]) -> Tuple[bool, str]:
    language = str(row.get("language", "") or "").strip().lower()
    format_type = str(row.get("format_type", "") or "").strip().lower()
    alias_type = str(row.get("alias_type", "") or "").strip().lower()
    alias_group = str(row.get("alias_group", "") or "").strip().lower()
    coda_type = str(row.get("coda_type", "") or "").strip().lower()

    if language != str(spec["language"]):
        return False, "language_mismatch"
    if format_type not in set(spec.get("formats") or []):
        return False, "format_mismatch"
    if alias_type not in set(spec.get("alias_types") or []):
        return False, "alias_type_filtered"
    allowed_alias_groups = spec.get("allowed_alias_groups")
    if allowed_alias_groups and alias_group not in set(str(v).strip().lower() for v in allowed_alias_groups):
        return False, "alias_group_filtered"
    if spec.get("require_train_keep") and int(float(row.get("train_keep_default", "0") or 0)) <= 0:
        return False, "train_keep_filtered"
    min_conf = float(spec.get("min_mapping_confidence", 0.0) or 0.0)
    if _has_nonempty_value(row, "mapping_confidence") and _to_float(row, "mapping_confidence", 0.0) < min_conf:
        return False, "low_mapping_confidence"
    if spec.get("exclude_nuclei_fallback") and _has_nonempty_value(row, "used_nuclei_fallback") and int(_to_float(row, "used_nuclei_fallback", 0.0)) > 0:
        return False, "nuclei_fallback_filtered"
    if spec.get("require_alias_occurrence_mapping") and _has_nonempty_value(row, "used_alias_occurrence_mapping"):
        if int(_to_float(row, "used_alias_occurrence_mapping", 0.0)) <= 0:
            return False, "alias_occurrence_required"
    allowed_coda = spec.get("allowed_coda_types_for_vc")
    if alias_type == "vc" and allowed_coda and coda_type not in set(allowed_coda):
        return False, "coda_type_filtered"
    max_abs_targets = spec.get("max_abs_targets") or {}
    for target_name, limit in max_abs_targets.items():
        if _has_nonempty_value(row, target_name):
            if abs(_to_float(row, target_name, 0.0)) > float(limit):
                return False, f"{target_name}_outlier"
    return True, ""


def _load_build_manifest(path: str) -> List[Dict[str, object]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_wav_dir_map(build_manifest: List[Dict[str, object]]) -> Dict[Tuple[str, str, str], List[str]]:
    out: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    def _add_key(vb_id: str, language: str, format_type: str, wav_dir: str) -> None:
        key = (vb_id, language, format_type)
        if wav_dir not in out[key]:
            out[key].append(wav_dir)

    for item in build_manifest or []:
        if str(item.get("status", "")) != "built":
            continue
        voicebank_id = str(item.get("voicebank_id", "") or "").strip()
        voicebank_norm = normalize_key(voicebank_id)
        language = str(item.get("language", "") or "").strip().lower()
        format_type = str(item.get("format_type", "") or "").strip().lower()
        wav_dir = str(item.get("wav_dir", "") or "").strip()
        if wav_dir and os.path.isdir(wav_dir):
            _add_key(voicebank_id, language, format_type, wav_dir)
            if voicebank_norm and voicebank_norm != voicebank_id:
                _add_key(voicebank_norm, language, format_type, wav_dir)
    return out


def _scan_dataset_voicebanks(dataset_root: str) -> Dict[str, List[str]]:
    voicebanks: Dict[str, List[str]] = defaultdict(list)

    def _register_key(key: str, wav_root: str) -> None:
        if not key:
            return
        if wav_root not in voicebanks[key]:
            voicebanks[key].append(wav_root)

    def _register_key_variants(raw_key: str, wav_root: str) -> None:
        key = str(raw_key or "").strip()
        if not key:
            return
        _register_key(key, wav_root)
        norm = normalize_key(key)
        if norm and norm != key:
            _register_key(norm, wav_root)

    if not dataset_root or not os.path.isdir(dataset_root):
        return voicebanks
    for language in os.listdir(dataset_root):
        lang_dir = os.path.join(dataset_root, language)
        if not os.path.isdir(lang_dir) or language.startswith("_"):
            continue
        for fmt in os.listdir(lang_dir):
            fmt_dir = os.path.join(lang_dir, fmt)
            if not os.path.isdir(fmt_dir):
                continue
            for root, dirs, files in os.walk(fmt_dir):
                wav_files = [fn for fn in files if fn.lower().endswith('.wav')]
                if wav_files:
                    _register_key(_ALL_WAV_DIRS_KEY, root)

                    # Register both this wav folder and parent folders as candidate voicebank keys.
                    current = root
                    while True:
                        if os.path.normcase(current) == os.path.normcase(fmt_dir):
                            break
                        base = os.path.basename(current)
                        _register_key_variants(base, root)
                        parent = os.path.dirname(current)
                        if parent == current:
                            break
                        current = parent
    return voicebanks


def _resolve_wav_path(row: Dict[str, str], wav_dir_map: Dict[Tuple[str, str, str], List[str]], dataset_voicebanks: Dict[str, List[str]], wav_cache: Dict[str, Dict[str, str]]) -> str:
    voicebank_id = str(row.get("voicebank_id", "") or "").strip()
    voicebank_norm = normalize_key(voicebank_id)
    language = str(row.get("language", "") or "").strip().lower()
    format_type = str(row.get("format_type", "") or "").strip().lower()
    wav_name = str(row.get("wav", "") or "").strip()
    wav_norm = normalize_key(wav_name)

    candidate_dirs: List[str] = []
    for vb_key in (voicebank_id, voicebank_norm):
        if not vb_key:
            continue
        for path in wav_dir_map.get((vb_key, language, format_type), []):
            if path not in candidate_dirs:
                candidate_dirs.append(path)
        for path in dataset_voicebanks.get(vb_key, []):
            if path not in candidate_dirs:
                candidate_dirs.append(path)

    # Last-resort fallback: search all staged wav folders in this dataset.
    for path in dataset_voicebanks.get(_ALL_WAV_DIRS_KEY, []):
        if path not in candidate_dirs:
            candidate_dirs.append(path)

    for wav_dir in candidate_dirs:
        if wav_dir not in wav_cache:
            index = {}
            for root, _dirs, files in os.walk(wav_dir):
                for fn in files:
                    if fn.lower().endswith('.wav'):
                        index[normalize_key(fn)] = os.path.join(root, fn)
            wav_cache[wav_dir] = index
        hit = wav_cache[wav_dir].get(wav_norm)
        if hit and os.path.isfile(hit):
            return hit
    return ""


def _target_center_ms(row: Dict[str, str]) -> float:
    base_offset = _to_float(row, "base_offset", 0.0)
    base_pre = _to_float(row, "base_pre", 0.0)
    expected = _to_float(row, "expected_anchor_ms", 0.0)
    center = base_offset + base_pre
    if center <= 0.0 and expected > 0.0:
        center = expected
    return max(center, 0.0)


def _clipped_target_array(row: Dict[str, str], language: str) -> np.ndarray:
    clip_limits = get_delta_clip_limits(language)
    vals: List[float] = []
    for name in TARGET_NAMES:
        v = _to_float(row, name, 0.0)
        lo_hi = clip_limits.get(name)
        if lo_hi and len(lo_hi) == 2:
            lo, hi = float(lo_hi[0]), float(lo_hi[1])
            v = max(lo, min(hi, v))
        vals.append(float(v))
    return np.array(vals, dtype=np.float32)


def build_torch_dataset(
    task_name: str,
    dataset_csv: str,
    out_dir: str,
    dataset_root: str = "",
    build_manifest_path: str = "",
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    mel_bins: int = DEFAULT_MEL_BINS,
) -> TorchDatasetSummary:
    spec = _task_spec(task_name)
    dataset_root = dataset_root or _default_dataset_root()
    build_manifest_path = build_manifest_path or _default_build_manifest()
    rows = _read_csv_rows(dataset_csv)
    build_manifest = _load_build_manifest(build_manifest_path)
    wav_dir_map = _build_wav_dir_map(build_manifest)
    dataset_voicebanks = _scan_dataset_voicebanks(dataset_root)
    wav_cache: Dict[str, Dict[str, str]] = {}

    os.makedirs(out_dir, exist_ok=True)
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    skipped_rows = 0
    skip_reasons: Dict[str, int] = defaultdict(int)
    for row in rows:
        keep, reason = _keep_row_for_task(row, spec)
        if not keep:
            skipped_rows += 1
            skip_reasons[reason] += 1
            continue
        groups[str(row.get("voicebank_id", "unknown") or "unknown")].append(row)

    numeric_names = get_numeric_feature_names()
    categorical_names = list(CATEGORICAL_FEATURES)
    shard_entries: List[Dict[str, object]] = []
    total_rows = 0
    total_shards = 0

    for voicebank_id, shard_rows in sorted(groups.items()):
        numeric_buf: List[np.ndarray] = []
        mel_buf: List[np.ndarray] = []
        target_buf: List[np.ndarray] = []
        cat_buf: List[List[str]] = []
        aliases: List[str] = []
        wavs: List[str] = []
        wav_paths: List[str] = []
        line_indices: List[int] = []
        format_types: List[str] = []
        languages: List[str] = []
        alias_types: List[str] = []
        alias_groups: List[str] = []

        for row in shard_rows:
            wav_path = _resolve_wav_path(row, wav_dir_map, dataset_voicebanks, wav_cache)
            if not wav_path:
                skipped_rows += 1
                skip_reasons["wav_not_found"] += 1
                continue
            center_ms = _target_center_ms(row)
            try:
                mel_window, audio_stats = extract_centered_mel_window(wav_path, center_ms, window_frames=window_frames, mel_bins=mel_bins)
            except Exception:
                skipped_rows += 1
                skip_reasons["mel_window_failed"] += 1
                continue
            merged = dict(row)
            merged.update(audio_stats)
            numeric, categorical = build_tabular_parts(merged, categorical_features=categorical_names)
            target = _clipped_target_array(row, language=str(row.get("language", "") or ""))

            mel_buf.append(mel_window.astype(np.float16))
            numeric_buf.append(numeric.astype(np.float32))
            target_buf.append(target)
            cat_buf.append([categorical.get(name, "") for name in categorical_names])
            aliases.append(str(row.get("alias", "") or ""))
            wavs.append(str(row.get("wav", "") or ""))
            wav_paths.append(wav_path)
            line_indices.append(int(float(row.get("line_index", "0") or 0)))
            format_types.append(str(row.get("format_type", "") or ""))
            languages.append(str(row.get("language", "") or ""))
            alias_types.append(str(row.get("alias_type", "") or ""))
            alias_groups.append(str(row.get("alias_group", "") or ""))

        if not mel_buf:
            continue

        shard_path = os.path.join(shard_dir, f"{normalize_key(voicebank_id) or 'voicebank'}.npz")
        np.savez_compressed(
            shard_path,
            mel_windows=np.stack(mel_buf, axis=0),
            numeric_features=np.stack(numeric_buf, axis=0),
            categorical_features=np.array(cat_buf, dtype='<U128'),
            targets=np.stack(target_buf, axis=0),
            alias=np.array(aliases, dtype='<U256'),
            wav=np.array(wavs, dtype='<U256'),
            wav_path=np.array(wav_paths, dtype='<U512'),
            line_index=np.array(line_indices, dtype=np.int32),
            voicebank_id=np.array([voicebank_id] * len(mel_buf), dtype='<U128'),
            language=np.array(languages, dtype='<U32'),
            format_type=np.array(format_types, dtype='<U32'),
            alias_type=np.array(alias_types, dtype='<U64'),
            alias_group=np.array(alias_groups, dtype='<U64'),
        )
        count = len(mel_buf)
        total_rows += count
        total_shards += 1
        shard_entries.append({
            "voicebank_id": voicebank_id,
            "path": shard_path,
            "rows": count,
        })

    index = {
        "task_name": task_name,
        "language": str(spec["language"]),
        "formats": sorted(spec.get("formats") or []),
        "window_frames": int(window_frames),
        "mel_bins": int(mel_bins),
        "numeric_feature_names": numeric_names,
        "categorical_feature_names": categorical_names,
        "target_names": list(TARGET_NAMES),
        "shards": shard_entries,
        "summary": {
            "row_count": total_rows,
            "shard_count": total_shards,
            "voicebank_count": len(shard_entries),
            "skipped_rows": skipped_rows,
            "skip_reasons": dict(skip_reasons),
        },
    }
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return TorchDatasetSummary(
        task_name=task_name,
        out_dir=os.path.abspath(out_dir),
        row_count=total_rows,
        shard_count=total_shards,
        voicebank_count=len(shard_entries),
        skipped_rows=skipped_rows,
        index_path=index_path,
    )


def load_torch_dataset_index(index_path: str) -> Dict[str, object]:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)
