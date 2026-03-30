"""
OTO ML feature extraction helpers.

This module builds row-level structured features from:
- auto-generated OTO rows
- TextGrid boundaries
- WAV-derived mel/F0 summaries
- alias/format metadata

The output is shared by dataset generation, model training, and runtime inference.

.. note::
    Schema, constants, and cache logic are now also available from
    ``core.oto_ml.features.schema`` and ``core.oto_ml.features.caches``
    for modular, fine-grained access. This file retains all definitions
    for full backward compatibility.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import hashlib
import tempfile
from typing import Dict, List, Optional, Tuple

import textgrid

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

from core.lab_generator import load_custom_phonemes
from core.kr_oto_rules import should_ignore_korean_alias
from core.oto_ml_mapping_quality import augment_mapping_quality_features
from core.oto_ml_reliability import blank_risk_flag, compute_blank_risk_score
from core.format_type_utils import normalize_format_type
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key
from core.prefix_map_utils import find_prefix_map_path, strip_prefix_map_affixes
from core.pipeline_status import has_textgrid_files
from core.oto_ml.features.mel_patches import (
    make_mel_patch_debug_key,
    make_mel_patch_key,
    resolve_mel_patch_anchors,
)

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v17"
TRAIN_ROW_MATCH_VERSION = "v17"
TARGET_NAMES = ["delta_offset", "delta_cons", "delta_cutoff", "delta_pre", "delta_ovl"]
ANCHOR_TARGET_NAMES = ["delta_offset", "delta_pre", "delta_cutoff"]
DELTA_TARGET_NAMES = ["delta_cons", "delta_ovl"]
AUX_TARGET_NAMES = ["aux_vowel_start_rel", "aux_vowel_end_rel", "aux_next_onset_rel"]

FEATURE_NAMES = [
    "language", "format_type", "alias_type", "alias_group", "row_index_in_wav", "row_ratio_in_wav",
    "file_row_count", "file_cv_count", "file_vc_count", "file_vv_count", "file_vcv_count",
    "file_br_count", "file_mono_count", "file_cv_ratio", "file_vc_ratio",
    "file_vc_cv_ratio", "file_cv_vc_balance", "file_mean_energy", "file_voiced_ratio",
    "file_mean_syllable_dur_ms",
    "is_head_row", "is_tail_row", "wav_duration_ms", "base_offset", "base_cons",
    "base_cutoff_abs", "base_pre", "base_ovl", "base_cons_gap", "base_cut_gap",
    "base_ovl_ratio", "curr_phone_start_ms", "curr_phone_end_ms", "curr_phone_len_ms",
    "curr_vowel_start_ms", "curr_vowel_end_ms", "curr_vowel_len_ms", "syllable_start_ms",
    "syllable_end_ms", "syllable_len_ms", "prev_phone_gap_ms", "next_phone_gap_ms",
    "expected_anchor_ms", "base_offset_to_expected_ms", "base_pre_to_expected_ms",
    "base_cutoff_to_next_anchor_ms", "energy_mean", "energy_min", "energy_max",
    "energy_slope_pre", "energy_slope_post", "valley_energy", "valley_dist_from_cutoff_ms",
    "db_mean", "db_min", "db_silence_ratio", "f0_voicing_mean", "f0_voicing_near_pre",
    "f0_valid_ratio", "f0_gap_ratio", "f0_continuity", "rms_norm_wav", "rms_norm_vb",
    "zcr_mean", "spectral_flux_mean", "onset_class", "voicing_class", "is_tense",
    "is_diphthong", "coda_type", "vowel_class", "mora_position", "bridge_type",
    "is_nasal_or_sonorant", "prev_alias_type", "next_alias_type", "prev_base_pre",
    "next_base_pre", "prev_base_offset", "next_base_offset", "prev_base_cutoff_abs",
    "next_base_cutoff_abs",
    "m_map_ok", "m_offset_hint_abs", "m_cutoff_hint_abs", "m_hint_rank",
    "mapping_confidence", "used_alias_occurrence_mapping", "used_exact_vowel_fix",
    "used_nuclei_fallback", "used_alias_based_syllables", "words_vs_alias_score_margin",
    "jump_blocked_flag", "mapping_reason_code",
    "local_peak_db", "local_valley_db", "mel_window_energy_mean", "mel_window_silence_ratio",
    "mel_voiced_formant_ratio", "mel_silence_sparse_ratio", "mel_unvoiced_diffuse_ratio",
    "mel_breath_like_ratio", "blank_span_confidence", "mel_offset_candidate_ms",
    "mel_cutoff_candidate_ms", "mel_voiced_onset_ms", "mel_vowel_nucleus_start_ms",
    "mel_vowel_nucleus_end_ms", "mel_voiced_onset_to_pre_ms", "mel_safety_clamped",
    "onset_patch_energy_mean", "onset_patch_voiced_ratio",
    "onset_patch_unvoiced_ratio", "tail_patch_energy_mean", "tail_patch_silence_ratio",
    "syllable_blank_confidence", "syllable_mel_voiced_conf", "syllable_mel_silence_conf",
    "syllable_mel_unvoiced_conf", "syllable_mel_breath_conf",
]

CATEGORICAL_FEATURES = [
    "language", "format_type", "alias_type", "alias_group", "onset_class", "voicing_class",
    "coda_type", "vowel_class", "mora_position", "bridge_type", "prev_alias_type",
    "next_alias_type", "mapping_reason_code",
]

FEATURE_DEFAULTS = {name: 0.0 for name in FEATURE_NAMES}
for _name in CATEGORICAL_FEATURES:
    FEATURE_DEFAULTS[_name] = ""

def _stable_source_oto_id(path: str) -> str:
    abs_path = os.path.abspath(path or "")
    digest = hashlib.sha1(abs_path.encode("utf-8", errors="replace")).hexdigest()
    return digest[:16]


KR_DELTA_CLIP_LIMITS = {
    "delta_offset": [-220.0, 220.0],
    "delta_cons": [-220.0, 220.0],
    "delta_cutoff": [-260.0, 260.0],
    "delta_pre": [-180.0, 180.0],
    "delta_ovl": [-140.0, 140.0],
}

# alias_type별 세분화된 delta 클리핑 범위
KR_DELTA_CLIP_LIMITS_BY_TYPE = {
    "cv": {
        "delta_offset": [-36.0, 40.0],
        "delta_cons": [-110.0, 110.0],
        "delta_cutoff": [-140.0, 120.0],
        "delta_pre": [-28.0, 34.0],
        "delta_ovl": [-56.0, 56.0],
    },
    "cv_head": {
        "delta_offset": [-42.0, 48.0],
        "delta_cons": [-120.0, 120.0],
        "delta_cutoff": [-150.0, 130.0],
        "delta_pre": [-32.0, 38.0],
        "delta_ovl": [-60.0, 60.0],
    },
    "vc": {
        "delta_offset": [-200.0, 200.0],
        "delta_cons": [-180.0, 180.0],
        "delta_cutoff": [-240.0, 240.0],
        "delta_pre": [-160.0, 160.0],
        "delta_ovl": [-120.0, 120.0],
    },
    "vv": {
        "delta_offset": [-220.0, 220.0],
        "delta_cons": [-200.0, 200.0],
        "delta_cutoff": [-260.0, 260.0],
        "delta_pre": [-120.0, 120.0],
        "delta_ovl": [-110.0, 110.0],
    },
    "vcv": {
        "delta_offset": [-200.0, 200.0],
        "delta_cons": [-200.0, 200.0],
        "delta_cutoff": [-260.0, 260.0],
        "delta_pre": [-180.0, 180.0],
        "delta_ovl": [-140.0, 140.0],
    },
}

JA_DELTA_CLIP_LIMITS = {
    "delta_offset": [-260.0, 260.0],
    "delta_cons": [-220.0, 220.0],
    "delta_cutoff": [-280.0, 280.0],
    "delta_pre": [-220.0, 220.0],
    "delta_ovl": [-160.0, 160.0],
}

_GENERIC_VOWELS = {
    "a", "i", "u", "e", "o", "ɯ", "ʌ", "ɛ", "ə", "æ", "ɑ", "ɐ", "ɔ", "ɪ", "ʊ", "ø", "œ",
}


def normalize_key(name: str) -> str:
    return normalize_wav_key(name)


def get_delta_clip_limits(language: str, alias_type: str = "") -> Dict[str, List[float]]:
    if str(language).strip().lower().startswith("ja") or str(language).strip().lower() == "japanese":
        return dict(JA_DELTA_CLIP_LIMITS)
    # alias_type별 세분화된 범위 사용
    a_type = str(alias_type or "").strip().lower()
    if a_type and a_type in KR_DELTA_CLIP_LIMITS_BY_TYPE:
        return dict(KR_DELTA_CLIP_LIMITS_BY_TYPE[a_type])
    return dict(KR_DELTA_CLIP_LIMITS)


def get_feature_schema() -> Dict[str, object]:
    return {
        "version": 1,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "target_names": list(TARGET_NAMES),
        "anchor_target_names": list(ANCHOR_TARGET_NAMES),
        "delta_target_names": list(DELTA_TARGET_NAMES),
        "aux_target_names": list(AUX_TARGET_NAMES),
    }


def _default_feature_cache_dir() -> str:
    configured = os.environ.get("UTOA_OTO_ML_CACHE_DIR", "").strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".cache", "oto_ml_features")
    os.makedirs(path, exist_ok=True)
    return path


def _default_training_row_cache_dir() -> str:
    configured = os.environ.get("UTOA_OTO_ML_ROW_CACHE_DIR", "").strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".cache", "oto_ml_training_rows")
    os.makedirs(path, exist_ok=True)
    return path


def _has_usable_oto_lines(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) <= 0:
            return False
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(64):
                line = f.readline()
                if not line:
                    break
                if "=" in line and "," in line:
                    return True
    except Exception:
        return False
    return False


def _infer_format_type_from_paths(language: str, *paths: str) -> str:
    lang = str(language or "").strip().lower()
    for path in paths:
        if not path:
            continue
        try:
            parts = [p for p in os.path.abspath(path).split(os.sep) if p]
        except Exception:
            continue
        for part in reversed(parts):
            fmt = normalize_format_type(lang, part)
            if fmt:
                return fmt
    return ""


def _normalize_auto_oto_policy(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raw = str(os.environ.get("UTOA_ML_AUTO_OTO_POLICY", "") or "").strip().lower()
    if raw in {"require", "required", "on", "true", "1"}:
        return "require"
    if raw in {"generate", "auto", "temp", "runtime", "generate-temp"}:
        return "generate-temp"
    if raw in {"persist", "save", "write", "generate-persist"}:
        return "generate-persist"
    return "require"


def _resolve_auto_oto_for_training(
    *,
    language: str,
    auto_oto_path: str,
    manual_oto_path: str,
    tg_dir: str,
    wav_dir: str,
    format_type_override: str,
    auto_oto_policy: str,
) -> Tuple[str, Dict[str, object]]:
    policy = _normalize_auto_oto_policy(auto_oto_policy)
    candidate_auto = auto_oto_path
    if not candidate_auto:
        if manual_oto_path:
            candidate_auto = os.path.join(os.path.dirname(manual_oto_path), "oto_auto_ml.ini")
        elif tg_dir:
            candidate_auto = os.path.join(tg_dir, "oto_auto_ml.ini")
        elif wav_dir:
            candidate_auto = os.path.join(wav_dir, "oto_auto_ml.ini")
    if _has_usable_oto_lines(candidate_auto):
        return candidate_auto, {"status": "existing", "policy": policy}
    if policy == "require":
        return (candidate_auto or auto_oto_path), {"status": "missing_auto_oto", "policy": policy}
    if not manual_oto_path or not os.path.isfile(manual_oto_path):
        return auto_oto_path, {"status": "missing_manual_oto", "policy": policy}
    if not tg_dir or not has_textgrid_files(tg_dir):
        return auto_oto_path, {"status": "missing_textgrid", "policy": policy}

    fmt = normalize_format_type(language, format_type_override)
    if not fmt:
        fmt = _infer_format_type_from_paths(language, manual_oto_path, tg_dir, wav_dir)
    if not fmt:
        fmt = "cvvc"

    temp_dir = None
    if policy == "generate-persist":
        work_dir = os.path.dirname(auto_oto_path) if auto_oto_path else ""
        if not work_dir:
            work_dir = os.path.dirname(manual_oto_path) if manual_oto_path else ""
        if not work_dir:
            work_dir = tg_dir or wav_dir
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="utoa_auto_oto_")
        work_dir = temp_dir.name

    try:
        from core.oto_ml_prepare_steps import _generate_auto_oto
        from core.oto_ml_prepare_types import PreparedAutoPair

        item = PreparedAutoPair(
            language=str(language or "").strip().lower() or "korean",
            format_type=str(fmt),
            stage_root=work_dir,
            work_dir=work_dir,
            manual_oto=manual_oto_path,
            tg_dir=tg_dir,
        )
        logs: List[str] = []
        _generate_auto_oto(item, logs)
        if _has_usable_oto_lines(item.auto_oto):
            return item.auto_oto, {
                "status": "generated",
                "policy": policy,
                "temp_dir": temp_dir,
                "log_count": len(logs),
            }
    except Exception as exc:
        logger.warning("auto oto generation failed: %s", exc)

    if temp_dir is not None:
        try:
            temp_dir.cleanup()
        except Exception:
            pass
    return auto_oto_path, {"status": "generate_failed", "policy": policy}


def _path_signature(path: str) -> Dict[str, object]:
    if not path:
        return {"path": "", "exists": False}
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        st = os.stat(abs_path)
        return {
            "path": abs_path,
            "exists": True,
            "kind": "file",
            "size": int(st.st_size),
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        }
    if os.path.isdir(abs_path):
        count = 0
        latest = 0
        total_size = 0
        for dp, dns, fns in os.walk(abs_path):
            for fn in fns:
                if not (fn.lower().endswith(".wav") or fn.lower().endswith(".textgrid") or fn.lower().endswith(".lab") or fn.lower().endswith(".txt")):
                    continue
                full = os.path.join(dp, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                count += 1
                total_size += int(st.st_size)
                latest = max(latest, int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))
        return {
            "path": abs_path,
            "exists": True,
            "kind": "dir",
            "count": count,
            "size": total_size,
            "mtime_ns": latest,
        }
    return {"path": abs_path, "exists": False}


def _feature_cache_path(language: str, oto_path: str, tg_dir: str, wav_dir: str, custom_phonemes_path: str = "", voicebank_id: str = "", prefix_map_path: str = "") -> str:
    key_obj = {
        "feature_version": FEATURE_VERSION,
        "language": str(language).strip().lower(),
        "oto": _path_signature(oto_path),
        "tg_dir": _path_signature(tg_dir),
        "wav_dir": _path_signature(wav_dir),
        "custom_phonemes": _path_signature(custom_phonemes_path),
        "prefix_map": _path_signature(prefix_map_path),
        "voicebank_id": str(voicebank_id or ""),
    }
    raw = json.dumps(key_obj, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()
    return os.path.join(_default_feature_cache_dir(), f"{digest}.json")


def _training_row_cache_path(language: str, auto_oto_path: str, manual_oto_path: str, tg_dir: str, wav_dir: str, custom_phonemes_path: str = "", voicebank_id: str = "", auto_prefix_map_path: str = "", manual_prefix_map_path: str = "") -> str:
    key_obj = {
        "feature_version": FEATURE_VERSION,
        "row_match_version": TRAIN_ROW_MATCH_VERSION,
        "language": str(language).strip().lower(),
        "auto_oto": _path_signature(auto_oto_path),
        "manual_oto": _path_signature(manual_oto_path),
        "tg_dir": _path_signature(tg_dir),
        "wav_dir": _path_signature(wav_dir),
        "custom_phonemes": _path_signature(custom_phonemes_path),
        "auto_prefix_map": _path_signature(auto_prefix_map_path),
        "manual_prefix_map": _path_signature(manual_prefix_map_path),
        "voicebank_id": str(voicebank_id or ""),
    }
    raw = json.dumps(key_obj, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()
    return os.path.join(_default_training_row_cache_dir(), f"{digest}.json")


def _load_feature_cache(path: str) -> Optional[List[Dict[str, object]]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("rows")
        if isinstance(rows, list):
            return rows
    except Exception:
        return None
    return None


def _save_feature_cache(path: str, rows: List[Dict[str, object]]) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"feature_version": FEATURE_VERSION, "rows": rows}, f, ensure_ascii=False)
    except Exception:
        logger.debug("Failed to save OTO ML feature cache: %s", path, exc_info=True)


def _load_training_row_cache(path: str) -> Optional[Tuple[List[Dict[str, object]], Dict[str, int]]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("rows")
        stats = payload.get("stats")
        if isinstance(rows, list) and isinstance(stats, dict):
            return rows, {str(k): int(v) for k, v in stats.items()}
    except Exception:
        return None
    return None


def _save_training_row_cache(path: str, rows: List[Dict[str, object]], stats: Dict[str, int]) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"feature_version": FEATURE_VERSION, "rows": rows, "stats": stats},
                f,
                ensure_ascii=False,
            )
    except Exception:
        logger.debug("Failed to save OTO ML training-row cache: %s", path, exc_info=True)


def write_feature_schema(path: str) -> str:
    schema = get_feature_schema()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    return path


def canonicalize_feature_row(row: Dict[str, object], feature_names: Optional[List[str]] = None) -> Dict[str, object]:
    names = feature_names or FEATURE_NAMES
    out = {}
    for name in names:
        val = row.get(name, FEATURE_DEFAULTS.get(name, 0.0))
        if name in CATEGORICAL_FEATURES:
            out[name] = "" if val is None else str(val)
        else:
            try:
                out[name] = float(val)
            except Exception:
                out[name] = 0.0
    return out


def _read_text_with_fallback(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "cp932", "euc-kr", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _classify_for_match(language: str, alias: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    try:
        return classify_alias_type(language or "korean", alias, custom_map=custom_map)
    except Exception:
        return ""


def _is_matchable_alias_type(alias_type: str) -> bool:
    return alias_type in {"br", "cv", "cv_head", "vc", "vv", "vcv", "mono"}


def _strip_known_pitch_suffix(text: str) -> str:
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:#|b)?[0-8])$", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:sharp|flat)?[0-8])$", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _strip_bracket_suffix(text: str) -> str:
    return re.sub(r"\s*[\(\[\{（【].*?[\)\]\}）】]\s*$", "", text).strip()


def _try_strip_trailing_separator_suffix(text: str, language: str, custom_map: Optional[Dict[str, str]], base_type: str) -> str:
    for sep in ("_", " "):
        if sep not in text:
            continue
        parts = text.rsplit(sep, 1)
        if len(parts) != 2:
            continue
        prefix, suffix = parts[0].strip(), parts[1].strip()
        if not prefix or not suffix:
            continue
        prefix_type = _classify_for_match(language, prefix, custom_map=custom_map)
        if prefix_type == base_type and _is_matchable_alias_type(prefix_type):
            return prefix
    return text


def _try_strip_attached_suffix(text: str, language: str, custom_map: Optional[Dict[str, str]], base_type: str) -> str:
    m = re.search(r"([A-Za-z가-힣ぁ-んァ-ヶ一-龯]+)$", text)
    if not m:
        return text
    suffix = m.group(1)
    if not re.search(r"[가-힣ぁ-んァ-ヶ一-龯]", suffix):
        return text
    if len(suffix) <= 1 and not re.search(r"[一-龯]", suffix):
        return text
    run_start = len(text) - len(suffix)
    first_suffix_idx = next(
        (idx for idx, ch in enumerate(suffix) if re.match(r"[가-힣ぁ-んァ-ヶ一-龯]", ch)),
        None,
    )
    if first_suffix_idx is None:
        return text
    for cut in range(run_start + first_suffix_idx, len(text)):
        prefix = text[:cut].rstrip(" _-")
        if not prefix:
            continue
        prefix_type = _classify_for_match(language, prefix, custom_map=custom_map)
        if prefix_type == base_type and _is_matchable_alias_type(prefix_type):
            return prefix
    return text


def _normalize_alias_for_match(alias: str, language: str = "", custom_map: Optional[Dict[str, str]] = None) -> str:
    return canonicalize_alias_for_matching(language or "korean", alias, custom_map=custom_map)


def parse_oto_rows(path: str, language: str = "", custom_map: Optional[Dict[str, str]] = None, prefix_map_path: str = "", prefix_context_paths: Optional[List[str]] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path or not os.path.exists(path):
        return rows
    text = _read_text_with_fallback(path)
    context_paths = list(prefix_context_paths or [])
    source_oto_id = _stable_source_oto_id(path)
    for line_index, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or "=" not in line or "," not in line:
            continue
        left, right = line.split("=", 1)
        parts = [p.strip() for p in right.split(",")]
        if len(parts) < 6:
            continue
        alias_raw = parts[0]
        if not str(alias_raw or "").strip():
            alias_raw = os.path.splitext(os.path.basename(left.strip()))[0]
        alias = strip_prefix_map_affixes(
            alias_raw,
            prefix_map_path=prefix_map_path,
            wav_name=left.strip(),
            context_paths=context_paths,
        )
        if str(language or "").strip().lower() == "korean" and should_ignore_korean_alias(alias):
            continue
        try:
            row = {
                "line_index": line_index,
                "source_oto_id": source_oto_id,
                "source_row_id": f"{source_oto_id}:{line_index}",
                "raw_line": raw,
                "wav": left.strip(),
                "alias_raw": alias_raw,
                "alias": alias,
                "offset": float(parts[1]),
                "cons": float(parts[2]),
                "cutoff": float(parts[3]),
                "pre": float(parts[4]),
                "ovl": float(parts[5]),
            }
        except Exception:
            continue
        row["wav_norm"] = normalize_key(str(row["wav"]))
        row["alias_norm"] = _normalize_alias_for_match(str(row["alias"]), language=language, custom_map=custom_map)
        rows.append(row)
    return attach_occurrence_indices(rows)


def attach_occurrence_indices(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        base = (str(row.get("wav_norm", "")), str(row.get("alias_norm", "")))
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        row["occurrence_index"] = idx
    return rows


def build_occurrence_map(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str, int], Dict[str, object]]:
    return {
        (str(r.get("wav_norm", "")), str(r.get("alias_norm", "")), int(r.get("occurrence_index", 0))): r
        for r in rows
    }


def infer_voicebank_id(auto_oto_path: str, manual_oto_path: str = "", tg_dir: str = "", wav_dir: str = "", voicebank_id: str = "") -> str:
    if voicebank_id:
        return str(voicebank_id).strip()
    for candidate in (wav_dir, tg_dir, auto_oto_path, manual_oto_path):
        if not candidate:
            continue
        p = candidate if os.path.isdir(candidate) else os.path.dirname(os.path.abspath(candidate))
        base = os.path.basename(os.path.abspath(p))
        if base:
            return base
    return "unknown_voicebank"


def load_custom_map(custom_phonemes_path: str = "") -> Dict[str, str]:
    if not custom_phonemes_path:
        return {}
    try:
        return load_custom_phonemes(custom_phonemes_path) or {}
    except Exception:
        return {}


def classify_alias_type(language: str, alias: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    lang = str(language).strip().lower()
    if lang == "japanese":
        from core.ja_oto_mapping import classify_ja_alias
        return str(classify_ja_alias(alias, custom_map))
    from core.oto_generator import classify_alias
    return str(classify_alias(alias, custom_map))


def detect_format_type(language: str, aliases: List[str], custom_map: Optional[Dict[str, str]] = None) -> str:
    lang = str(language).strip().lower()
    if lang == "japanese":
        from core.ja_oto_mapping import detect_ja_alias_format
        return normalize_format_type(lang, str(detect_ja_alias_format(aliases, custom_map=custom_map)))
    from core.oto_generator import detect_alias_format
    return normalize_format_type(lang, str(detect_alias_format(aliases, custom_map=custom_map)))


def _clean_mark(mark: str) -> str:
    return re.sub(r"[0-9]", "", (mark or "").strip().lower())


def _is_vowel_mark(language: str, mark: str) -> bool:
    clean = _clean_mark(mark)
    if clean in _GENERIC_VOWELS:
        return True
    if language == "japanese":
        return clean in {"a", "i", "u", "e", "o", "ɯ", "ɴ"}
    return clean in _GENERIC_VOWELS


def _build_tg_index(tg_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not tg_dir or not os.path.isdir(tg_dir):
        return index
    for dp, _dns, fns in os.walk(tg_dir):
        for fn in fns:
            if fn.lower().endswith(".textgrid"):
                index[normalize_key(fn)] = os.path.join(dp, fn)
    return index


def _load_textgrid_tiers(tg_path: str) -> Tuple[List[object], List[object]]:
    phones: List[object] = []
    words: List[object] = []
    if not tg_path or not os.path.exists(tg_path):
        return phones, words
    tg = textgrid.TextGrid.fromFile(tg_path)
    phone_tier = None
    word_tier = None
    for tier in tg:
        name = str(getattr(tier, "name", "")).strip().lower()
        if name == "phones":
            phone_tier = tier
        elif name == "words":
            word_tier = tier
    if phone_tier is None:
        for tier in tg:
            name = str(getattr(tier, "name", "")).strip().lower()
            if "phone" in name:
                phone_tier = tier
                break
    if word_tier is None:
        for tier in tg:
            name = str(getattr(tier, "name", "")).strip().lower()
            if "word" in name:
                word_tier = tier
                break
    if phone_tier is not None:
        phones = [iv for iv in phone_tier if str(getattr(iv, "mark", "")).strip()]
    if word_tier is not None:
        words = [iv for iv in word_tier if str(getattr(iv, "mark", "")).strip()]
    return phones, words


def _find_interval_index(intervals: List[object], t_ms: float) -> int:
    if not intervals:
        return -1
    t_sec = float(t_ms) / 1000.0
    for idx, iv in enumerate(intervals):
        if float(iv.minTime) <= t_sec <= float(iv.maxTime):
            return idx
    best_idx = -1
    best_dist = float("inf")
    for idx, iv in enumerate(intervals):
        mid = (float(iv.minTime) + float(iv.maxTime)) * 500.0
        dist = abs(mid - t_ms)
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def _nearest_interval(intervals: List[object], t_ms: float, predicate=None) -> Optional[object]:
    if not intervals:
        return None
    pool = intervals if predicate is None else [iv for iv in intervals if predicate(iv)]
    if not pool:
        return None
    idx = _find_interval_index(pool, t_ms)
    if idx < 0:
        return None
    return pool[idx]


def _interval_bounds_ms(interval) -> Tuple[float, float]:
    return float(interval.minTime) * 1000.0, float(interval.maxTime) * 1000.0


def _window_mean(arr) -> float:
    if np is None or arr is None or len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def _window_min(arr) -> float:
    if np is None or arr is None or len(arr) == 0:
        return 0.0
    return float(np.min(arr))


def _window_max(arr) -> float:
    if np is None or arr is None or len(arr) == 0:
        return 0.0
    return float(np.max(arr))


def _select_mask(times_ms, start_ms: float, end_ms: float):
    if np is None or times_ms is None or len(times_ms) == 0:
        return np.array([], dtype=np.int64) if np is not None else []
    return np.where((times_ms >= start_ms) & (times_ms <= end_ms))[0]


def _compute_segment_stats(mel_ctx, offset_ms: float, pre_abs: float, cut_abs: float, audio, sr) -> Dict[str, float]:
    stats = {
        "energy_mean": 0.0, "energy_min": 0.0, "energy_max": 0.0, "energy_slope_pre": 0.0,
        "energy_slope_post": 0.0, "valley_energy": 0.0, "valley_dist_from_cutoff_ms": 0.0,
        "db_mean": 0.0, "db_min": 0.0, "db_silence_ratio": 0.0, "f0_voicing_mean": 0.0,
        "f0_voicing_near_pre": 0.0, "f0_valid_ratio": 0.0, "f0_gap_ratio": 0.0,
        "f0_continuity": 0.0, "zcr_mean": 0.0, "spectral_flux_mean": 0.0,
        "mel_voiced_formant_ratio": 0.0,
        "mel_silence_sparse_ratio": 0.0,
        "mel_unvoiced_diffuse_ratio": 0.0,
        "mel_breath_like_ratio": 0.0,
        "blank_span_confidence": 0.0,
        "mel_offset_candidate_ms": float(max(0.0, offset_ms)),
        "mel_cutoff_candidate_ms": float(max(cut_abs, pre_abs + 12.0)),
        "mel_voiced_onset_ms": 0.0,
        "mel_vowel_nucleus_start_ms": 0.0,
        "mel_vowel_nucleus_end_ms": 0.0,
        "mel_voiced_onset_to_pre_ms": 0.0,
        "mel_safety_clamped": 0.0,
        "onset_patch_energy_mean": 0.0,
        "onset_patch_voiced_ratio": 0.0,
        "onset_patch_unvoiced_ratio": 0.0,
        "tail_patch_energy_mean": 0.0,
        "tail_patch_silence_ratio": 0.0,
    }
    if np is None or not mel_ctx:
        return stats
    times_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    db_arr = mel_ctx.get("db_db")
    f0_arr = mel_ctx.get("f0_voicing")
    cls_voiced = mel_ctx.get("cls_voiced_formant")
    cls_silence = mel_ctx.get("cls_silence_sparse")
    cls_unvoiced = mel_ctx.get("cls_unvoiced_diffuse")
    cls_breath = mel_ctx.get("cls_breath_like")
    if times_ms is None or en is None or len(times_ms) == 0:
        return stats

    def _ratio(arr, idxs):
        if arr is None or len(arr) != len(en) or len(idxs) <= 0:
            return 0.0
        return float(np.mean(np.clip(np.asarray(arr)[idxs], 0.0, 1.0)))

    seg_mask = _select_mask(times_ms, max(offset_ms, pre_abs - 40.0), max(pre_abs + 40.0, cut_abs))
    if len(seg_mask) > 0:
        local_e = en[seg_mask]
        stats["energy_mean"] = _window_mean(local_e)
        stats["energy_min"] = _window_min(local_e)
        stats["energy_max"] = _window_max(local_e)
        valley_idx = int(seg_mask[int(np.argmin(local_e))])
        stats["valley_energy"] = float(en[valley_idx])
        stats["valley_dist_from_cutoff_ms"] = float(cut_abs - float(times_ms[valley_idx]))
        if db_arr is not None and len(db_arr) == len(en):
            local_db = db_arr[seg_mask]
            stats["db_mean"] = _window_mean(local_db)
            stats["db_min"] = _window_min(local_db)
            silence_th = float(mel_ctx.get("db_silence_th", -42.0))
            stats["db_silence_ratio"] = float(np.mean(local_db <= silence_th))
        if f0_arr is not None and len(f0_arr) == len(en):
            local_f0 = np.clip(np.asarray(f0_arr)[seg_mask], 0.0, 1.0)
            stats["f0_voicing_mean"] = _window_mean(local_f0)
            valid_th = float(os.environ.get("UTOA_ML_F0_VALID_TH", "0.50") or 0.50)
            valid_mask = local_f0 >= valid_th
            stats["f0_valid_ratio"] = float(np.mean(valid_mask)) if len(valid_mask) else 0.0
            stats["f0_gap_ratio"] = float(1.0 - stats["f0_valid_ratio"])
            cont_scope_mask = np.ones((len(seg_mask),), dtype=bool)
            if cls_unvoiced is not None and len(cls_unvoiced) == len(en):
                cont_scope_mask &= (np.asarray(cls_unvoiced)[seg_mask] < 0.5)
            if cls_breath is not None and len(cls_breath) == len(en):
                cont_scope_mask &= (np.asarray(cls_breath)[seg_mask] < 0.5)
            scoped_valid = valid_mask[cont_scope_mask]
            if len(scoped_valid) >= 2:
                transitions = float(np.sum(scoped_valid[1:] != scoped_valid[:-1]))
                trans_ratio = transitions / float(max(1, len(scoped_valid) - 1))
                cont = max(0.0, 1.0 - trans_ratio)
                cont *= float(np.mean(scoped_valid))
                stats["f0_continuity"] = max(0.0, min(1.0, cont))
            elif len(scoped_valid) == 1:
                stats["f0_continuity"] = float(scoped_valid[0])
        stats["mel_voiced_formant_ratio"] = _ratio(cls_voiced, seg_mask)
        stats["mel_silence_sparse_ratio"] = _ratio(cls_silence, seg_mask)
        stats["mel_unvoiced_diffuse_ratio"] = _ratio(cls_unvoiced, seg_mask)
        stats["mel_breath_like_ratio"] = _ratio(cls_breath, seg_mask)

    pre_mask = _select_mask(times_ms, pre_abs - 30.0, pre_abs + 10.0)
    post_mask = _select_mask(times_ms, pre_abs + 10.0, pre_abs + 60.0)
    if len(pre_mask) >= 2:
        stats["energy_slope_pre"] = float(en[pre_mask[-1]] - en[pre_mask[0]])
    if len(post_mask) >= 2:
        stats["energy_slope_post"] = float(en[post_mask[-1]] - en[post_mask[0]])
    if len(pre_mask) > 0 and f0_arr is not None and len(f0_arr) == len(en):
        stats["f0_voicing_near_pre"] = _window_mean(f0_arr[pre_mask])
    if len(seg_mask) >= 2:
        stats["spectral_flux_mean"] = float(np.mean(np.abs(np.diff(en[seg_mask]))))

    onset_mask = _select_mask(times_ms, pre_abs - 60.0, pre_abs + 40.0)
    if len(onset_mask) > 0:
        stats["onset_patch_energy_mean"] = _window_mean(en[onset_mask])
        stats["onset_patch_voiced_ratio"] = _ratio(cls_voiced, onset_mask)
        stats["onset_patch_unvoiced_ratio"] = _ratio(cls_unvoiced, onset_mask)

    tail_mask = _select_mask(times_ms, cut_abs - 80.0, cut_abs + 40.0)
    if len(tail_mask) > 0:
        stats["tail_patch_energy_mean"] = _window_mean(en[tail_mask])
        stats["tail_patch_silence_ratio"] = _ratio(cls_silence, tail_mask)

    blank_conf = (
        (0.65 * float(stats["mel_silence_sparse_ratio"]))
        + (0.20 * float(stats["mel_breath_like_ratio"]))
        + max(0.0, float(stats["db_silence_ratio"]) - float(stats["mel_unvoiced_diffuse_ratio"])) * 0.20
        - (0.45 * float(stats["mel_voiced_formant_ratio"]))
    )
    stats["blank_span_confidence"] = max(0.0, min(1.0, float(blank_conf)))

    # Candidate boundaries inferred from class transitions.
    # offset candidate: first strong non-silence frame near pre region
    # cutoff candidate: last silence-sparse frame before current cutoff
    onset_probe_mask = _select_mask(times_ms, max(0.0, offset_ms - 80.0), pre_abs + 40.0)
    if len(onset_probe_mask) > 0:
        onset_idx = None
        for idx in onset_probe_mask:
            voiced_v = float(cls_voiced[idx]) if cls_voiced is not None and len(cls_voiced) == len(en) else 0.0
            unvoiced_v = float(cls_unvoiced[idx]) if cls_unvoiced is not None and len(cls_unvoiced) == len(en) else 0.0
            silence_v = float(cls_silence[idx]) if cls_silence is not None and len(cls_silence) == len(en) else 0.0
            if (voiced_v >= 0.5 or unvoiced_v >= 0.5) and silence_v < 0.5:
                onset_idx = int(idx)
                break
        if onset_idx is not None:
            cand = float(times_ms[onset_idx]) - 12.0
            stats["mel_offset_candidate_ms"] = max(0.0, min(pre_abs - 8.0, cand))

    cutoff_probe_mask = _select_mask(times_ms, pre_abs + 20.0, cut_abs + 120.0)
    if len(cutoff_probe_mask) > 0:
        last_sil_idx = None
        for idx in cutoff_probe_mask:
            silence_v = float(cls_silence[idx]) if cls_silence is not None and len(cls_silence) == len(en) else 0.0
            if silence_v >= 0.5:
                last_sil_idx = int(idx)
        if last_sil_idx is not None:
            cand_cut = float(times_ms[last_sil_idx]) + 4.0
            stats["mel_cutoff_candidate_ms"] = max(pre_abs + 12.0, min(cut_abs, cand_cut))

    try:
        from core.oto_generator import _estimate_mel_voiced_onset, _estimate_mel_vowel_nucleus

        mel_onset = _estimate_mel_voiced_onset(mel_ctx, pre_abs)
        if mel_onset is not None:
            stats["mel_voiced_onset_ms"] = float(mel_onset)
            stats["mel_voiced_onset_to_pre_ms"] = float(pre_abs - float(mel_onset))
            nuc_start, nuc_end = _estimate_mel_vowel_nucleus(mel_ctx, mel_onset)
            if nuc_start is not None:
                stats["mel_vowel_nucleus_start_ms"] = float(nuc_start)
            if nuc_end is not None:
                stats["mel_vowel_nucleus_end_ms"] = float(nuc_end)
    except Exception:
        pass

    if audio is not None and sr and sr > 0:
        start_s = int(max(offset_ms, pre_abs - 20.0) * sr / 1000.0)
        end_s = int(max(start_s + 1, min(cut_abs, pre_abs + 60.0) * sr / 1000.0))
        end_s = min(end_s, len(audio))
        start_s = max(0, min(start_s, len(audio)))
        segment = audio[start_s:end_s]
        if len(segment) >= 2:
            signs = np.signbit(segment)
            stats["zcr_mean"] = float(np.mean(signs[1:] != signs[:-1]))

    return stats


def _safe_ratio(a: float, b: float) -> float:
    if abs(float(b)) < 1e-9:
        return 0.0
    return float(a) / float(b)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _compute_audio_rms(audio) -> float:
    if np is None or audio is None:
        return 0.0
    try:
        arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    except Exception:
        return 0.0
    if arr.size <= 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _compute_segment_rms(audio, sr: int, start_ms: float, end_ms: float) -> float:
    if np is None or audio is None or not sr or sr <= 0:
        return 0.0
    try:
        arr = np.asarray(audio, dtype=np.float64).reshape(-1)
    except Exception:
        return 0.0
    if arr.size <= 0:
        return 0.0
    s = int(max(0.0, float(start_ms)) * float(sr) / 1000.0)
    e = int(max(float(start_ms), float(end_ms)) * float(sr) / 1000.0)
    s = max(0, min(s, int(arr.size)))
    e = max(s + 1, min(e, int(arr.size)))
    seg = arr[s:e]
    if seg.size <= 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(seg))))


def _lazy_wav_helpers():
    from core.oto_generator import _find_wav_path_for_name, _mel_envelope, _read_wav_mono_np
    return _find_wav_path_for_name, _mel_envelope, _read_wav_mono_np


def _extract_kr_structure(alias: str, alias_type: str) -> Dict[str, object]:
    from core.oto_generator import (
        KR_PLOSIVE_ONSETS, KR_SIBILANT_ONSETS, KR_SONORANT_CONSONANTS,
        KR_TENSE_CONSONANTS, KR_VOICED_ONSETS, KR_VOICELESS_ONSETS,
        _extract_alias_onset, _extract_kr_cv_alias_token, _split_kr_syllable_parts, is_diphthong,
    )

    token = _extract_kr_cv_alias_token(alias)
    onset, vowel, coda = _split_kr_syllable_parts(token)
    onset = (onset or _extract_alias_onset(alias) or "").lower()
    onset_class = "other"
    if onset in KR_PLOSIVE_ONSETS:
        onset_class = "plosive"
    elif onset in KR_SIBILANT_ONSETS:
        onset_class = "sibilant"
    elif onset in KR_SONORANT_CONSONANTS:
        onset_class = "sonorant"

    voicing_class = "unknown"
    if onset in KR_VOICED_ONSETS:
        voicing_class = "voiced"
    elif onset in KR_VOICELESS_ONSETS:
        voicing_class = "voiceless"

    coda = (coda or "").lower()
    if not coda:
        coda_type = "none"
    elif coda in {"n", "m", "ng"}:
        coda_type = "nasal"
    elif coda in {"r", "l"}:
        coda_type = "liquid"
    elif coda in {"k", "t", "p"}:
        coda_type = "stop"
    else:
        coda_type = "other"

    return {
        "onset_class": onset_class,
        "voicing_class": voicing_class,
        "is_tense": 1.0 if onset in KR_TENSE_CONSONANTS else 0.0,
        "is_diphthong": 1.0 if is_diphthong(alias) else 0.0,
        "coda_type": coda_type,
        "vowel_class": vowel or "",
        "bridge_type": alias_type if alias_type in {"vc", "vcv", "vv"} else "",
        "is_nasal_or_sonorant": 1.0 if onset in KR_SONORANT_CONSONANTS else 0.0,
    }


def _extract_ja_structure(alias: str, alias_type: str) -> Dict[str, object]:
    from core.ja_oto_mapping import (
        _extract_ja_cv_target_syllable, _extract_ja_onset_token,
        _extract_vcv_target_syllable, split_ja_romaji_syllable,
    )

    target = _extract_ja_cv_target_syllable(alias)
    if alias_type == "vcv":
        target = _extract_vcv_target_syllable(alias) or target
    onset = (_extract_ja_onset_token(target or alias) or "").lower()
    _, vowel = split_ja_romaji_syllable((target or "").lower())

    onset_class = "other"
    if onset in {"k", "g", "t", "d", "p", "b", "ch", "ts", "dz", "ky", "gy", "py", "by"}:
        onset_class = "plosive"
    elif onset in {"s", "z", "sh", "h", "f", "hy"}:
        onset_class = "fricative"
    elif onset in {"m", "n", "ny", "r", "l", "w", "y"}:
        onset_class = "sonorant"
    elif onset in {"y", "w"}:
        onset_class = "glide"

    voicing_class = "unknown"
    if onset in {"g", "z", "d", "b", "j", "m", "n", "ny", "r", "l", "w", "y"}:
        voicing_class = "voiced"
    elif onset in {"k", "s", "sh", "t", "h", "f", "p", "ch", "ts"}:
        voicing_class = "voiceless"

    return {
        "onset_class": onset_class,
        "voicing_class": voicing_class,
        "is_tense": 1.0 if onset in {"kk", "ss", "tt", "pp"} else 0.0,
        "is_diphthong": 0.0,
        "coda_type": "none",
        "vowel_class": vowel or "",
        "bridge_type": alias_type if alias_type in {"vc", "vcv", "vv"} else "",
        "is_nasal_or_sonorant": 1.0 if onset in {"m", "n", "ny", "r", "l", "w", "y"} else 0.0,
    }


def _mora_position(row_index: int, total_rows: int) -> str:
    if total_rows <= 1 or row_index <= 0:
        return "head"
    if row_index >= (total_rows - 1):
        return "tail"
    return "mid"


def _extract_structure_features(language: str, alias: str, alias_type: str, row_index: int, total_rows: int) -> Dict[str, object]:
    if language == "japanese":
        out = _extract_ja_structure(alias, alias_type)
    else:
        out = _extract_kr_structure(alias, alias_type)
    out["mora_position"] = _mora_position(row_index, total_rows)
    return out


def _classify_kr_breath_alias_kind(alias: str, alias_type: str = "") -> str:
    """
    한국어 숨소리 계열을 분리한다.
    - tail_breath: 어미 끝숨 (예: "a R", "oH")
    - standalone_breath: 독립 숨소리 (예: "br", "bre")
    """
    text = str(alias or "").strip()
    if not text:
        return ""

    parts = [p for p in re.split(r"\s+", text) if p]
    if len(parts) >= 2 and parts[0].lower() in _GENERIC_VOWELS and parts[-1] in {"R", "H"}:
        return "tail_breath"

    compact = re.sub(r"[\s_\-]+", "", text)
    if len(compact) >= 2 and compact[-1] in {"R", "H"} and compact[:-1].lower() in _GENERIC_VOWELS:
        return "tail_breath"

    low = text.lower()
    if re.fullmatch(r"br\d*", low) or low in {"br", "bre", "breath"}:
        return "standalone_breath"
    if str(alias_type or "").strip().lower() == "br":
        return "standalone_breath"
    return ""


def _derive_alias_group(language: str, feat: Dict[str, object], alias: str = "") -> str:
    lang = str(language or "").strip().lower()
    alias_type = str(feat.get("alias_type", "") or "").strip().lower()
    coda_type = str(feat.get("coda_type", "") or "").strip().lower()
    is_diph = float(feat.get("is_diphthong", 0.0) or 0.0) >= 0.5
    onset_class = str(feat.get("onset_class", "") or "").strip().lower()

    if lang == "korean":
        if alias_type == "cv":
            return "cv_glide" if is_diph else "cv"
        if alias_type == "cv_head":
            return "cv_head_glide" if is_diph else "cv_head"
        if alias_type == "vc":
            breath_kind = _classify_kr_breath_alias_kind(alias, alias_type=alias_type)
            if breath_kind == "tail_breath":
                return "vc_tail_breath"
            if coda_type == "stop":
                return "vc_stop"
            if coda_type in {"nasal", "liquid"}:
                return "vc_sonorant"
            return "vc_other"
        if alias_type == "vv":
            return "vv"
        if alias_type == "vcv":
            return "vcv"
        if alias_type == "mono":
            return "mono"
        if alias_type == "br":
            return "br_standalone"
        return alias_type or "other"

    if alias_type == "vc":
        return f"vc_{onset_class or 'other'}"
    if alias_type == "cv":
        return f"cv_{onset_class or 'other'}"
    if alias_type == "cv_head":
        return f"cv_head_{onset_class or 'other'}"
    if alias_type in {"vv", "vcv", "mono", "br"}:
        return alias_type
    return alias_type or "other"


def _feature_row_from_context(language: str, format_type: str, row: Dict[str, object], row_index: int, total_rows: int, alias_type: str, phones: List[object], words: List[object], mel_ctx, audio, sr: int, prev_row: Optional[Dict[str, object]], next_row: Optional[Dict[str, object]], file_stats: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    feat = dict(FEATURE_DEFAULTS)
    feat["language"] = language
    feat["format_type"] = format_type
    feat["alias_type"] = alias_type
    feat["row_index_in_wav"] = float(row_index)
    feat["row_ratio_in_wav"] = _safe_ratio(row_index, max(total_rows - 1, 1))
    if file_stats:
        feat["file_row_count"] = float(file_stats.get("row_count", 0.0) or 0.0)
        feat["file_cv_count"] = float(file_stats.get("cv_count", 0.0) or 0.0)
        feat["file_vc_count"] = float(file_stats.get("vc_count", 0.0) or 0.0)
        feat["file_vv_count"] = float(file_stats.get("vv_count", 0.0) or 0.0)
        feat["file_vcv_count"] = float(file_stats.get("vcv_count", 0.0) or 0.0)
        feat["file_br_count"] = float(file_stats.get("br_count", 0.0) or 0.0)
        feat["file_mono_count"] = float(file_stats.get("mono_count", 0.0) or 0.0)
        feat["file_cv_ratio"] = float(file_stats.get("cv_ratio", 0.0) or 0.0)
        feat["file_vc_ratio"] = float(file_stats.get("vc_ratio", 0.0) or 0.0)
        feat["file_vc_cv_ratio"] = float(file_stats.get("vc_cv_ratio", 0.0) or 0.0)
        feat["file_cv_vc_balance"] = float(file_stats.get("cv_vc_balance", 0.0) or 0.0)
        feat["file_mean_energy"] = float(file_stats.get("file_mean_energy", 0.0) or 0.0)
        feat["file_voiced_ratio"] = float(file_stats.get("file_voiced_ratio", 0.0) or 0.0)
        feat["file_mean_syllable_dur_ms"] = float(file_stats.get("file_mean_syllable_dur_ms", 0.0) or 0.0)
    feat["is_head_row"] = 1.0 if row_index == 0 else 0.0
    feat["is_tail_row"] = 1.0 if row_index == (total_rows - 1) else 0.0
    if audio is not None and sr and sr > 0:
        feat["wav_duration_ms"] = (len(audio) * 1000.0) / float(sr)

    offset = float(row.get("offset", 0.0))
    cons = float(row.get("cons", 0.0))
    cutoff = float(row.get("cutoff", 0.0))
    pre = float(row.get("pre", 0.0))
    ovl = float(row.get("ovl", 0.0))
    cut_abs = offset + abs(cutoff)
    pre_abs = offset + pre

    feat["base_offset"] = offset
    feat["base_cons"] = cons
    feat["base_cutoff_abs"] = cut_abs
    feat["base_pre"] = pre
    feat["base_ovl"] = ovl
    feat["base_cons_gap"] = max(cons - pre, 0.0)
    feat["base_cut_gap"] = max(abs(cutoff) - cons, 0.0)
    feat["base_ovl_ratio"] = _safe_ratio(ovl, max(pre, 1e-6))

    phone_idx = _find_interval_index(phones, pre_abs)
    curr_phone = phones[phone_idx] if 0 <= phone_idx < len(phones) else None
    curr_vowel = _nearest_interval(phones, pre_abs, predicate=lambda iv: _is_vowel_mark(language, getattr(iv, "mark", "")))
    curr_word = _nearest_interval(words, pre_abs) if words else None

    if curr_phone is not None:
        p_start, p_end = _interval_bounds_ms(curr_phone)
        feat["curr_phone_start_ms"] = p_start
        feat["curr_phone_end_ms"] = p_end
        feat["curr_phone_len_ms"] = max(p_end - p_start, 0.0)
        if phone_idx > 0:
            feat["prev_phone_gap_ms"] = max(p_start - (float(phones[phone_idx - 1].maxTime) * 1000.0), 0.0)
        if phone_idx + 1 < len(phones):
            feat["next_phone_gap_ms"] = max((float(phones[phone_idx + 1].minTime) * 1000.0) - p_end, 0.0)
    if curr_vowel is not None:
        v_start, v_end = _interval_bounds_ms(curr_vowel)
        feat["curr_vowel_start_ms"] = v_start
        feat["curr_vowel_end_ms"] = v_end
        feat["curr_vowel_len_ms"] = max(v_end - v_start, 0.0)
        feat["expected_anchor_ms"] = v_start
    elif curr_phone is not None:
        feat["expected_anchor_ms"] = feat["curr_phone_start_ms"]
    else:
        feat["expected_anchor_ms"] = pre_abs

    if curr_word is not None:
        w_start, w_end = _interval_bounds_ms(curr_word)
        feat["syllable_start_ms"] = w_start
        feat["syllable_end_ms"] = w_end
        feat["syllable_len_ms"] = max(w_end - w_start, 0.0)
    elif curr_vowel is not None:
        feat["syllable_start_ms"] = feat["curr_vowel_start_ms"]
        feat["syllable_end_ms"] = feat["curr_vowel_end_ms"]
        feat["syllable_len_ms"] = feat["curr_vowel_len_ms"]

    if mel_ctx:
        try:
            from core.oto_generator import (
                _estimate_kr_blank_confidence_at_time,
                _estimate_kr_mel_class_scores_at_time,
            )
            t_ms = float(feat.get("syllable_start_ms", 0.0) or 0.0)
            if t_ms <= 0.0:
                t_ms = float(feat.get("expected_anchor_ms", 0.0) or 0.0)
            if t_ms <= 0.0:
                t_ms = float(pre_abs)
            feat["syllable_blank_confidence"] = _estimate_kr_blank_confidence_at_time(mel_ctx, t_ms)
            mel_scores = _estimate_kr_mel_class_scores_at_time(mel_ctx, t_ms)
            feat["syllable_mel_voiced_conf"] = float(mel_scores.get("mel_voiced_formant_conf", 0.0) or 0.0)
            feat["syllable_mel_silence_conf"] = float(mel_scores.get("mel_silence_sparse_conf", 0.0) or 0.0)
            feat["syllable_mel_unvoiced_conf"] = float(mel_scores.get("mel_unvoiced_diffuse_conf", 0.0) or 0.0)
            feat["syllable_mel_breath_conf"] = float(mel_scores.get("mel_breath_like_conf", 0.0) or 0.0)
        except Exception:
            pass

    feat["base_offset_to_expected_ms"] = offset - feat["expected_anchor_ms"]
    feat["base_pre_to_expected_ms"] = pre_abs - feat["expected_anchor_ms"]

    next_anchor_ms = 0.0
    if phone_idx >= 0 and phone_idx + 1 < len(phones):
        next_anchor_ms = float(phones[phone_idx + 1].minTime) * 1000.0
    elif next_row is not None:
        next_anchor_ms = float(next_row.get("offset", 0.0)) + float(next_row.get("pre", 0.0))
    feat["base_cutoff_to_next_anchor_ms"] = next_anchor_ms - cut_abs if next_anchor_ms else 0.0

    feat.update(_compute_segment_stats(mel_ctx, offset, pre_abs, cut_abs, audio, sr))
    seg_rms = _compute_segment_rms(audio, sr, max(0.0, offset), max(cut_abs, pre_abs + 12.0))
    wav_rms = float(file_stats.get("wav_rms_linear", 0.0) or 0.0) if file_stats else 0.0
    vb_rms = float(file_stats.get("voicebank_rms_linear", 0.0) or 0.0) if file_stats else 0.0
    feat["rms_norm_wav"] = _safe_ratio(seg_rms, max(wav_rms, 1e-8)) if wav_rms > 0.0 else 0.0
    feat["rms_norm_vb"] = _safe_ratio(seg_rms, max(vb_rms, 1e-8)) if vb_rms > 0.0 else feat["rms_norm_wav"]
    mel_safety = 0.0
    if alias_type in {"cv", "cv_head", "vcv"}:
        blank_conf = float(feat.get("blank_span_confidence", 0.0) or 0.0)
        cand_off = float(feat.get("mel_offset_candidate_ms", 0.0) or 0.0)
        if blank_conf >= 0.72 and offset < cand_off - 12.0:
            mel_safety = 1.0
    if alias_type in {"vc", "vv"}:
        tail_sil = float(feat.get("tail_patch_silence_ratio", 0.0) or 0.0)
        cand_cut = float(feat.get("mel_cutoff_candidate_ms", 0.0) or 0.0)
        if tail_sil >= 0.70 and (offset + abs(float(cutoff))) > cand_cut + 8.0:
            mel_safety = 1.0
    feat["mel_safety_clamped"] = mel_safety
    feat.update(_extract_structure_features(language, str(row.get("alias", "")), alias_type, row_index, total_rows))
    feat["alias_group"] = _derive_alias_group(
        language,
        feat,
        alias=str(row.get("alias", "") or ""),
    )
    feat = augment_mapping_quality_features(language, format_type, alias_type, feat)

    if prev_row is not None:
        feat["prev_alias_type"] = str(prev_row.get("alias_type", ""))
        feat["prev_base_pre"] = float(prev_row.get("pre", 0.0))
        feat["prev_base_offset"] = float(prev_row.get("offset", 0.0))
        feat["prev_base_cutoff_abs"] = float(prev_row.get("offset", 0.0)) + abs(float(prev_row.get("cutoff", 0.0)))
    if next_row is not None:
        feat["next_alias_type"] = str(next_row.get("alias_type", ""))
        feat["next_base_pre"] = float(next_row.get("pre", 0.0))
        feat["next_base_offset"] = float(next_row.get("offset", 0.0))
        feat["next_base_cutoff_abs"] = float(next_row.get("offset", 0.0)) + abs(float(next_row.get("cutoff", 0.0)))

    return feat


def _compute_file_context_stats(alias_types: List[str]) -> Dict[str, float]:
    total = len(alias_types)
    cv_count = 0
    vc_count = 0
    vv_count = 0
    vcv_count = 0
    br_count = 0
    mono_count = 0
    for a_type in alias_types:
        if a_type in {"cv", "cv_head"}:
            cv_count += 1
        elif a_type == "vc":
            vc_count += 1
        elif a_type == "vv":
            vv_count += 1
        elif a_type == "vcv":
            vcv_count += 1
        elif a_type == "br":
            br_count += 1
        elif a_type == "mono":
            mono_count += 1
    denom = max(total, 1)
    return {
        "row_count": float(total),
        "cv_count": float(cv_count),
        "vc_count": float(vc_count),
        "vv_count": float(vv_count),
        "vcv_count": float(vcv_count),
        "br_count": float(br_count),
        "mono_count": float(mono_count),
        "cv_ratio": float(cv_count) / float(denom),
        "vc_ratio": float(vc_count) / float(denom),
        "vc_cv_ratio": _safe_ratio(vc_count + 1.0, cv_count + 1.0),
        "cv_vc_balance": float(cv_count - vc_count) / float(denom),
    }


def _compute_mel_file_stats(
    mel_ctx,
    phones: List[object],
    language: str,
    *,
    wav_rms_linear: float = 0.0,
    voicebank_rms_linear: float = 0.0,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    stats["wav_rms_linear"] = float(max(0.0, wav_rms_linear))
    stats["voicebank_rms_linear"] = float(max(0.0, voicebank_rms_linear))
    if mel_ctx:
        en = mel_ctx.get("energy")
        f0v = mel_ctx.get("f0_voicing")
        if en is not None and len(en):
            if np is not None:
                stats["file_mean_energy"] = float(np.mean(np.asarray(en, dtype=np.float64)))
            else:
                stats["file_mean_energy"] = float(sum(float(x) for x in en) / max(len(en), 1))
        if f0v is not None and len(f0v):
            if np is not None:
                stats["file_voiced_ratio"] = float(np.mean(np.asarray(f0v, dtype=np.float64) >= 0.5))
            else:
                voiced_count = sum(1 for x in f0v if float(x) >= 0.5)
                stats["file_voiced_ratio"] = float(voiced_count) / float(max(len(f0v), 1))

    lengths = []
    for iv in phones or []:
        try:
            if _is_vowel_mark(language, getattr(iv, "mark", "")):
                lengths.append((float(iv.maxTime) - float(iv.minTime)) * 1000.0)
        except Exception:
            continue
    if not lengths:
        for iv in phones or []:
            try:
                lengths.append((float(iv.maxTime) - float(iv.minTime)) * 1000.0)
            except Exception:
                continue
    if lengths:
        stats["file_mean_syllable_dur_ms"] = float(sum(lengths) / max(len(lengths), 1))
    return stats


def _build_mapping_syllables_info(language: str, words: List[object], phones: List[object], mel_ctx) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    lang = str(language or "").strip().lower()
    if not words:
        words = []
    try:
        from core.oto_generator import (
            _estimate_kr_blank_confidence_at_time,
            _estimate_kr_mel_class_scores_at_time,
        )
    except Exception:
        _estimate_kr_blank_confidence_at_time = None
        _estimate_kr_mel_class_scores_at_time = None

    for w in words:
        mark = str(getattr(w, "mark", "") or "").strip()
        if not mark:
            continue
        w_start = float(getattr(w, "minTime", 0.0) or 0.0)
        w_end = float(getattr(w, "maxTime", 0.0) or 0.0)
        if w_end <= w_start:
            continue
        s_phones: List[object] = []
        for p in phones or []:
            p_start = float(getattr(p, "minTime", 0.0) or 0.0)
            p_end = float(getattr(p, "maxTime", 0.0) or 0.0)
            if p_start >= (w_start - 0.01) and p_end <= (w_end + 0.01):
                s_phones.append(p)

        row: Dict[str, object] = {
            "word": mark,
            "start_time": float(w_start),
            "end_time": float(w_end),
            "phones": s_phones,
        }
        if lang == "korean":
            try:
                from core.lab_generator import decompose_hangul_to_roman
                from core.kr_oto_rules import _kr_cv_kernel

                roman_parts: List[str] = []
                for ch in mark:
                    roman_parts.extend(decompose_hangul_to_roman(ch))
                roman_raw = "".join(roman_parts).lower()
                row["roman"] = roman_raw
                row["roman_cv"] = _kr_cv_kernel(roman_raw)
            except Exception:
                row["roman"] = mark.lower()
                row["roman_cv"] = mark.lower()
        else:
            try:
                from core.ja_oto_mapping import _normalize_ja_syllable_token

                row["roman"] = _normalize_ja_syllable_token(mark)
            except Exception:
                row["roman"] = mark.lower()

        if mel_ctx and _estimate_kr_blank_confidence_at_time is not None and _estimate_kr_mel_class_scores_at_time is not None:
            t_ms = float(w_start * 1000.0)
            try:
                row["blank_confidence"] = float(_estimate_kr_blank_confidence_at_time(mel_ctx, t_ms))
                mel_scores = _estimate_kr_mel_class_scores_at_time(mel_ctx, t_ms)
                row["mel_voiced_formant_conf"] = float(mel_scores.get("mel_voiced_formant_conf", 0.0) or 0.0)
                row["mel_silence_sparse_conf"] = float(mel_scores.get("mel_silence_sparse_conf", 0.0) or 0.0)
                row["mel_unvoiced_diffuse_conf"] = float(mel_scores.get("mel_unvoiced_diffuse_conf", 0.0) or 0.0)
                row["mel_breath_like_conf"] = float(mel_scores.get("mel_breath_like_conf", 0.0) or 0.0)
            except Exception:
                row["blank_confidence"] = 0.0
        out.append(row)
    if len(out) <= 1 and phones:
        fallback: List[Dict[str, object]] = []
        for p in phones:
            mark = str(getattr(p, "mark", "") or "").strip()
            if not mark:
                continue
            clean = _clean_mark(mark)
            if clean in {"", "sil", "sp", "spn", "pau"}:
                continue
            p_start = float(getattr(p, "minTime", 0.0) or 0.0)
            p_end = float(getattr(p, "maxTime", 0.0) or 0.0)
            if p_end <= p_start:
                continue
            row: Dict[str, object] = {
                "word": mark,
                "start_time": float(p_start),
                "end_time": float(p_end),
                "phones": [p],
            }
            if lang == "korean":
                try:
                    from core.kr_oto_rules import _kr_cv_kernel

                    row["roman"] = clean
                    row["roman_cv"] = _kr_cv_kernel(clean)
                except Exception:
                    row["roman"] = clean
                    row["roman_cv"] = clean
            else:
                try:
                    from core.ja_oto_mapping import _normalize_ja_syllable_token

                    row["roman"] = _normalize_ja_syllable_token(clean)
                except Exception:
                    row["roman"] = clean
            if mel_ctx and _estimate_kr_blank_confidence_at_time is not None and _estimate_kr_mel_class_scores_at_time is not None:
                t_ms = float(p_start * 1000.0)
                try:
                    row["blank_confidence"] = float(_estimate_kr_blank_confidence_at_time(mel_ctx, t_ms))
                    mel_scores = _estimate_kr_mel_class_scores_at_time(mel_ctx, t_ms)
                    row["mel_voiced_formant_conf"] = float(mel_scores.get("mel_voiced_formant_conf", 0.0) or 0.0)
                    row["mel_silence_sparse_conf"] = float(mel_scores.get("mel_silence_sparse_conf", 0.0) or 0.0)
                    row["mel_unvoiced_diffuse_conf"] = float(mel_scores.get("mel_unvoiced_diffuse_conf", 0.0) or 0.0)
                    row["mel_breath_like_conf"] = float(mel_scores.get("mel_breath_like_conf", 0.0) or 0.0)
                except Exception:
                    row["blank_confidence"] = 0.0
            fallback.append(row)
        if len(fallback) > len(out):
            out = fallback
    return out


def _candidate_anchor_ms_from_syllable(language: str, syl: Dict[str, object]) -> float:
    phones = list((syl or {}).get("phones") or [])
    start_ms = float((syl or {}).get("start_time", 0.0) or 0.0) * 1000.0
    if not phones:
        return float(start_ms)
    first_any = None
    first_vowel = None
    lang = str(language or "").strip().lower()
    for p in phones:
        mark = str(getattr(p, "mark", "") or "")
        clean = _clean_mark(mark)
        p_start = float(getattr(p, "minTime", 0.0) or 0.0) * 1000.0
        if first_any is None and clean not in {"", "sil", "sp", "spn", "pau"}:
            first_any = p_start
        if first_vowel is None:
            if lang == "japanese":
                try:
                    from core.ja_oto_mapping import _is_nucleus_phone

                    if _is_nucleus_phone(clean):
                        first_vowel = p_start
                except Exception:
                    pass
            else:
                if _is_vowel_mark(language, clean):
                    first_vowel = p_start
    if first_vowel is not None:
        return float(first_vowel)
    if first_any is not None:
        return float(first_any)
    return float(start_ms)


def _extract_wav_mapping_hints(
    language: str,
    format_type: str,
    wav_rows: List[Dict[str, object]],
    alias_types: List[str],
    words: List[object],
    phones: List[object],
    mel_ctx,
) -> Dict[int, Dict[str, float]]:
    hints: Dict[int, Dict[str, float]] = {}
    if not wav_rows:
        return hints
    lang = str(language or "").strip().lower()
    mapping_rows: List[Tuple[int, str, str]] = []

    for idx, row in enumerate(wav_rows):
        alias = str(row.get("alias", "") or "")
        alias_type = str(alias_types[idx] if idx < len(alias_types) else "").strip().lower()
        if alias_type not in {"cv", "cv_head", "vcv", "mono"}:
            continue
        token = ""
        try:
            if lang == "japanese":
                from core.ja_oto_mapping import _alias_to_ja_cv_target

                token = str(_alias_to_ja_cv_target(alias, alias_type) or "")
            else:
                from core.kr_oto_mapping import _alias_to_cv_target

                token = str(_alias_to_cv_target(alias, alias_type) or "")
        except Exception:
            token = ""
        if token:
            mapping_rows.append((idx, alias_type, token))

    if not mapping_rows:
        return hints

    def _fallback_hint_for_row(row_idx: int) -> Dict[str, float]:
        row = wav_rows[row_idx] if 0 <= int(row_idx) < len(wav_rows) else {}
        try:
            base_offset = float(row.get("offset", 0.0) or 0.0)
        except Exception:
            base_offset = 0.0
        try:
            base_cutoff = abs(float(row.get("cutoff", 0.0) or 0.0))
        except Exception:
            base_cutoff = 0.0
        cutoff_abs = max(base_offset + 24.0, base_offset + base_cutoff)
        return {
            "m_map_ok": 0.50,
            "m_offset_hint_abs": float(base_offset),
            "m_cutoff_hint_abs": float(cutoff_abs),
            "m_hint_rank": 1.0,
        }

    syllables_info = _build_mapping_syllables_info(lang, words, phones, mel_ctx)
    if not syllables_info:
        return hints

    expected_tokens = [tok for _idx, _alias_type, tok in mapping_rows]
    try:
        if lang == "japanese":
            from core.ja_mapping_v2 import build_ja_cv_anchor_plan

            plan = build_ja_cv_anchor_plan(expected_tokens, syllables_info, use_mel=True, format_type=format_type)
        else:
            from core.kr_mapping_v2 import build_kr_cv_anchor_plan

            plan = build_kr_cv_anchor_plan(expected_tokens, syllables_info, use_mel=True, format_type=format_type)
    except Exception:
        plan = {}

    indices = list((plan or {}).get("indices") or [])
    score_rows = list((plan or {}).get("score_rows") or [])
    feature_rows = list((plan or {}).get("feature_rows") or [])
    if not indices or len(indices) != len(mapping_rows):
        for row_idx, _alias_type, _tok in mapping_rows:
            hints[int(row_idx)] = _fallback_hint_for_row(int(row_idx))
        return hints

    topk_rows = []
    top1_indices = []
    if feature_rows and len(feature_rows) == len(score_rows):
        try:
            from core.oto_ml_mapping_runtime import predict_mapping_hint_topk

            hint_payload = predict_mapping_hint_topk(
                language=lang,
                format_type=format_type,
                score_rows=score_rows,
                feature_rows=feature_rows,
                top_k=3,
            )
            if bool(((hint_payload or {}).get("meta") or {}).get("applied", False)):
                topk_rows = list((hint_payload or {}).get("topk_rows") or [])
                top1_indices = list((hint_payload or {}).get("top1_indices") or [])
        except Exception:
            topk_rows = []
            top1_indices = []

    for target_idx, (row_idx, _alias_type, _tok) in enumerate(mapping_rows):
        cand_idx = int(indices[target_idx]) if target_idx < len(indices) else -1
        model_top = None
        if target_idx < len(topk_rows):
            cand_rows = list(topk_rows[target_idx] or [])
            if cand_rows:
                model_top = dict(cand_rows[0])
                if target_idx < len(top1_indices):
                    cand_idx = int(top1_indices[target_idx])
                else:
                    cand_idx = int(model_top.get("cand_idx", cand_idx))
        if not (0 <= cand_idx < len(syllables_info)):
            hints[int(row_idx)] = _fallback_hint_for_row(int(row_idx))
            continue
        syl = syllables_info[cand_idx]
        anchor_ms = _candidate_anchor_ms_from_syllable(lang, syl)
        syl_start_ms = float((syl or {}).get("start_time", 0.0) or 0.0) * 1000.0
        syl_end_ms = float((syl or {}).get("end_time", 0.0) or 0.0) * 1000.0
        syl_dur_ms = max(16.0, syl_end_ms - syl_start_ms)
        try:
            pre = float(wav_rows[row_idx].get("pre", 0.0) or 0.0)
        except Exception:
            pre = 0.0
        offset_hint_abs = max(0.0, anchor_ms - pre)
        cutoff_hint_abs = max(syl_end_ms, offset_hint_abs + max(28.0, syl_dur_ms * 0.70))

        rank = 1
        p_map_ok = 0.0
        if target_idx < len(score_rows):
            row_scores = list(score_rows[target_idx] or [])
            if row_scores and 0 <= cand_idx < len(row_scores):
                ranked = sorted(range(len(row_scores)), key=lambda idx_: row_scores[idx_], reverse=True)
                try:
                    rank = ranked.index(cand_idx) + 1
                except Exception:
                    rank = 1
                best = float(row_scores[cand_idx])
                second = max(
                    (float(v) for j, v in enumerate(row_scores) if j != cand_idx),
                    default=(best - 12.0),
                )
                gap = float(best - second)
                p_map_ok = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, gap / 18.0))))
        if isinstance(model_top, dict):
            try:
                p_map_ok = float(model_top.get("p_map_ok", p_map_ok) or p_map_ok)
            except Exception:
                pass
            try:
                rank = int(model_top.get("m_rank", rank) or rank)
            except Exception:
                pass
            try:
                pred_off = float(model_top.get("m_offset_hint", 0.0) or 0.0)
                if pred_off > 0.0:
                    offset_hint_abs = float(pred_off)
            except Exception:
                pass
            try:
                pred_cut = float(model_top.get("m_cutoff_hint", 0.0) or 0.0)
                if pred_cut > (offset_hint_abs + 4.0):
                    cutoff_hint_abs = float(pred_cut)
            except Exception:
                pass

        hints[int(row_idx)] = {
            "m_map_ok": float(max(0.0, min(1.0, p_map_ok))),
            "m_offset_hint_abs": float(offset_hint_abs),
            "m_cutoff_hint_abs": float(cutoff_hint_abs),
            "m_hint_rank": float(rank),
        }
    return hints


def extract_feature_rows(language: str, oto_path: str, tg_dir: str, wav_dir: str, custom_phonemes_path: str = "", voicebank_id: str = "", format_type_override: str = "") -> List[Dict[str, object]]:
    lang = str(language).strip().lower()
    custom_map = load_custom_map(custom_phonemes_path)
    fmt_tag = normalize_format_type(lang, str(format_type_override or "").strip().lower())
    m_hint_toggle = "on" if _env_bool("UTOA_M_HINT_ENABLE", True) else "off"
    prefix_map_path = find_prefix_map_path(oto_path, wav_dir, tg_dir)
    cache_path = _feature_cache_path(
        lang,
        oto_path,
        tg_dir,
        wav_dir,
        custom_phonemes_path,
        f"{voicebank_id}|fmt={fmt_tag}|m_hint={m_hint_toggle}",
        prefix_map_path,
    )
    cached_rows = _load_feature_cache(cache_path)
    if cached_rows is not None:
        return cached_rows
    rows = parse_oto_rows(
        oto_path,
        language=lang,
        custom_map=custom_map,
        prefix_map_path=prefix_map_path,
        prefix_context_paths=[oto_path, wav_dir, tg_dir],
    )
    if not rows:
        return []

    tg_index = _build_tg_index(tg_dir)
    find_wav_path_for_name, mel_envelope, read_wav_mono_np = _lazy_wav_helpers()

    wav_index = {}
    if wav_dir and os.path.isdir(wav_dir):
        for fn in os.listdir(wav_dir):
            if fn.lower().endswith(".wav"):
                wav_index[normalize_key(fn)] = os.path.join(wav_dir, fn)

    out_rows: List[Dict[str, object]] = []
    rows_by_wav: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        rows_by_wav.setdefault(str(row["wav"]), []).append(row)

    vb_id = infer_voicebank_id(oto_path, tg_dir=tg_dir, wav_dir=wav_dir, voicebank_id=voicebank_id)
    mel_cache: Dict[str, object] = {}
    audio_cache: Dict[str, Tuple[object, int]] = {}
    wav_rms_by_name: Dict[str, float] = {}

    for wav_name in rows_by_wav.keys():
        wav_path = find_wav_path_for_name(wav_name, wav_dir, wav_index)
        if not wav_path:
            continue
        if wav_path not in audio_cache:
            audio_cache[wav_path] = read_wav_mono_np(wav_path)
        audio, sr = audio_cache[wav_path]
        wav_rms_by_name[wav_name] = _compute_audio_rms(audio)
        if wav_path not in mel_cache:
            mel_cache[wav_path] = mel_envelope(audio, sr)

    vb_rms_vals = [float(v) for v in wav_rms_by_name.values() if float(v) > 0.0]
    if vb_rms_vals:
        if np is not None:
            voicebank_rms_ref = float(np.median(np.asarray(vb_rms_vals, dtype=np.float64)))
        else:
            vs = sorted(vb_rms_vals)
            mid = len(vs) // 2
            voicebank_rms_ref = float(vs[mid])
    else:
        voicebank_rms_ref = 0.0

    for wav_name, wav_rows in rows_by_wav.items():
        aliases = [str(r["alias"]) for r in wav_rows]
        format_type = fmt_tag or detect_format_type(lang, aliases, custom_map=custom_map)
        tg_path = tg_index.get(normalize_key(wav_name), "")
        phones, words = _load_textgrid_tiers(tg_path)

        wav_path = find_wav_path_for_name(wav_name, wav_dir, wav_index)
        mel_ctx = None
        audio = None
        sr = 0
        if wav_path:
            if wav_path not in audio_cache:
                audio_cache[wav_path] = read_wav_mono_np(wav_path)
            audio, sr = audio_cache[wav_path]
            if wav_path not in mel_cache:
                mel_cache[wav_path] = mel_envelope(audio, sr)
            mel_ctx = mel_cache[wav_path]

        alias_types = []
        for row in wav_rows:
            alias = str(row["alias"])
            alias_type = classify_alias_type(lang, alias, custom_map=custom_map)
            row["alias_type"] = alias_type
            alias_types.append(alias_type)
        mapping_hint_enabled = _env_bool("UTOA_M_HINT_ENABLE", True)
        mapping_hints_by_idx = (
            _extract_wav_mapping_hints(
                lang,
                format_type,
                wav_rows,
                alias_types,
                words,
                phones,
                mel_ctx,
            )
            if mapping_hint_enabled
            else {}
        )
        file_stats = _compute_file_context_stats(alias_types)
        file_stats.update(
            _compute_mel_file_stats(
                mel_ctx,
                phones,
                lang,
                wav_rms_linear=float(wav_rms_by_name.get(wav_name, 0.0) or 0.0),
                voicebank_rms_linear=float(voicebank_rms_ref),
            )
        )

        for idx, row in enumerate(wav_rows):
            alias_type = alias_types[idx]
            feat = _feature_row_from_context(
                lang, format_type, row, idx, len(wav_rows), alias_type,
                phones, words, mel_ctx, audio, sr,
                wav_rows[idx - 1] if idx > 0 else None,
                wav_rows[idx + 1] if idx + 1 < len(wav_rows) else None,
                file_stats,
            )
            hint = mapping_hints_by_idx.get(int(idx), {}) if mapping_hint_enabled else {}
            if mapping_hint_enabled:
                m_map_ok = float(hint.get("m_map_ok", feat.get("mapping_confidence", 0.0)) or 0.0)
                m_offset_hint_abs = float(
                    hint.get(
                        "m_offset_hint_abs",
                        feat.get("mel_offset_candidate_ms", feat.get("base_offset", 0.0)),
                    )
                    or 0.0
                )
                m_cutoff_hint_abs = float(
                    hint.get(
                        "m_cutoff_hint_abs",
                        feat.get("mel_cutoff_candidate_ms", feat.get("base_cutoff_abs", 0.0)),
                    )
                    or 0.0
                )
                if m_offset_hint_abs <= 0.0:
                    m_offset_hint_abs = float(feat.get("base_offset", 0.0) or 0.0)
                if m_cutoff_hint_abs <= (m_offset_hint_abs + 4.0):
                    m_cutoff_hint_abs = max(
                        float(feat.get("base_cutoff_abs", 0.0) or 0.0),
                        m_offset_hint_abs + 24.0,
                    )
                feat["m_map_ok"] = float(max(0.0, min(1.0, m_map_ok)))
                feat["m_offset_hint_abs"] = float(m_offset_hint_abs)
                feat["m_cutoff_hint_abs"] = float(m_cutoff_hint_abs)
                feat["m_hint_rank"] = float(hint.get("m_hint_rank", 99.0) or 99.0)
            else:
                feat["m_map_ok"] = 0.0
                feat["m_offset_hint_abs"] = float(feat.get("base_offset", 0.0) or 0.0)
                feat["m_cutoff_hint_abs"] = float(feat.get("base_cutoff_abs", 0.0) or 0.0)
                feat["m_hint_rank"] = 99.0
            if "mapping_confidence" in row:
                try:
                    feat["mapping_confidence"] = float(row.get("mapping_confidence", feat.get("mapping_confidence", 0.0)))
                except Exception:
                    pass
            if "mapping_reason_code" in row and str(row.get("mapping_reason_code", "")).strip():
                feat["mapping_reason_code"] = str(row.get("mapping_reason_code", "")).strip()
            if "jump_blocked_flag" in row:
                try:
                    feat["jump_blocked_flag"] = 1.0 if float(row.get("jump_blocked_flag", 0.0) or 0.0) > 0.0 else 0.0
                except Exception:
                    feat["jump_blocked_flag"] = 0.0
            if not str(feat.get("mapping_reason_code", "")).strip():
                feat["mapping_reason_code"] = "unknown"
            feat["voicebank_id"] = vb_id
            feat["wav"] = row["wav"]
            feat["wav_norm"] = row["wav_norm"]
            feat["alias"] = row["alias"]
            feat["alias_norm"] = row["alias_norm"]
            feat["occurrence_index"] = int(row["occurrence_index"])
            feat["line_index"] = int(row["line_index"])
            feat["source_oto_id"] = str(row.get("source_oto_id", "") or "")
            feat["source_row_id"] = str(row.get("source_row_id", "") or "")
            feat["raw_line"] = row["raw_line"]
            onset_anchor_ms, tail_anchor_ms, patch_source = resolve_mel_patch_anchors(feat)
            feat["mel_onset_anchor_ms"] = float(onset_anchor_ms)
            feat["mel_tail_anchor_ms"] = float(tail_anchor_ms)
            feat["mel_patch_source"] = str(patch_source)
            feat["mel_patch_key"] = make_mel_patch_key(
                language=lang,
                format_type=format_type,
                voicebank_id=vb_id,
                wav_norm=str(row["wav_norm"]),
                alias_norm=str(row["alias_norm"]),
                occurrence_index=int(row["occurrence_index"]),
                row_index_in_wav=idx,
            )
            feat["mel_patch_debug_key"] = make_mel_patch_debug_key(
                voicebank_id=vb_id,
                wav=str(row["wav"]),
                alias=str(row["alias"]),
                alias_norm=str(row["alias_norm"]),
                occurrence_index=int(row["occurrence_index"]),
                row_index_in_wav=idx,
            )
            out_rows.append(feat)

    _save_feature_cache(cache_path, out_rows)
    return out_rows


def _evaluate_training_row_quality(language: str, row: Dict[str, object]) -> Tuple[int, str, float]:
    lang = str(language or "").strip().lower()
    alias_type = str(row.get("alias_type", "") or "").strip().lower()
    alias_group = str(row.get("alias_group", "") or "").strip().lower()
    coda_type = str(row.get("coda_type", "") or "").strip().lower()
    is_diph = float(row.get("is_diphthong", 0.0) or 0.0) >= 0.5

    d_off = abs(float(row.get("delta_offset", 0.0) or 0.0))
    d_cons = abs(float(row.get("delta_cons", 0.0) or 0.0))
    d_cut = abs(float(row.get("delta_cutoff", 0.0) or 0.0))
    d_pre = abs(float(row.get("delta_pre", 0.0) or 0.0))
    d_ovl = abs(float(row.get("delta_ovl", 0.0) or 0.0))
    base_off_to_exp = abs(float(row.get("base_offset_to_expected_ms", 0.0) or 0.0))

    reasons: List[str] = []
    if lang == "korean":
        if alias_type in {"cv", "cv_head"}:
            off_lim = 110.0 if is_diph or "glide" in alias_group else 135.0
            pre_lim = 110.0 if is_diph or "glide" in alias_group else 125.0
            cons_lim = 170.0
            if d_off > off_lim:
                reasons.append("cv_offset_outlier")
            if d_pre > pre_lim:
                reasons.append("cv_pre_outlier")
            if d_cons > cons_lim:
                reasons.append("cv_cons_outlier")
            if base_off_to_exp > 320.0:
                reasons.append("cv_base_anchor_far")
        elif alias_type == "vv":
            if d_off > 125.0:
                reasons.append("vv_offset_outlier")
            if d_pre > 105.0:
                reasons.append("vv_pre_outlier")
            if d_cut > 150.0:
                reasons.append("vv_cutoff_outlier")
        elif alias_type == "vc":
            if alias_group == "vc_tail_breath":
                if d_cut > 360.0:
                    reasons.append("vc_tail_breath_cutoff_outlier")
                if d_pre > 210.0:
                    reasons.append("vc_tail_breath_pre_outlier")
                if d_ovl > 160.0:
                    reasons.append("vc_tail_breath_ovl_outlier")
            else:
                cut_lim = 120.0 if coda_type == "stop" else 155.0
                pre_lim = 100.0 if coda_type == "stop" else 125.0
                ovl_lim = 75.0 if coda_type == "stop" else 90.0
                if d_cut > cut_lim:
                    reasons.append("vc_cutoff_outlier")
                if d_pre > pre_lim:
                    reasons.append("vc_pre_outlier")
                if d_ovl > ovl_lim:
                    reasons.append("vc_ovl_outlier")
        elif alias_type == "vcv":
            if d_off > 150.0:
                reasons.append("vcv_offset_outlier")
            if d_pre > 120.0:
                reasons.append("vcv_pre_outlier")
        elif alias_type == "br" or alias_group == "br_standalone":
            if d_cut > 420.0:
                reasons.append("br_cutoff_outlier")
            if d_pre > 240.0:
                reasons.append("br_pre_outlier")

        gross_limit = 420.0 if alias_group in {"vc_tail_breath", "br_standalone"} else 235.0
        if max(d_off, d_cons, d_cut, d_pre, d_ovl) > gross_limit:
            reasons.append("gross_outlier")
    else:
        if max(d_off, d_cons, d_cut, d_pre, d_ovl) > 300.0:
            reasons.append("gross_outlier")

    keep = 0 if reasons else 1
    score = max(0.0, 100.0 - (18.0 * len(reasons)) - min(max(d_off, d_pre) / 20.0, 20.0))
    return keep, ";".join(reasons), float(score)


def _compute_training_sample_weight(row: Dict[str, object]) -> float:
    try:
        quality = float(row.get("train_quality_score", 100.0) or 100.0)
    except Exception:
        quality = 100.0
    try:
        mapping_conf = float(row.get("mapping_confidence", 1.0) or 1.0)
    except Exception:
        mapping_conf = 1.0
    try:
        keep_default = int(row.get("train_keep_default", 1) or 1)
    except Exception:
        keep_default = 1
    alias_type = str(row.get("alias_type", "") or "").strip().lower()
    mapping_reason = str(row.get("mapping_reason_code", "") or "").strip().lower()
    try:
        jump_blocked = int(float(row.get("jump_blocked_flag", 0.0) or 0.0) > 0.0)
    except Exception:
        jump_blocked = 0
    try:
        alias_occurrence = int(float(row.get("used_alias_occurrence_mapping", 0.0) or 0.0) > 0.0)
    except Exception:
        alias_occurrence = 0
    try:
        nuclei_fallback = int(float(row.get("used_nuclei_fallback", 0.0) or 0.0) > 0.0)
    except Exception:
        nuclei_fallback = 0
    try:
        alias_based = int(float(row.get("used_alias_based_syllables", 0.0) or 0.0) > 0.0)
    except Exception:
        alias_based = 0

    quality_norm = max(0.2, min(1.0, quality / 100.0))
    mapping_norm = max(0.2, min(1.0, mapping_conf))
    weight = quality_norm * (0.60 + (0.40 * mapping_norm))
    if keep_default <= 0:
        weight *= 0.55

    if mapping_conf < 0.42:
        weight *= 0.82

    hard_boost = 1.0
    if alias_type in {"vc", "vv", "vcv"} and mapping_conf >= 0.52:
        hard_boost *= 1.08
    if jump_blocked > 0:
        hard_boost *= 1.08
    if alias_occurrence > 0 and mapping_conf >= 0.56:
        hard_boost *= 1.06
    if nuclei_fallback > 0 and mapping_conf >= 0.56:
        hard_boost *= 1.05
    if alias_based > 0 and mapping_conf >= 0.58:
        hard_boost *= 1.04
    if mapping_reason in {
        "order_locked_glide_mismatch",
        "order_locked_length_mismatch",
        "order_locked_low_phone_quality",
        "alias_based_recover",
        "alias_based_empty_words",
    } and mapping_conf >= 0.54:
        hard_boost *= 1.06
    if mapping_reason in {"words_keep_high_conf", "filename_sequence_lock"} and mapping_conf >= 0.76:
        hard_boost *= 1.03
    weight *= min(hard_boost, 1.36)

    blank_score = float(compute_blank_risk_score(row))
    if alias_type in {"cv", "cv_head"} and blank_score >= 0.55:
        blank_penalty = min(0.60, 0.12 + ((blank_score - 0.55) * 0.85))
        weight *= max(0.40, 1.0 - blank_penalty)

    return float(max(0.18, min(weight, 1.80)))


def _mapping_max_shift_ms(feature_row: Dict[str, object]) -> float:
    wav_ms = float(feature_row.get("wav_duration_ms", 0.0) or 0.0)
    if wav_ms > 0.0:
        return max(500.0, min(3500.0, wav_ms * 0.30))
    return 1800.0


def _group_manual_rows_for_match(manual_rows: List[Dict[str, object]]) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in manual_rows:
        key = (str(row.get("wav_norm", "")), str(row.get("alias_norm", "")))
        grouped.setdefault(key, []).append(row)
    for key in list(grouped.keys()):
        grouped[key].sort(key=lambda r: (float(r.get("offset", 0.0)), int(r.get("line_index", 0))))
    return grouped


def _group_auto_indices_for_match(auto_feats: List[Dict[str, object]]) -> Dict[Tuple[str, str], List[int]]:
    grouped: Dict[Tuple[str, str], List[int]] = {}
    for idx, feat in enumerate(auto_feats):
        key = (str(feat.get("wav_norm", "")), str(feat.get("alias_norm", "")))
        grouped.setdefault(key, []).append(idx)
    for key in list(grouped.keys()):
        grouped[key].sort(key=lambda i: (float(auto_feats[i].get("base_offset", 0.0)), int(auto_feats[i].get("line_index", 0))))
    return grouped


def _build_manual_matches_by_time(
    auto_feats: List[Dict[str, object]],
    manual_rows: List[Dict[str, object]],
) -> Tuple[Dict[int, Dict[str, object]], Dict[str, int]]:
    manual_groups = _group_manual_rows_for_match(manual_rows)
    auto_groups = _group_auto_indices_for_match(auto_feats)
    matches: Dict[int, Dict[str, object]] = {}
    stats = {
        "single_direct": 0,
        "occurrence_direct": 0,
        "time_nearest": 0,
        "skip_far": 0,
        "skip_unmatched": 0,
    }

    for key, auto_indices in auto_groups.items():
        manual_group = manual_groups.get(key) or []
        if not manual_group:
            stats["skip_unmatched"] += len(auto_indices)
            continue

        if len(auto_indices) == 1 and len(manual_group) == 1:
            auto_idx = auto_indices[0]
            diff = abs(float(manual_group[0].get("offset", 0.0)) - float(auto_feats[auto_idx].get("base_offset", 0.0)))
            if diff > _mapping_max_shift_ms(auto_feats[auto_idx]):
                stats["skip_far"] += 1
            else:
                matches[auto_idx] = manual_group[0]
                stats["single_direct"] += 1
            continue

        used_manual = set()
        manual_occ_idx: Dict[int, int] = {}
        for j, manual_row in enumerate(manual_group):
            try:
                occ = int(manual_row.get("occurrence_index", -1))
            except Exception:
                occ = -1
            if occ >= 0 and occ not in manual_occ_idx:
                manual_occ_idx[occ] = j

        # Repeated aliases are best aligned by occurrence index first.
        # If the occurrence candidate is too far, fall back to time-nearest.
        for auto_idx in auto_indices:
            try:
                auto_occ = int(auto_feats[auto_idx].get("occurrence_index", -1))
            except Exception:
                auto_occ = -1
            if auto_occ < 0:
                continue
            manual_j = manual_occ_idx.get(auto_occ, -1)
            if manual_j < 0 or manual_j in used_manual:
                continue
            manual_row = manual_group[manual_j]
            diff = abs(float(manual_row.get("offset", 0.0)) - float(auto_feats[auto_idx].get("base_offset", 0.0)))
            if diff > _mapping_max_shift_ms(auto_feats[auto_idx]):
                continue
            used_manual.add(manual_j)
            matches[auto_idx] = manual_row
            stats["occurrence_direct"] += 1

        for auto_idx in auto_indices:
            if auto_idx in matches:
                continue
            auto_off = float(auto_feats[auto_idx].get("base_offset", 0.0))
            best_j = -1
            best_diff = None
            for j, manual_row in enumerate(manual_group):
                if j in used_manual:
                    continue
                diff = abs(float(manual_row.get("offset", 0.0)) - auto_off)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_j = j
            if best_j < 0:
                stats["skip_unmatched"] += 1
                continue
            if float(best_diff or 0.0) > _mapping_max_shift_ms(auto_feats[auto_idx]):
                stats["skip_far"] += 1
                continue
            used_manual.add(best_j)
            matches[auto_idx] = manual_group[best_j]
            stats["time_nearest"] += 1

    return matches, stats


def _resolve_manual_cutoff_abs(
    feature_row: Dict[str, object],
    manual_offset: float,
    manual_cutoff: float,
) -> Tuple[Optional[float], str]:
    # Standard negative cutoff: relative tail distance from offset.
    if manual_cutoff < 0.0:
        return float(manual_offset + abs(manual_cutoff)), "negative_relative"

    base_cut_abs = float(feature_row.get("base_cutoff_abs", 0.0) or 0.0)
    wav_ms = float(feature_row.get("wav_duration_ms", 0.0) or 0.0)
    cutoff_abs_raw = abs(float(manual_cutoff))

    # Positive cutoff appears in mixed styles:
    # 1) relative-from-offset: offset + cutoff
    # 2) absolute-in-file: cutoff itself
    candidates = [
        ("positive_relative", float(manual_offset + cutoff_abs_raw)),
        ("positive_absolute", float(cutoff_abs_raw)),
    ]
    if wav_ms > 0.0:
        # Some legacy banks store large positive values near wav end.
        # This branch interprets cutoff as "end-referenced positive style".
        candidates.append(
            ("positive_from_wav_end", float(manual_offset + max(wav_ms - cutoff_abs_raw, 0.0)))
        )

    scored: List[Tuple[float, str, float]] = []
    for mode, cand in candidates:
        score = abs(cand - base_cut_abs)
        if wav_ms > 0.0:
            if cand > (wav_ms + 80.0):
                score += 2000.0 + (cand - wav_ms)
            if cand < (manual_offset + 4.0):
                score += 800.0 + ((manual_offset + 4.0) - cand)
        scored.append((score, mode, cand))
    scored.sort(key=lambda x: x[0])

    chosen_mode, chosen_abs = scored[0][1], float(scored[0][2])
    if wav_ms > 0.0:
        chosen_abs = min(chosen_abs, max(0.0, wav_ms - 2.0))
    if chosen_abs <= (manual_offset + 2.0):
        return None, "invalid_cutoff"
    return chosen_abs, chosen_mode


def build_training_rows(
    language: str,
    auto_oto_path: str,
    manual_oto_path: str,
    tg_dir: str,
    wav_dir: str,
    custom_phonemes_path: str = "",
    voicebank_id: str = "",
    format_type_override: str = "",
    auto_oto_policy: str = "",
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    fmt_tag = str(format_type_override or "").strip().lower()
    resolved_auto, auto_meta = _resolve_auto_oto_for_training(
        language=language,
        auto_oto_path=auto_oto_path,
        manual_oto_path=manual_oto_path,
        tg_dir=tg_dir,
        wav_dir=wav_dir,
        format_type_override=fmt_tag,
        auto_oto_policy=auto_oto_policy,
    )
    auto_oto_path = resolved_auto
    temp_dir = auto_meta.get("temp_dir")

    try:
        auto_prefix_map_path = find_prefix_map_path(auto_oto_path, wav_dir, tg_dir)
        manual_prefix_map_path = find_prefix_map_path(manual_oto_path, wav_dir, tg_dir)
        cache_path = _training_row_cache_path(
            language=language,
            auto_oto_path=auto_oto_path,
            manual_oto_path=manual_oto_path,
            tg_dir=tg_dir,
            wav_dir=wav_dir,
            custom_phonemes_path=custom_phonemes_path,
            auto_prefix_map_path=auto_prefix_map_path,
            manual_prefix_map_path=manual_prefix_map_path,
            voicebank_id=f"{voicebank_id}|fmt={fmt_tag}",
        )
        cached = _load_training_row_cache(cache_path)
        if cached is not None:
            return cached
        auto_feats = extract_feature_rows(
            language,
            auto_oto_path,
            tg_dir=tg_dir,
            wav_dir=wav_dir,
            custom_phonemes_path=custom_phonemes_path,
            voicebank_id=voicebank_id,
            format_type_override=fmt_tag,
        )
        manual_rows = parse_oto_rows(
            manual_oto_path,
            language=language,
            custom_map=load_custom_map(custom_phonemes_path),
            prefix_map_path=manual_prefix_map_path,
            prefix_context_paths=[manual_oto_path, wav_dir, tg_dir],
        )
        manual_matches, match_stats = _build_manual_matches_by_time(auto_feats, manual_rows)
        matched_rows: List[Dict[str, object]] = []
        skipped = 0
        skipped_cutoff = 0
        label_counts = {"manual": 0}

        for feat_idx, feat in enumerate(auto_feats):
            manual_row = manual_matches.get(feat_idx)
            if not manual_row:
                skipped += 1
                continue
            row = dict(feat)
            if not str(row.get("mapping_reason_code", "")).strip():
                row["mapping_reason_code"] = "unknown"
            try:
                row["jump_blocked_flag"] = 1.0 if float(row.get("jump_blocked_flag", 0.0) or 0.0) > 0.0 else 0.0
            except Exception:
                row["jump_blocked_flag"] = 0.0

            row["manual_offset"] = float(manual_row["offset"])
            row["manual_cons"] = float(manual_row["cons"])
            row["manual_cutoff"] = float(manual_row["cutoff"])
            row["manual_pre"] = float(manual_row["pre"])
            row["manual_ovl"] = float(manual_row["ovl"])
            resolved_cutoff_abs, cutoff_mode = _resolve_manual_cutoff_abs(
                feature_row=feat,
                manual_offset=float(row["manual_offset"]),
                manual_cutoff=float(row["manual_cutoff"]),
            )
            if resolved_cutoff_abs is None:
                skipped += 1
                skipped_cutoff += 1
                continue
            row["manual_cutoff_abs"] = float(resolved_cutoff_abs)
            row["manual_cutoff_mode"] = cutoff_mode
            row["delta_offset"] = row["manual_offset"] - row["base_offset"]
            row["delta_cons"] = row["manual_cons"] - row["base_cons"]
            row["delta_cutoff"] = row["manual_cutoff_abs"] - row["base_cutoff_abs"]
            row["delta_pre"] = row["manual_pre"] - row["base_pre"]
            row["delta_ovl"] = row["manual_ovl"] - row["base_ovl"]
            try:
                manual_offset = float(row.get("manual_offset", 0.0) or 0.0)
            except Exception:
                manual_offset = 0.0
            try:
                vowel_start = float(row.get("curr_vowel_start_ms", 0.0) or 0.0)
            except Exception:
                vowel_start = 0.0
            try:
                vowel_end = float(row.get("curr_vowel_end_ms", 0.0) or 0.0)
            except Exception:
                vowel_end = 0.0
            try:
                next_anchor_abs = float(row.get("base_cutoff_abs", 0.0) or 0.0) + float(
                    row.get("base_cutoff_to_next_anchor_ms", 0.0) or 0.0
                )
            except Exception:
                next_anchor_abs = 0.0
            row["aux_vowel_start_rel"] = (vowel_start - manual_offset) if vowel_start > 0.0 else 0.0
            row["aux_vowel_end_rel"] = (vowel_end - manual_offset) if vowel_end > 0.0 else 0.0
            row["aux_next_onset_rel"] = (next_anchor_abs - manual_offset) if next_anchor_abs > 0.0 else 0.0
            keep_default, skip_reason, quality_score = _evaluate_training_row_quality(language, row)
            row["train_keep_default"] = int(keep_default)
            row["train_skip_reason"] = skip_reason
            row["train_quality_score"] = float(quality_score)
            mapping_conf = float(row.get("mapping_confidence", 0.0) or 0.0)
            jump_blocked = int(float(row.get("jump_blocked_flag", 0.0) or 0.0) > 0.0)
            label_source = "manual"
            row["label_source"] = label_source
            row["sample_weight"] = _compute_training_sample_weight(row)
            label_counts["manual"] += 1
            row["blank_risk_score"] = float(compute_blank_risk_score(row))
            row["blank_risk_flag"] = int(blank_risk_flag(row))
            row["skipped_reason"] = ""
            matched_rows.append(row)

        stats = {
            "auto_rows": len(auto_feats),
            "manual_rows": len(manual_rows),
            "matched_rows": len(matched_rows),
            "skipped_rows": skipped,
            "skipped_cutoff_rows": skipped_cutoff,
            "matched_single_direct": int(match_stats.get("single_direct", 0)),
            "matched_occurrence_direct": int(match_stats.get("occurrence_direct", 0)),
            "matched_time_nearest": int(match_stats.get("time_nearest", 0)),
            "skip_mapping_far": int(match_stats.get("skip_far", 0)),
            "skip_mapping_unmatched": int(match_stats.get("skip_unmatched", 0)),
            "label_manual": int(label_counts["manual"]),
        }
        if isinstance(auto_meta, dict):
            stats["auto_oto_status"] = str(auto_meta.get("status", ""))
            stats["auto_oto_policy"] = str(auto_meta.get("policy", ""))
        _save_training_row_cache(cache_path, matched_rows, stats)
        return matched_rows, stats
    finally:
        if temp_dir is not None:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


def dataset_fieldnames() -> List[str]:
    return [
        "voicebank_id", "wav", "alias", "wav_norm", "alias_norm", "occurrence_index", "line_index", "source_oto_id", "source_row_id",
        "mel_patch_key", "mel_patch_debug_key", "mel_onset_anchor_ms", "mel_tail_anchor_ms", "mel_patch_source",
        *FEATURE_NAMES,
        "manual_offset", "manual_cons", "manual_cutoff", "manual_cutoff_abs", "manual_cutoff_mode", "manual_pre", "manual_ovl",
        *TARGET_NAMES,
        *AUX_TARGET_NAMES,
        "label_source", "sample_weight", "blank_risk_score", "blank_risk_flag",
        "train_keep_default", "train_skip_reason", "train_quality_score", "skipped_reason",
    ]


def write_dataset_csv(path: str, rows: List[Dict[str, object]], append: bool = False) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mode = "a" if append and os.path.exists(path) else "w"
    fieldnames = dataset_fieldnames()
    with open(path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            merged = dict(row)
            merged.update(canonicalize_feature_row(row))
            writer.writerow(merged)
    return path
