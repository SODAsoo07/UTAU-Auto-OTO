"""
OTO ML feature & training row caches.

피처 추출 결과 및 학습 행 매칭 결과를 디스크에 캐시하여 반복 계산을 방지합니다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from core.oto_ml.features.schema import FEATURE_VERSION

logger = logging.getLogger(__name__)

TRAIN_ROW_MATCH_VERSION = "v9"


# ── Cache directory helpers ──────────────────────────────────────────────────

def _default_feature_cache_dir() -> str:
    configured = os.environ.get("UTOA_OTO_ML_CACHE_DIR", "").strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, ".cache", "oto_ml_features")
    os.makedirs(path, exist_ok=True)
    return path


def _default_training_row_cache_dir() -> str:
    configured = os.environ.get("UTOA_OTO_ML_ROW_CACHE_DIR", "").strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, ".cache", "oto_ml_training_rows")
    os.makedirs(path, exist_ok=True)
    return path


# ── Path signature ───────────────────────────────────────────────────────────

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


# ── Cache key computation ────────────────────────────────────────────────────

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


# ── Load / Save ──────────────────────────────────────────────────────────────

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
