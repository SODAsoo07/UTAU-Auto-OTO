from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


def _workspace_root() -> Path:
    # 개발 실행: .../Auto_OTO/core/conversion_tables.py -> parent.parent
    root = Path(__file__).resolve().parent.parent
    if (root / "assets" / "profiles").exists():
        return root

    # 배포 실행(PyInstaller): _MEIPASS 내부 assets 탐색
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        mp = Path(meipass)
        if (mp / "assets" / "profiles").exists():
            return mp

    # 마지막 폴백: 현재 작업 디렉터리
    return Path(os.getcwd())


def _profiles_dir() -> Path:
    return _workspace_root() / "assets" / "profiles"


def japanese_conversion_table_path() -> str:
    return str(_profiles_dir() / "ja_conversion_table.json")


def korean_conversion_table_path() -> str:
    return str(_profiles_dir() / "ko_conversion_table.json")


@lru_cache(maxsize=4)
def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def load_japanese_conversion_table() -> Dict[str, Any]:
    raw = _load_json(japanese_conversion_table_path())
    node = raw.get("japanese", raw)
    return node if isinstance(node, dict) else {}


def load_korean_conversion_table() -> Dict[str, Any]:
    raw = _load_json(korean_conversion_table_path())
    node = raw.get("korean", raw)
    return node if isinstance(node, dict) else {}

