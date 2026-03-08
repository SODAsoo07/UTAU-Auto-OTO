from __future__ import annotations

import os
from typing import List

from core.oto_ml_prepare_types import PreparedAutoPair


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


def _has_textgrid_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    for _dp, _dns, fns in os.walk(path):
        if any(fn.lower().endswith(".textgrid") for fn in fns):
            return True
    return False


def _discover_work_items(dataset_root: str) -> List[PreparedAutoPair]:
    items: List[PreparedAutoPair] = []
    for language in ("korean", "japanese"):
        lang_root = os.path.join(dataset_root, language)
        if not os.path.isdir(lang_root):
            continue
        for format_type in os.listdir(lang_root):
            fmt_root = os.path.join(lang_root, format_type)
            if not os.path.isdir(fmt_root):
                continue
            for voicebank in os.listdir(fmt_root):
                vb_root = os.path.join(fmt_root, voicebank)
                if not os.path.isdir(vb_root):
                    continue
                for dp, dns, fns in os.walk(vb_root):
                    lower = {fn.lower(): fn for fn in fns}
                    manual = ""
                    candidates = []
                    if "oto.ini" in lower:
                        candidates.append(os.path.join(dp, lower["oto.ini"]))
                    base_candidates = [
                        os.path.join(dp, fn)
                        for fn in fns
                        if fn.lower().endswith(".ini") and ("oto" in fn.lower() or "base" in fn.lower())
                    ]
                    candidates.extend(sorted(base_candidates))
                    for candidate in candidates:
                        if _has_usable_oto_lines(candidate):
                            manual = candidate
                            break
                    wavs = [fn for fn in fns if fn.lower().endswith(".wav")]
                    if manual and wavs:
                        items.append(
                            PreparedAutoPair(
                                language=language,
                                format_type=format_type,
                                stage_root=vb_root,
                                work_dir=dp,
                                manual_oto=manual,
                            )
                        )
    return items


__all__ = ["_discover_work_items", "_has_textgrid_files", "_has_usable_oto_lines"]
