"""
Stage training-source files into the repository-local dataset folder.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, asdict
from typing import Iterable, List

from core.oto_ml_collection import load_training_roots


ALLOWED_EXTENSIONS = {".wav", ".ini", ".lab", ".txt", ".yaml", ".yml", ".map"}
TEXT_NAME_HINTS = ("custom", "phoneme", "phone", "character", "dict", "reclist", "prefix")


@dataclass
class StagedVoicebank:
    language: str
    format_type: str
    source_root: str
    stage_root: str
    copied_files: int = 0
    copied_wavs: int = 0
    copied_otos: int = 0
    status: str = "pending"
    reason: str = ""


def _should_copy(path: str) -> bool:
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1]
    if ext not in ALLOWED_EXTENSIONS:
        return False
    if ext == ".wav" or ext == ".lab":
        return True
    if ext == ".ini":
        return "oto" in name or "base" in name
    if ext == ".map":
        return name == "prefix.map"
    if ext in {".txt", ".yaml", ".yml"}:
        return any(hint in name for hint in TEXT_NAME_HINTS)
    return False


def _safe_name(text: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in text)
    return out.strip().rstrip(".") or "voicebank"


def stage_training_sources(config_path: str, dataset_root: str) -> List[StagedVoicebank]:
    roots = load_training_roots(config_path)
    results: List[StagedVoicebank] = []
    for language, groups in roots.items():
        for format_type, root_list in groups.items():
            for root in root_list:
                stage_root = os.path.join(dataset_root, language, format_type, _safe_name(os.path.basename(root)))
                item = StagedVoicebank(
                    language=language,
                    format_type=format_type,
                    source_root=root,
                    stage_root=stage_root,
                )
                if not os.path.isdir(root):
                    item.status = "skip"
                    item.reason = "missing_root"
                    results.append(item)
                    continue
                if os.path.exists(stage_root):
                    shutil.rmtree(stage_root)
                os.makedirs(stage_root, exist_ok=True)
                copied = 0
                copied_wavs = 0
                copied_otos = 0
                for dp, dns, fns in os.walk(root):
                    for fn in fns:
                        src = os.path.join(dp, fn)
                        if not _should_copy(src):
                            continue
                        rel = os.path.relpath(src, root)
                        dst = os.path.join(stage_root, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                        copied += 1
                        low = fn.lower()
                        if low.endswith(".wav"):
                            copied_wavs += 1
                        elif low.endswith(".ini"):
                            copied_otos += 1
                item.copied_files = copied
                item.copied_wavs = copied_wavs
                item.copied_otos = copied_otos
                item.status = "staged" if copied > 0 else "skip"
                item.reason = "" if copied > 0 else "no_matching_files"
                results.append(item)
    return results


def write_stage_manifest_json(path: str, rows: Iterable[StagedVoicebank]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)
    return path


def write_stage_manifest_csv(path: str, rows: Iterable[StagedVoicebank]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(StagedVoicebank.__dataclass_fields__.keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path
