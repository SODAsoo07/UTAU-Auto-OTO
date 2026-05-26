from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VoicebankBatchTarget:
    wav_dir: str
    out_path: str
    rel_path: str
    display_name: str
    index: int
    total: int


def has_top_level_wavs(folder_path: str) -> bool:
    try:
        return any(str(name).lower().endswith(".wav") for name in os.listdir(folder_path))
    except Exception:
        return False


def discover_recursive_voicebank_dirs(root_dir: str) -> list[str]:
    root = os.path.abspath(str(root_dir or "").strip())
    if not root or not os.path.isdir(root):
        return []

    candidates: list[str] = []
    for cur_root, _dirs, files in os.walk(root):
        if any(str(name).lower().endswith(".wav") for name in files):
            candidates.append(os.path.abspath(cur_root))

    if not candidates:
        return []

    deduped = sorted(set(candidates), key=lambda path: (len(path), path.lower()))
    leaf_only: list[str] = []
    for current in deduped:
        current_norm = os.path.normcase(os.path.abspath(current)).rstrip("\\/")
        child_prefix = current_norm + os.sep
        has_child_candidate = False
        for other in deduped:
            if other == current:
                continue
            other_norm = os.path.normcase(os.path.abspath(other))
            if other_norm.startswith(child_prefix):
                has_child_candidate = True
                break
        if not has_child_candidate:
            leaf_only.append(current)

    return leaf_only or deduped


def resolve_recursive_out_path(
    root_wav_dir: str,
    base_out_path: str,
    target_wav_dir: str,
    target_count: int,
) -> str:
    target_abs = os.path.abspath(str(target_wav_dir or "").strip())
    out_raw = str(base_out_path or "").strip()
    if target_count <= 1:
        if out_raw:
            if os.path.isdir(out_raw) or out_raw.endswith(("\\", "/")):
                return os.path.join(os.path.abspath(out_raw), "oto.ini")
            return out_raw
        return os.path.join(target_abs, "oto.ini")

    if not out_raw:
        return os.path.join(target_abs, "oto.ini")

    out_abs = os.path.abspath(out_raw)
    out_ext = str(os.path.splitext(out_abs)[1] or "").lower()
    if out_ext == ".ini":
        out_root = os.path.dirname(out_abs)
        out_name = os.path.basename(out_abs) or "oto.ini"
    else:
        out_root = out_abs
        out_name = "oto.ini"
    root_abs = os.path.abspath(str(root_wav_dir or "").strip())
    rel = ""
    try:
        rel = os.path.relpath(target_abs, root_abs)
    except Exception:
        rel = ""
    if not rel or rel.startswith(".."):
        rel = os.path.basename(target_abs)
    if rel in {".", ""}:
        return os.path.join(out_root, out_name)
    return os.path.join(out_root, rel, out_name)


def resolve_voicebank_batch_targets(
    root_wav_dir: str,
    base_out_path: str = "",
    *,
    batch_scan_enabled: bool = False,
) -> list[VoicebankBatchTarget]:
    root_abs = os.path.abspath(str(root_wav_dir or "").strip())
    if not root_abs or not os.path.isdir(root_abs):
        return []

    if batch_scan_enabled:
        wav_dirs = discover_recursive_voicebank_dirs(root_abs)
    else:
        wav_dirs = [root_abs]

    if not wav_dirs:
        return []

    total = len(wav_dirs)
    targets: list[VoicebankBatchTarget] = []
    for idx, wav_dir in enumerate(wav_dirs):
        wav_abs = os.path.abspath(str(wav_dir or "").strip())
        try:
            rel_path = os.path.relpath(wav_abs, root_abs)
        except Exception:
            rel_path = os.path.basename(wav_abs)
        if not rel_path or rel_path.startswith(".."):
            rel_path = os.path.basename(wav_abs)
        if rel_path in {".", ""}:
            rel_path = os.path.basename(wav_abs) or "."
        targets.append(
            VoicebankBatchTarget(
                wav_dir=wav_abs,
                out_path=resolve_recursive_out_path(root_abs, base_out_path, wav_abs, total),
                rel_path=rel_path,
                display_name=rel_path if rel_path not in {".", ""} else os.path.basename(wav_abs),
                index=idx,
                total=total,
            )
        )
    return targets


__all__ = [
    "VoicebankBatchTarget",
    "discover_recursive_voicebank_dirs",
    "has_top_level_wavs",
    "resolve_recursive_out_path",
    "resolve_voicebank_batch_targets",
]
