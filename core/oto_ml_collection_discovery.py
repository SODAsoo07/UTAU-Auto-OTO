from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

import yaml

from core.oto_ml_collection_types import TrainingCandidate


MANUAL_NAME_PRIORITY = [
    "oto.ini",
    "base_oto.ini",
    "krcvvc_base.ini",
    "krcvc_base.ini",
]

AUTO_KEYWORDS = ["autotest", "auto", "tuned", "generated", "output", "out"]
TEXTGRID_DIR_NAMES = ["textgrids", "textgrid"]


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


def _norm_name(path: str) -> str:
    return os.path.basename(path).strip().lower()


def _load_yaml(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_training_roots(config_path: str) -> Dict[str, Dict[str, List[str]]]:
    raw = _load_yaml(config_path)
    out: Dict[str, Dict[str, List[str]]] = {}
    allowed_languages = {"korean", "japanese"}
    for language, groups in raw.items():
        if str(language) not in allowed_languages:
            continue
        if not isinstance(groups, dict):
            continue
        lang_groups: Dict[str, List[str]] = {}
        for format_type, roots in groups.items():
            if not isinstance(roots, list):
                continue
            lang_groups[str(format_type)] = [str(root) for root in roots if str(root).strip()]
        if lang_groups:
            out[str(language)] = lang_groups
    return out


def _find_custom_phoneme_map(root: str) -> str:
    for name in ("custom_phonemes.txt", "custom_phones.txt", "phonemes.txt"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    return ""


def _find_custom_phoneme_map_chain(*roots: str) -> str:
    for root in roots:
        if root:
            path = _find_custom_phoneme_map(root)
            if path:
                return path
    return ""


def _find_textgrid_dir(base: str) -> str:
    for dir_name in TEXTGRID_DIR_NAMES:
        candidate = os.path.join(base, dir_name)
        if os.path.isdir(candidate) and any(fn.lower().endswith(".textgrid") for fn in os.listdir(candidate)):
            return candidate
    found: Dict[str, int] = {}
    for dp, dns, fns in os.walk(base):
        count = sum(1 for fn in fns if fn.lower().endswith(".textgrid"))
        if count > 0:
            found[dp] = count
    if not found:
        return ""
    return max(found.items(), key=lambda item: item[1])[0]


def _find_wav_dir(base: str, prefer_dir: str = "") -> str:
    counts: Dict[str, int] = {}
    for dp, dns, fns in os.walk(base):
        count = sum(1 for fn in fns if fn.lower().endswith(".wav"))
        if count > 0:
            counts[dp] = count
    if not counts:
        return ""
    if prefer_dir and prefer_dir in counts:
        return prefer_dir
    return max(counts.items(), key=lambda item: item[1])[0]


def _score_manual_name(path: str) -> Tuple[int, int]:
    name = _norm_name(path)
    if name in MANUAL_NAME_PRIORITY:
        return (100 - MANUAL_NAME_PRIORITY.index(name), -len(path))
    if name == "oto.ini":
        return (99, -len(path))
    if "base" in name:
        return (95, -len(path))
    return (10, -len(path))


def _score_auto_name(path: str) -> Tuple[int, int]:
    name = _norm_name(path)
    score = 0
    for idx, kw in enumerate(AUTO_KEYWORDS):
        if kw in name:
            score = max(score, 100 - idx)
    if score == 0:
        return (0, -len(path))
    return (score, -len(path))


def _discover_oto_files(base: str) -> List[str]:
    found = []
    for dp, dns, fns in os.walk(base):
        for fn in fns:
            low = fn.lower()
            if low.endswith(".ini") and ("oto" in low or "base" in low):
                full = os.path.join(dp, fn)
                if _has_usable_oto_lines(full):
                    found.append(full)
    return found


def _pick_pair(oto_files: List[str]) -> Tuple[str, str]:
    if not oto_files:
        return "", ""
    manual = ""
    auto = ""
    manual_candidates = sorted(oto_files, key=_score_manual_name, reverse=True)
    auto_candidates = [p for p in oto_files if _score_auto_name(p)[0] > 0]
    auto_candidates.sort(key=_score_auto_name, reverse=True)
    if manual_candidates:
        manual = manual_candidates[0]
    for candidate in auto_candidates:
        if os.path.abspath(candidate) != os.path.abspath(manual):
            auto = candidate
            break
    return manual, auto


def _build_candidate(
    language: str,
    format_type: str,
    root: str,
    folder: str,
    manual: str,
    auto: str,
    tg_dir: str,
    wav_dir: str,
    custom_phonemes: str,
    voicebank_id: str,
) -> TrainingCandidate:
    candidate = TrainingCandidate(
        language=language,
        format_type=format_type,
        root=root,
        folder=folder,
        manual_oto=manual,
        auto_oto=auto,
        tg_dir=tg_dir,
        wav_dir=wav_dir,
        custom_phonemes=custom_phonemes,
        voicebank_id=voicebank_id,
    )
    missing = []
    if not manual:
        missing.append("manual_oto")
    if not auto:
        missing.append("auto_oto")
    if not tg_dir:
        missing.append("textgrids")
    if not wav_dir:
        missing.append("wavs")
    if missing:
        candidate.status = "skip"
        candidate.reason = "missing_" + "_".join(missing)
    else:
        candidate.status = "ready"
    return candidate


def discover_training_candidates(config_path: str) -> List[TrainingCandidate]:
    roots = load_training_roots(config_path)
    candidates: List[TrainingCandidate] = []
    for language, groups in roots.items():
        for format_type, root_list in groups.items():
            for root in root_list:
                if not os.path.isdir(root):
                    candidates.append(TrainingCandidate(language, format_type, root, root, status="skip", reason="missing_root"))
                    continue
                oto_files = _discover_oto_files(root)
                if not oto_files:
                    candidates.append(TrainingCandidate(language, format_type, root, root, status="skip", reason="missing_oto"))
                    continue
                manual, auto = _pick_pair(oto_files)
                tg_dir = _find_textgrid_dir(root)
                wav_dir = _find_wav_dir(root, prefer_dir=os.path.dirname(manual) if manual else "")
                candidates.append(
                    _build_candidate(
                        language=language,
                        format_type=format_type,
                        root=root,
                        folder=root,
                        manual=manual,
                        auto=auto,
                        tg_dir=tg_dir,
                        wav_dir=wav_dir,
                        custom_phonemes=_find_custom_phoneme_map(root),
                        voicebank_id=os.path.basename(os.path.abspath(root)),
                    )
                )
    return candidates


def discover_training_candidates_from_dataset_root(dataset_root: str) -> List[TrainingCandidate]:
    candidates: List[TrainingCandidate] = []
    for language in ("korean", "japanese"):
        lang_root = os.path.join(dataset_root, language)
        if not os.path.isdir(lang_root):
            continue
        for format_type in sorted(os.listdir(lang_root)):
            fmt_root = os.path.join(lang_root, format_type)
            if not os.path.isdir(fmt_root):
                continue
            for voicebank in sorted(os.listdir(fmt_root)):
                vb_root = os.path.join(fmt_root, voicebank)
                if not os.path.isdir(vb_root):
                    continue
                for dp, dns, fns in os.walk(vb_root):
                    wavs = [fn for fn in fns if fn.lower().endswith(".wav")]
                    if not wavs:
                        continue
                    ini_files = []
                    for fn in fns:
                        low = fn.lower()
                        if low.endswith(".ini") and ("oto" in low or "base" in low):
                            full = os.path.join(dp, fn)
                            if _has_usable_oto_lines(full):
                                ini_files.append(full)
                    if not ini_files:
                        continue
                    manual, auto = _pick_pair(ini_files)
                    tg_dir = _find_textgrid_dir(dp)
                    wav_dir = _find_wav_dir(dp, prefer_dir=dp)
                    candidates.append(
                        _build_candidate(
                            language=language,
                            format_type=format_type,
                            root=vb_root,
                            folder=dp,
                            manual=manual,
                            auto=auto,
                            tg_dir=tg_dir,
                            wav_dir=wav_dir,
                            custom_phonemes=_find_custom_phoneme_map_chain(dp, vb_root),
                            voicebank_id=os.path.basename(vb_root),
                        )
                    )
    return candidates


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def write_candidate_manifest(path: str, candidates: Iterable[TrainingCandidate]) -> str:
    _ensure_parent(path)
    rows = [asdict(c) for c in candidates]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path


def write_candidate_csv(path: str, candidates: Iterable[TrainingCandidate]) -> str:
    _ensure_parent(path)
    rows = [asdict(c) for c in candidates]
    fieldnames = list(TrainingCandidate.__dataclass_fields__.keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


__all__ = [
    "TrainingCandidate",
    "discover_training_candidates",
    "discover_training_candidates_from_dataset_root",
    "load_training_roots",
    "write_candidate_csv",
    "write_candidate_manifest",
]
