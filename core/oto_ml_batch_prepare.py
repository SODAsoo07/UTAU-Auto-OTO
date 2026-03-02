"""
Batch preparation of auto OTO/TextGrid assets for staged training sources.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List

from core.ja_lab_generator import generate_ja_dictionary, generate_ja_labs
from core.ja_oto_generator import generate_ja_oto
from core.lab_generator import generate_dictionary, generate_labs
from core.mfa_runner import find_mfa_executable, run_mfa_align
from core.oto_generator import generate_oto


@dataclass
class PreparedAutoPair:
    language: str
    format_type: str
    stage_root: str
    work_dir: str
    manual_oto: str
    auto_oto: str = ""
    tg_dir: str = ""
    dict_path: str = ""
    mfa_path: str = ""
    status: str = "pending"
    reason: str = ""


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
                    base_candidates = [os.path.join(dp, fn) for fn in fns if fn.lower().endswith(".ini") and ("oto" in fn.lower() or "base" in fn.lower())]
                    candidates.extend(sorted(base_candidates))
                    for candidate in candidates:
                        if _has_usable_oto_lines(candidate):
                            manual = candidate
                            break
                    wavs = [fn for fn in fns if fn.lower().endswith(".wav")]
                    if manual and wavs:
                        items.append(PreparedAutoPair(language=language, format_type=format_type, stage_root=vb_root, work_dir=dp, manual_oto=manual))
    return items


def _has_textgrid_files(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and any(fn.lower().endswith(".textgrid") for fn in os.listdir(path))


def _prepare_lab_and_dict(item: PreparedAutoPair, logs: List[str]) -> None:
    if item.language == "japanese":
        generate_ja_labs(item.work_dir, callback=logs.append)
        item.dict_path = os.path.join(item.work_dir, "dictionary_auto.txt")
        generate_ja_dictionary(item.work_dir, item.dict_path, callback=logs.append)
    else:
        generate_labs(item.work_dir, callback=logs.append)
        item.dict_path = os.path.join(item.work_dir, "dictionary_auto.txt")
        generate_dictionary(item.work_dir, item.dict_path, callback=logs.append)


def _generate_auto_oto(item: PreparedAutoPair, logs: List[str]) -> None:
    item.auto_oto = os.path.join(item.work_dir, "oto_auto_ml.ini")
    if item.language == "japanese":
        generate_ja_oto(
            tg_folder=item.tg_dir,
            tpl_path=item.manual_oto,
            out_path=item.auto_oto,
            fallback_format=item.format_type,
            auto_format=item.format_type,
            callback=logs.append,
        )
    else:
        generate_oto(
            tg_folder=item.tg_dir,
            tpl_path=item.manual_oto,
            out_path=item.auto_oto,
            fallback_format=item.format_type,
            auto_format=item.format_type,
            callback=logs.append,
        )


def prepare_staged_auto_pairs(
    dataset_root: str,
    dry_run: bool = False,
    limit: int = 0,
    progress_callback: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    items = _discover_work_items(dataset_root)
    if limit > 0:
        items = items[:limit]
    mfa_path = find_mfa_executable() or ""
    results: List[PreparedAutoPair] = []
    logs_by_item: Dict[str, List[str]] = {}
    summary = {
        "total_items": len(items),
        "prepared": 0,
        "skipped": 0,
        "dry_run": bool(dry_run),
        "mfa_path": mfa_path,
    }

    def emit(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    emit(
        f"[Prepare] 시작: total={len(items)} dry_run={bool(dry_run)} "
        f"mfa={'OK' if mfa_path else 'MISSING'}"
    )

    for index, item in enumerate(items, start=1):
        key = os.path.relpath(item.work_dir, dataset_root)
        logs: List[str] = []
        logs_by_item[key] = logs
        item.mfa_path = mfa_path
        item.tg_dir = os.path.join(item.work_dir, "textgrids_auto")
        item.auto_oto = os.path.join(item.work_dir, "oto_auto_ml.ini")
        item.dict_path = os.path.join(item.work_dir, "dictionary_auto.txt")
        emit(
            f"[Prepare] ({index}/{len(items)}) {item.language}/{item.format_type} "
            f"{key} 처리 시작"
        )
        if dry_run:
            item.status = "dry_run"
            emit(f"[Prepare] ({index}/{len(items)}) {key} dry-run")
            results.append(item)
            continue
        if _has_textgrid_files(item.tg_dir) and _has_usable_oto_lines(item.auto_oto):
            item.status = "prepared_existing"
            summary["prepared"] += 1
            emit(f"[Prepare] ({index}/{len(items)}) {key} 기존 결과 재사용")
            results.append(item)
            continue
        if not mfa_path:
            item.status = "skip"
            item.reason = "missing_mfa"
            summary["skipped"] += 1
            emit(f"[Prepare] ({index}/{len(items)}) {key} 건너뜀: missing_mfa")
            results.append(item)
            continue
        try:
            _prepare_lab_and_dict(item, logs)
            item.tg_dir = os.path.join(item.work_dir, "textgrids_auto")
            os.makedirs(item.tg_dir, exist_ok=True)
            ok, err = run_mfa_align(
                mfa_path=mfa_path,
                wav_folder=item.work_dir,
                dict_path=item.dict_path,
                output_folder=item.tg_dir,
                language=item.language,
                callback=logs.append,
            )
            if not ok:
                item.status = "skip"
                item.reason = f"align_failed:{err}"
                summary["skipped"] += 1
                emit(
                    f"[Prepare] ({index}/{len(items)}) {key} 건너뜀: "
                    f"align_failed:{err}"
                )
                results.append(item)
                continue
            _generate_auto_oto(item, logs)
            item.status = "prepared"
            summary["prepared"] += 1
            emit(f"[Prepare] ({index}/{len(items)}) {key} 완료")
            results.append(item)
        except Exception as exc:
            item.status = "skip"
            item.reason = f"exception:{exc}"
            summary["skipped"] += 1
            emit(f"[Prepare] ({index}/{len(items)}) {key} 예외: {exc}")
            results.append(item)

    emit(
        f"[Prepare] 종료: prepared={summary['prepared']} skipped={summary['skipped']} "
        f"total={summary['total_items']}"
    )

    return {
        "summary": summary,
        "items": results,
        "logs": logs_by_item,
    }


def write_prepare_report(path: str, result: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "summary": result.get("summary", {}),
        "items": [asdict(item) for item in result.get("items", [])],
        "logs": result.get("logs", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
