"""Batch preparation of auto OTO/TextGrid assets for staged training sources."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable, Dict, List

from core.mfa_runner import find_mfa_executable, run_mfa_align
from core.oto_ml_prepare_discovery import (
    _discover_work_items,
    _has_textgrid_files,
    _has_usable_oto_lines,
)
from core.oto_ml_prepare_steps import _generate_auto_oto, _prepare_lab_and_dict
from core.oto_ml_prepare_types import PreparedAutoPair


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
