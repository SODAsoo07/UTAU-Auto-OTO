"""Batch preparation of auto OTO/TextGrid assets for staged training sources."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Callable, Dict, List, Optional

from core.mfa_runner import find_mfa_executable, run_mfa_align
from core.oto_ml_prepare_discovery import (
    _discover_work_items,
    _has_textgrid_files,
    _has_usable_oto_lines,
)
from core.oto_ml_prepare_steps import _generate_auto_oto, _prepare_lab_and_dict
from core.oto_ml_prepare_types import PreparedAutoPair

_PREPARED_DONE_STATUSES = {"prepared", "prepared_existing"}


def _normalize_prepare_item_key(work_dir: str) -> str:
    return os.path.normcase(os.path.abspath(str(work_dir or "")))


def _coerce_prepared_item(raw: object) -> Optional[PreparedAutoPair]:
    if isinstance(raw, PreparedAutoPair):
        return raw
    if not isinstance(raw, dict):
        return None
    field_names = PreparedAutoPair.__dataclass_fields__.keys()
    payload = {name: raw.get(name, "") for name in field_names}
    try:
        return PreparedAutoPair(**payload)
    except Exception:
        return None


def _atomic_write_json(path: str, payload: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def load_prepare_report(path: str) -> Dict[str, object]:
    if not path or not os.path.isfile(path):
        return {"summary": {}, "items": [], "logs": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
    except Exception:
        return {"summary": {}, "items": [], "logs": {}}

    items = []
    for raw_item in payload.get("items", []) or []:
        item = _coerce_prepared_item(raw_item)
        if item is not None:
            items.append(item)

    logs = payload.get("logs", {})
    if not isinstance(logs, dict):
        logs = {}

    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    return {
        "summary": summary,
        "items": items,
        "logs": logs,
    }


def _build_prepare_summary(
    *,
    total_items: int,
    items: List[PreparedAutoPair],
    dry_run: bool,
    mfa_path: str,
    resume: bool,
    retry_failed: bool,
    resume_skipped: int,
) -> Dict[str, object]:
    prepared = sum(1 for item in items if item.status in _PREPARED_DONE_STATUSES)
    skipped = sum(1 for item in items if item.status == "skip")
    dry_run_items = sum(1 for item in items if item.status == "dry_run")
    processed = len(items)
    pending = max(int(total_items) - processed, 0)
    return {
        "total_items": int(total_items),
        "prepared": int(prepared),
        "skipped": int(skipped),
        "dry_run_items": int(dry_run_items),
        "pending": int(pending),
        "dry_run": bool(dry_run),
        "mfa_path": str(mfa_path or ""),
        "resume": bool(resume),
        "retry_failed": bool(retry_failed),
        "resume_skipped": int(resume_skipped),
    }


def _write_prepare_checkpoint(
    path: str,
    *,
    total_items: int,
    items: List[PreparedAutoPair],
    logs: Dict[str, List[str]],
    dry_run: bool,
    mfa_path: str,
    resume: bool,
    retry_failed: bool,
    resume_skipped: int,
) -> str:
    summary = _build_prepare_summary(
        total_items=total_items,
        items=items,
        dry_run=dry_run,
        mfa_path=mfa_path,
        resume=resume,
        retry_failed=retry_failed,
        resume_skipped=resume_skipped,
    )
    payload = {
        "summary": summary,
        "items": [asdict(item) for item in items],
        "logs": logs,
    }
    return _atomic_write_json(path, payload)


def _should_retry_mfa_with_fallback(err: str) -> bool:
    text = str(err or "").lower()
    if not text:
        return False
    if "missing korean tokenizer dependencies" in text:
        return False
    if "missing japanese tokenizer dependencies" in text:
        return False
    if "mfa executable not found" in text:
        return False
    return (
        "code: 3221225477" in text
        or "code: -1073741819" in text
        or "access violation" in text
    )


def _preferred_mfa_profiles_for_prepare(language: str) -> list[str]:
    raw = str(os.environ.get("UTOA_ML_PREPARE_MFA_PROFILE", "")).strip().lower()
    valid = ["default", "accurate", "fast"]
    if raw in valid:
        return [raw] + [p for p in valid if p != raw]

    lang = str(language or "").strip().lower()
    if lang == "korean":
        return ["default", "accurate", "fast"]
    return ["accurate", "default", "fast"]


def _run_mfa_align_with_fallbacks(
    *,
    mfa_path: str,
    item: PreparedAutoPair,
    logs: List[str],
) -> tuple[bool, str, str]:
    profiles = _preferred_mfa_profiles_for_prepare(item.language)
    last_err = ""

    for idx, profile in enumerate(profiles):
        if idx > 0 and not _should_retry_mfa_with_fallback(last_err):
            break
        logs.append(f"[MFA] Align attempt profile={profile}")
        ok, err = run_mfa_align(
            mfa_path=mfa_path,
            wav_folder=item.work_dir,
            dict_path=item.dict_path,
            output_folder=item.tg_dir,
            language=item.language,
            callback=logs.append,
            align_profile=profile,
        )
        if ok:
            return True, "", profile
        last_err = err
        logs.append(f"[MFA] Align attempt failed profile={profile}: {err}")

    return False, last_err, ""


def _prepare_one_item(item: PreparedAutoPair, mfa_path: str) -> tuple[PreparedAutoPair, List[str]]:
    logs: List[str] = []
    item.mfa_path = mfa_path
    item.tg_dir = item.work_dir
    item.auto_oto = os.path.join(item.work_dir, "oto_auto_ml.ini")
    item.dict_path = os.path.join(item.work_dir, "dictionary_auto.txt")

    if _has_textgrid_files(item.tg_dir):
        try:
            _generate_auto_oto(item, logs)
            item.status = "prepared"
            item.reason = "textgrid_existing"
        except Exception as exc:
            item.status = "skip"
            item.reason = f"auto_oto_failed:{exc}"
        return item, logs

    if not mfa_path:
        item.status = "skip"
        item.reason = "missing_mfa"
        return item, logs

    try:
        _prepare_lab_and_dict(item, logs)
        item.tg_dir = item.work_dir
        os.makedirs(item.tg_dir, exist_ok=True)
        ok, err, used_profile = _run_mfa_align_with_fallbacks(
            mfa_path=mfa_path,
            item=item,
            logs=logs,
        )
        if not ok:
            item.status = "skip"
            item.reason = f"align_failed:{err}"
            return item, logs
        _generate_auto_oto(item, logs)
        item.status = "prepared"
        if used_profile:
            item.reason = f"align_profile:{used_profile}"
        return item, logs
    except Exception as exc:
        item.status = "skip"
        item.reason = f"exception:{exc}"
        return item, logs


def prepare_staged_auto_pairs(
    dataset_root: str,
    dry_run: bool = False,
    limit: int = 0,
    progress_callback: Callable[[str], None] | None = None,
    report_path: str = "",
    resume: bool = False,
    retry_failed: bool = False,
    workers: int = 0,
) -> Dict[str, object]:
    items = _discover_work_items(dataset_root)
    if limit > 0:
        items = items[:limit]
    report_path = os.path.abspath(
        report_path or os.path.join(dataset_root, "_manifest", "prepared_auto_pairs.json")
    )
    if workers and workers > 1:
        os.environ.setdefault("UTOA_MFA_ROOT_DIR_MODE", "per_process")
    mfa_path = find_mfa_executable() or ""
    results: List[PreparedAutoPair] = []
    logs_by_item: Dict[str, List[str]] = {}
    resume_skipped = 0
    previous_payload = load_prepare_report(report_path) if resume else {"items": [], "logs": {}}
    previous_items = previous_payload.get("items", []) or []
    previous_logs = previous_payload.get("logs", {}) or {}
    previous_by_key = {
        _normalize_prepare_item_key(item.work_dir): item
        for item in previous_items
        if item.work_dir
    }

    def emit(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    emit(
        f"[Prepare] start: total={len(items)} dry_run={bool(dry_run)} "
        f"mfa={'OK' if mfa_path else 'MISSING'} resume={bool(resume)}"
    )

    work_items: List[tuple[str, PreparedAutoPair, List[str]]] = []

    for index, item in enumerate(items, start=1):
        key = os.path.relpath(item.work_dir, dataset_root)
        prev_item = previous_by_key.get(_normalize_prepare_item_key(item.work_dir))
        prev_logs: List[str] = []
        if prev_item is not None and isinstance(previous_logs, dict):
            prev_key = os.path.relpath(prev_item.work_dir, dataset_root)
            raw_prev_logs = previous_logs.get(prev_key, [])
            if isinstance(raw_prev_logs, list):
                prev_logs = [str(entry) for entry in raw_prev_logs]
        logs: List[str] = list(prev_logs)
        logs_by_item[key] = logs
        item.mfa_path = mfa_path
        item.tg_dir = item.work_dir
        item.auto_oto = os.path.join(item.work_dir, "oto_auto_ml.ini")
        item.dict_path = os.path.join(item.work_dir, "dictionary_auto.txt")
        if prev_item is not None:
            if prev_item.auto_oto:
                item.auto_oto = prev_item.auto_oto
            if prev_item.tg_dir:
                item.tg_dir = prev_item.tg_dir
            if prev_item.dict_path:
                item.dict_path = prev_item.dict_path
            if prev_item.reason:
                item.reason = prev_item.reason
        emit(
            f"[Prepare] ({index}/{len(items)}) {item.language}/{item.format_type} "
            f"{key} start"
        )

        tg_exists = _has_textgrid_files(item.tg_dir)
        auto_oto_exists = _has_usable_oto_lines(item.auto_oto)

        if resume and prev_item is not None:
            if (
                prev_item.status in _PREPARED_DONE_STATUSES
                and auto_oto_exists
                and (not tg_exists)
            ):
                item.status = prev_item.status
                item.reason = prev_item.reason
                resume_skipped += 1
                emit(
                    f"[Prepare] ({index}/{len(items)}) {key} checkpoint: "
                    f"{item.status}"
                )
                results.append(item)
                _write_prepare_checkpoint(
                    report_path,
                    total_items=len(items),
                    items=results,
                    logs=logs_by_item,
                    dry_run=dry_run,
                    mfa_path=mfa_path,
                    resume=resume,
                    retry_failed=retry_failed,
                    resume_skipped=resume_skipped,
                )
                continue
            if prev_item.status == "skip" and not retry_failed:
                item.status = "skip"
                item.reason = prev_item.reason
                resume_skipped += 1
                emit(
                    f"[Prepare] ({index}/{len(items)}) {key} skip(previous): "
                    f"{item.reason or 'skip'}"
                )
                results.append(item)
                _write_prepare_checkpoint(
                    report_path,
                    total_items=len(items),
                    items=results,
                    logs=logs_by_item,
                    dry_run=dry_run,
                    mfa_path=mfa_path,
                    resume=resume,
                    retry_failed=retry_failed,
                    resume_skipped=resume_skipped,
                )
                continue

        if dry_run:
            item.status = "dry_run"
            emit(f"[Prepare] ({index}/{len(items)}) {key} dry-run")
            results.append(item)
            _write_prepare_checkpoint(
                report_path,
                total_items=len(items),
                items=results,
                logs=logs_by_item,
                dry_run=dry_run,
                mfa_path=mfa_path,
                resume=resume,
                retry_failed=retry_failed,
                resume_skipped=resume_skipped,
            )
            continue

        if auto_oto_exists:
            item.status = "skip"
            item.reason = "auto_oto_exists"
            emit(f"[Prepare] ({index}/{len(items)}) {key} skip(existing_auto_oto)")
            results.append(item)
            _write_prepare_checkpoint(
                report_path,
                total_items=len(items),
                items=results,
                logs=logs_by_item,
                dry_run=dry_run,
                mfa_path=mfa_path,
                resume=resume,
                retry_failed=retry_failed,
                resume_skipped=resume_skipped,
            )
            continue

        work_items.append((key, item, list(logs)))

    if work_items:
        if workers and workers > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                future_map = {
                    executor.submit(_prepare_one_item, item, mfa_path): (key, prev_logs)
                    for key, item, prev_logs in work_items
                }
                for future in as_completed(future_map):
                    key, prev_logs = future_map[future]
                    item, item_logs = future.result()
                    logs_by_item[key] = list(prev_logs) + list(item_logs or [])
                    if item.status == "prepared" and item.reason == "textgrid_existing":
                        emit(f"[Prepare] {key} textgrid_existing -> auto_oto regenerated")
                    elif item.status == "prepared" and str(item.reason or "").startswith("align_profile:"):
                        profile = str(item.reason).split(":", 1)[-1]
                        emit(f"[Prepare] {key} done (align_profile={profile})")
                    elif item.status == "prepared":
                        emit(f"[Prepare] {key} done")
                    else:
                        emit(f"[Prepare] {key} skip: {item.reason}")
                    results.append(item)
                    _write_prepare_checkpoint(
                        report_path,
                        total_items=len(items),
                        items=results,
                        logs=logs_by_item,
                        dry_run=dry_run,
                        mfa_path=mfa_path,
                        resume=resume,
                        retry_failed=retry_failed,
                        resume_skipped=resume_skipped,
                    )
        else:
            for key, item, prev_logs in work_items:
                item, item_logs = _prepare_one_item(item, mfa_path)
                logs_by_item[key] = list(prev_logs) + list(item_logs or [])
                if item.status == "prepared" and item.reason == "textgrid_existing":
                    emit(f"[Prepare] {key} textgrid_existing -> auto_oto regenerated")
                elif item.status == "prepared" and str(item.reason or "").startswith("align_profile:"):
                    profile = str(item.reason).split(":", 1)[-1]
                    emit(f"[Prepare] {key} done (align_profile={profile})")
                elif item.status == "prepared":
                    emit(f"[Prepare] {key} done")
                else:
                    emit(f"[Prepare] {key} skip: {item.reason}")
                results.append(item)
                _write_prepare_checkpoint(
                    report_path,
                    total_items=len(items),
                    items=results,
                    logs=logs_by_item,
                    dry_run=dry_run,
                    mfa_path=mfa_path,
                    resume=resume,
                    retry_failed=retry_failed,
                    resume_skipped=resume_skipped,
                )
    summary = _build_prepare_summary(
        total_items=len(items),
        items=results,
        dry_run=dry_run,
        mfa_path=mfa_path,
        resume=resume,
        retry_failed=retry_failed,
        resume_skipped=resume_skipped,
    )
    emit(
        f"[Prepare] end: prepared={summary['prepared']} skipped={summary['skipped']} "
        f"pending={summary['pending']} total={summary['total_items']}"
    )

    return {
        "summary": summary,
        "items": results,
        "logs": logs_by_item,
    }


def write_prepare_report(path: str, result: Dict[str, object]) -> str:
    payload = {
        "summary": result.get("summary", {}),
        "items": [asdict(item) for item in result.get("items", [])],
        "logs": result.get("logs", {}),
    }
    return _atomic_write_json(path, payload)
