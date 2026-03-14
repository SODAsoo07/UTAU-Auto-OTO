from __future__ import annotations

import os
from typing import Callable, Optional

from core.pipeline_status import ML_INFER_FAILED


def resolve_wav_dir_from_tg_folder(tg_folder: str) -> str:
    return os.path.dirname(os.path.abspath(str(tg_folder or "").rstrip("\\/")))


def log_changed_lines(
    log_fn: Optional[Callable[[str], None]],
    tag: str,
    changed: int,
    description: str,
) -> None:
    if changed > 0 and callable(log_fn):
        log_fn(f"{tag} {description}: {changed} lines")


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _log_ml_report_summary(log_fn: Optional[Callable[[str], None]], ml_report: object) -> None:
    if not callable(log_fn) or not isinstance(ml_report, dict):
        return
    route = str(ml_report.get("route", "") or "").strip()
    model_conf = _safe_float(ml_report.get("model_confidence", 0.0), 0.0)
    blank_conf = _safe_float(ml_report.get("blank_confidence", 0.0), 0.0)
    fallback_reason = str(ml_report.get("fallback_reason", "") or "").strip()
    constraint_adjust_count = int(_safe_float(ml_report.get("constraint_adjust_count", 0), 0))
    fallback_used = bool(ml_report.get("fallback_used", False))
    status = str(ml_report.get("status", "") or "").strip()

    details = []
    if route:
        details.append(f"route={route}")
    details.append(f"model_conf={model_conf:.3f}")
    details.append(f"blank_conf={blank_conf:.3f}")
    details.append(f"constraint_adjust={constraint_adjust_count}")
    if status:
        details.append(f"status={status}")
    if fallback_used:
        details.append("fallback=1")
    if fallback_reason:
        details.append(f"reason={fallback_reason}")
    if details:
        log_fn("[OTO-ML] report " + ", ".join(details))


def run_ml_post_stage(
    *,
    language: str,
    out_path: str,
    tg_folder: str,
    custom_phonemes_path: str,
    enable_ml_correction: bool,
    format_override: str,
    ml_policy,
    runtime_report,
    log_fn: Optional[Callable[[str], None]],
    ml_route: str = "",
) -> int:
    wav_dir = resolve_wav_dir_from_tg_folder(tg_folder)
    try:
        from core.oto_ml_refiner import apply_oto_ml_to_oto_file

        ml_report = {}
        ml_changed = apply_oto_ml_to_oto_file(
            str(language or ""),
            out_path,
            tg_dir=tg_folder,
            wav_dir=wav_dir,
            custom_phonemes_path=custom_phonemes_path,
            callback=log_fn,
            enabled=enable_ml_correction,
            format_override=format_override,
            policy=ml_policy,
            report=ml_report,
            ml_route=ml_route,
        )
        if isinstance(runtime_report, dict):
            runtime_report["ml"] = dict(ml_report)
        log_changed_lines(log_fn, "[OTO-ML]", ml_changed, "ML refine changed")
        _log_ml_report_summary(log_fn, ml_report)
        return int(ml_changed or 0)
    except Exception as exc:
        if callable(log_fn):
            log_fn(f"[OTO-ML] ML refine failed: {exc}")
        if isinstance(runtime_report, dict):
            runtime_report["ml"] = {
                "stage": "ml",
                "code": ML_INFER_FAILED,
                "message": str(exc),
                "status": "fallback",
                "fallback_used": True,
                "policy": str(ml_policy or ""),
            }
        return 0


__all__ = [
    "log_changed_lines",
    "resolve_wav_dir_from_tg_folder",
    "run_ml_post_stage",
]
