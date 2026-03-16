from __future__ import annotations

import os
from typing import Callable, Optional

from core.pipeline_status import ML_APPLIED, ML_INFER_FAILED

_ML_LOADING_NOTICE = (
    "[OTO-ML] Loading correction models. "
    "Coupled+ML mode may take longer on the first run; please wait."
)


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


def _normalize_ml_route(value: str) -> str:
    _ = str(value or "").strip().lower()
    return "legacy"


def _join_reason(*items: object) -> str:
    out = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return "; ".join(out)


def _compose_combined_ml_report(
    *,
    route_name: str,
    legacy_report: dict,
    legacy_changed: int,
) -> dict:
    combined: dict = {
        "stage": "ml",
        "ml_route": route_name,
        "route": "legacy",
        "status": "skipped",
        "fallback_used": False,
        "policy": str(legacy_report.get("policy", "") or ""),
        "code": str(legacy_report.get("code", "") or ""),
        "message": str(legacy_report.get("message", "") or ""),
        "changed_lines": int(max(0, legacy_changed)),
        "model_confidence": _safe_float(legacy_report.get("model_confidence", 0.0), 0.0),
        "blank_confidence": _safe_float(legacy_report.get("blank_confidence", 0.0), 0.0),
        "constraint_adjust_count": int(_safe_float(legacy_report.get("constraint_adjust_count", 0), 0)),
        "fallback_reason": str(legacy_report.get("fallback_reason", "") or ""),
        "legacy_report": dict(legacy_report),
    }
    combined["fallback_used"] = bool(combined.get("fallback_used", False)) or bool(legacy_report.get("fallback_used", False))

    if combined["changed_lines"] > 0:
        combined["status"] = "applied"
        combined["code"] = ML_APPLIED
        combined["message"] = f"{combined['changed_lines']} lines adjusted"
        return combined

    statuses = [str(legacy_report.get("status", "") or "")]

    if any(status == "fallback" for status in statuses):
        combined["status"] = "fallback"
        if not str(combined.get("code", "")).strip():
            combined["code"] = str(legacy_report.get("code", "") or "")
        if not str(combined.get("message", "")).strip():
            combined["message"] = _join_reason(legacy_report.get("message", ""))
        return combined

    if any(status == "no_change" for status in statuses):
        combined["status"] = "no_change"
        combined["code"] = ML_APPLIED
        combined["message"] = "변경이 필요하지 않았습니다."
        return combined

    combined["status"] = "skipped"
    combined["code"] = str(legacy_report.get("code", "") or "")
    combined["message"] = str(legacy_report.get("message", "") or "")
    return combined


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
    route_name = _normalize_ml_route(ml_route or os.environ.get("UTOA_ML_ROUTE", "legacy"))
    legacy_report = {}
    legacy_changed = 0
    if callable(log_fn):
        log_fn(_ML_LOADING_NOTICE)
    try:
        from core.oto_ml_refiner import apply_oto_ml_to_oto_file

        legacy_changed = int(
            apply_oto_ml_to_oto_file(
                str(language or ""),
                out_path,
                tg_dir=tg_folder,
                wav_dir=wav_dir,
                custom_phonemes_path=custom_phonemes_path,
                callback=log_fn,
                enabled=enable_ml_correction,
                format_override=format_override,
                policy=ml_policy,
                report=legacy_report,
                ml_route="legacy",
            )
            or 0
        )
        log_changed_lines(log_fn, "[OTO-ML]", legacy_changed, "legacy ML refine changed")

        ml_report = _compose_combined_ml_report(
            route_name=route_name,
            legacy_report=legacy_report if isinstance(legacy_report, dict) else {},
            legacy_changed=int(legacy_changed),
        )
        if isinstance(runtime_report, dict):
            runtime_report["ml"] = dict(ml_report)
        _log_ml_report_summary(log_fn, ml_report)
        return int(ml_report.get("changed_lines", 0) or 0)
    except Exception as exc:
        if callable(log_fn):
            log_fn(f"[OTO-ML] ML refine failed: {exc}")
        if isinstance(runtime_report, dict):
            runtime_report["ml"] = {
                "stage": "ml",
                "route": route_name,
                "code": ML_INFER_FAILED,
                "message": str(exc),
                "status": "fallback",
                "fallback_used": True,
                "policy": str(ml_policy or ""),
                "legacy_report": dict(legacy_report) if isinstance(legacy_report, dict) else {},
            }
        return int(max(0, legacy_changed))


__all__ = [
    "log_changed_lines",
    "resolve_wav_dir_from_tg_folder",
    "run_ml_post_stage",
]
