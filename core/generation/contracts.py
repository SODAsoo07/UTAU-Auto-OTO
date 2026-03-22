from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional


@dataclass
class GenerationRequest:
    """Language-agnostic normalized request passed to generators."""

    language: str
    tg_folder: str
    tpl_path: str
    out_path: str
    fallback_format: str
    custom_phonemes_path: str = ""
    alias_suffix: str = ""
    auto_format: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


def initialize_runtime_report(
    runtime_report: Optional[MutableMapping[str, Any]],
    *,
    language: str,
    stage: str = "generate",
) -> Optional[MutableMapping[str, Any]]:
    """Initialize shared runtime report schema in-place if dict-like object is provided."""
    if not isinstance(runtime_report, MutableMapping):
        return None
    runtime_report.setdefault("schema_version", "1.0")
    runtime_report.setdefault("stage", str(stage or "generate").strip().lower())
    runtime_report.setdefault("language", str(language or "").strip().lower())
    runtime_report.setdefault("started_at", dt.datetime.now().isoformat(timespec="seconds"))
    runtime_report.setdefault("code", "OK")
    runtime_report.setdefault("message", "")
    runtime_report.setdefault("errors", [])
    runtime_report.setdefault("details", {})
    return runtime_report


def attach_request_metadata(
    runtime_report: Optional[MutableMapping[str, Any]],
    request: GenerationRequest,
) -> None:
    """Attach normalized request metadata to runtime report."""
    if not isinstance(runtime_report, MutableMapping):
        return
    runtime_report.setdefault(
        "request",
        {
            "language": request.language,
            "tg_folder": request.tg_folder,
            "tpl_path": request.tpl_path,
            "out_path": request.out_path,
            "fallback_format": request.fallback_format,
            "auto_format": request.auto_format,
            "alias_suffix": request.alias_suffix,
            "options": dict(request.options),
        },
    )


def finalize_runtime_report(
    runtime_report: Optional[MutableMapping[str, Any]],
    *,
    processed: int,
    total: int,
    errors: List[str],
) -> None:
    """Finalize shared runtime report schema."""
    if not isinstance(runtime_report, MutableMapping):
        return
    error_list = [str(err) for err in (errors or []) if str(err).strip()]
    runtime_report["processed"] = int(processed or 0)
    runtime_report["total"] = int(total or 0)
    runtime_report["error_count"] = len(error_list)
    runtime_report["errors"] = error_list
    runtime_report["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    if error_list:
        runtime_report["code"] = "GENERATOR_ERROR"
        if not str(runtime_report.get("message", "")).strip():
            runtime_report["message"] = "generation finished with errors"
    else:
        runtime_report["code"] = "OK"
        if not str(runtime_report.get("message", "")).strip():
            runtime_report["message"] = "generation complete"
