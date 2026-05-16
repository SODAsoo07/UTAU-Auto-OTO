from __future__ import annotations

import os
import sys


def _normalize_nonempty(path: str) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    try:
        return os.path.abspath(value)
    except Exception:
        return ""


def _dedupe_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in paths:
        normalized = _normalize_nonempty(candidate)
        if not normalized:
            continue
        key = os.path.normcase(normalized)
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _collect_runtime_root_candidates(
    *,
    app_dir: str = "",
    app_data_dir: str = "",
    writable_data_dir: str = "",
    frozen: bool | None = None,
    executable_path: str = "",
) -> list[str]:
    roots: list[str] = []
    app_root = _normalize_nonempty(app_dir)
    if not app_root:
        app_root = _normalize_nonempty(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if app_root:
        roots.append(app_root)
        roots.append(os.path.dirname(app_root))

    frozen_mode = bool(getattr(sys, "frozen", False) if frozen is None else frozen)
    if frozen_mode:
        exe_path = _normalize_nonempty(executable_path or getattr(sys, "executable", ""))
        if exe_path:
            exe_dir = os.path.dirname(exe_path)
            roots.append(exe_dir)
            roots.append(os.path.dirname(exe_dir))

    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    if local_app_data:
        roots.append(os.path.join(local_app_data, "UTAU_Auto_OTO"))
        roots.append(os.path.join(local_app_data, "UTAU_Auto_OTO_v3"))

    if app_data_dir:
        roots.append(app_data_dir)
    if writable_data_dir:
        roots.append(writable_data_dir)

    try:
        roots.append(os.getcwd())
    except Exception:
        pass

    return _dedupe_paths(roots)


def resolve_setup_script_path(
    script_name: str,
    *,
    app_dir: str = "",
    app_data_dir: str = "",
    writable_data_dir: str = "",
    frozen: bool | None = None,
    executable_path: str = "",
) -> str:
    script = str(script_name or "").strip()
    if not script:
        return ""
    candidates: list[str] = []
    for root in _collect_runtime_root_candidates(
        app_dir=app_dir,
        app_data_dir=app_data_dir,
        writable_data_dir=writable_data_dir,
        frozen=frozen,
        executable_path=executable_path,
    ):
        candidates.append(os.path.join(root, script))
    for candidate in _dedupe_paths(candidates):
        if os.path.isfile(candidate):
            return candidate
    return ""


def resolve_setup_mfa_script_path(**kwargs) -> str:
    return resolve_setup_script_path("setup_mfa.bat", **kwargs)


__all__ = [
    "resolve_setup_script_path",
    "resolve_setup_mfa_script_path",
]
