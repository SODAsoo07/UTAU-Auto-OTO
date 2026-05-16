"""
Install exported OTO ML bundles into a runtime search path.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from typing import Dict, Optional

from core.oto_ml_export import read_bundle_meta, validate_bundle_dir
from core.oto_ml_policy import normalize_alias_family


def default_installed_bundle_root() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "models_installed", "oto_ml")
    os.makedirs(path, exist_ok=True)
    return path


def _build_install_subdir(language: str, format_type: str, model_version: str, alias_family: str = "") -> str:
    parts = [language, format_type]
    if alias_family:
        parts.extend(["families", alias_family])
    parts.append(model_version)
    return os.path.join(*parts)


def _infer_alias_family_from_path(bundle_dir: str, format_type: str) -> str:
    parts = [part.strip().lower() for part in os.path.normpath(bundle_dir).split(os.sep) if part.strip()]
    if "families" in parts:
        idx = parts.index("families")
        if idx + 1 < len(parts):
            return normalize_alias_family(parts[idx + 1])
    prefix = f"{str(format_type or '').strip().lower()}_"
    for part in parts:
        if prefix and part.startswith(prefix):
            return normalize_alias_family(part[len(prefix):])
    return ""


def _bundle_lang_fmt(bundle_dir: str) -> Optional[Dict[str, str]]:
    manifest_path = os.path.join(bundle_dir, "bundle_manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    meta = read_bundle_meta(bundle_dir)
    if not meta:
        return None
    language = str(manifest.get("language", "") or meta.get("language", "")).strip().lower()
    format_type = str(manifest.get("format_type", "") or meta.get("format_type", "")).strip().lower()
    alias_family = normalize_alias_family(
        manifest.get("alias_family", "")
        or meta.get("alias_family", "")
        or (meta.get("filters") or {}).get("alias_family", "")
    )
    model_version = str(manifest.get("model_version", "") or meta.get("model_version", "")).strip() or (
        os.path.basename(os.path.abspath(bundle_dir)) or "v1"
    )
    install_subdir = str(manifest.get("install_subdir", "")).strip().replace("/", os.sep)
    if not install_subdir and language and format_type:
        if not alias_family:
            alias_family = _infer_alias_family_from_path(bundle_dir, format_type)
        install_subdir = _build_install_subdir(language, format_type, model_version, alias_family=alias_family)
    return {
        "language": language,
        "format_type": format_type,
        "alias_family": alias_family,
        "model_version": model_version,
        "install_subdir": install_subdir,
    }


def install_exported_bundle(bundle_source: str, install_root: str = "") -> Dict[str, object]:
    src = os.path.abspath(bundle_source)
    install_root = os.path.abspath(install_root or default_installed_bundle_root())
    unpack_root = ""
    if os.path.isfile(src) and src.lower().endswith(".zip"):
        unpack_root = os.path.join(install_root, "_tmp_unpack")
        if os.path.isdir(unpack_root):
            shutil.rmtree(unpack_root, ignore_errors=True)
        os.makedirs(unpack_root, exist_ok=True)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(unpack_root)
        dirs = [os.path.join(unpack_root, name) for name in os.listdir(unpack_root) if os.path.isdir(os.path.join(unpack_root, name))]
        if not dirs:
            raise FileNotFoundError(f"No bundle directory found in zip: {src}")
        src = dirs[0]
    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    missing = validate_bundle_dir(src)
    if missing:
        raise FileNotFoundError(f"Invalid bundle, missing: {', '.join(missing)}")
    info = _bundle_lang_fmt(src)
    if not info or not info["language"] or not info["format_type"]:
        raise RuntimeError(f"Could not determine language/format for bundle: {src}")
    target_dir = os.path.join(install_root, info["install_subdir"])
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    shutil.copytree(src, target_dir)
    if unpack_root and os.path.isdir(unpack_root):
        shutil.rmtree(unpack_root, ignore_errors=True)
    return {
        "source": bundle_source,
        "installed_dir": target_dir,
        "language": info["language"],
        "format_type": info["format_type"],
        "alias_family": info["alias_family"],
        "model_version": info["model_version"],
    }
