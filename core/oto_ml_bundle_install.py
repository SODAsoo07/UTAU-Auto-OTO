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


def default_installed_bundle_root() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "models_installed", "oto_ml")
    os.makedirs(path, exist_ok=True)
    return path


def _bundle_lang_fmt(bundle_dir: str) -> Optional[Dict[str, str]]:
    manifest_path = os.path.join(bundle_dir, "bundle_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return {
            "language": str(obj.get("language", "")).strip().lower(),
            "format_type": str(obj.get("format_type", "")).strip().lower(),
        }
    meta = read_bundle_meta(bundle_dir)
    if not meta:
        return None
    return {
        "language": str(meta.get("language", "")).strip().lower(),
        "format_type": str(meta.get("format_type", "")).strip().lower(),
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
    target_dir = os.path.join(install_root, info["language"], info["format_type"], "v1")
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
    }
