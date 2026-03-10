"""
Export helpers for distributable OTO ML model bundles.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from typing import Dict, List, Optional

from core.oto_ml_policy import normalize_alias_family


COMMON_BUNDLE_FILES = [
    "feature_schema.json",
    "model_meta.json",
    "eval_summary.json",
]

LIGHTGBM_BUNDLE_FILES = [
    "model_offset.txt",
    "model_cons.txt",
    "model_cutoff.txt",
    "model_pre.txt",
    "model_ovl.txt",
]

COUPLED_BUNDLE_FILES = [
    "coupled_model.pt",
]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bundle_meta(model_dir: str) -> Dict[str, object]:
    meta_path = os.path.join(model_dir, "model_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def required_bundle_files_for_backend(backend: str) -> List[str]:
    backend_norm = str(backend or "").strip().lower()
    if backend_norm == "coupled_nn_v1":
        return list(COMMON_BUNDLE_FILES) + list(COUPLED_BUNDLE_FILES)
    return list(COMMON_BUNDLE_FILES) + list(LIGHTGBM_BUNDLE_FILES)


def validate_bundle_dir(model_dir: str, meta: Optional[Dict[str, object]] = None) -> List[str]:
    if meta is None:
        try:
            meta = read_bundle_meta(model_dir)
        except Exception:
            meta = {}
    required_files = required_bundle_files_for_backend((meta or {}).get("backend", ""))
    missing = []
    for name in required_files:
        if not os.path.isfile(os.path.join(model_dir, name)):
            missing.append(name)
    return missing


def resolve_bundle_dir(asset_root: str, bundle_spec: str) -> str:
    cleaned = str(bundle_spec or "").strip().replace("/", os.sep).replace("\\", os.sep)
    return os.path.abspath(os.path.join(asset_root, cleaned))


def _bundle_routing_meta(model_dir: str, meta: Dict[str, object]) -> Dict[str, str]:
    language = str(meta.get("language", "")).strip().lower() or "unknown"
    format_type = str(meta.get("format_type", "")).strip().lower() or "unknown"
    alias_family = normalize_alias_family(
        meta.get("alias_family", "") or (meta.get("filters") or {}).get("alias_family", "")
    )
    model_version = str(meta.get("model_version", "")).strip() or (os.path.basename(os.path.abspath(model_dir)) or "v1")
    return {
        "language": language,
        "format_type": format_type,
        "alias_family": alias_family,
        "model_version": model_version,
    }


def bundle_install_subdir(model_dir: str, meta: Dict[str, object]) -> str:
    route = _bundle_routing_meta(model_dir, meta)
    parts = [route["language"], route["format_type"]]
    if route["alias_family"]:
        parts.extend(["families", route["alias_family"]])
    parts.append(route["model_version"])
    return os.path.join(*parts)


def _default_bundle_slug(model_dir: str, meta: Dict[str, object]) -> str:
    route = _bundle_routing_meta(model_dir, meta)
    family_part = f"_{route['alias_family']}" if route["alias_family"] else ""
    return f"{route['language']}_{route['format_type']}{family_part}_{route['model_version']}"


def export_model_bundle(
    model_dir: str,
    export_root: str,
    bundle_slug: str = "",
    create_zip: bool = False,
) -> Dict[str, object]:
    model_dir = os.path.abspath(model_dir)
    export_root = os.path.abspath(export_root)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(model_dir)

    meta = read_bundle_meta(model_dir)
    required_files = required_bundle_files_for_backend(meta.get("backend", ""))
    missing = validate_bundle_dir(model_dir, meta=meta)
    if missing:
        raise FileNotFoundError(f"Missing required bundle files in {model_dir}: {', '.join(missing)}")

    route = _bundle_routing_meta(model_dir, meta)
    slug = bundle_slug.strip() or _default_bundle_slug(model_dir, meta)
    out_dir = os.path.join(export_root, slug)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    files: List[Dict[str, object]] = []
    for name in required_files:
        src = os.path.join(model_dir, name)
        dst = os.path.join(out_dir, name)
        shutil.copy2(src, dst)
        files.append({
            "name": name,
            "size": int(os.path.getsize(dst)),
            "sha256": _file_sha256(dst),
        })

    manifest = {
        "bundle_slug": slug,
        "source_dir": model_dir,
        "export_dir": out_dir,
        "language": route["language"],
        "format_type": route["format_type"],
        "alias_family": route["alias_family"],
        "backend": meta.get("backend", ""),
        "model_version": route["model_version"],
        "feature_version": meta.get("feature_version", ""),
        "train_rows": meta.get("train_rows", 0),
        "voicebank_count": meta.get("voicebank_count", 0),
        "required_files": required_files,
        "install_subdir": bundle_install_subdir(model_dir, meta).replace("\\", "/"),
        "files": files,
    }
    manifest_path = os.path.join(out_dir, "bundle_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    zip_path = ""
    if create_zip:
        zip_path = os.path.join(export_root, f"{slug}.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fn in required_files + ["bundle_manifest.json"]:
                abs_path = os.path.join(out_dir, fn)
                zf.write(abs_path, arcname=os.path.join(slug, fn))
        manifest["zip_path"] = zip_path
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def export_multiple_model_bundles(
    model_dirs: List[str],
    export_root: str,
    create_zip: bool = False,
) -> Dict[str, object]:
    export_root = os.path.abspath(export_root)
    os.makedirs(export_root, exist_ok=True)
    exports = []
    for model_dir in model_dirs:
        exports.append(export_model_bundle(model_dir, export_root, create_zip=create_zip))
    summary = {
        "export_root": export_root,
        "bundle_count": len(exports),
        "bundles": exports,
    }
    summary_path = os.path.join(export_root, "export_manifest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
