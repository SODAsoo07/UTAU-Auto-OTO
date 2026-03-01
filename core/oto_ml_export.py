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


REQUIRED_BUNDLE_FILES = [
    "feature_schema.json",
    "model_meta.json",
    "eval_summary.json",
    "model_offset.txt",
    "model_cons.txt",
    "model_cutoff.txt",
    "model_pre.txt",
    "model_ovl.txt",
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


def validate_bundle_dir(model_dir: str) -> List[str]:
    missing = []
    for name in REQUIRED_BUNDLE_FILES:
        if not os.path.isfile(os.path.join(model_dir, name)):
            missing.append(name)
    return missing


def resolve_bundle_dir(asset_root: str, bundle_spec: str) -> str:
    cleaned = str(bundle_spec or "").strip().replace("/", os.sep).replace("\\", os.sep)
    return os.path.abspath(os.path.join(asset_root, cleaned))


def _default_bundle_slug(model_dir: str, meta: Dict[str, object]) -> str:
    language = str(meta.get("language", "")).strip().lower() or "unknown"
    format_type = str(meta.get("format_type", "")).strip().lower() or "unknown"
    version = os.path.basename(os.path.abspath(model_dir)) or "v1"
    return f"{language}_{format_type}_{version}"


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

    missing = validate_bundle_dir(model_dir)
    if missing:
        raise FileNotFoundError(f"Missing required bundle files in {model_dir}: {', '.join(missing)}")

    meta = read_bundle_meta(model_dir)
    slug = bundle_slug.strip() or _default_bundle_slug(model_dir, meta)
    out_dir = os.path.join(export_root, slug)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    files: List[Dict[str, object]] = []
    for name in REQUIRED_BUNDLE_FILES:
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
        "language": meta.get("language", ""),
        "format_type": meta.get("format_type", ""),
        "backend": meta.get("backend", ""),
        "model_version": meta.get("model_version", ""),
        "feature_version": meta.get("feature_version", ""),
        "train_rows": meta.get("train_rows", 0),
        "voicebank_count": meta.get("voicebank_count", 0),
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
            for fn in REQUIRED_BUNDLE_FILES + ["bundle_manifest.json"]:
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
