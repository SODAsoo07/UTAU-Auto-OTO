from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from typing import Dict, List

REQUIRED_TORCH_BUNDLE_FILES = [
    "model.pt",
    "torch_config.json",
    "feature_schema.json",
    "model_meta.json",
    "normalizer.json",
    "voicebank_vocab.json",
    "categorical_vocabs.json",
    "eval_summary.json",
]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_torch_bundle_dir(model_dir: str) -> List[str]:
    missing = []
    for name in REQUIRED_TORCH_BUNDLE_FILES:
        if not os.path.isfile(os.path.join(model_dir, name)):
            missing.append(name)
    return missing


def export_torch_bundle(model_dir: str, export_root: str, bundle_slug: str = "", create_zip: bool = False) -> Dict[str, object]:
    model_dir = os.path.abspath(model_dir)
    export_root = os.path.abspath(export_root)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(model_dir)
    missing = validate_torch_bundle_dir(model_dir)
    if missing:
        raise FileNotFoundError(f"Missing required torch bundle files in {model_dir}: {', '.join(missing)}")
    with open(os.path.join(model_dir, "model_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    slug = bundle_slug.strip() or f"{meta.get('language','unknown')}_{meta.get('task_name','task')}_{os.path.basename(model_dir)}"
    out_dir = os.path.join(export_root, slug)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for name in REQUIRED_TORCH_BUNDLE_FILES:
        src = os.path.join(model_dir, name)
        dst = os.path.join(out_dir, name)
        shutil.copy2(src, dst)
        files.append({"name": name, "size": int(os.path.getsize(dst)), "sha256": _file_sha256(dst)})
    manifest = {
        "bundle_slug": slug,
        "source_dir": model_dir,
        "export_dir": out_dir,
        "language": meta.get("language", ""),
        "format_type": meta.get("format_type", ""),
        "task_name": meta.get("task_name", ""),
        "backend": meta.get("backend", ""),
        "feature_version": meta.get("feature_version", ""),
        "train_rows": meta.get("train_rows", 0),
        "voicebank_count": meta.get("voicebank_count", 0),
        "files": files,
    }
    manifest_path = os.path.join(out_dir, "bundle_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if create_zip:
        zip_path = os.path.join(export_root, f"{slug}.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in REQUIRED_TORCH_BUNDLE_FILES + ["bundle_manifest.json"]:
                zf.write(os.path.join(out_dir, name), arcname=os.path.join(slug, name))
        manifest["zip_path"] = zip_path
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
