from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

APP_DIR = str(Path(__file__).resolve().parents[1])
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from core.oto_ml.coupled.inference import load_coupled_bundle
from core.oto_ml.coupled.model import (
    COUPLED_BACKEND,
    COUPLED_BACKEND_RAWMEL,
    COUPLED_MODEL_FILE,
    _import_torch,
)
from core.oto_ml.coupled.training import _export_coupled_onnx


def _is_coupled_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, COUPLED_MODEL_FILE)):
        return False
    meta_path = os.path.join(path, "model_meta.json")
    if not os.path.isfile(meta_path):
        return True
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
        backend = str(meta.get("backend", "") or "").strip().lower()
        return backend in {COUPLED_BACKEND, COUPLED_BACKEND_RAWMEL}
    except Exception:
        return True


def _collect_target_dirs(root: str) -> List[str]:
    if _is_coupled_dir(root):
        return [os.path.abspath(root)]
    out: List[str] = []
    if not root or not os.path.isdir(root):
        return out
    for current_root, _dirs, _files in os.walk(root):
        if _is_coupled_dir(current_root):
            out.append(os.path.abspath(current_root))
    return sorted(set(out))


def _load_backend(model_dir: str) -> str:
    meta_path = os.path.join(model_dir, "model_meta.json")
    if not os.path.isfile(meta_path):
        return ""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
    except Exception:
        return ""
    return str(meta.get("backend", "") or "").strip().lower()


def _resolve_coupled_export_dir(model_dir: str) -> Optional[str]:
    if not model_dir or not os.path.isdir(model_dir):
        return None
    backend = _load_backend(model_dir)
    if backend == "ensemble_v1":
        coupled_dir = os.path.join(model_dir, "coupled")
        return os.path.abspath(coupled_dir) if _is_coupled_dir(coupled_dir) else None
    if backend in {COUPLED_BACKEND, COUPLED_BACKEND_RAWMEL}:
        return os.path.abspath(model_dir) if _is_coupled_dir(model_dir) else None
    return None


def _collect_runtime_coupled_dirs() -> List[str]:
    from core.oto_ml_refiner import _resolve_model_dir

    langs = ["korean", "japanese"]
    formats = ["cv", "cvc", "cvvc", "vcv", "general"]
    families = ["", "cv", "bridge", "vc", "vowel"]
    out: Set[str] = set()
    for lang in langs:
        for fmt in formats:
            for family in families:
                try:
                    resolved = _resolve_model_dir(lang, fmt, alias_family=family)
                except Exception:
                    resolved = None
                coupled_dir = _resolve_coupled_export_dir(str(resolved or ""))
                if coupled_dir:
                    out.add(coupled_dir)
    return sorted(out)


def _collect_ui_selectable_coupled_dirs() -> List[str]:
    from core.oto_ml_refiner import _collect_model_dir_candidates

    langs = ["korean", "japanese"]
    formats = ["cv", "cvc", "cvvc", "vcv", "general"]
    families = ["", "cv", "bridge", "vc", "vowel"]
    out: Set[str] = set()
    for lang in langs:
        for fmt in formats:
            for family in families:
                try:
                    candidates = _collect_model_dir_candidates(lang, fmt, alias_family=family)
                except Exception:
                    candidates = []
                for candidate in candidates:
                    coupled_dir = _resolve_coupled_export_dir(str(candidate or ""))
                    if coupled_dir:
                        out.add(coupled_dir)
    return sorted(out)


def _export_one(path: str) -> Dict[str, object]:
    # Force torch loader path for migration from legacy .pt-only bundles.
    os.environ["UTOA_ML_COUPLED_ONNX_ENABLE"] = "0"
    bundle = load_coupled_bundle(path, device="cpu")
    if str(bundle.get("runtime", "")).strip().lower() != "torch":
        return {"status": "skipped", "reason": "torch payload unavailable", "path": path}
    torch, _nn, _F = _import_torch()
    return {
        "path": path,
        **_export_coupled_onnx(
            torch,
            bundle["model"],
            path,
            feature_names=list(bundle.get("feature_names") or []),
            categorical_features=list(bundle.get("categorical_features") or []),
            categorical_bucket_sizes=[int(v) for v in (bundle.get("categorical_bucket_sizes") or [])],
            patch_features=list(bundle.get("patch_features") or []),
            head_mode=str(bundle.get("head_mode", "single") or "single"),
            anchor_targets=list(bundle.get("anchor_targets") or []),
            delta_targets=list(bundle.get("delta_targets") or []),
            rawmel_enabled=bool(bundle.get("rawmel_enabled", False)),
            mel_bins=int(bundle.get("mel_bins", 80) or 80),
            onset_frames=int(bundle.get("onset_frames", 8) or 8),
            tail_frames=int(bundle.get("tail_frames", 8) or 8),
            mel_patch_spec=dict(bundle.get("mel_patch_spec") or {}),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export coupled .pt bundles to ONNX for runtime torch removal.")
    parser.add_argument("--model-dir", default="", help="Single model directory or root directory to scan.")
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Export only coupled bundles selected by current runtime model routing.",
    )
    parser.add_argument(
        "--ui-selectable",
        action="store_true",
        help="Export all coupled bundles that can be selected by UI/runtime routing candidates (primary + fallback).",
    )
    args = parser.parse_args()

    if bool(args.runtime_only) and bool(args.ui_selectable):
        print("Use either --runtime-only or --ui-selectable, not both.")
        return 2

    if bool(args.runtime_only):
        dirs = _collect_runtime_coupled_dirs()
    elif bool(args.ui_selectable):
        dirs = _collect_ui_selectable_coupled_dirs()
    else:
        target = str(args.model_dir or "").strip()
        if not target:
            print("No model directory provided. Use --model-dir <path>, --runtime-only, or --ui-selectable.")
            return 2
        dirs = _collect_target_dirs(target)
    if not dirs:
        if bool(args.runtime_only):
            print("No coupled runtime model directories found for current routing.")
        elif bool(args.ui_selectable):
            print("No coupled UI-selectable model directories found for current routing candidates.")
        else:
            print(f"No coupled model directories found: {target}")
        return 1

    ok_count = 0
    fail_count = 0
    for model_dir in dirs:
        try:
            result = _export_one(model_dir)
        except Exception as exc:
            fail_count += 1
            print(f"[FAIL] {model_dir} :: {exc}")
            continue
        status = str(result.get("status", "") or "")
        if status == "ok":
            ok_count += 1
            print(f"[OK] {model_dir} -> {result.get('onnx_path', '')}")
        elif status == "disabled":
            print(f"[SKIP] {model_dir} :: ONNX export disabled")
        else:
            fail_count += 1
            reason = str(result.get("error", "") or result.get("reason", "") or "unknown")
            print(f"[FAIL] {model_dir} :: {reason}")

    print(f"Done. success={ok_count}, failed={fail_count}, total={len(dirs)}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
