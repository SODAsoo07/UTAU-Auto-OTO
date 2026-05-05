from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple


_DICT_CANDIDATES = ("japanese_dict.txt", "korean_dict.txt")


@dataclass
class CopyItem:
    source_wav: str
    source_lab: str
    out_wav: str
    out_lab: str


def _abs(path: str) -> str:
    return os.path.abspath(path)


def _safe_mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _list_wavs_recursive(root: str) -> List[str]:
    out: List[str] = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".wav"):
                out.append(os.path.join(base, name))
    out.sort()
    return out


def _unique_stem(stem: str, used: Dict[str, int]) -> str:
    key = str(stem or "").strip() or "sample"
    if key not in used:
        used[key] = 1
        return key
    idx = used[key]
    used[key] = idx + 1
    return f"{key}__dup{idx:02d}"


def _maybe_copy_dictionary(source_roots: List[str], out_dir: str) -> str:
    for root in source_roots:
        for base, _dirs, files in os.walk(root):
            file_set = {str(x) for x in files}
            for name in _DICT_CANDIDATES:
                if name in file_set:
                    src = os.path.join(base, name)
                    dst = os.path.join(out_dir, name)
                    shutil.copy2(src, dst)
                    return dst
    return ""


def _copy_items(source_roots: List[str], out_dir: str) -> Tuple[List[CopyItem], str]:
    used: Dict[str, int] = {}
    copied: List[CopyItem] = []

    for root in source_roots:
        wavs = _list_wavs_recursive(root)
        for src_wav in wavs:
            stem = os.path.splitext(os.path.basename(src_wav))[0]
            out_stem = _unique_stem(stem, used)
            out_wav = os.path.join(out_dir, out_stem + ".wav")
            shutil.copy2(src_wav, out_wav)

            src_lab = os.path.splitext(src_wav)[0] + ".lab"
            out_lab = os.path.join(out_dir, out_stem + ".lab")
            if os.path.isfile(src_lab):
                shutil.copy2(src_lab, out_lab)
            else:
                out_lab = ""

            copied.append(
                CopyItem(
                    source_wav=src_wav,
                    source_lab=src_lab if os.path.isfile(src_lab) else "",
                    out_wav=out_wav,
                    out_lab=out_lab,
                )
            )

    dict_path = _maybe_copy_dictionary(source_roots, out_dir)
    return copied, dict_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect test pronunciations into one flat folder for alignment runs."
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        required=True,
        help="Source folder (repeatable). WAV/LAB will be collected recursively.",
    )
    parser.add_argument("--out-dir", required=True, help="Output folder (flat).")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing WAV/LAB/manifest in out-dir before copy.",
    )
    args = parser.parse_args()

    sources = [_abs(p) for p in (args.sources or [])]
    for src in sources:
        if not os.path.isdir(src):
            raise SystemExit(f"source not found: {src}")

    out_dir = _safe_mkdir(_abs(args.out_dir))
    if args.clean:
        for name in os.listdir(out_dir):
            low = name.lower()
            if low.endswith(".wav") or low.endswith(".lab") or low in {
                "japanese_dict.txt",
                "korean_dict.txt",
                "manifest.json",
            }:
                try:
                    os.remove(os.path.join(out_dir, name))
                except Exception:
                    pass

    copied, dict_path = _copy_items(sources, out_dir)
    payload = {
        "out_dir": out_dir,
        "source_count": len(sources),
        "copied_count": len(copied),
        "dictionary": dict_path,
        "items": [
            {
                "source_wav": item.source_wav,
                "source_lab": item.source_lab,
                "out_wav": item.out_wav,
                "out_lab": item.out_lab,
            }
            for item in copied
        ],
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[DONE] out_dir={out_dir}")
    print(f"[DONE] copied_count={len(copied)}")
    print(f"[DONE] dictionary={dict_path or '(none)'}")
    print(f"[DONE] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
