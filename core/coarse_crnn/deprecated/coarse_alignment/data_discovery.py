from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from core.coarse_crnn.deprecated.coarse_alignment.lab_io import find_alignment_label, label_compatibility_ratio, load_aligned_segments
from core.coarse_crnn.lang import infer_language_from_path, normalize_language


def discover_dataset_staged(root: str) -> list[dict[str, object]]:
    base = Path(root)
    if not base.exists():
        return []
    out: list[dict[str, object]] = []
    for language_dir, language in (("korean", "korean"), ("japanese", "japanese")):
        lang_root = base / language_dir
        if not lang_root.exists():
            continue
        for wav in lang_root.rglob("*.wav"):
            label_path, label_format = find_alignment_label(str(wav))
            if not label_path:
                continue
            segments = load_aligned_segments(label_path, label_format)
            if label_compatibility_ratio(segments, language=language) < 0.65:
                continue
            out.append(
                {
                    "audio": str(wav.resolve()),
                    "language": language,
                    "label_path": label_path,
                    "label_format": label_format,
                    "source": "dataset_staged",
                    "weight": 0.9,
                }
            )
    return out


_PUBLIC_SKIP_MARKERS = (
    "english",
    "_en",
    " eng",
    "english-dataset",
    "ngyy_eng",
    "tiger_en",
    "project-aidol-public-english",
)


def discover_public_sinsy(root: str, *, scan_all: bool = False) -> list[dict[str, object]]:
    base = Path(root)
    if not base.exists():
        return []
    out: list[dict[str, object]] = []
    search_roots = _public_search_roots(base, scan_all=scan_all)
    for search_root in search_roots:
        for lab in search_root.rglob("*.lab"):
            _append_public_lab(out, lab)
    return out


def _public_search_roots(base: Path, *, scan_all: bool) -> list[Path]:
    if scan_all:
        return [base]
    roots: list[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        lowered = str(child.name).lower()
        if any(marker in lowered for marker in _PUBLIC_SKIP_MARKERS):
            continue
        language = infer_language_from_path(str(child))
        if language in {"korean", "japanese"}:
            roots.append(child)
            continue
        # Known public singing datasets in this workspace that carry Sinsy-style
        # phone labs while not always advertising the language in the path.
        if lowered in {"csd", "tiger_ja", "ofutonp", "ofuton_p_utagoe_db", "oniku_kurumi_utagoe_db"}:
            roots.append(child)
            continue
        if any(token in lowered for token in ("utagoe", "kiritan", "nit070", "豕｢髻ｳ", "豁悟｣ｰ")):
            roots.append(child)
            continue
    return roots


def _append_public_lab(out: list[dict[str, object]], lab: Path) -> None:
    lowered = str(lab).replace("\\", "/").lower()
    if any(x in lowered for x in ("/english/", "_en/", "/en/", "english-dataset", "ngyy_eng", "tiger_en")):
        return
    language = infer_language_from_path(str(lab))
    if language not in {"korean", "japanese"}:
        return
    stem = lab.with_suffix("")
    wav_candidates = [stem.with_suffix(".wav")]
    if str(lab.name).endswith(".sinsy.lab"):
        wav_candidates.insert(0, Path(str(lab)[: -len(".sinsy.lab")] + ".wav"))
    wav_path = next((p for p in wav_candidates if p.is_file()), None)
    if wav_path is None:
        return
    segments = load_aligned_segments(str(lab), "timed_lab")
    if len(segments) < 2:
        return
    if label_compatibility_ratio(segments, language=language) < 0.65:
        return
    out.append(
        {
            "audio": str(wav_path.resolve()),
            "language": language,
            "label_path": str(lab.resolve()),
            "label_format": "timed_lab",
            "source": "public_sinsy",
            "weight": 1.0,
        }
    )


def write_manifest(path: str, rows: Iterable[dict[str, object]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_manifest(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = str(raw or "").strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            payload["language"] = normalize_language(payload.get("language"))
            rows.append(payload)
    return rows


__all__ = ["discover_dataset_staged", "discover_public_sinsy", "read_manifest", "write_manifest"]

