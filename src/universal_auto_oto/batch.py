"""Batch generation: one independent oto.ini per WAV folder under a root folder.

Each folder is generated on its own (its own format detection, its own
oto.ini as alias reference when present, its own output). Pitch folders of the
same bank (A4, C4, ...) whose format cannot be detected share the format the
other folders agree on. A failing folder is reported and the rest continue.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from .bank_detection import FormatDetection, detect_format
from .oto_io import parse_oto


@dataclass(frozen=True)
class BankFolder:
    path: Path
    relative: str
    wav_count: int
    has_oto: bool
    detection: FormatDetection

    @property
    def format(self) -> str | None:
        return self.detection.format

    def to_dict(self) -> dict[str, object]:
        return {
            "folder": str(self.path),
            "relative": self.relative,
            "wav_count": self.wav_count,
            "has_oto": self.has_oto,
            "detection": self.detection.to_dict(),
        }


def discover_bank_folders(
    root: Path,
    *,
    language: str | None = None,
    use_folder_oto: bool = True,
) -> list[BankFolder]:
    """Every folder under ``root`` (itself included) that holds WAV files directly."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    folders: list[BankFolder] = []
    for current, directories, files in os.walk(root):
        directories[:] = sorted(
            name for name in directories
            if not name.startswith(".") and not name.endswith("_auto_oto") and name != "timelines"
        )
        wavs = sorted(name for name in files if name.casefold().endswith(".wav"))
        if not wavs:
            continue
        path = Path(current)
        oto = path / "oto.ini"
        aliases: list[str] = []
        if use_folder_oto and oto.is_file():
            try:
                aliases = [row.alias for row in parse_oto(oto)[0]]
            except (OSError, ValueError):
                aliases = []
        detection = detect_format(path, wavs, base_aliases=aliases or None, language=language)
        relative = path.relative_to(root).as_posix()
        folders.append(BankFolder(path, "." if relative == "." else relative, len(wavs), oto.is_file(), detection))
    return _share_detected_format(folders)


def _share_detected_format(folders: list[BankFolder]) -> list[BankFolder]:
    detected = Counter(folder.format for folder in folders if folder.format is not None)
    if not detected:
        return folders
    shared, votes = detected.most_common(1)[0]
    if votes < max(1, len([folder for folder in folders if folder.format is not None]) // 2):
        return folders
    result = []
    for folder in folders:
        if folder.format is None:
            note = f"다른 폴더 {votes}개가 {shared.upper()}로 감지되어 같은 형식을 씁니다."
            folder = replace(
                folder,
                detection=FormatDetection(shared, "low", "sibling_folders", (*folder.detection.evidence, note)),
            )
        result.append(folder)
    return result


@dataclass(frozen=True)
class BatchItemResult:
    folder: str
    ok: bool
    output: str | None
    summary: dict[str, object] | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {"folder": self.folder, "ok": self.ok, "output": self.output, "summary": self.summary, "error": self.error}


def batch_output_folder(root: Path, output_root: Path, folder: BankFolder, *, in_place: bool) -> Path:
    if in_place:
        return folder.path
    return Path(output_root) if folder.relative == "." else Path(output_root) / folder.relative


def run_batch(
    folders: Sequence[BankFolder],
    *,
    root: Path,
    output_root: Path,
    in_place: bool,
    generate_one: Callable[[BankFolder, Path], dict[str, object]],
    progress: Callable[[int, int, BankFolder], None] | None = None,
) -> dict[str, object]:
    """Run ``generate_one(folder, output)`` for every folder; failures do not stop the batch."""

    results: list[BatchItemResult] = []
    for index, folder in enumerate(folders, start=1):
        if progress is not None:
            progress(index, len(folders), folder)
        output = batch_output_folder(root, output_root, folder, in_place=in_place)
        try:
            summary = dict(generate_one(folder, output))
            results.append(BatchItemResult(folder.relative, True, str(output), summary, None))
        except Exception as error:  # one folder must not stop the others
            results.append(BatchItemResult(folder.relative, False, str(output), None, f"{type(error).__name__}: {error}"))
    return {
        "batch": True,
        "root": str(root),
        "output_root": str(output_root),
        "folder_count": len(results),
        "succeeded": sum(1 for item in results if item.ok),
        "failed": sum(1 for item in results if not item.ok),
        "generated_rows": sum(int((item.summary or {}).get("generated_rows", 0) or 0) for item in results),
        "results": [item.to_dict() for item in results],
    }


__all__ = ["BankFolder", "BatchItemResult", "batch_output_folder", "discover_bank_folders", "run_batch"]
