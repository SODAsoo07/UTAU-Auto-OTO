"""Locate phoneme-boundary checkpoints for the UI dropdown.
Scans `ml_workspace/models/phoneme_boundary` (and analogous roots) for *.pt
files and returns dropdown-friendly metadata."""
from __future__ import annotations

import glob
import os

_GLOB = "*.pt"


def _roots() -> list[str]:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [
        os.path.join(root, "models", "phoneme_boundary"),
        os.path.join(root, "ml_workspace", "models", "phoneme_boundary"),
        os.path.join(root, "assets", "models", "phoneme_boundary"),
    ]


def list_available_phoneme_boundary_models() -> list[dict]:
    """Sorted newest-mtime first. Empty list when nothing is discoverable."""
    seen: set[str] = set()
    items: list[dict] = []
    for root in _roots():
        for path in glob.glob(os.path.join(root, _GLOB)):
            full = os.path.abspath(path)
            key = full.lower()
            if key in seen or not os.path.isfile(full):
                continue
            seen.add(key)
            name = os.path.basename(full)
            items.append({
                "name": name,
                "path": full,
                "label": name,
                "mtime": float(os.path.getmtime(full)),
            })
    items.sort(key=lambda item: float(item.get("mtime") or 0.0), reverse=True)
    return items


__all__ = ["list_available_phoneme_boundary_models"]
