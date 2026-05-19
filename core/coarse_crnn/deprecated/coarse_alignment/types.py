from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Segment:
    label: str
    start: float
    end: float
    score: float | None = None


@dataclass
class PhoneSegment:
    phone: str
    start: float
    end: float
    coarse: str
    confidence: float | None = None


@dataclass
class ManifestItem:
    audio: str
    language: str
    label_path: str
    label_format: str
    source: str
    weight: float = 1.0
    meta: dict[str, Any] | None = None


__all__ = ["ManifestItem", "PhoneSegment", "Segment"]
