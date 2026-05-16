from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoopPrepCommonResult:
    status: str = "ok"
    meta: dict[str, object] = field(default_factory=dict)
    phone_quality: dict[str, object] = field(default_factory=dict)
    low_quality_reasons: list[str] = field(default_factory=list)
    low_phone_quality: bool = False
    textgrid_trust_score: float = 0.0
    textgrid_trust_tier: str = "low"
    prefer_filename_sequence: bool = False
    timeline_start_ms: float = 0.0
    timeline_end_ms: float = 0.0
    alignment_source: str = ""
    alignment_source_reason: str = ""
    alignment_source_meta: dict[str, object] = field(default_factory=dict)


__all__ = [
    "LoopPrepCommonResult",
]
