from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping


TIMEBASE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TimebaseContract:
    sample_rate: int
    analysis_sample_rate: int
    frame_shift_ms: float
    window_size_ms: float
    frame_time_anchor: str = "left"
    feature_latency_ms: float = 0.0
    padding_policy: str = "reflect"
    resample_method: str = ""
    rounding: str = "round_half_up_0.01ms"
    schema_version: int = TIMEBASE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def cache_fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def frame_to_ms(self, frame_index: int | float) -> float:
        idx = float(frame_index)
        if self.frame_time_anchor.strip().lower() == "left":
            anchor_offset = 0.0
        elif self.frame_time_anchor.strip().lower() == "right":
            anchor_offset = float(self.window_size_ms)
        else:
            anchor_offset = 0.5 * float(self.window_size_ms)
        return idx * float(self.frame_shift_ms) + anchor_offset + float(self.feature_latency_ms)


def timebase_from_mapping(data: Mapping[str, object]) -> TimebaseContract:
    def _int(name: str, default: int) -> int:
        try:
            return int(data.get(name, default))
        except Exception:
            return int(default)

    def _float(name: str, default: float) -> float:
        try:
            return float(data.get(name, default))
        except Exception:
            return float(default)

    return TimebaseContract(
        sample_rate=_int("sample_rate", 0),
        analysis_sample_rate=_int("analysis_sample_rate", _int("sample_rate", 0)),
        frame_shift_ms=_float("frame_shift_ms", 5.0),
        window_size_ms=_float("window_size_ms", 25.0),
        frame_time_anchor=str(data.get("frame_time_anchor", "left") or "left"),
        feature_latency_ms=_float("feature_latency_ms", 0.0),
        padding_policy=str(data.get("padding_policy", "reflect") or "reflect"),
        resample_method=str(data.get("resample_method", "") or ""),
        rounding=str(data.get("rounding", "round_half_up_0.01ms") or "round_half_up_0.01ms"),
        schema_version=_int("schema_version", TIMEBASE_SCHEMA_VERSION),
    )


def cache_key_matches_timebase(cache_meta: Mapping[str, object], contract: TimebaseContract) -> bool:
    return str(cache_meta.get("timebase_fingerprint", "") or "") == contract.cache_fingerprint()


__all__ = [
    "TIMEBASE_SCHEMA_VERSION",
    "TimebaseContract",
    "cache_key_matches_timebase",
    "timebase_from_mapping",
]
