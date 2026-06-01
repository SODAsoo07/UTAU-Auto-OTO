from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from core.generation.common.oto_alias_family import alias_family
from core.generation.common.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.phoneme_boundary.inference import infer_boundary_scores_with_model
from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint
from core.phoneme_boundary.types import BoundaryFrameScores

DEFAULT_ROLE_LABELS: dict[str, str] = {
    "cv": "vowel_onset",
    "vc": "vowel_end",
}


@dataclass(frozen=True)
class OtoLine:
    index: int
    raw: str
    parsed: bool
    wav: str = ""
    alias: str = ""
    offset: float = 0.0
    consonant: float = 0.0
    cutoff: float = 0.0
    preutterance: float = 0.0
    overlap: float = 0.0
    family: str = ""


@dataclass(frozen=True)
class CandidateConfig:
    role_labels: Mapping[str, str]
    search_radius_ms: float = 120.0
    max_shift_ms: float = 60.0
    min_shift_ms: float = 8.0
    min_score: float = 0.18
    min_score_margin: float = 0.035
    min_quality: float = 0.0
    min_new_offset_ms: float = 0.0
    skip_edge_peaks: bool = True
    center_penalty_per_ms: float = 0.0
    role_max_shift_ms: Mapping[str, float] = field(default_factory=dict)
    role_min_score_margin: Mapping[str, float] = field(default_factory=dict)
    neighbor_guard_enabled: bool = False
    neighbor_guard_roles: tuple[str, ...] = ("vc",)
    neighbor_guard_min_gap_ms: float = 10.0


@dataclass(frozen=True)
class CandidateDecision:
    row_index: int
    wav: str
    alias: str
    alias_family: str
    label: str = ""
    action: str = "skip"
    reason: str = ""
    current_anchor_ms: float = 0.0
    selected_anchor_ms: float = 0.0
    current_score: float = 0.0
    selected_score: float = 0.0
    selected_quality: float = 0.0
    shift_ms: float = 0.0
    old_offset_ms: float = 0.0
    new_offset_ms: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "row_index": int(self.row_index),
            "wav": self.wav,
            "alias": self.alias,
            "alias_family": self.alias_family,
            "label": self.label,
            "action": self.action,
            "reason": self.reason,
            "current_anchor_ms": round(float(self.current_anchor_ms), 3),
            "selected_anchor_ms": round(float(self.selected_anchor_ms), 3),
            "current_score": round(float(self.current_score), 6),
            "selected_score": round(float(self.selected_score), 6),
            "selected_quality": round(float(self.selected_quality), 6),
            "shift_ms": round(float(self.shift_ms), 3),
            "old_offset_ms": round(float(self.old_offset_ms), 3),
            "new_offset_ms": round(float(self.new_offset_ms), 3),
        }


@dataclass(frozen=True)
class PeakCandidate:
    time_ms: float
    score: float
    quality: float
    at_edge: bool


def read_oto_lines(path: str) -> list[OtoLine]:
    text = read_text_with_fallback(path)
    return read_oto_lines_from_text(text)


def read_oto_lines_from_text(text: str) -> list[OtoLine]:
    rows: list[OtoLine] = []
    for index, raw in enumerate(str(text or "").splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            rows.append(OtoLine(index=index, raw=raw, parsed=False))
            continue
        alias = str(parsed.get("alias") or "")
        rows.append(
            OtoLine(
                index=index,
                raw=raw,
                parsed=True,
                wav=str(parsed.get("wav") or ""),
                alias=alias,
                offset=float(parsed.get("offset") or 0.0),
                consonant=float(parsed.get("cons") or 0.0),
                cutoff=float(parsed.get("cutoff") or 0.0),
                preutterance=float(parsed.get("pre") or 0.0),
                overlap=float(parsed.get("ovl") or 0.0),
                family=alias_family(alias),
            )
        )
    return rows


def apply_boundary_candidates(
    rows: list[OtoLine],
    score_maps: Mapping[str, BoundaryFrameScores],
    config: CandidateConfig,
) -> tuple[list[str], dict[str, Any]]:
    out_lines: list[str] = []
    decisions: list[CandidateDecision] = []
    adjusted_by_family_label: dict[str, Counter[str]] = defaultdict(Counter)
    skipped = Counter()
    shifts: list[float] = []
    neighbor_bounds = _row_neighbor_bounds(rows) if bool(config.neighbor_guard_enabled) else {}

    for row in rows:
        if not row.parsed:
            out_lines.append(row.raw)
            skipped["unparsed"] += 1
            continue
        score_map = _lookup_score_map(score_maps, row.wav)
        if score_map is None:
            out_lines.append(row.raw)
            skipped["missing_score_map"] += 1
            decisions.append(_skip(row, "missing_score_map"))
            continue
        decision = boundary_candidate_for_row(
            row,
            score_map,
            config,
            neighbor_bounds=neighbor_bounds.get(row.index),
        )
        decisions.append(decision)
        if decision.action == "adjust":
            out_lines.append(_format_oto_line(row, offset=decision.new_offset_ms))
            adjusted_by_family_label[row.family][decision.label] += 1
            shifts.append(float(decision.shift_ms))
        else:
            out_lines.append(row.raw)
            skipped[decision.reason or "skipped"] += 1

    parsed_rows = sum(1 for row in rows if row.parsed)
    adjusted_rows = sum(1 for item in decisions if item.action == "adjust")
    report = {
        "summary": {
            "rows_total": len(rows),
            "parsed_rows": parsed_rows,
            "adjusted_rows": adjusted_rows,
            "adjusted_ratio": round(adjusted_rows / max(1, parsed_rows), 6),
            "skipped_by_reason": dict(sorted(skipped.items())),
            "adjusted_by_family_label": {
                family: dict(sorted(labels.items())) for family, labels in sorted(adjusted_by_family_label.items())
            },
            "shift_stats_ms": _stats(shifts),
        },
        "config": config_to_json(config),
        "decisions": [item.to_json() for item in decisions],
    }
    return out_lines, report


def boundary_candidate_for_row(
    row: OtoLine,
    score_map: BoundaryFrameScores,
    config: CandidateConfig,
    *,
    neighbor_bounds: tuple[float | None, float | None] | None = None,
) -> CandidateDecision:
    label = str(config.role_labels.get(row.family, "") or "")
    if not label:
        return _skip(row, "role_not_enabled")
    current_anchor = float(row.offset + row.preutterance)
    current_score = _score_at(score_map, label, current_anchor)
    peak = _select_peak(
        score_map,
        label,
        center_ms=current_anchor,
        radius_ms=float(config.search_radius_ms),
        center_penalty_per_ms=float(config.center_penalty_per_ms),
    )
    if peak is None:
        return _skip(row, "no_peak_in_window", label=label, current_anchor=current_anchor, current_score=current_score)
    shift = float(peak.time_ms - current_anchor)
    new_offset = float(row.offset + shift)
    base = CandidateDecision(
        row_index=row.index,
        wav=row.wav,
        alias=row.alias,
        alias_family=row.family,
        label=label,
        current_anchor_ms=current_anchor,
        selected_anchor_ms=float(peak.time_ms),
        current_score=current_score,
        selected_score=float(peak.score),
        selected_quality=float(peak.quality),
        shift_ms=shift,
        old_offset_ms=float(row.offset),
        new_offset_ms=new_offset,
    )
    min_score_margin = float(config.role_min_score_margin.get(row.family, config.min_score_margin))
    max_shift_ms = float(config.role_max_shift_ms.get(row.family, config.max_shift_ms))
    if bool(config.skip_edge_peaks) and peak.at_edge:
        return _replace_decision(base, reason="peak_at_search_edge")
    if float(peak.score) < float(config.min_score):
        return _replace_decision(base, reason="score_below_min")
    if float(peak.quality) < float(config.min_quality):
        return _replace_decision(base, reason="quality_below_min")
    if float(peak.score) - float(current_score) < min_score_margin:
        return _replace_decision(base, reason="score_margin_too_small")
    if abs(shift) < float(config.min_shift_ms):
        return _replace_decision(base, reason="shift_too_small")
    if abs(shift) > max_shift_ms:
        return _replace_decision(base, reason="shift_too_large")
    if not math.isfinite(new_offset):
        return _replace_decision(base, reason="new_offset_nonfinite")
    if new_offset < float(config.min_new_offset_ms):
        return _replace_decision(base, reason="new_offset_below_min")
    if _neighbor_guard_applies(row, config):
        guard_reason = _neighbor_guard_reject_reason(
            selected_anchor=float(peak.time_ms),
            neighbor_bounds=neighbor_bounds,
            min_gap_ms=float(config.neighbor_guard_min_gap_ms),
        )
        if guard_reason:
            return _replace_decision(base, reason=guard_reason)
    return replace(base, action="adjust", reason="accepted")


def load_score_maps_for_oto(
    rows: list[OtoLine],
    *,
    wav_dir: str,
    model_path: str,
    device: str,
    use_acoustic_heuristics: bool,
    heuristic_weight: float,
) -> dict[str, BoundaryFrameScores]:
    torch = __import__("torch")
    torch_device = _resolve_torch_device(torch, device)
    model, config, _meta = load_phoneme_boundary_checkpoint(model_path, map_location=str(torch_device))
    score_maps: dict[str, BoundaryFrameScores] = {}
    wav_names = sorted({row.wav for row in rows if row.parsed and row.wav})
    for wav_name in wav_names:
        wav_path = os.path.join(wav_dir, wav_name)
        if not os.path.isfile(wav_path):
            continue
        score_maps[wav_name] = infer_boundary_scores_with_model(
            model=model,
            config=config,
            wav_path=wav_path,
            device=str(torch_device),
            use_acoustic_heuristics=bool(use_acoustic_heuristics),
            heuristic_weight=float(heuristic_weight),
        )
    return score_maps


def apply_boundary_candidates_to_oto_file(
    *,
    oto_path: str,
    wav_dir: str,
    model_path: str,
    config: CandidateConfig,
    out_path: str = "",
    device: str = "auto",
    use_acoustic_heuristics: bool = True,
    heuristic_weight: float = 0.28,
) -> dict[str, Any]:
    rows = read_oto_lines(oto_path)
    score_maps = load_score_maps_for_oto(
        rows,
        wav_dir=wav_dir,
        model_path=model_path,
        device=device,
        use_acoustic_heuristics=use_acoustic_heuristics,
        heuristic_weight=heuristic_weight,
    )
    out_lines, report = apply_boundary_candidates(rows, score_maps, config)
    target_path = str(out_path or oto_path)
    if target_path:
        write_oto_lines_atomic(target_path, out_lines)
    report["inputs"] = {
        "pred": os.path.abspath(oto_path),
        "wav_dir": os.path.abspath(wav_dir),
        "model": os.path.abspath(model_path),
    }
    report["score_maps"] = {
        "requested_wavs": len({row.wav for row in rows if row.parsed and row.wav}),
        "loaded_wavs": len(score_maps),
    }
    return report


def parse_role_labels(value: str, *, enable_vv: bool = False) -> dict[str, str]:
    labels = dict(DEFAULT_ROLE_LABELS)
    if enable_vv:
        labels["vv"] = "vowel_onset"
    text = str(value or "").strip()
    if not text:
        return labels
    parsed: dict[str, str] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"invalid role label item: {item!r}; expected family:label")
        family, label = item.split(":", 1)
        family = family.strip().lower()
        label = label.strip().lower()
        if not family or not label:
            raise ValueError(f"invalid role label item: {item!r}; expected family:label")
        parsed[family] = label
    return parsed


def parse_float_map(value: str) -> dict[str, float]:
    text = str(value or "").strip()
    if not text:
        return {}
    parsed: dict[str, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"invalid float-map item: {item!r}; expected key:value")
        key, raw_value = item.split(":", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError(f"invalid float-map item: {item!r}; empty key")
        parsed[key] = float(str(raw_value).strip())
    return parsed


def config_to_json(config: CandidateConfig) -> dict[str, Any]:
    return {
        "role_labels": dict(config.role_labels),
        "search_radius_ms": float(config.search_radius_ms),
        "max_shift_ms": float(config.max_shift_ms),
        "min_shift_ms": float(config.min_shift_ms),
        "min_score": float(config.min_score),
        "min_score_margin": float(config.min_score_margin),
        "min_quality": float(config.min_quality),
        "min_new_offset_ms": float(config.min_new_offset_ms),
        "skip_edge_peaks": bool(config.skip_edge_peaks),
        "center_penalty_per_ms": float(config.center_penalty_per_ms),
        "role_max_shift_ms": {key: float(value) for key, value in dict(config.role_max_shift_ms).items()},
        "role_min_score_margin": {key: float(value) for key, value in dict(config.role_min_score_margin).items()},
        "neighbor_guard_enabled": bool(config.neighbor_guard_enabled),
        "neighbor_guard_roles": list(config.neighbor_guard_roles),
        "neighbor_guard_min_gap_ms": float(config.neighbor_guard_min_gap_ms),
    }


def write_oto_lines_atomic(path: str, lines: list[str]) -> None:
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    temp_path = f"{target}.boundary_candidates.tmp"
    try:
        write_oto_lines(temp_path, lines)
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def write_oto_lines(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def _select_peak(
    score_map: BoundaryFrameScores,
    label: str,
    *,
    center_ms: float,
    radius_ms: float,
    center_penalty_per_ms: float,
) -> PeakCandidate | None:
    times = list(score_map.times_ms or [])
    values = list((score_map.scores or {}).get(label, []) or [])
    if not times or not values:
        return None
    lo = float(center_ms) - float(radius_ms)
    hi = float(center_ms) + float(radius_ms)
    indices = [idx for idx, time_ms in enumerate(times[: len(values)]) if lo <= float(time_ms) <= hi]
    if not indices:
        return None
    best_idx = max(
        indices,
        key=lambda idx: float(values[idx]) - (abs(float(times[idx]) - float(center_ms)) * float(center_penalty_per_ms)),
    )
    quality = _quality_at_index(score_map, best_idx)
    return PeakCandidate(
        time_ms=float(times[best_idx]),
        score=max(0.0, min(1.0, float(values[best_idx]))),
        quality=quality,
        at_edge=bool(best_idx == indices[0] or best_idx == indices[-1]),
    )


def _score_at(score_map: BoundaryFrameScores, label: str, time_ms: float) -> float:
    times = list(score_map.times_ms or [])
    values = list((score_map.scores or {}).get(label, []) or [])
    if not times or not values:
        return 0.0
    limit = min(len(times), len(values))
    idx = min(range(limit), key=lambda i: abs(float(times[i]) - float(time_ms)))
    return max(0.0, min(1.0, float(values[idx])))


def _quality_at_index(score_map: BoundaryFrameScores, index: int) -> float:
    values = list(score_map.quality_scores or [])
    if not values:
        return 1.0
    idx = min(max(0, int(index)), len(values) - 1)
    return max(0.0, min(1.0, float(values[idx])))


def _lookup_score_map(score_maps: Mapping[str, BoundaryFrameScores], wav_name: str) -> BoundaryFrameScores | None:
    direct = score_maps.get(wav_name)
    if direct is not None:
        return direct
    lowered = str(wav_name or "").lower()
    for key, value in score_maps.items():
        if str(key).lower() == lowered:
            return value
        if os.path.basename(str(key)).lower() == lowered:
            return value
    return None


def _format_oto_line(row: OtoLine, *, offset: float) -> str:
    return ",".join(
        [
            f"{row.wav}={row.alias}",
            _fmt_number(offset),
            _fmt_number(row.consonant),
            _fmt_number(row.cutoff),
            _fmt_number(row.preutterance),
            _fmt_number(row.overlap),
        ]
    )


def _fmt_number(value: float) -> str:
    v = float(value)
    if abs(v - round(v)) < 0.0005:
        return str(int(round(v)))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _skip(
    row: OtoLine,
    reason: str,
    *,
    label: str = "",
    current_anchor: float | None = None,
    current_score: float = 0.0,
) -> CandidateDecision:
    anchor = float(row.offset + row.preutterance) if current_anchor is None else float(current_anchor)
    return CandidateDecision(
        row_index=row.index,
        wav=row.wav,
        alias=row.alias,
        alias_family=row.family,
        label=label,
        action="skip",
        reason=str(reason),
        current_anchor_ms=anchor,
        selected_anchor_ms=anchor,
        current_score=float(current_score),
        selected_score=float(current_score),
        old_offset_ms=float(row.offset),
        new_offset_ms=float(row.offset),
    )


def _replace_decision(decision: CandidateDecision, *, reason: str) -> CandidateDecision:
    return CandidateDecision(
        row_index=decision.row_index,
        wav=decision.wav,
        alias=decision.alias,
        alias_family=decision.alias_family,
        label=decision.label,
        action="skip",
        reason=str(reason),
        current_anchor_ms=decision.current_anchor_ms,
        selected_anchor_ms=decision.selected_anchor_ms,
        current_score=decision.current_score,
        selected_score=decision.selected_score,
        selected_quality=decision.selected_quality,
        shift_ms=decision.shift_ms,
        old_offset_ms=decision.old_offset_ms,
        new_offset_ms=decision.new_offset_ms,
    )


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    abs_values = sorted(abs(float(item)) for item in values)
    raw_values = [float(item) for item in values]
    return {
        "n": len(values),
        "mean_ms": round(sum(raw_values) / len(raw_values), 3),
        "mean_abs_ms": round(sum(abs_values) / len(abs_values), 3),
        "median_abs_ms": round(abs_values[len(abs_values) // 2], 3),
        "p90_abs_ms": round(abs_values[int(0.9 * (len(abs_values) - 1))], 3),
        "max_abs_ms": round(max(abs_values), 3),
    }


def _row_neighbor_bounds(rows: list[OtoLine]) -> dict[int, tuple[float | None, float | None]]:
    by_wav: dict[str, list[OtoLine]] = defaultdict(list)
    for row in rows:
        if row.parsed and row.wav:
            by_wav[str(row.wav).lower()].append(row)
    out: dict[int, tuple[float | None, float | None]] = {}
    for group in by_wav.values():
        anchor_rows = [row for row in group if _is_neighbor_anchor_family(row.family)]
        for row in group:
            prev_anchor: float | None = None
            next_anchor: float | None = None
            for candidate in anchor_rows:
                if candidate.index >= row.index:
                    continue
                prev_anchor = float(candidate.offset + candidate.preutterance)
            for candidate in anchor_rows:
                if candidate.index <= row.index:
                    continue
                next_anchor = float(candidate.offset + candidate.preutterance)
                break
            out[row.index] = (prev_anchor, next_anchor)
    return out


def _is_neighbor_anchor_family(family: str) -> bool:
    return str(family or "").strip().lower() in {"cv", "cv_head", "vv", "vcv"}


def _neighbor_guard_applies(row: OtoLine, config: CandidateConfig) -> bool:
    if not bool(config.neighbor_guard_enabled):
        return False
    roles = {str(item).strip().lower() for item in config.neighbor_guard_roles}
    return str(row.family or "").strip().lower() in roles


def _neighbor_guard_reject_reason(
    *,
    selected_anchor: float,
    neighbor_bounds: tuple[float | None, float | None] | None,
    min_gap_ms: float,
) -> str:
    if neighbor_bounds is None:
        return "neighbor_guard_no_bounds"
    prev_anchor, next_anchor = neighbor_bounds
    if prev_anchor is None and next_anchor is None:
        return "neighbor_guard_no_bounds"
    if prev_anchor is not None and next_anchor is not None and prev_anchor >= next_anchor:
        return "neighbor_guard_bad_bounds"
    gap = max(0.0, float(min_gap_ms))
    if prev_anchor is not None and float(selected_anchor) <= float(prev_anchor) + gap:
        return "neighbor_guard_crosses_previous"
    if next_anchor is not None and float(selected_anchor) >= float(next_anchor) - gap:
        return "neighbor_guard_crosses_next"
    return ""


def _resolve_torch_device(torch, device: str):
    text = str(device or "auto").strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


__all__ = [
    "CandidateConfig",
    "CandidateDecision",
    "DEFAULT_ROLE_LABELS",
    "OtoLine",
    "apply_boundary_candidates",
    "apply_boundary_candidates_to_oto_file",
    "boundary_candidate_for_row",
    "config_to_json",
    "load_score_maps_for_oto",
    "parse_float_map",
    "parse_role_labels",
    "read_oto_lines",
    "read_oto_lines_from_text",
    "write_oto_lines",
    "write_oto_lines_atomic",
]
