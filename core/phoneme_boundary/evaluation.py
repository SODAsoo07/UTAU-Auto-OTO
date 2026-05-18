from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.phoneme_boundary.inference import infer_boundary_scores_with_model
from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint
from core.phoneme_boundary.targets import events_from_manifest_row, resolve_wav_path


def evaluate_phoneme_boundary_manifest(
    rows: list[dict[str, Any]],
    *,
    model_path: str,
    manifest_dir: str = "",
    device: str = "auto",
    search_radius_ms: float = 120.0,
) -> dict[str, Any]:
    torch = __import__("torch")
    torch_device = _resolve_torch_device(torch, device)
    model, config, _meta = load_phoneme_boundary_checkpoint(model_path, map_location=str(torch_device))
    model = model.to(torch_device).eval()
    by_label: dict[str, list[float]] = defaultdict(list)
    missed: dict[str, int] = defaultdict(int)
    total_events = 0
    used_rows = 0
    for row in rows:
        wav_path = resolve_wav_path(row, manifest_dir=manifest_dir)
        events = events_from_manifest_row(row)
        if not wav_path or not events:
            continue
        score_map = infer_boundary_scores_with_model(model=model, config=config, wav_path=wav_path, device=str(torch_device))
        used_rows += 1
        for event in events:
            total_events += 1
            predicted = _best_time_for_label(score_map.times_ms, score_map.scores.get(event.label, []), event.time_ms, search_radius_ms)
            if predicted is None:
                missed[event.label] += 1
                continue
            by_label[event.label].append(abs(float(predicted) - float(event.time_ms)))
    label_metrics = {
        label: {
            "count": len(values),
            "mae_ms": _mean(values),
            "p50_ms": _percentile(values, 50.0),
            "p90_ms": _percentile(values, 90.0),
            "missed": int(missed.get(label, 0)),
        }
        for label, values in sorted(by_label.items())
    }
    all_values = [v for values in by_label.values() for v in values]
    return {
        "rows": int(used_rows),
        "events": int(total_events),
        "matched_events": int(len(all_values)),
        "mae_ms": _mean(all_values),
        "p50_ms": _percentile(all_values, 50.0),
        "p90_ms": _percentile(all_values, 90.0),
        "by_label": label_metrics,
        "missed_by_label": dict(sorted(missed.items())),
    }


def _best_time_for_label(times: list[float], values: list[float], target_ms: float, radius_ms: float) -> float | None:
    best_time = None
    best_score = -1.0
    lo = float(target_ms) - float(radius_ms)
    hi = float(target_ms) + float(radius_ms)
    for idx, score in enumerate(values):
        if idx >= len(times):
            break
        time_ms = float(times[idx])
        if time_ms < lo or time_ms > hi:
            continue
        s = float(score)
        if s > best_score:
            best_score = s
            best_time = time_ms
    return best_time


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = int(round((len(ordered) - 1) * max(0.0, min(100.0, pct)) / 100.0))
    return float(ordered[idx])


def _resolve_torch_device(torch, device: str):
    text = str(device or "auto").strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


__all__ = ["evaluate_phoneme_boundary_manifest"]
