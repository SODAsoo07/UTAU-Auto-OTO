from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

from core.coarse_crnn.oto_inference import predict_oto_with_model
from core.coarse_crnn.oto_model import load_oto_checkpoint
from core.coarse_crnn.oto_targets import OTO_ANCHOR_NAMES, OtoAnchors, anchors_to_oto_params, extract_alias_features
from core.coarse_crnn.training import resolve_torch_device


@dataclass
class OtoEvalConfig:
    model_path: str
    device: str = "auto"
    max_items: int = 0
    seed: int = 1337
    language: str = ""
    format_type: str = ""


def evaluate_oto_manifest(rows: list[dict[str, Any]], config: OtoEvalConfig) -> dict[str, Any]:
    torch = __import__("torch")
    device = resolve_torch_device(torch, config.device)
    model, model_config, meta = load_oto_checkpoint(config.model_path, map_location=str(device))
    model = model.to(device).eval()
    selected = _select_rows(rows, config)
    files: list[dict[str, Any]] = []
    failures: list[str] = []
    anchor_errors: dict[str, list[float]] = {name: [] for name in OTO_ANCHOR_NAMES}
    param_errors: dict[str, list[float]] = {
        "offset": [],
        "consonant": [],
        "cutoff_abs": [],
        "preutterance": [],
        "overlap": [],
    }
    pre_hits_20 = 0
    pre_hits_50 = 0
    pre_count = 0

    for row in selected:
        try:
            result = _evaluate_one(row, config, model=model, model_config=model_config, device=str(device))
        except Exception as exc:
            failures.append(f"{row.get('audio', '')}: {exc}")
            continue
        files.append(result)
        for name, value in result["anchor_abs_errors_ms"].items():
            anchor_errors[name].append(float(value))
        for name, value in result["param_abs_errors_ms"].items():
            param_errors[name].append(float(value))
        pre_error = float(result["param_abs_errors_ms"]["preutterance"])
        pre_hits_20 += 1 if pre_error <= 20.0 else 0
        pre_hits_50 += 1 if pre_error <= 50.0 else 0
        pre_count += 1

    summary = {
        "model_path": os.path.abspath(config.model_path),
        "model_meta": meta,
        "device": str(device),
        "requested_items": len(selected),
        "evaluated_items": len(files),
        "failed_items": len(failures),
        "anchor_mae_ms": {name: _mean(values) for name, values in anchor_errors.items()},
        "param_mae_ms": {name: _mean(values) for name, values in param_errors.items()},
        "preutterance_acc_20ms": float(pre_hits_20) / float(pre_count) if pre_count else None,
        "preutterance_acc_50ms": float(pre_hits_50) / float(pre_count) if pre_count else None,
        "by_language": _aggregate_by(files, "language"),
        "by_format": _aggregate_by(files, "format_type"),
        "by_alias_context": _aggregate_by_alias_context(files),
        "by_alias_role": _aggregate_by_alias_role(files),
        "failures": failures[:20],
        "files": files,
    }
    return summary


def _evaluate_one(row: dict[str, Any], config: OtoEvalConfig, *, model, model_config, device: str) -> dict[str, Any]:
    prediction = predict_oto_with_model(
        model=model,
        config=model_config,
        wav_path=str(row.get("audio", "") or ""),
        language=str(row.get("language", "") or ""),
        format_type=str(row.get("format_type", "") or ""),
        alias=str(row.get("alias", "") or ""),
        prev_alias=str(row.get("prev_alias", "") or ""),
        next_alias=str(row.get("next_alias", "") or ""),
        row_index_in_wav=int(row.get("row_index_in_wav", 0) or 0),
        file_row_count=int(row.get("file_row_count", 1) or 1),
        device=device,
    )
    target_anchors = OtoAnchors(
        offset=float(row.get("anchor_offset_ms", 0.0) or 0.0),
        overlap=float(row.get("anchor_overlap_ms", 0.0) or 0.0),
        preutterance=float(row.get("anchor_preutterance_ms", 0.0) or 0.0),
        consonant=float(row.get("anchor_consonant_ms", 0.0) or 0.0),
        cutoff=float(row.get("anchor_cutoff_ms", row.get("target_cutoff_abs_ms", 0.0)) or 0.0),
    )
    target_params = anchors_to_oto_params(target_anchors, duration_ms=float(row.get("duration_ms", prediction.duration_ms) or prediction.duration_ms))
    pred_anchor_dict = prediction.anchors.to_dict()
    target_anchor_dict = target_anchors.to_dict()
    anchor_errors = {name: abs(float(pred_anchor_dict[name]) - float(target_anchor_dict[name])) for name in OTO_ANCHOR_NAMES}
    param_errors = {
        "offset": abs(float(prediction.params["offset"]) - float(target_params["offset"])),
        "consonant": abs(float(prediction.params["consonant"]) - float(target_params["consonant"])),
        "cutoff_abs": abs(float(prediction.anchors.cutoff) - float(target_anchors.cutoff)),
        "preutterance": abs(float(prediction.params["preutterance"]) - float(target_params["preutterance"])),
        "overlap": abs(float(prediction.params["overlap"]) - float(target_params["overlap"])),
    }
    return {
        "audio": str(row.get("audio", "") or ""),
        "alias": str(row.get("alias", "") or ""),
        "language": str(row.get("language", "") or ""),
        "format_type": str(row.get("format_type", "") or ""),
        "alias_type": str(row.get("alias_type", "") or "other"),
        "transition_type": str(row.get("transition_type", "") or "other"),
        **_resolve_alias_role_fields(row),
        "duration_ms": float(row.get("duration_ms", prediction.duration_ms) or prediction.duration_ms),
        "confidence": float(prediction.confidence),
        "anchor_abs_errors_ms": anchor_errors,
        "param_abs_errors_ms": param_errors,
    }


def _select_rows(rows: list[dict[str, Any]], config: OtoEvalConfig) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if (not config.language or str(row.get("language", "") or "") == config.language)
        and (not config.format_type or str(row.get("format_type", "") or "") == config.format_type)
    ]
    rng = random.Random(int(config.seed))
    rng.shuffle(candidates)
    if int(config.max_items) > 0:
        candidates = candidates[: int(config.max_items)]
    return candidates


def _aggregate_by(files: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        groups.setdefault(str(row.get(key, "") or "unknown"), []).append(row)
    out: dict[str, Any] = {}
    for group, rows in groups.items():
        pre_errors = [float(row["param_abs_errors_ms"]["preutterance"]) for row in rows]
        out[group] = {
            "files": len(rows),
            "preutterance_mae_ms": _mean(pre_errors),
            "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
            "offset_mae_ms": _mean([float(row["param_abs_errors_ms"]["offset"]) for row in rows]),
            "cutoff_abs_mae_ms": _mean([float(row["param_abs_errors_ms"]["cutoff_abs"]) for row in rows]),
        }
    return out


def _aggregate_by_alias_context(files: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        key = "|".join(
            [
                str(row.get("format_type", "") or "unknown"),
                str(row.get("alias_type", "") or "unknown"),
                str(row.get("transition_type", "") or "unknown"),
            ]
        )
        groups.setdefault(key, []).append(row)
    out: dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        pre_errors = [float(row["param_abs_errors_ms"]["preutterance"]) for row in rows]
        out[group] = {
            "files": len(rows),
            "format_type": group.split("|", 2)[0],
            "alias_type": group.split("|", 2)[1],
            "transition_type": group.split("|", 2)[2],
            "preutterance_mae_ms": _mean(pre_errors),
            "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
            "offset_mae_ms": _mean([float(row["param_abs_errors_ms"]["offset"]) for row in rows]),
            "consonant_mae_ms": _mean([float(row["param_abs_errors_ms"]["consonant"]) for row in rows]),
            "cutoff_abs_mae_ms": _mean([float(row["param_abs_errors_ms"]["cutoff_abs"]) for row in rows]),
            "overlap_mae_ms": _mean([float(row["param_abs_errors_ms"]["overlap"]) for row in rows]),
        }
    return out


def _aggregate_by_alias_role(files: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        role = str(row.get("alias_role", "") or "other")
        diphthong = "diph" if bool(row.get("is_diphthong", False)) else "mono"
        special = "special" if bool(row.get("is_special", False)) else "normal"
        key = "|".join([role, diphthong, special])
        groups.setdefault(key, []).append(row)
    out: dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        role, diph_flag, special_flag = group.split("|", 2)
        pre_errors = [float(row["param_abs_errors_ms"]["preutterance"]) for row in rows]
        out[group] = {
            "files": len(rows),
            "alias_role": role,
            "is_diphthong": diph_flag == "diph",
            "is_special": special_flag == "special",
            "preutterance_mae_ms": _mean(pre_errors),
            "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
            "offset_mae_ms": _mean([float(row["param_abs_errors_ms"]["offset"]) for row in rows]),
            "consonant_mae_ms": _mean([float(row["param_abs_errors_ms"]["consonant"]) for row in rows]),
            "cutoff_abs_mae_ms": _mean([float(row["param_abs_errors_ms"]["cutoff_abs"]) for row in rows]),
            "overlap_mae_ms": _mean([float(row["param_abs_errors_ms"]["overlap"]) for row in rows]),
        }
    return out


def _resolve_alias_role_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return alias_role, is_diphthong, is_special for an eval result row.

    Falls back to extract_alias_features() when the manifest was built before
    the alias_role feature was added (values are None rather than a string).
    """
    if row.get("alias_role") is not None:
        return {
            "alias_role": str(row["alias_role"] or "other"),
            "is_diphthong": bool(float(row.get("is_diphthong", 0.0) or 0.0) >= 0.5),
            "is_special": bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
        }
    feats = extract_alias_features(
        str(row.get("alias", "") or ""),
        language=str(row.get("language", "") or ""),
    )
    return {
        "alias_role": str(feats.get("alias_role", "") or "other"),
        "is_diphthong": bool(float(feats.get("is_diphthong", 0.0) or 0.0) >= 0.5),
        "is_special": bool(float(feats.get("is_special", 0.0) or 0.0) >= 0.5),
    }


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / float(len(values))) if values else None


def _hit_rate(values: list[float], threshold: float) -> float | None:
    return float(sum(1 for value in values if float(value) <= float(threshold))) / float(len(values)) if values else None


def write_oto_eval_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = ["OtoEvalConfig", "evaluate_oto_manifest", "write_oto_eval_json"]
