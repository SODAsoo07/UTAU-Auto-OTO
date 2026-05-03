from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.oto_model import alias_type_id, format_id, language_id, load_oto_checkpoint, right_boundary_prior_blend_for, transition_type_id, uses_relative_param_head
from core.coarse_crnn.oto_param_priors import decode_relative_oto_params, relative_params_to_anchors
from core.coarse_crnn.oto_targets import OTO_ANCHOR_NAMES, OtoAnchors, anchors_to_oto_params, extract_alias_features, repair_anchors
from core.coarse_crnn.oto_windowing import crop_oto_target_window, should_use_vcv_target_window, target_window_frames_for
from core.coarse_crnn.training import resolve_torch_device


@dataclass
class OtoPrediction:
    anchors: OtoAnchors
    params: dict[str, float]
    confidence: float
    duration_ms: float
    heatmap_confidence: dict[str, float]


def predict_oto(
    *,
    wav_path: str,
    model_path: str,
    language: str,
    format_type: str,
    alias: str = "",
    prev_alias: str = "",
    next_alias: str = "",
    row_index_in_wav: int = 0,
    file_row_count: int = 1,
    device: str = "auto",
) -> OtoPrediction:
    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    model, config, _meta = load_oto_checkpoint(model_path, map_location=str(torch_device))
    model = model.to(torch_device).eval()
    return predict_oto_with_model(
        model=model,
        config=config,
        wav_path=wav_path,
        language=language,
        format_type=format_type,
        alias=alias,
        prev_alias=prev_alias,
        next_alias=next_alias,
        row_index_in_wav=row_index_in_wav,
        file_row_count=file_row_count,
        device=str(torch_device),
    )


def predict_oto_with_model(
    *,
    model,
    config,
    wav_path: str,
    language: str,
    format_type: str,
    alias: str = "",
    prev_alias: str = "",
    next_alias: str = "",
    row_index_in_wav: int = 0,
    file_row_count: int = 1,
    device: str = "cpu",
) -> OtoPrediction:
    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    model = model.to(torch_device).eval()
    samples, sr, duration_sec = load_wav_mono(wav_path, target_sr=int(config.sample_rate))
    features, hop_sec = log_mel_spectrogram(
        samples,
        sr,
        n_mels=int(config.n_mels),
        frame_ms=float(config.frame_ms),
        hop_ms=float(config.hop_ms),
    )
    if features.shape[0] <= 0:
        raise ValueError(f"empty audio features: {wav_path}")
    full_duration_ms = max(1.0, float(duration_sec) * 1000.0)
    prediction_duration_ms = full_duration_ms
    crop_start_ms = 0.0
    using_vcv_window = False
    alias_features = extract_alias_features(alias, language=language)
    if should_use_vcv_target_window(
        format_type,
        enabled=bool(getattr(config, "enable_vcv_target_window", False)),
        formats=tuple(getattr(config, "target_window_formats", ("vcv",))),
        alias_type=alias_features.get("alias_type", ""),
        cvvc_alias_types=tuple(getattr(config, "cvvc_target_window_alias_types", ("vc", "vv"))),
    ):
        using_vcv_window = True
        features, _anchors, prediction_duration_ms, start_frame = crop_oto_target_window(
            features,
            None,
            hop_sec=float(hop_sec),
            duration_ms=full_duration_ms,
            row_index_in_wav=int(row_index_in_wav),
            file_row_count=int(file_row_count),
            window_frames=target_window_frames_for(config, format_type, int(getattr(config, "vcv_target_window_frames", 0))),
        )
        crop_start_ms = float(start_frame) * max(float(hop_sec) * 1000.0, 1e-3)
    lang = torch.tensor([language_id(config, language)], dtype=torch.long, device=torch_device)
    fmt = torch.tensor([format_id(config, format_type)], dtype=torch.long, device=torch_device)
    prev_alias_features = extract_alias_features(prev_alias, language=language)
    next_alias_features = extract_alias_features(next_alias, language=language)
    alias_id = torch.tensor([alias_type_id(config, alias_features.get("alias_type", ""))], dtype=torch.long, device=torch_device)
    transition_id = torch.tensor(
        [transition_type_id(config, alias_features.get("transition_type", ""))],
        dtype=torch.long,
        device=torch_device,
    )
    prev_alias_id = torch.tensor([alias_type_id(config, prev_alias_features.get("alias_type", ""))], dtype=torch.long, device=torch_device)
    next_alias_id = torch.tensor([alias_type_id(config, next_alias_features.get("alias_type", ""))], dtype=torch.long, device=torch_device)
    prev_transition_id = torch.tensor(
        [transition_type_id(config, prev_alias_features.get("transition_type", ""))],
        dtype=torch.long,
        device=torch_device,
    )
    next_transition_id = torch.tensor(
        [transition_type_id(config, next_alias_features.get("transition_type", ""))],
        dtype=torch.long,
        device=torch_device,
    )
    context = torch.tensor(
        [
            [
                _row_ratio(row_index_in_wav, file_row_count),
                min(1.0, max(1.0, float(file_row_count)) / 64.0),
                min(1.0, max(0.0, float(alias_features.get("alias_phone_count", 0.0) or 0.0)) / 6.0),
                float(alias_features.get("alias_starts_vowel", 0.0) or 0.0),
                float(alias_features.get("alias_ends_vowel", 0.0) or 0.0),
                float(alias_features.get("alias_has_space", 0.0) or 0.0),
                float(alias_features.get("alias_is_vc", 0.0) or 0.0),
                float(alias_features.get("alias_is_cv", 0.0) or 0.0),
                float(alias_features.get("alias_is_vv", 0.0) or 0.0),
                1.0 if int(row_index_in_wav) <= 0 else 0.0,
                1.0 if int(row_index_in_wav) >= max(0, int(file_row_count) - 1) else 0.0,
                float(prev_alias_features.get("alias_ends_vowel", 0.0) or 0.0),
            ]
        ],
        dtype=torch.float32,
        device=torch_device,
    )
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(torch_device)
        outputs = model(
            x,
            lang,
            fmt,
            context,
            alias_id,
            transition_id,
            prev_alias_id,
            next_alias_id,
            prev_transition_id,
            next_transition_id,
        )
        heat = torch.sigmoid(outputs["heatmap_logits"]).squeeze(0).detach().cpu().numpy().astype(np.float32)
        scalar = torch.sigmoid(outputs["scalar_logits"]).squeeze(0).detach().cpu().numpy().astype(np.float32)
    heat_ms, heat_conf = _heatmap_to_anchor_ms(heat, hop_sec=float(hop_sec))
    heatmap_blend = _prediction_heatmap_blend(config, using_vcv_window=using_vcv_window)
    if uses_relative_param_head(config):
        anchor_ms = _relative_scalar_to_anchor_ms(
            scalar,
            heat_ms,
            duration_ms=float(prediction_duration_ms),
            format_type=format_type,
            alias_type=alias_features.get("alias_type", ""),
            transition_type=alias_features.get("transition_type", ""),
            heatmap_blend=heatmap_blend,
            prior_blend=right_boundary_prior_blend_for(config, format_type),
        ) + float(crop_start_ms)
    else:
        scalar_ms = scalar * max(float(prediction_duration_ms), 1.0)
        anchor_ms = ((heat_ms * heatmap_blend) + (scalar_ms * (1.0 - heatmap_blend))) + float(crop_start_ms)
    anchors = repair_anchors(
        OtoAnchors(
            offset=float(anchor_ms[0]),
            overlap=float(anchor_ms[1]),
            preutterance=float(anchor_ms[2]),
            consonant=float(anchor_ms[3]),
            cutoff=float(anchor_ms[4]),
        ),
        duration_ms=full_duration_ms,
    )
    params = anchors_to_oto_params(anchors, duration_ms=full_duration_ms)
    conf_map = {name: float(heat_conf[idx]) for idx, name in enumerate(OTO_ANCHOR_NAMES)}
    confidence = float(sum(conf_map.values()) / max(1, len(conf_map)))
    return OtoPrediction(anchors=anchors, params=params, confidence=confidence, duration_ms=full_duration_ms, heatmap_confidence=conf_map)


def write_prediction_json(path: str, prediction: OtoPrediction, *, meta: dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "meta": dict(meta or {}),
        "duration_ms": float(prediction.duration_ms),
        "confidence": float(prediction.confidence),
        "heatmap_confidence": dict(prediction.heatmap_confidence),
        "anchors": prediction.anchors.to_dict(),
        "oto_params": dict(prediction.params),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def format_oto_line(wav_name: str, alias: str, params: dict[str, float]) -> str:
    return (
        f"{wav_name}={alias},"
        f"{float(params['offset']):.3f},"
        f"{float(params['consonant']):.3f},"
        f"{float(params['cutoff']):.3f},"
        f"{float(params['preutterance']):.3f},"
        f"{float(params['overlap']):.3f}"
    )


def _heatmap_to_anchor_ms(heat: np.ndarray, *, hop_sec: float) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(heat, dtype=np.float32)
    if h.ndim != 2 or h.shape[0] <= 0:
        return np.zeros((len(OTO_ANCHOR_NAMES),), dtype=np.float32), np.zeros((len(OTO_ANCHOR_NAMES),), dtype=np.float32)
    out = np.zeros((h.shape[1],), dtype=np.float32)
    conf = np.zeros((h.shape[1],), dtype=np.float32)
    hop_ms = max(float(hop_sec) * 1000.0, 1e-3)
    for idx in range(h.shape[1]):
        col = h[:, idx]
        center = int(np.argmax(col))
        lo = max(0, center - 2)
        hi = min(h.shape[0], center + 3)
        weights = col[lo:hi].astype(np.float64)
        if float(np.sum(weights)) > 1e-8:
            local = np.arange(lo, hi, dtype=np.float64)
            out[idx] = float(np.sum(local * weights) / np.sum(weights)) * hop_ms
        else:
            out[idx] = float(center) * hop_ms
        conf[idx] = float(col[center])
    return out, conf


def _relative_scalar_to_anchor_ms(
    scalar: np.ndarray,
    heat_ms: np.ndarray,
    *,
    duration_ms: float,
    format_type: object,
    alias_type: object,
    transition_type: object,
    heatmap_blend: float,
    prior_blend: float,
) -> np.ndarray:
    params = decode_relative_oto_params(
        scalar,
        duration_ms=float(duration_ms),
        format_type=format_type,
        alias_type=alias_type,
        transition_type=transition_type,
        prior_blend=float(prior_blend),
    )
    anchors = np.asarray(relative_params_to_anchors(params, duration_ms=float(duration_ms)), dtype=np.float32)
    blend = max(0.0, min(1.0, float(heatmap_blend)))
    if blend > 0.0:
        heat = np.asarray(heat_ms, dtype=np.float32)
        original_pre = float(anchors[2])
        # Keep onset-related fields heatmap-aware, but keep consonant/cutoff on the constrained param head.
        anchors[:3] = (heat[:3] * blend) + (anchors[:3] * (1.0 - blend))
        pre_shift = float(anchors[2]) - original_pre
        anchors[3:] = anchors[3:] + pre_shift
    repaired = repair_anchors(
        OtoAnchors(
            offset=float(anchors[0]),
            overlap=float(anchors[1]),
            preutterance=float(anchors[2]),
            consonant=float(anchors[3]),
            cutoff=float(anchors[4]),
        ),
        duration_ms=float(duration_ms),
    )
    return np.asarray(
        [repaired.offset, repaired.overlap, repaired.preutterance, repaired.consonant, repaired.cutoff],
        dtype=np.float32,
    )


def _row_ratio(row_index_in_wav: int, file_row_count: int) -> float:
    count = max(1, int(file_row_count))
    if count <= 1:
        return 0.0
    return max(0.0, min(1.0, float(int(row_index_in_wav)) / float(count - 1)))


def _prediction_heatmap_blend(config, *, using_vcv_window: bool) -> float:
    key = "vcv_window_heatmap_blend" if using_vcv_window else "anchor_heatmap_blend"
    try:
        value = float(getattr(config, key))
    except Exception:
        value = 0.70
    return max(0.0, min(1.0, value))


__all__ = ["OtoPrediction", "format_oto_line", "predict_oto", "predict_oto_with_model", "write_prediction_json"]
