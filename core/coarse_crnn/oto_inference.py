from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.oto_model import (
    alias_role_id,
    alias_type_id,
    format_id,
    language_id,
    load_oto_checkpoint,
    right_boundary_prior_blend_for_context,
    transition_type_id,
    uses_relative_param_head,
)
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
    predicted_error_ms: float | None = None
    low_confidence: bool = False
    confidence_components: dict[str, float] | None = None


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
    is_special: bool = False,
    prev_is_special: bool = False,
    next_is_special: bool = False,
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
        is_special=is_special,
        prev_is_special=prev_is_special,
        next_is_special=next_is_special,
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
    is_special: bool = False,
    prev_is_special: bool = False,
    next_is_special: bool = False,
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
    alias_features = extract_alias_features(alias, language=language, is_special=bool(is_special))
    if should_use_vcv_target_window(
        format_type,
        enabled=bool(getattr(config, "enable_vcv_target_window", False)),
        formats=tuple(getattr(config, "target_window_formats", ("vcv",))),
        alias_type=alias_features.get("alias_type", ""),
        alias_role=alias_features.get("alias_role", ""),
        cvvc_alias_types=tuple(getattr(config, "cvvc_target_window_alias_types", ("vc", "vv"))),
        cvvc_alias_roles=tuple(getattr(config, "cvvc_target_window_alias_roles", ("vc", "vv"))),
    ):
        using_vcv_window = True
        features, _anchors, prediction_duration_ms, start_frame = crop_oto_target_window(
            features,
            None,
            hop_sec=float(hop_sec),
            duration_ms=full_duration_ms,
            row_index_in_wav=int(row_index_in_wav),
            file_row_count=int(file_row_count),
            window_frames=target_window_frames_for(
                config,
                format_type,
                int(getattr(config, "vcv_target_window_frames", 0)),
                alias_role=alias_features.get("alias_role", ""),
            ),
        )
        crop_start_ms = float(start_frame) * max(float(hop_sec) * 1000.0, 1e-3)
    lang = torch.tensor([language_id(config, language)], dtype=torch.long, device=torch_device)
    fmt = torch.tensor([format_id(config, format_type)], dtype=torch.long, device=torch_device)
    prev_alias_features = extract_alias_features(prev_alias, language=language, is_special=bool(prev_is_special))
    next_alias_features = extract_alias_features(next_alias, language=language, is_special=bool(next_is_special))
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
    role_text = str(alias_features.get("alias_role", "") or "")
    prev_role_text = str(prev_alias_features.get("alias_role", "") or "")
    next_role_text = str(next_alias_features.get("alias_role", "") or "")
    is_diphthong_flag = bool(float(alias_features.get("is_diphthong", 0.0) or 0.0) >= 0.5)
    is_special_flag = bool(float(alias_features.get("is_special", 0.0) or 0.0) >= 0.5) or bool(is_special)
    if normalize_role(role_text) == "special":
        is_special_flag = True
    role_tensor = torch.tensor([alias_role_id(config, role_text)], dtype=torch.long, device=torch_device)
    prev_role_tensor = torch.tensor([alias_role_id(config, prev_role_text)], dtype=torch.long, device=torch_device)
    next_role_tensor = torch.tensor([alias_role_id(config, next_role_text)], dtype=torch.long, device=torch_device)
    extra_flags_tensor = torch.tensor(
        [[1.0 if is_diphthong_flag else 0.0, 1.0 if is_special_flag else 0.0]],
        dtype=torch.float32,
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
    pass1 = _run_oto_forward_pass(
        model=model,
        features=features,
        torch_device=torch_device,
        lang=lang,
        fmt=fmt,
        context=context,
        alias_id=alias_id,
        transition_id=transition_id,
        prev_alias_id=prev_alias_id,
        next_alias_id=next_alias_id,
        prev_transition_id=prev_transition_id,
        next_transition_id=next_transition_id,
        role_tensor=role_tensor,
        prev_role_tensor=prev_role_tensor,
        next_role_tensor=next_role_tensor,
        extra_flags_tensor=extra_flags_tensor,
    )
    heat = pass1["heat"]
    scalar = pass1["scalar"]
    scalar_logvar = pass1["scalar_logvar"]
    conf_head = float(pass1["conf_head"])
    if bool(getattr(config, "enable_two_stage_refine", False)):
        refine_frames = int(getattr(config, "two_stage_refine_window_frames", 0) or 0)
        if refine_frames > 0:
            heat_ms_pass1, _heat_conf_pass1 = _heatmap_to_anchor_ms(heat, hop_sec=float(hop_sec))
            refine = _two_stage_crop_from_heatmap(
                features=features,
                heat_ms=heat_ms_pass1,
                hop_sec=float(hop_sec),
                window_frames=refine_frames,
            )
            if refine is not None:
                refined_features, refine_start_ms, refine_duration_ms = refine
                pass2 = _run_oto_forward_pass(
                    model=model,
                    features=refined_features,
                    torch_device=torch_device,
                    lang=lang,
                    fmt=fmt,
                    context=context,
                    alias_id=alias_id,
                    transition_id=transition_id,
                    prev_alias_id=prev_alias_id,
                    next_alias_id=next_alias_id,
                    prev_transition_id=prev_transition_id,
                    next_transition_id=next_transition_id,
                    role_tensor=role_tensor,
                    prev_role_tensor=prev_role_tensor,
                    next_role_tensor=next_role_tensor,
                    extra_flags_tensor=extra_flags_tensor,
                )
                heat = pass2["heat"]
                scalar = pass2["scalar"]
                scalar_logvar = pass2["scalar_logvar"]
                conf_head = float(pass2["conf_head"])
                crop_start_ms = float(crop_start_ms) + float(refine_start_ms)
                prediction_duration_ms = float(refine_duration_ms)
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
            prior_blend=right_boundary_prior_blend_for_context(
                config,
                format_type,
                alias_role=role_text,
                is_special=is_special_flag,
            ),
            alias_role=role_text,
            is_diphthong=is_diphthong_flag,
            is_special=is_special_flag,
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
    heat_confidence = float(sum(conf_map.values()) / max(1, len(conf_map)))
    predicted_error_ms = None
    uncertainty_confidence = None
    if scalar_logvar is not None:
        scalar_std = np.sqrt(np.exp(np.asarray(scalar_logvar, dtype=np.float32)))
        predicted_error_ms = float(np.mean(scalar_std) * max(float(prediction_duration_ms), 1.0))
        error_scale = max(1.0, float(getattr(config, "confidence_error_scale_ms", 75.0)))
        uncertainty_confidence = float(np.exp(-predicted_error_ms / error_scale))
    if uncertainty_confidence is not None:
        confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    (heat_confidence * 0.30)
                    + (conf_head * 0.35)
                    + (uncertainty_confidence * 0.35),
                ),
            )
        )
    else:
        confidence = float(max(0.0, min(1.0, (heat_confidence * 0.55) + (conf_head * 0.45))))
    low_confidence = bool(confidence < 0.58)
    if predicted_error_ms is not None and predicted_error_ms > 80.0:
        low_confidence = True
    components = {
        "heatmap": float(heat_confidence),
        "confidence_head": float(conf_head),
    }
    if uncertainty_confidence is not None:
        components["uncertainty"] = float(uncertainty_confidence)
    return OtoPrediction(
        anchors=anchors,
        params=params,
        confidence=confidence,
        duration_ms=full_duration_ms,
        heatmap_confidence=conf_map,
        predicted_error_ms=predicted_error_ms,
        low_confidence=low_confidence,
        confidence_components=components,
    )


def _run_oto_forward_pass(
    *,
    model,
    features: np.ndarray,
    torch_device,
    lang,
    fmt,
    context,
    alias_id,
    transition_id,
    prev_alias_id,
    next_alias_id,
    prev_transition_id,
    next_transition_id,
    role_tensor,
    prev_role_tensor,
    next_role_tensor,
    extra_flags_tensor,
) -> dict[str, Any]:
    torch = __import__("torch")
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
            alias_role_ids=role_tensor,
            prev_alias_role_ids=prev_role_tensor,
            next_alias_role_ids=next_role_tensor,
            extra_alias_flags=extra_flags_tensor,
        )
    heat = torch.sigmoid(outputs["heatmap_logits"]).squeeze(0).detach().cpu().numpy().astype(np.float32)
    scalar = torch.sigmoid(outputs["scalar_logits"]).squeeze(0).detach().cpu().numpy().astype(np.float32)
    scalar_logvar = outputs.get("scalar_logvar")
    if scalar_logvar is not None:
        scalar_logvar = scalar_logvar.squeeze(0).detach().cpu().numpy().astype(np.float32)
    conf_logits = outputs.get("confidence_logits")
    conf_head = float(torch.sigmoid(conf_logits).squeeze(0).detach().cpu().item()) if conf_logits is not None else 0.0
    return {
        "heat": heat,
        "scalar": scalar,
        "scalar_logvar": scalar_logvar,
        "conf_head": conf_head,
    }


def _two_stage_crop_from_heatmap(
    *,
    features: np.ndarray,
    heat_ms: np.ndarray,
    hop_sec: float,
    window_frames: int,
) -> tuple[np.ndarray, float, float] | None:
    frame_count = int(features.shape[0])
    width = max(1, min(int(window_frames), frame_count))
    if frame_count <= width + 8:
        return None
    heat_arr = np.asarray(heat_ms, dtype=np.float32)
    pre_ms = float(heat_arr[2]) if heat_arr.shape[0] > 2 else float(np.median(heat_arr))
    cons_ms = float(heat_arr[3]) if heat_arr.shape[0] > 3 else pre_ms
    center_ms = max(0.0, (pre_ms + cons_ms) * 0.5)
    hop_ms = max(float(hop_sec) * 1000.0, 1e-3)
    center_frame = int(round(center_ms / hop_ms))
    start = max(0, min(frame_count - width, center_frame - (width // 2)))
    end = start + width
    cropped = features[start:end]
    start_ms = float(start) * hop_ms
    duration_ms = float(width) * hop_ms
    return cropped, start_ms, duration_ms


def write_prediction_json(path: str, prediction: OtoPrediction, *, meta: dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "meta": dict(meta or {}),
        "duration_ms": float(prediction.duration_ms),
        "confidence": float(prediction.confidence),
        "predicted_error_ms": (
            float(prediction.predicted_error_ms) if prediction.predicted_error_ms is not None else None
        ),
        "low_confidence": bool(prediction.low_confidence),
        "confidence_components": dict(prediction.confidence_components or {}),
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
    alias_role: object = "",
    is_diphthong: bool = False,
    is_special: bool = False,
) -> np.ndarray:
    params = decode_relative_oto_params(
        scalar,
        duration_ms=float(duration_ms),
        format_type=format_type,
        alias_type=alias_type,
        transition_type=transition_type,
        prior_blend=float(prior_blend),
        alias_role=alias_role,
        is_diphthong=is_diphthong,
        is_special=is_special,
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
