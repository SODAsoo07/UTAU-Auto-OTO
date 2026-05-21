from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from .decode import decode_monotonic_events
from .features import extract_features
from .model import MfaFreeFrameModelConfig, build_frame_model
from .slot_viterbi import ExpectedSlot, SlotViterbiResult, assign_slots_viterbi, slot_assignments_to_decoded_events
from .types import DecodedEvent, EVENT_LABELS, FRAME_LABELS, FramePosterior

WORLD_V1_CHECKPOINT_FORMAT = "mfa_free_oto_frame_model_world_v1_v1"
WORLD_V1_FEATURE_SET = "world_v1"
WORLD_V1_FRAME_MS = 25.0
WORLD_V1_HOP_MS = 10.0


@dataclass(frozen=True)
class RuntimeInferenceConfig:
    checkpoint_path: str
    encoder: str | None = None
    device: str | None = None
    use_slot_viterbi: bool = True


@dataclass(frozen=True)
class RuntimePrediction:
    posterior: FramePosterior
    decoded_events: tuple[DecodedEvent, ...]
    slot_result: SlotViterbiResult | None

    def to_json_dict(self) -> dict:
        return {
            "wav_path": self.posterior.wav_path,
            "times_ms": list(self.posterior.times_ms),
            "class_probs": {key: list(value) for key, value in self.posterior.class_probs.items()},
            "event_scores": {key: list(value) for key, value in self.posterior.event_scores.items()},
            "acoustic_scores": {key: list(value) for key, value in self.posterior.acoustic_scores.items()},
            "metadata": dict(self.posterior.metadata),
            "decoded_events": [
                {
                    "label": event.label,
                    "selected_time_ms": event.selected_time_ms,
                    "score": event.score,
                    "expected_phone": event.expected_phone,
                    "frame_index": event.frame_index,
                }
                for event in self.decoded_events
            ],
            "slot_assignments": [
                {
                    "slot_index": assignment.slot_index,
                    "phone_index": assignment.phone_index,
                    "phone": assignment.phone,
                    "role": assignment.role,
                    "event_label": assignment.event_label,
                    "selected_time_ms": assignment.selected_time_ms,
                    "score": assignment.score,
                    "frame_index": assignment.frame_index,
                    "expected_time_ms": assignment.expected_time_ms,
                }
                for assignment in (self.slot_result.assignments if self.slot_result is not None else ())
            ],
            "slot_warnings": list(self.slot_result.warnings if self.slot_result is not None else ()),
            "slot_average_score": self.slot_result.average_score if self.slot_result is not None else None,
        }


def predict_wav(
    wav_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    expected_phones: Sequence[str] | None = None,
    expected_slots: Sequence[ExpectedSlot] | None = None,
    encoder: str | None = None,
    device: str | None = None,
    use_slot_viterbi: bool = True,
    language: str = "",
) -> RuntimePrediction:
    checkpoint_file = str(Path(checkpoint_path)) if checkpoint_path else ""
    encoder_name = str(encoder or "acoustic_world_v1")
    checkpoint = None
    if checkpoint_file and os.path.isfile(checkpoint_file):
        try:
            checkpoint, model, target_device = _load_runtime_checkpoint(checkpoint_file, device)
            encoder_name = str(encoder or checkpoint.get("encoder") or encoder_name)
            _validate_checkpoint_metadata_for_world_v1(checkpoint=checkpoint, encoder_name=encoder_name)
            posterior = _predict_posterior_with_loaded_model(
                wav_path,
                checkpoint=checkpoint,
                model=model,
                target_device=target_device,
                encoder=encoder_name,
                requested_device=device,
            )
        except Exception as exc:
            posterior = _predict_posterior_rule_based(
                wav_path,
                encoder=encoder_name,
                metadata={
                    "rule_based": True,
                    "rule_fallback_reason": f"checkpoint_inference_failed:{exc}",
                    "acoustic_feature_set": "world_v1",
                },
            )
    else:
        posterior = _predict_posterior_rule_based(
            wav_path,
            encoder=encoder_name,
            metadata={
                "rule_based": True,
                "rule_fallback_reason": "checkpoint_missing_or_unset",
                "acoustic_feature_set": "world_v1",
            },
        )
    slot_result = (
        assign_slots_viterbi(
            posterior,
            expected_phones=expected_phones,
            expected_slots=expected_slots,
            language=language,
        )
        if use_slot_viterbi and (expected_slots or expected_phones)
        else None
    )
    decoded = (
        slot_assignments_to_decoded_events(slot_result)
        if slot_result is not None and slot_result.assignments
        else decode_monotonic_events(posterior, expected_phones=expected_phones, language=language)
    )
    return RuntimePrediction(
        posterior=posterior,
        decoded_events=tuple(decoded),
        slot_result=slot_result,
    )


def predict_row(
    row: dict,
    *,
    checkpoint_path: str | Path,
    encoder: str | None = None,
    device: str | None = None,
    use_slot_viterbi: bool = True,
) -> RuntimePrediction:
    return predict_wav(
        row["wav_path"],
        checkpoint_path=checkpoint_path,
        expected_phones=_expected_phones(row),
        encoder=encoder,
        device=device,
        use_slot_viterbi=use_slot_viterbi,
    )


@lru_cache(maxsize=4)
def _load_runtime_checkpoint(checkpoint_path: str, device: str | None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MFA-free OTO runtime inference requires torch.") from exc
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    cfg = MfaFreeFrameModelConfig.from_dict(checkpoint["model_config"])
    model = build_frame_model(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()
    return checkpoint, model, target_device


def _predict_posterior_with_loaded_model(
    wav_path: str | Path,
    *,
    checkpoint: dict,
    model,
    target_device: str,
    encoder: str,
    requested_device: str | None,
) -> FramePosterior:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MFA-free OTO runtime inference requires torch.") from exc
    batch = extract_features(wav_path, encoder=encoder, device=requested_device)
    with torch.no_grad():
        x = torch.from_numpy(batch.features[None, :, :]).float().to(target_device)
        output = model(x)
        frame_probs = torch.softmax(output["frame_logits"], dim=-1)[0].detach().cpu().numpy()
        event_scores = torch.sigmoid(output["event_logits"])[0].detach().cpu().numpy()
    return FramePosterior(
        wav_path=str(wav_path),
        times_ms=batch.times_ms.tolist(),
        class_probs={label: frame_probs[:, idx].astype(float).tolist() for idx, label in enumerate(FRAME_LABELS)},
        event_scores={label: event_scores[:, idx].astype(float).tolist() for idx, label in enumerate(EVENT_LABELS)},
        acoustic_scores={key: value.astype(float).tolist() for key, value in batch.acoustic_scores.items()},
        metadata={
            "encoder": encoder,
            "checkpoint_encoder": checkpoint.get("encoder"),
            "checkpoint_format_version": checkpoint.get("format_version") or checkpoint.get("format"),
            "checkpoint_acoustic_feature_set": checkpoint.get("acoustic_feature_set"),
            "acoustic_feature_set": WORLD_V1_FEATURE_SET if "world" in str(encoder).lower() else "",
            "rule_based": False,
        },
    )


def _predict_posterior_rule_based(
    wav_path: str | Path,
    *,
    encoder: str,
    metadata: dict[str, object] | None = None,
) -> FramePosterior:
    batch = extract_features(wav_path, encoder=encoder)
    times = np.asarray(batch.times_ms, dtype=np.float32)
    frame_count = int(times.shape[0])
    if frame_count <= 0:
        return FramePosterior(
            wav_path=str(wav_path),
            times_ms=[],
            class_probs={label: [] for label in FRAME_LABELS},
            event_scores={label: [] for label in EVENT_LABELS},
            acoustic_scores={},
            metadata={**(metadata or {}), "encoder": encoder, "rule_based": True},
        )
    silence = _track(batch.acoustic_scores, "silence_likelihood", frame_count)
    voicing = _track(batch.acoustic_scores, "voicing", frame_count)
    transition = _track(batch.acoustic_scores, "transition_likelihood", frame_count)
    sonorant = _track(batch.acoustic_scores, "sonorant_onset_likelihood", frame_count)
    nucleus = _track(batch.acoustic_scores, "nucleus_likelihood", frame_count)
    if not np.any(nucleus):
        nucleus = _track(batch.acoustic_scores, "world_nucleus", frame_count)
    if not np.any(nucleus):
        nucleus = voicing
    vowel = np.clip((0.58 * voicing) + (0.22 * nucleus) + (0.20 * (1.0 - silence)), 0.0, 1.0)
    consonant = np.clip((0.52 * transition) + (0.28 * (1.0 - voicing)) + (0.20 * (1.0 - silence)), 0.0, 1.0)
    raw_other = np.clip(1.0 - (silence + vowel + consonant), 0.0, 1.0)
    class_stack = np.stack([silence, consonant, vowel, raw_other], axis=1)
    denom = np.maximum(np.sum(class_stack, axis=1, keepdims=True), 1e-6)
    class_norm = (class_stack / denom).astype(np.float32)
    cv = np.clip(
        (0.38 * transition) + (0.30 * class_norm[:, 2]) + (0.16 * (1.0 - silence)) + (0.16 * sonorant),
        0.0,
        1.0,
    )
    vn = np.clip((0.55 * nucleus) + (0.30 * voicing) + (0.15 * (1.0 - silence)), 0.0, 1.0)
    pc = np.clip((0.58 * transition) + (0.22 * (1.0 - silence)) + (0.20 * (1.0 - voicing)), 0.0, 1.0)
    acoustic_scores = {key: np.asarray(value, dtype=np.float32).tolist() for key, value in batch.acoustic_scores.items()}
    return FramePosterior(
        wav_path=str(wav_path),
        times_ms=times.astype(float).tolist(),
        class_probs={
            "silence": class_norm[:, 0].astype(float).tolist(),
            "consonant": class_norm[:, 1].astype(float).tolist(),
            "vowel": class_norm[:, 2].astype(float).tolist(),
            "other": class_norm[:, 3].astype(float).tolist(),
        },
        event_scores={
            "cv_boundary": cv.astype(float).tolist(),
            "vowel_nucleus": vn.astype(float).tolist(),
            "phone_change": pc.astype(float).tolist(),
        },
        acoustic_scores=acoustic_scores,
        metadata={
            "encoder": encoder,
            "rule_based": True,
            "acoustic_feature_set": WORLD_V1_FEATURE_SET,
            **(metadata or {}),
        },
    )


def _track(scores: dict[str, np.ndarray] | dict[str, object], key: str, frame_count: int) -> np.ndarray:
    values = np.asarray(scores.get(key, []), dtype=np.float32)
    if values.shape[0] != frame_count:
        return np.zeros((frame_count,), dtype=np.float32)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def _expected_phones(row: dict) -> list[str]:
    phones: list[str] = []
    for item in row.get("expected_phones") or []:
        phone = str(item.get("phone") if isinstance(item, dict) else item).strip()
        if phone:
            phones.append(phone)
    return phones


def _validate_checkpoint_metadata_for_world_v1(*, checkpoint: dict, encoder_name: str) -> None:
    normalized_encoder = str(encoder_name or "").strip().lower().replace("_", "-")
    if "world" not in normalized_encoder:
        return
    format_version = _required_checkpoint_str(checkpoint, "format_version", fallback_key="format")
    if format_version != WORLD_V1_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"checkpoint_format_mismatch:{format_version}!= {WORLD_V1_CHECKPOINT_FORMAT}"
        )
    feature_set = _required_checkpoint_str(checkpoint, "acoustic_feature_set")
    if feature_set.lower() != WORLD_V1_FEATURE_SET:
        raise RuntimeError(f"checkpoint_feature_set_mismatch:{feature_set}!={WORLD_V1_FEATURE_SET}")
    frame_labels = tuple(str(item) for item in checkpoint.get("frame_labels") or ())
    event_labels = tuple(str(item) for item in checkpoint.get("event_labels") or ())
    if frame_labels != FRAME_LABELS:
        raise RuntimeError("checkpoint_frame_labels_mismatch")
    if event_labels != EVENT_LABELS:
        raise RuntimeError("checkpoint_event_labels_mismatch")
    acoustic_cfg = checkpoint.get("acoustic_config")
    if not isinstance(acoustic_cfg, dict):
        raise RuntimeError("checkpoint_missing:acoustic_config")
    frame_ms = _required_checkpoint_float(acoustic_cfg, "frame_ms")
    hop_ms = _required_checkpoint_float(acoustic_cfg, "hop_ms")
    if abs(frame_ms - WORLD_V1_FRAME_MS) > 1e-6:
        raise RuntimeError(f"checkpoint_frame_ms_mismatch:{frame_ms}!={WORLD_V1_FRAME_MS}")
    if abs(hop_ms - WORLD_V1_HOP_MS) > 1e-6:
        raise RuntimeError(f"checkpoint_hop_ms_mismatch:{hop_ms}!={WORLD_V1_HOP_MS}")


def _required_checkpoint_str(payload: dict, key: str, *, fallback_key: str | None = None) -> str:
    if key in payload:
        value = payload.get(key)
    elif fallback_key and fallback_key in payload:
        value = payload.get(fallback_key)
    else:
        raise RuntimeError(f"checkpoint_missing:{key}")
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"checkpoint_invalid:{key}")
    return text


def _required_checkpoint_float(payload: dict, key: str) -> float:
    if key not in payload:
        raise RuntimeError(f"checkpoint_missing:{key}")
    raw = payload.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"checkpoint_invalid:{key}") from exc
