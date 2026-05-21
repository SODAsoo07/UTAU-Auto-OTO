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
) -> RuntimePrediction:
    checkpoint_file = str(Path(checkpoint_path)) if checkpoint_path else ""
    encoder_name = str(encoder or "acoustic_world_v1")
    checkpoint = None
    if checkpoint_file and os.path.isfile(checkpoint_file):
        try:
            checkpoint, model, target_device = _load_runtime_checkpoint(checkpoint_file, device)
            encoder_name = encoder or str(checkpoint.get("encoder") or encoder_name)
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
        assign_slots_viterbi(posterior, expected_phones=expected_phones, expected_slots=expected_slots)
        if use_slot_viterbi and (expected_slots or expected_phones)
        else None
    )
    decoded = (
        slot_assignments_to_decoded_events(slot_result)
        if slot_result is not None and slot_result.assignments
        else decode_monotonic_events(posterior, expected_phones=expected_phones)
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
            "acoustic_feature_set": "world_v1" if "world" in str(encoder).lower() else "",
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
    cv = np.clip((0.46 * transition) + (0.34 * class_norm[:, 2]) + (0.20 * (1.0 - silence)), 0.0, 1.0)
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
            "acoustic_feature_set": "world_v1",
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
