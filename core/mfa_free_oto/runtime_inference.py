from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

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
    checkpoint_path: str | Path,
    expected_phones: Sequence[str] | None = None,
    expected_slots: Sequence[ExpectedSlot] | None = None,
    encoder: str | None = None,
    device: str | None = None,
    use_slot_viterbi: bool = True,
) -> RuntimePrediction:
    checkpoint, model, target_device = _load_runtime_checkpoint(str(Path(checkpoint_path)), device)
    encoder_name = encoder or str(checkpoint.get("encoder") or "acoustic")
    posterior = _predict_posterior_with_loaded_model(
        wav_path,
        checkpoint=checkpoint,
        model=model,
        target_device=target_device,
        encoder=encoder_name,
        requested_device=device,
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
        metadata={"encoder": encoder, "checkpoint_encoder": checkpoint.get("encoder")},
    )


def _expected_phones(row: dict) -> list[str]:
    phones: list[str] = []
    for item in row.get("expected_phones") or []:
        phone = str(item.get("phone") if isinstance(item, dict) else item).strip()
        if phone:
            phones.append(phone)
    return phones
