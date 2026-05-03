from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from core.coarse_crnn.audio import estimate_active_region_for_wav, load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.decode import greedy_segments, viterbi_align_phones
from core.coarse_crnn.export import write_debug_json, write_phone_lab, write_textgrid
from core.coarse_crnn.lang import normalize_language, phones_from_text, phones_from_wav_name
from core.coarse_crnn.lab_io import read_transcript_for_wav
from core.coarse_crnn.model import load_checkpoint
from core.coarse_crnn.reclist import estimate_reclist_phone_bounds
from core.coarse_crnn.training import resolve_torch_device
from core.coarse_crnn.types import PhoneSegment, Segment


@dataclass
class CoarseCrnnAlignmentResult:
    success: bool
    wav_path: str
    textgrid_path: str
    lab_path: str
    debug_path: str
    overall_confidence: float
    warnings: list[str]
    error: str = ""


def predict_outputs(model, config, wav_path: str, *, device: str = "cpu") -> tuple[np.ndarray, np.ndarray | None, float, float]:
    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    samples, sr, duration_sec = load_wav_mono(wav_path, target_sr=int(config.sample_rate))
    features, hop_sec = log_mel_spectrogram(
        samples,
        sr,
        n_mels=int(config.n_mels),
        frame_ms=float(config.frame_ms),
        hop_ms=float(config.hop_ms),
    )
    if features.shape[0] <= 0:
        return np.zeros((0, len(config.labels)), dtype=np.float32), None, float(hop_sec), float(duration_sec)
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(torch_device)
        if hasattr(model, "forward_all"):
            outputs = model.to(torch_device).forward_all(x)
            logits = outputs["logits"]
            boundary_logits = outputs.get("boundary_logits")
        else:
            logits = model.to(torch_device)(x)
            boundary_logits = None
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        boundary_probs = (
            torch.sigmoid(boundary_logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
            if boundary_logits is not None
            else None
        )
    return probs, boundary_probs, float(hop_sec), float(duration_sec)


def predict_posteriors(model, config, wav_path: str, *, device: str = "cpu") -> tuple[np.ndarray, float, float]:
    probs, _boundary_probs, hop_sec, duration_sec = predict_outputs(model, config, wav_path, device=device)
    return probs, hop_sec, duration_sec


def align_wav(
    *,
    wav_path: str,
    output_dir: str,
    model_path: str,
    language: str,
    transcript: str = "",
    device: str = "cpu",
    export_debug: bool = True,
) -> CoarseCrnnAlignmentResult:
    try:
        lang = normalize_language(language)
        if lang not in {"korean", "japanese"}:
            return CoarseCrnnAlignmentResult(False, wav_path, "", "", "", 0.0, [], f"unsupported language: {language}")
        if not os.path.isfile(wav_path):
            return CoarseCrnnAlignmentResult(False, wav_path, "", "", "", 0.0, [], f"wav not found: {wav_path}")
        if not os.path.isfile(model_path):
            return CoarseCrnnAlignmentResult(False, wav_path, "", "", "", 0.0, [], f"coarse_crnn model not found: {model_path}")

        torch = __import__("torch")
        torch_device = resolve_torch_device(torch, device)
        model, config, meta = load_checkpoint(model_path, map_location=str(torch_device))
        probs, boundary_probs, hop_sec, duration_sec = predict_outputs(model, config, wav_path, device=str(torch_device))
        if probs.shape[0] <= 0 or duration_sec <= 0.0:
            return CoarseCrnnAlignmentResult(False, wav_path, "", "", "", 0.0, [], "empty model output")

        transcript_text = str(transcript or "").strip() or read_transcript_for_wav(wav_path)
        phones = phones_from_text(transcript_text, lang)
        if not phones:
            phones = phones_from_wav_name(wav_path, lang)
        warnings: list[str] = []
        if not phones:
            warnings.append("transcript phones empty; coarse-only output generated")

        coarse_segments = greedy_segments(probs, labels=list(config.labels), hop_sec=hop_sec)
        phone_segments: list[PhoneSegment] = []
        if phones:
            active_region = estimate_active_region_for_wav(wav_path, target_sr=int(config.sample_rate))
            boundary_priors = estimate_reclist_phone_bounds(
                wav_path,
                phones,
                language=lang,
                active_range_sec=active_region,
            )
            phone_segments = viterbi_align_phones(
                probs,
                phones,
                labels=list(config.labels),
                hop_sec=hop_sec,
                language=lang,
                duration_sec=duration_sec,
                active_range_sec=active_region,
                boundary_priors_sec=boundary_priors,
                boundary_probs=boundary_probs,
            )

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        textgrid_path = os.path.join(output_dir, stem + ".TextGrid")
        lab_path = os.path.join(output_dir, stem + ".coarse_crnn.lab")
        debug_path = os.path.join(output_dir, stem + ".coarse_crnn.debug.json")
        write_textgrid(
            textgrid_path,
            duration_sec=duration_sec,
            phone_segments=phone_segments,
            coarse_segments=coarse_segments,
            transcript=transcript_text,
        )
        write_phone_lab(lab_path, phone_segments)
        if export_debug:
            write_debug_json(
                debug_path,
                coarse_segments=coarse_segments,
                phone_segments=phone_segments,
                meta={
                    "backend": "coarse_crnn_v1",
                    "language": lang,
                    "transcript": transcript_text,
                    "model_path": os.path.abspath(model_path),
                    "model_meta": meta,
                    "hop_sec": hop_sec,
                },
            )
        else:
            debug_path = ""
        confidence = _overall_confidence(phone_segments, coarse_segments)
        return CoarseCrnnAlignmentResult(
            True,
            wav_path,
            textgrid_path,
            lab_path,
            debug_path,
            confidence,
            warnings,
            "",
        )
    except Exception as exc:
        return CoarseCrnnAlignmentResult(False, wav_path, "", "", "", 0.0, [], str(exc))


def _overall_confidence(phone_segments: list[PhoneSegment], coarse_segments: list[Segment]) -> float:
    vals = [float(seg.confidence) for seg in phone_segments if seg.confidence is not None]
    if vals:
        return max(0.0, min(1.0, float(sum(vals) / float(len(vals)))))
    vals2 = [float(seg.score) for seg in coarse_segments if seg.score is not None]
    if vals2:
        return max(0.0, min(1.0, float(sum(vals2) / float(len(vals2)))))
    return 0.0


__all__ = ["CoarseCrnnAlignmentResult", "align_wav", "predict_outputs", "predict_posteriors"]
