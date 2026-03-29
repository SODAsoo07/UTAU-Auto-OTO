import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import onnxruntime as ort
import torch
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.wfl_asr import BIOPhonemeTagger

HTK_TIME_FACTOR = 1e7


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    mel_min = _hz_to_mel(np.array([f_min]))[0]
    mel_max = _hz_to_mel(np.array([f_max]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i - 1, j] = (j - left) / float(center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i - 1, j] = (right - j) / float(right - center)
    return fb


def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    pad_to: Optional[int] = 480000,
) -> np.ndarray:
    if pad_to is not None:
        if len(audio) < pad_to:
            audio = np.pad(audio, (0, pad_to - len(audio)), mode="constant")
        elif len(audio) > pad_to:
            audio = audio[:pad_to]

    window = np.hanning(n_fft).astype(np.float32)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode="constant")
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
        writeable=False,
    )
    frames = frames * window
    fft = np.fft.rfft(frames, n=n_fft)
    power = (np.abs(fft) ** 2).astype(np.float32)
    fb = _mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
    mel = np.dot(power, fb.T)
    mel = np.maximum(mel, 1e-10)
    log_mel = np.log10(mel)
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    log_mel = log_mel.T
    target_frames = 3000
    if log_mel.shape[1] < target_frames:
        pad = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")
    elif log_mel.shape[1] > target_frames:
        log_mel = log_mel[:, :target_frames]
    return log_mel


def load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)
        sr = target_sr
    return audio.astype(np.float32), sr


def run_compare(
    audio_path: str,
    encoder_path: str,
    head_path: str,
    config_path: str,
    checkpoint_path: str,
    phonemes_path: str,
    lang_id: int,
    provider: str,
):
    config = _load_yaml(config_path)
    labels = load_labels(phonemes_path)
    device = torch.device("cpu")

    audio, sr = load_audio(audio_path, int(config["data"].get("sample_rate", 16000)))
    log_mel = log_mel_spectrogram(audio, sample_rate=sr)
    input_features = log_mel[np.newaxis, ...].astype(np.float32)

    model = BIOPhonemeTagger(config, labels)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        input_t = torch.from_numpy(input_features).to(device)
        hidden = model.encoder(input_t).last_hidden_state
        lang_t = torch.tensor([lang_id], dtype=torch.long)
        # bypass model.forward to avoid feature extractor; reuse internal layers
        lang_embed = model.lang_emb(lang_t).unsqueeze(1).expand(-1, hidden.size(1), -1)
        head_in = torch.cat([hidden, lang_embed], dim=-1)
        head_in = model.lang_proj(head_in)
        if model.enable_bilstm and model.bilstm is not None:
            head_in, _ = model.bilstm(head_in)
        out = head_in
        for layer in model.conformer_layers:
            out = layer(out)
        if model.enable_dilated_conv:
            out = model.dilated_conv_stack(out.transpose(1, 2)).transpose(1, 2)
        logits_t = model.classifier(out)
        offsets_t = model.boundary_offset_head(out.transpose(1, 2)).transpose(1, 2)

    providers = ort.get_available_providers()
    if provider == "auto":
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
    sess_opts = ort.SessionOptions()
    encoder_sess = ort.InferenceSession(encoder_path, sess_options=sess_opts, providers=[provider])
    head_sess = ort.InferenceSession(head_path, sess_options=sess_opts, providers=[provider])
    hidden_onnx = encoder_sess.run(None, {"input_features": input_features})[0]
    logits_onnx, offsets_onnx = head_sess.run(
        None,
        {"hidden_states": hidden_onnx.astype(np.float32), "lang_id": np.array([lang_id], dtype=np.int64)},
    )

    logits_t = logits_t.cpu().numpy()
    offsets_t = offsets_t.cpu().numpy()

    def _stats(name: str, a: np.ndarray, b: np.ndarray):
        diff = np.abs(a - b)
        return {
            "name": name,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "p99": float(np.percentile(diff, 99.0)),
        }

    stats = [
        _stats("logits", logits_t, logits_onnx),
        _stats("offsets", offsets_t, offsets_onnx),
    ]

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch vs ONNX outputs for WFL-ASR.")
    parser.add_argument("--audio", required=True, help="Path to wav file")
    parser.add_argument("--encoder", required=True, help="Path to encoder.onnx")
    parser.add_argument("--head", required=True, help="Path to head.onnx")
    parser.add_argument("--config", required=True, help="Path to config_infer.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--phonemes", required=True, help="Path to phonemes.txt")
    parser.add_argument("--lang-id", type=int, default=0, help="Language id (jp=0, kr=1)")
    parser.add_argument("--provider", default="auto", help="auto|CPUExecutionProvider|CUDAExecutionProvider")
    args = parser.parse_args()

    stats = run_compare(
        audio_path=args.audio,
        encoder_path=args.encoder,
        head_path=args.head,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        phonemes_path=args.phonemes,
        lang_id=args.lang_id,
        provider=args.provider,
    )
    for item in stats:
        print(f"{item['name']}: max_abs={item['max_abs']:.6f}, mean_abs={item['mean_abs']:.6f}, p99={item['p99']:.6f}")


if __name__ == "__main__":
    main()
