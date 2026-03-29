import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

import onnxruntime as ort

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

HTK_TIME_FACTOR = 1e7
FRAME_DURATION = 0.02
MAX_SEGMENT_DURATION = 30.0


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else _load_yaml(path)


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_langs(path: str) -> Dict[str, int]:
    lang2id = {}
    if not path or not os.path.exists(path):
        return lang2id
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lang, idx = line.strip().split(",")
            lang2id[lang] = int(idx)
    return lang2id


def load_phoneme_merge_map(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_to_lang(phoneme: str, lang: str, merge_map: Optional[dict]) -> str:
    if not merge_map:
        return phoneme
    if phoneme in merge_map:
        return merge_map[phoneme].get(lang, phoneme)
    return phoneme


def decode_bio_tags(tags: Sequence[str], frame_duration: float, offsets: Optional[np.ndarray] = None):
    segments = []
    current_ph = None
    start_idx = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if current_ph is not None:
                end_idx = i
                start_time = (start_idx + 0.5) * frame_duration
                end_time = (end_idx + 0.5) * frame_duration
                if offsets is not None:
                    start_time = (start_idx + float(offsets[start_idx][0])) * frame_duration
                    end_time = (end_idx + float(offsets[end_idx][1])) * frame_duration
                segments.append((start_time, end_time, current_ph))
                current_ph = None
                start_idx = None
            continue
        if tag.startswith("B-"):
            if current_ph is not None:
                end_idx = i
                start_time = (start_idx + 0.5) * frame_duration
                end_time = (end_idx + 0.5) * frame_duration
                if offsets is not None:
                    start_time = (start_idx + float(offsets[start_idx][0])) * frame_duration
                    end_time = (end_idx + float(offsets[end_idx][1])) * frame_duration
                segments.append((start_time, end_time, current_ph))
            current_ph = tag[2:]
            start_idx = i
        elif tag.startswith("I-"):
            ph = tag[2:]
            if current_ph != ph:
                if current_ph is not None:
                    end_idx = i
                    start_time = (start_idx + 0.5) * frame_duration
                    end_time = (end_idx + 0.5) * frame_duration
                    if offsets is not None:
                        start_time = (start_idx + float(offsets[start_idx][0])) * frame_duration
                        end_time = (end_idx + float(offsets[end_idx][1])) * frame_duration
                    segments.append((start_time, end_time, current_ph))
                current_ph = ph
                start_idx = i
    if current_ph is not None and start_idx is not None:
        end_idx = len(tags) - 1
        start_time = (start_idx + 0.5) * frame_duration
        end_time = (end_idx + 0.5) * frame_duration
        if offsets is not None and start_idx < len(offsets) and end_idx < len(offsets):
            start_time = (start_idx + float(offsets[start_idx][0])) * frame_duration
            end_time = (end_idx + float(offsets[end_idx][1])) * frame_duration
        segments.append((start_time, end_time, current_ph))
    return segments


def merge_adjacent_segments(segments: Sequence[Tuple[float, float, str]], mode: str = "right"):
    if not segments or mode == "none":
        return list(segments)
    segments = list(segments)
    if mode == "right":
        merged = [segments[0]]
        for start, end, ph in segments[1:]:
            last_start, last_end, last_ph = merged[-1]
            if ph == last_ph:
                merged[-1] = (last_start, end, ph)
            else:
                merged.append((start, end, ph))
        return merged
    if mode == "left":
        merged = []
        i = 0
        while i < len(segments):
            if i > 0 and segments[i][2] == segments[i - 1][2]:
                prev_start, _prev_end, ph = merged.pop()
                merged.append((prev_start, segments[i][1], ph))
            else:
                merged.append(segments[i])
            i += 1
        return merged
    if mode == "previous":
        merged = []
        i = 0
        while i < len(segments):
            if i > 1 and segments[i - 1][2] == segments[i][2]:
                if len(merged) >= 2:
                    p0 = merged[-2]
                    _p1 = merged.pop()
                    merged[-1] = (p0[0], segments[i][1], p0[2])
                else:
                    merged.append(segments[i])
            else:
                merged.append(segments[i])
            i += 1
        return merged
    raise ValueError(f"Unsupported merge mode: {mode}")


def save_lab(path: str, segments: Sequence[Tuple[float, float, str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for start, end, ph in segments:
            start_htk = int(start * HTK_TIME_FACTOR)
            end_htk = int(end * HTK_TIME_FACTOR)
            f.write(f"{start_htk} {end_htk} {ph}\n")


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
    # Whisper encoder expects 3000 frames for 30s input; enforce for static ONNX.
    target_frames = 3000
    if log_mel.shape[1] < target_frames:
        pad = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")
    elif log_mel.shape[1] > target_frames:
        log_mel = log_mel[:, :target_frames]
    return log_mel


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def suppress_low_confidence(logits: np.ndarray, id2label: Dict[int, str], threshold: float) -> List[str]:
    probs = _softmax(logits)
    max_probs = np.max(probs, axis=-1)
    pred_ids = np.argmax(probs, axis=-1)
    tags = []
    for p, idx in zip(max_probs, pred_ids):
        if float(p) < threshold:
            tags.append("O")
        else:
            tags.append(id2label[int(idx)])
    return tags


def split_audio(audio: np.ndarray, sr: int, max_duration: float) -> List[np.ndarray]:
    total_samples = len(audio)
    samples_per_segment = int(max_duration * sr)
    return [audio[start : min(start + samples_per_segment, total_samples)] for start in range(0, total_samples, samples_per_segment)]


def _run_head(session: ort.InferenceSession, hidden_states: np.ndarray, lang_id: int) -> Tuple[np.ndarray, np.ndarray]:
    outputs = session.run(
        None,
        {
            "hidden_states": hidden_states.astype(np.float32),
            "lang_id": np.array([lang_id], dtype=np.int64),
        },
    )
    logits, offsets = outputs
    return logits, offsets


def infer_segments(
    encoder_sess: ort.InferenceSession,
    head_sess: ort.InferenceSession,
    segments: List[np.ndarray],
    sr: int,
    config: dict,
    labels: List[str],
    lang_id: Optional[int],
    lang2id: Dict[str, int],
    merge_map: Optional[dict],
    lang_name: Optional[str],
    confidence_threshold: float,
) -> List[Tuple[float, float, str]]:
    id2label = {i: label for i, label in enumerate(labels)}
    all_segments = []
    current_time = 0.0
    for segment in segments:
        if len(segment) == 0:
            continue
        seg = segment.astype(np.float32)
        seg = seg / (np.max(np.abs(seg)) + 1e-8)
        log_mel = log_mel_spectrogram(seg, sample_rate=sr)
        input_features = log_mel[np.newaxis, ...]
        hidden_states = encoder_sess.run(None, {"input_features": input_features.astype(np.float32)})[0]

        if lang_id is not None:
            logits, offsets = _run_head(head_sess, hidden_states, lang_id)
        else:
            logits_list = []
            offsets_list = []
            for lid in lang2id.values():
                l_logits, l_offsets = _run_head(head_sess, hidden_states, int(lid))
                logits_list.append(l_logits)
                offsets_list.append(l_offsets)
            logits = np.mean(np.stack(logits_list, axis=0), axis=0)
            offsets = np.mean(np.stack(offsets_list, axis=0), axis=0)

        logits = logits.squeeze(0)
        offsets = offsets.squeeze(0) if offsets is not None else None

        pred_tags = suppress_low_confidence(logits, id2label, threshold=confidence_threshold)
        pred_ids = np.array([labels.index(t) if t in labels else labels.index("O") for t in pred_tags], dtype=np.int64)
        if int(config["postprocess"].get("median_filter", 1)) > 1:
            pred_ids = median_filter(pred_ids, size=int(config["postprocess"]["median_filter"]))
        pred_tags = [id2label[int(i)] for i in pred_ids]
        segments_pred = decode_bio_tags(pred_tags, frame_duration=FRAME_DURATION, offsets=offsets)
        if merge_map and lang_name:
            segments_pred = [(s, e, canonical_to_lang(ph, lang_name, merge_map)) for s, e, ph in segments_pred]
        shifted = [(s + current_time, e + current_time, ph) for s, e, ph in segments_pred]
        all_segments.extend(shifted)
        current_time += len(segment) / sr
    return all_segments


def infer_audio(
    encoder_path: str,
    head_path: str,
    audio_path: str,
    config_path: str,
    phonemes_path: str,
    langs_path: str,
    merge_map_path: str,
    output_lab_path: str,
    lang_id: Optional[int],
    provider: str,
    confidence_threshold: Optional[float],
) -> List[Tuple[float, float, str]]:
    config = _load_yaml(config_path)
    labels = load_labels(phonemes_path)
    lang2id = load_langs(langs_path)
    merge_map = load_phoneme_merge_map(merge_map_path)

    lang_name = None
    if lang_id is not None:
        for name, idx in lang2id.items():
            if idx == lang_id:
                lang_name = name
                break

    providers = ort.get_available_providers()
    if provider == "auto":
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
    sess_opts = ort.SessionOptions()
    encoder_sess = ort.InferenceSession(encoder_path, sess_options=sess_opts, providers=[provider])
    head_sess = ort.InferenceSession(head_path, sess_options=sess_opts, providers=[provider])

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    target_sr = int(config["data"].get("sample_rate", 16000))
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)
        sr = target_sr

    if len(audio) / sr > MAX_SEGMENT_DURATION:
        segments = split_audio(audio, sr, MAX_SEGMENT_DURATION)
    else:
        segments = [audio]

    conf_th = confidence_threshold if confidence_threshold is not None else float(
        config.get("postprocess", {}).get("confidence_threshold", 0.0)
    )

    segments_pred = infer_segments(
        encoder_sess,
        head_sess,
        segments,
        sr,
        config,
        labels,
        lang_id,
        lang2id,
        merge_map,
        lang_name,
        conf_th,
    )

    merge_mode = str(config.get("postprocess", {}).get("merge_segments", "none")).strip().lower()
    if merge_mode != "none":
        segments_pred = merge_adjacent_segments(segments_pred, mode=merge_mode)

    if output_lab_path:
        save_lab(output_lab_path, segments_pred)
    return segments_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="WFL-ASR ONNX runtime inference (minimal deps).")
    parser.add_argument("path", help="Path to wav file or directory")
    parser.add_argument("--encoder", required=True, help="Path to encoder.onnx")
    parser.add_argument("--head", required=True, help="Path to head.onnx")
    parser.add_argument("--config", required=True, help="Path to config_infer.yaml")
    parser.add_argument("--phonemes", required=True, help="Path to phonemes.txt")
    parser.add_argument("--langs", required=True, help="Path to langs.txt")
    parser.add_argument("--merge-map", default="", help="Path to phoneme_merge_map.json")
    parser.add_argument("--output", default=".", help="Output directory for .lab files")
    parser.add_argument("--lang-id", type=int, default=None, help="Language ID (e.g., kr=1, jp=0)")
    parser.add_argument("--provider", default="auto", help="auto|CPUExecutionProvider|CUDAExecutionProvider")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="Override confidence threshold")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        wav_files = [f for f in os.listdir(args.path) if f.lower().endswith(".wav")]
        os.makedirs(args.output, exist_ok=True)
        for wav in wav_files:
            src = os.path.join(args.path, wav)
            out_lab = os.path.join(args.output, wav.replace(".wav", ".lab"))
            infer_audio(
                encoder_path=args.encoder,
                head_path=args.head,
                audio_path=src,
                config_path=args.config,
                phonemes_path=args.phonemes,
                langs_path=args.langs,
                merge_map_path=args.merge_map,
                output_lab_path=out_lab,
                lang_id=args.lang_id,
                provider=args.provider,
                confidence_threshold=args.confidence_threshold,
            )
            print(f"Wrote: {out_lab}")
    else:
        out_lab = args.output
        if os.path.isdir(out_lab):
            base = os.path.splitext(os.path.basename(args.path))[0]
            out_lab = os.path.join(out_lab, base + ".lab")
        infer_audio(
            encoder_path=args.encoder,
            head_path=args.head,
            audio_path=args.path,
            config_path=args.config,
            phonemes_path=args.phonemes,
            langs_path=args.langs,
            merge_map_path=args.merge_map,
            output_lab_path=out_lab,
            lang_id=args.lang_id,
            provider=args.provider,
            confidence_threshold=args.confidence_threshold,
        )
        print(f"Wrote: {out_lab}")


if __name__ == "__main__":
    main()
