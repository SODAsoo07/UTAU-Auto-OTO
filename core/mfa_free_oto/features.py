from __future__ import annotations

import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

SSL_MODEL_IDS = {
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "xlsr-300m": "facebook/wav2vec2-xls-r-300m",
}

_SSL_MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


@dataclass(frozen=True)
class FeatureBatch:
    times_ms: np.ndarray
    features: np.ndarray
    sample_rate: int
    duration_ms: float
    encoder: str
    acoustic_scores: Mapping[str, np.ndarray] = field(default_factory=dict)
    source_sample_rate: int = 0
    frame_ms: float = 0.0
    hop_ms: float = 0.0


@dataclass(frozen=True)
class AcousticFeatureConfig:
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    fft_bins: int = 24
    aux_fft_bins: int = 32


def read_wav_mono(path: str | Path, *, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        try:
            data, sample_rate = _read_ieee_float_wav_mono(path)
        except ValueError:
            raise exc
        if target_sample_rate and target_sample_rate != sample_rate:
            data = _resample_linear(data, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        return np.asarray(data, dtype=np.float32), int(sample_rate)
    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8)
        if raw.size % 3 != 0:
            raise ValueError(f"Malformed 24-bit wav byte count: {raw.size}")
        triples = raw.reshape(-1, 3)
        padded = np.zeros((triples.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = triples
        padded[:, 3] = np.where(triples[:, 2] >= 128, 255, 0).astype(np.uint8)
        data = padded.reshape(-1, 4).view("<i4").reshape(-1).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported wav sample width: {sample_width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if target_sample_rate and target_sample_rate != sample_rate:
        data = _resample_linear(data, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def _read_ieee_float_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    raw = Path(path).read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not_riff_wave")

    fmt: bytes | None = None
    data_chunk: bytes | None = None
    offset = 12
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        chunk_size = int.from_bytes(raw[offset + 4 : offset + 8], "little", signed=False)
        data_start = offset + 8
        data_end = min(data_start + chunk_size, len(raw))
        if chunk_id == b"fmt ":
            fmt = raw[data_start:data_end]
        elif chunk_id == b"data":
            data_chunk = raw[data_start:data_end]
        offset = data_start + chunk_size + (chunk_size % 2)

    if fmt is None or data_chunk is None or len(fmt) < 16:
        raise ValueError("missing_float_wav_chunks")
    format_tag, channels, sample_rate, _byte_rate, block_align, bits_per_sample = struct.unpack_from("<HHIIHH", fmt, 0)
    if format_tag == 0xFFFE and len(fmt) >= 40:
        subtype = fmt[24:40]
        ieee_float_guid = bytes.fromhex("0300000000001000800000aa00389b71")
        if subtype == ieee_float_guid:
            format_tag = 3
    if format_tag != 3:
        raise ValueError(f"unsupported_wave_format:{format_tag}")
    if channels <= 0 or sample_rate <= 0:
        raise ValueError("invalid_float_wav_format")

    if bits_per_sample == 32:
        dtype = "<f4"
    elif bits_per_sample == 64:
        dtype = "<f8"
    else:
        raise ValueError(f"unsupported_float_wav_bits:{bits_per_sample}")
    bytes_per_sample = bits_per_sample // 8
    if block_align and block_align != channels * bytes_per_sample:
        raise ValueError("float_wav_block_align_mismatch")
    usable = (len(data_chunk) // bytes_per_sample) * bytes_per_sample
    if usable <= 0:
        raise ValueError("empty_float_wav_data")
    data = np.frombuffer(data_chunk[:usable], dtype=dtype).astype(np.float32)
    if data.size % channels != 0:
        data = data[: data.size - (data.size % channels)]
    if data.size == 0:
        raise ValueError("empty_float_wav_samples")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.clip(np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32), int(sample_rate)


def extract_features(
    wav_path: str | Path,
    *,
    encoder: str = "acoustic",
    acoustic_config: AcousticFeatureConfig | None = None,
    device: str | None = None,
) -> FeatureBatch:
    normalized_encoder = encoder.replace("_", "-").lower()
    source_sample_rate = _wav_sample_rate_if_available(wav_path)
    if normalized_encoder in {"acoustic-world", "acoustic-world-v1", "world", "world-v1"}:
        config = acoustic_config or AcousticFeatureConfig()
        samples, sample_rate = read_wav_mono(wav_path, target_sample_rate=config.sample_rate)
        world = acoustic_world_features(samples, sample_rate, config=config)
        duration_ms = 1000.0 * float(samples.shape[0]) / float(sample_rate)
        return FeatureBatch(
            times_ms=world.times_ms,
            features=world.features,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            encoder="acoustic_world_v1",
            acoustic_scores=world.scores,
            source_sample_rate=source_sample_rate or sample_rate,
            frame_ms=float(config.frame_ms),
            hop_ms=float(config.hop_ms),
        )
    if normalized_encoder == "acoustic":
        config = acoustic_config or AcousticFeatureConfig()
        samples, sample_rate = read_wav_mono(wav_path, target_sample_rate=config.sample_rate)
        times, feats = acoustic_frame_features(samples, sample_rate, config=config)
        aux = acoustic_aux_features(samples, sample_rate, config=config)
        duration_ms = 1000.0 * float(samples.shape[0]) / float(sample_rate)
        return FeatureBatch(
            times_ms=times,
            features=feats,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            encoder=encoder,
            acoustic_scores=aux.scores,
            source_sample_rate=source_sample_rate or sample_rate,
            frame_ms=float(config.frame_ms),
            hop_ms=float(config.hop_ms),
        )
    if normalized_encoder in {"acoustic-aux", "aux", "logmel-aux"}:
        config = acoustic_config or AcousticFeatureConfig()
        samples, sample_rate = read_wav_mono(wav_path, target_sample_rate=config.sample_rate)
        aux = acoustic_aux_features(samples, sample_rate, config=config)
        duration_ms = 1000.0 * float(samples.shape[0]) / float(sample_rate)
        return FeatureBatch(
            times_ms=aux.times_ms,
            features=aux.features,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            encoder=encoder,
            acoustic_scores=aux.scores,
            source_sample_rate=source_sample_rate or sample_rate,
            frame_ms=float(config.frame_ms),
            hop_ms=float(config.hop_ms),
        )
    if normalized_encoder.endswith("+aux"):
        base_encoder = encoder[: -len("+aux")]
        return extract_ssl_plus_aux_features(
            wav_path,
            encoder=base_encoder,
            acoustic_config=acoustic_config,
            device=device,
        )
    return extract_ssl_features(wav_path, encoder=encoder, device=device)


def feature_timebase_metadata(batch: FeatureBatch) -> dict[str, object]:
    source_sample_rate = int(batch.source_sample_rate or batch.sample_rate or 0)
    analysis_sample_rate = int(batch.sample_rate or source_sample_rate or 0)
    metadata: dict[str, object] = {
        "source_sample_rate": source_sample_rate,
        "analysis_sample_rate": analysis_sample_rate,
        "frame_time_anchor": "left",
        "feature_latency_ms": 0.0,
        "padding_policy": "reflect",
        "rounding": "round_half_up_0.01ms",
    }
    if float(batch.frame_ms or 0.0) > 0.0:
        metadata["window_size_ms"] = float(batch.frame_ms)
    if float(batch.hop_ms or 0.0) > 0.0:
        metadata["frame_shift_ms"] = float(batch.hop_ms)
    if source_sample_rate and analysis_sample_rate and source_sample_rate != analysis_sample_rate:
        metadata["resample_method"] = "linear"
    return metadata


@dataclass(frozen=True)
class AcousticAuxFeatures:
    times_ms: np.ndarray
    features: np.ndarray
    scores: Mapping[str, np.ndarray]


def acoustic_world_features(
    samples: np.ndarray,
    sample_rate: int,
    *,
    config: AcousticFeatureConfig | None = None,
) -> AcousticAuxFeatures:
    cfg = config or AcousticFeatureConfig(sample_rate=sample_rate)
    aux = acoustic_aux_features(samples, sample_rate, config=cfg)
    try:
        import librosa
        import pyworld as pw
        from scipy.ndimage import gaussian_filter1d
    except ImportError as exc:
        raise RuntimeError(
            "acoustic_world_v1 encoder requires pyworld, librosa, and scipy."
        ) from exc

    wav64 = samples.astype(np.float64)
    frame_period = float(cfg.hop_ms)
    raw_f0, time_axis = pw.dio(wav64, sample_rate, frame_period=frame_period)
    f0 = pw.stonemask(wav64, raw_f0, time_axis, sample_rate)
    sp = pw.cheaptrick(wav64, f0, time_axis, sample_rate)
    ap = pw.d4c(wav64, f0, time_axis, sample_rate)

    world_times_ms = (np.asarray(time_axis, dtype=np.float64) * 1000.0).astype(np.float32)
    voiced = (np.asarray(f0, dtype=np.float32) > 55.0).astype(np.float32)
    periodicity = 1.0 - np.clip(np.mean(np.asarray(ap, dtype=np.float32), axis=1), 0.0, 1.0)
    log_sp = np.log(np.maximum(np.asarray(sp, dtype=np.float32), 1e-7))
    sp_delta = np.mean(np.abs(np.diff(log_sp, axis=0)), axis=1).astype(np.float32)
    sp_delta = np.pad(sp_delta, (1, 0), mode="edge")
    spectral_stability = 1.0 - _robust_unit(sp_delta)

    hop = max(1, int(round(sample_rate * cfg.hop_ms / 1000.0)))
    onset = librosa.onset.onset_strength(y=samples.astype(np.float32), sr=sample_rate, hop_length=hop)
    rms = librosa.feature.rms(y=samples.astype(np.float32), frame_length=max(1, int(round(sample_rate * cfg.frame_ms / 1000.0))), hop_length=hop).squeeze(0)
    onset = _safe_smooth_track(onset.astype(np.float32), gaussian_filter1d)
    rms = _safe_smooth_track(rms.astype(np.float32), gaussian_filter1d)

    world_voicing = _interp_vector(world_times_ms, _safe_smooth_track(voiced, gaussian_filter1d), aux.times_ms)
    world_periodicity = _interp_vector(world_times_ms, _safe_smooth_track(periodicity.astype(np.float32), gaussian_filter1d), aux.times_ms)
    world_stability = _interp_vector(world_times_ms, _safe_smooth_track(spectral_stability.astype(np.float32), gaussian_filter1d), aux.times_ms)
    onset_times = (np.arange(onset.shape[0], dtype=np.float32) * float(hop) * 1000.0) / float(sample_rate)
    rms_times = (np.arange(rms.shape[0], dtype=np.float32) * float(hop) * 1000.0) / float(sample_rate)
    onset_interp = _interp_vector(onset_times, _robust_unit(onset), aux.times_ms)
    rms_interp = _interp_vector(rms_times, _robust_unit(rms), aux.times_ms)

    voiced_delta = np.abs(np.gradient(world_voicing.astype(np.float32))).astype(np.float32)
    transition_world = np.clip(
        0.45 * onset_interp + 0.25 * _positive_unit(voiced_delta) + 0.30 * (1.0 - world_stability),
        0.0,
        1.0,
    ).astype(np.float32)
    nucleus_world = np.clip(
        0.36 * world_voicing + 0.30 * world_periodicity + 0.20 * world_stability + 0.14 * rms_interp,
        0.0,
        1.0,
    ).astype(np.float32)

    merged_scores = dict(aux.scores)
    merged_scores.update(
        {
            "world_f0_hz": _interp_vector(world_times_ms, np.asarray(f0, dtype=np.float32), aux.times_ms),
            "world_voicing": world_voicing,
            "world_periodicity": world_periodicity,
            "world_spectral_stability": world_stability,
            "world_transition": transition_world,
            "world_nucleus": nucleus_world,
            "transition_likelihood": np.clip(0.55 * merged_scores["transition_likelihood"] + 0.45 * transition_world, 0.0, 1.0).astype(np.float32),
            "voicing": np.clip(0.55 * merged_scores["voicing"] + 0.45 * world_voicing, 0.0, 1.0).astype(np.float32),
            "nucleus_likelihood": nucleus_world,
            "onset_strength": onset_interp,
            "acoustic_feature_set_world_v1": np.ones_like(nucleus_world, dtype=np.float32),
        }
    )
    world_cols = np.stack(
        [
            world_voicing,
            world_periodicity,
            world_stability,
            transition_world,
            nucleus_world,
            onset_interp,
            rms_interp,
        ],
        axis=1,
    ).astype(np.float32)
    features = np.concatenate([aux.features, _standardize(world_cols)], axis=1).astype(np.float32)
    return AcousticAuxFeatures(
        times_ms=aux.times_ms,
        features=features,
        scores=merged_scores,
    )


def acoustic_frame_features(
    samples: np.ndarray,
    sample_rate: int,
    *,
    config: AcousticFeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or AcousticFeatureConfig(sample_rate=sample_rate)
    frame_size = max(1, int(round(sample_rate * cfg.frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * cfg.hop_ms / 1000.0)))
    if samples.shape[0] < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.shape[0]))
    frame_count = 1 + max(0, (samples.shape[0] - frame_size) // hop)
    window = np.hanning(frame_size).astype(np.float32)
    previous_mag: np.ndarray | None = None
    features: list[np.ndarray] = []
    for idx in range(frame_count):
        start = idx * hop
        frame = samples[start : start + frame_size]
        if frame.shape[0] < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.shape[0]))
        weighted = frame * window
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        log_energy = float(np.log1p(100.0 * rms))
        signs = np.signbit(frame)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if frame.shape[0] > 1 else 0.0
        mag = np.abs(np.fft.rfft(weighted)).astype(np.float32)
        if mag.sum() > 0.0:
            mag_norm = mag / (float(mag.sum()) + 1e-8)
            centroid = float(np.sum(np.linspace(0.0, 1.0, mag.shape[0]) * mag_norm))
        else:
            centroid = 0.0
        flux = 0.0 if previous_mag is None else float(np.mean(np.maximum(0.0, mag - previous_mag)))
        previous_mag = mag
        bands = _pool_spectrum(mag, cfg.fft_bins)
        features.append(np.concatenate([[log_energy, rms, zcr, centroid, flux], bands]).astype(np.float32))
    feats = np.stack(features, axis=0)
    feats = _standardize(feats)
    times_ms = (np.arange(frame_count, dtype=np.float32) * float(hop) * 1000.0) / float(sample_rate)
    return times_ms, feats


def acoustic_aux_features(
    samples: np.ndarray,
    sample_rate: int,
    *,
    config: AcousticFeatureConfig | None = None,
) -> AcousticAuxFeatures:
    cfg = config or AcousticFeatureConfig(sample_rate=sample_rate)
    frame_size = max(1, int(round(sample_rate * cfg.frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * cfg.hop_ms / 1000.0)))
    if samples.shape[0] < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.shape[0]))
    frame_count = 1 + max(0, (samples.shape[0] - frame_size) // hop)
    window = np.hanning(frame_size).astype(np.float32)
    raw_rows: list[np.ndarray] = []
    mel_rows: list[np.ndarray] = []
    previous_mag: np.ndarray | None = None
    previous_rms = 0.0
    for idx in range(frame_count):
        start = idx * hop
        frame = samples[start : start + frame_size]
        if frame.shape[0] < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.shape[0]))
        weighted = frame * window
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        log_rms = float(np.log10(rms + 1e-8))
        delta_rms = rms - previous_rms
        previous_rms = rms
        signs = np.signbit(frame)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if frame.shape[0] > 1 else 0.0
        mag = np.abs(np.fft.rfft(weighted)).astype(np.float32)
        if mag.sum() > 0.0:
            mag_norm = mag / (float(mag.sum()) + 1e-8)
            centroid = float(np.sum(np.linspace(0.0, 1.0, mag.shape[0]) * mag_norm))
        else:
            centroid = 0.0
        flux = 0.0 if previous_mag is None else float(np.mean(np.maximum(0.0, mag - previous_mag)))
        previous_mag = mag
        harmonicity = _autocorr_harmonicity(frame, sample_rate)
        mel = np.log1p(_pool_spectrum(mag, cfg.aux_fft_bins)).astype(np.float32)
        raw_rows.append(np.asarray([rms, log_rms, delta_rms, zcr, centroid, flux, harmonicity], dtype=np.float32))
        mel_rows.append(mel)
    raw = np.stack(raw_rows, axis=0).astype(np.float32)
    mel_bands = np.stack(mel_rows, axis=0).astype(np.float32)
    silence_score = _silence_likelihood(raw[:, 0])
    flux_score = _robust_unit(raw[:, 5])
    voicing_score = np.clip(raw[:, 6], 0.0, 1.0).astype(np.float32)
    energy_rise = _positive_unit(raw[:, 2])
    transition_score = np.clip(0.65 * flux_score + 0.35 * energy_rise, 0.0, 1.0).astype(np.float32)
    # Sonorant (m/n/l/r/w/y) onsets carry little spectral flux because voicing
    # is continuous across the boundary; an energy rise gated by voicing keeps a
    # cv-boundary cue where the flux-based transition score under-fires.
    sonorant_onset_score = np.clip(voicing_score * energy_rise, 0.0, 1.0).astype(np.float32)
    features = np.concatenate(
        [
            _standardize(raw),
            _standardize(mel_bands),
            silence_score[:, None],
            flux_score[:, None],
            voicing_score[:, None],
            transition_score[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    times_ms = (np.arange(frame_count, dtype=np.float32) * float(hop) * 1000.0) / float(sample_rate)
    return AcousticAuxFeatures(
        times_ms=times_ms,
        features=features,
        scores={
            "rms": raw[:, 0].astype(np.float32),
            "log_rms": raw[:, 1].astype(np.float32),
            "delta_rms": raw[:, 2].astype(np.float32),
            "zcr": raw[:, 3].astype(np.float32),
            "spectral_centroid": raw[:, 4].astype(np.float32),
            "spectral_flux": raw[:, 5].astype(np.float32),
            "voicing": voicing_score,
            "silence_likelihood": silence_score,
            "flux_likelihood": flux_score,
            "transition_likelihood": transition_score,
            "sonorant_onset_likelihood": sonorant_onset_score,
        },
    )


def spectrogram_image(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    bins: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = AcousticFeatureConfig(sample_rate=sample_rate, frame_ms=frame_ms, hop_ms=hop_ms, fft_bins=bins)
    frame_size = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    frame_count = 1 + max(0, (samples.shape[0] - frame_size) // hop)
    window = np.hanning(frame_size).astype(np.float32)
    rows: list[np.ndarray] = []
    for idx in range(frame_count):
        start = idx * hop
        frame = samples[start : start + frame_size]
        if frame.shape[0] < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.shape[0]))
        mag = np.log1p(np.abs(np.fft.rfft(frame * window))).astype(np.float32)
        rows.append(_pool_spectrum(mag, cfg.fft_bins))
    image = np.stack(rows, axis=0)
    image = image - float(image.min())
    if float(image.max()) > 0.0:
        image = image / float(image.max())
    times_ms = (np.arange(frame_count, dtype=np.float32) * float(hop) * 1000.0) / float(sample_rate)
    return times_ms, image


def extract_ssl_features(
    wav_path: str | Path,
    *,
    encoder: str,
    device: str | None = None,
) -> FeatureBatch:
    model_id = SSL_MODEL_IDS.get(encoder, encoder)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SSL encoder mode requires torch and transformers. Install ML dependencies or use --encoder acoustic."
        ) from exc
    source_sample_rate = _wav_sample_rate_if_available(wav_path)
    samples, sample_rate = read_wav_mono(wav_path, target_sample_rate=16000)
    processor, model = _get_ssl_model(model_id, device=device)
    target_device = next(model.parameters()).device
    inputs = processor(samples, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(target_device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[0].detach().cpu().float().numpy()
    duration_ms = 1000.0 * float(samples.shape[0]) / float(sample_rate)
    step = duration_ms / max(1, output.shape[0])
    times_ms = np.arange(output.shape[0], dtype=np.float32) * float(step)
    return FeatureBatch(
        times_ms=times_ms,
        features=output.astype(np.float32),
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        encoder=encoder,
        source_sample_rate=source_sample_rate or sample_rate,
    )


def _get_ssl_model(model_id: str, *, device: str | None = None):
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "SSL encoder mode requires torch and transformers. Install ML dependencies or use --encoder acoustic."
        ) from exc
    cache_device = str(device or "cpu")
    key = (model_id, cache_device)
    cached = _SSL_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    if device:
        model = model.to(device)
    model.eval()
    _SSL_MODEL_CACHE[key] = (processor, model)
    return processor, model


def extract_ssl_plus_aux_features(
    wav_path: str | Path,
    *,
    encoder: str,
    acoustic_config: AcousticFeatureConfig | None = None,
    device: str | None = None,
) -> FeatureBatch:
    ssl = extract_ssl_features(wav_path, encoder=encoder, device=device)
    cfg = acoustic_config or AcousticFeatureConfig()
    samples, sample_rate = read_wav_mono(wav_path, target_sample_rate=cfg.sample_rate)
    aux = acoustic_aux_features(samples, sample_rate, config=cfg)
    aux_interp = _interp_feature_matrix(aux.times_ms, aux.features, ssl.times_ms)
    scores = {
        key: _interp_vector(aux.times_ms, value, ssl.times_ms)
        for key, value in aux.scores.items()
    }
    return FeatureBatch(
        times_ms=ssl.times_ms,
        features=np.concatenate([ssl.features, aux_interp], axis=1).astype(np.float32),
        sample_rate=ssl.sample_rate,
        duration_ms=ssl.duration_ms,
        encoder=f"{encoder}+aux",
        acoustic_scores=scores,
        source_sample_rate=ssl.source_sample_rate or ssl.sample_rate,
        frame_ms=ssl.frame_ms,
        hop_ms=ssl.hop_ms,
    )


def _wav_sample_rate_if_available(path: str | Path) -> int:
    try:
        with wave.open(str(path), "rb") as handle:
            return int(handle.getframerate())
    except Exception:
        return 0


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32)
    source_times = np.arange(samples.shape[0], dtype=np.float64) / float(source_rate)
    target_count = max(1, int(round(samples.shape[0] * float(target_rate) / float(source_rate))))
    target_times = np.arange(target_count, dtype=np.float64) / float(target_rate)
    return np.interp(target_times, source_times, samples).astype(np.float32)


def _interp_feature_matrix(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if source_times.shape[0] == target_times.shape[0] and np.allclose(source_times, target_times):
        return values.astype(np.float32)
    cols = [
        np.interp(target_times.astype(np.float64), source_times.astype(np.float64), values[:, idx].astype(np.float64))
        for idx in range(values.shape[1])
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def _interp_vector(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if source_times.shape[0] == target_times.shape[0] and np.allclose(source_times, target_times):
        return values.astype(np.float32)
    return np.interp(target_times.astype(np.float64), source_times.astype(np.float64), values.astype(np.float64)).astype(np.float32)


def _pool_spectrum(mag: np.ndarray, bins: int) -> np.ndarray:
    if mag.shape[0] == 0:
        return np.zeros((bins,), dtype=np.float32)
    edges = np.linspace(0, mag.shape[0], bins + 1, dtype=np.int64)
    pooled = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            pooled.append(0.0)
        else:
            pooled.append(float(np.mean(mag[start:end])))
    return np.asarray(pooled, dtype=np.float32)


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-5)).astype(np.float32)


def _autocorr_harmonicity(frame: np.ndarray, sample_rate: int) -> float:
    centered = frame.astype(np.float32) - float(np.mean(frame))
    energy = float(np.dot(centered, centered)) + 1e-8
    if energy <= 1e-7:
        return 0.0
    min_lag = max(1, int(round(sample_rate / 600.0)))
    max_lag = min(centered.shape[0] - 1, int(round(sample_rate / 70.0)))
    if max_lag <= min_lag:
        return 0.0
    # FFT-based linear autocorrelation: autocorr[lag] == dot(centered[:-lag], centered[lag:]).
    # Zero-padding to >= 2*n avoids circular wrap-around.
    n = int(centered.shape[0])
    fft_size = 1 << int(np.ceil(np.log2(max(2, 2 * n))))
    spectrum = np.fft.rfft(centered, fft_size)
    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)
    lag_scores = autocorr[min_lag : max_lag + 1] / energy
    if lag_scores.size == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.max(lag_scores))))


def _robust_unit(values: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(values, 10.0))
    hi = float(np.percentile(values, 90.0))
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _positive_unit(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(values, 0.0)
    hi = float(np.percentile(positive, 90.0))
    if hi <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(positive / hi, 0.0, 1.0).astype(np.float32)


def _silence_likelihood(rms: np.ndarray) -> np.ndarray:
    speech = _robust_unit(rms)
    return np.clip(1.0 - speech, 0.0, 1.0).astype(np.float32)


def _safe_smooth_track(values: np.ndarray, gaussian_filter1d_func) -> np.ndarray:
    if values.size <= 2:
        return values.astype(np.float32)
    return gaussian_filter1d_func(values.astype(np.float32), sigma=1.0, mode="nearest").astype(np.float32)
