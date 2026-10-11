from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio import _robust_normalize, _select_main_activity_region, _smooth, file_sha256


LEGACY_FEATURE_BACKEND_ID = "numpy_spectral_hsmm_v1"
FEATURE_BACKEND_ID = "numpy_multiband_pitch_v2"
MULTIBAND_V3_FEATURE_BACKEND_ID = "numpy_multiband_pitch_v3"
PADDING_STABLE_FEATURE_BACKEND_ID = "numpy_multiband_pitch_padding_v1"
WORLD_FEATURE_BACKEND_ID = "world_fusion_v2"
NEURAL_FEATURE_BACKEND_ID = "onnx_neural_evidence_v1"
SUPPORTED_FEATURE_BACKENDS = frozenset(
    {
        LEGACY_FEATURE_BACKEND_ID,
        FEATURE_BACKEND_ID,
        MULTIBAND_V3_FEATURE_BACKEND_ID,
        PADDING_STABLE_FEATURE_BACKEND_ID,
        WORLD_FEATURE_BACKEND_ID,
        NEURAL_FEATURE_BACKEND_ID,
    }
)


@dataclass(frozen=True)
class AcousticEvidence:
    wav_path: Path
    wav_sha256: str
    source_sample_rate: int
    analysis_sample_rate: int
    duration_ms: float
    frame_times_ms: np.ndarray
    tracks: Mapping[str, np.ndarray]
    active_start_ms: float
    active_end_ms: float
    feature_config_sha256: str
    warnings: tuple[str, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _FrameAnalysis:
    mono: np.ndarray
    source_sample_rate: int
    analysis_sample_rate: int
    duration_ms: float
    frames: np.ndarray
    frame_times_ms: np.ndarray
    rms_db: np.ndarray
    magnitude: np.ndarray
    normalized_magnitude: np.ndarray
    frequencies: np.ndarray
    frame_ms: float
    hop_ms: float
    resample_method: str
    channel_mix_policy: str


def extract_acoustic_evidence(
    path: Path,
    *,
    internal_sample_rate: int,
    convert_to_mono: bool,
    resampler: str,
    feature_backend: str = FEATURE_BACKEND_ID,
    expected_activity_regions: int = 1,
    evidence_model: Path | None = None,
    evidence_provider: str = "cuda",
    preserve_leading_silence: bool = False,
    dense_speech_activity_floor: bool = False,
) -> AcousticEvidence:
    backend = str(feature_backend).strip().casefold()
    if backend not in SUPPORTED_FEATURE_BACKENDS:
        raise ValueError(f"지원하지 않는 acoustic feature backend입니다: {feature_backend}")
    frame_ms = 25.0 if backend == LEGACY_FEATURE_BACKEND_ID else 30.0
    padding_details: dict[str, object] = {}
    analysis = _frame_analysis(
        path,
        internal_sample_rate=internal_sample_rate,
        convert_to_mono=convert_to_mono,
        resampler=resampler,
        frame_ms=frame_ms,
        hop_ms=5.0,
        canonicalize_digital_padding=(backend == PADDING_STABLE_FEATURE_BACKEND_ID and not preserve_leading_silence),
    )
    if backend == PADDING_STABLE_FEATURE_BACKEND_ID:
        origin_ms = float(analysis.frame_times_ms[0]) - analysis.frame_ms / 2.0
        padding_details = {
            "digital_padding_policy": ("preserve_explicit_leading_silence_v1" if preserve_leading_silence
                                       else "exclude_exact_zero_prefix_before_resampling_v1"),
            "analysis_origin_ms": max(0.0, origin_ms),
            "alignment_duration_ms": max(0.0, analysis.duration_ms - origin_ms),
        }
    warnings: list[str] = []
    if backend == LEGACY_FEATURE_BACKEND_ID:
        tracks = _legacy_tracks(analysis)
        backend_details: dict[str, object] = {"pitch_method": "spectral_flatness_proxy"}
    else:
        tracks = _multiband_pitch_tracks(analysis)
        backend_details = {
            "pitch_method": "fft_autocorrelation_parabolic_v1",
            "pitch_floor_hz": 55.0,
            "pitch_ceiling_hz": 700.0,
            "spectral_bands_hz": [0.0, 500.0, 3000.0, float(analysis.analysis_sample_rate) / 2.0],
        }
        if backend == MULTIBAND_V3_FEATURE_BACKEND_ID:
            tracks = _multiband_pitch_v3_tracks(tracks)
            backend_details.update(
                {
                    "pitch_confidence_policy": "harmonicity_energy_voicing_v1",
                    "local_contrast_policy": "causal_acausal_20ms_v1",
                    "local_contrast_radius_frames": 4,
                }
            )
        if backend == NEURAL_FEATURE_BACKEND_ID:
            from .neural_evidence import infer_neural_tracks

            neural_tracks, neural_details = infer_neural_tracks(
                analysis,
                tracks,
                model_path=evidence_model,
                provider=evidence_provider,
            )
            tracks.update(neural_tracks)
            backend_details.update(neural_details)
        if backend == WORLD_FEATURE_BACKEND_ID:
            tracks, world_details = _fuse_world_tracks(analysis, tracks)
            backend_details.update(world_details)

    expected_activity_regions = max(1, int(expected_activity_regions))
    active_start_ms, active_end_ms, activity_details = _activity_region(
        analysis,
        tracks,
        warnings,
        expected_regions=expected_activity_regions,
        dense_speech_floor=dense_speech_activity_floor,
    )
    config = {
        "schema_version": 2 if backend != LEGACY_FEATURE_BACKEND_ID else 1,
        "feature_backend": backend,
        "source_sample_rate": int(analysis.source_sample_rate),
        "analysis_sample_rate": int(analysis.analysis_sample_rate),
        "frame_ms": analysis.frame_ms,
        "hop_ms": analysis.hop_ms,
        "frame_origin": "center",
        "feature_latency_ms": 0.0,
        "resample_method": analysis.resample_method,
        "channel_mix_policy": analysis.channel_mix_policy,
        "amplitude_normalization_policy": "none",
        "activity_region_policy": "sequence_aware_v2" + ("_dense_floor" if dense_speech_activity_floor else ""),
        "expected_activity_regions": expected_activity_regions,
        **backend_details,
        **padding_details,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AcousticEvidence(
        wav_path=path,
        wav_sha256=file_sha256(path),
        source_sample_rate=int(analysis.source_sample_rate),
        analysis_sample_rate=int(analysis.analysis_sample_rate),
        duration_ms=float(analysis.duration_ms),
        frame_times_ms=analysis.frame_times_ms.astype(np.float64, copy=False),
        tracks={name: np.asarray(values, dtype=np.float64) for name, values in tracks.items()},
        active_start_ms=float(active_start_ms),
        active_end_ms=float(active_end_ms),
        feature_config_sha256=config_hash,
        warnings=tuple(dict.fromkeys(warnings)),
        metadata={
            **config,
            **activity_details,
            "track_names": sorted(tracks),
            "timebase": {
                "schema_version": 1,
                "sample_rate": int(analysis.source_sample_rate),
                "analysis_sample_rate": int(analysis.analysis_sample_rate),
                "frame_shift_ms": analysis.hop_ms,
                "window_size_ms": analysis.frame_ms,
                "frame_time_anchor": "center",
                "feature_latency_ms": 0.0,
                "resample_method": analysis.resample_method,
            },
        },
    )


def write_acoustic_evidence_npz(
    evidence: AcousticEvidence,
    output: Path,
    *,
    force: bool = False,
) -> Path:
    target = output.resolve()
    if target.exists() and not force:
        raise FileExistsError(f"acoustic evidence artifact가 이미 있습니다: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    track_names = sorted(evidence.tracks)
    payload: dict[str, np.ndarray] = {
        "artifact_schema_version": np.asarray(1, dtype=np.int64),
        "wav_path": np.asarray(str(evidence.wav_path.resolve())),
        "wav_sha256": np.asarray(evidence.wav_sha256),
        "source_sample_rate": np.asarray(evidence.source_sample_rate, dtype=np.int64),
        "analysis_sample_rate": np.asarray(evidence.analysis_sample_rate, dtype=np.int64),
        "duration_ms": np.asarray(evidence.duration_ms, dtype=np.float64),
        "frame_times_ms": np.asarray(evidence.frame_times_ms, dtype=np.float64),
        "active_start_ms": np.asarray(evidence.active_start_ms, dtype=np.float64),
        "active_end_ms": np.asarray(evidence.active_end_ms, dtype=np.float64),
        "feature_config_sha256": np.asarray(evidence.feature_config_sha256),
        "warnings_json": np.asarray(json.dumps(list(evidence.warnings), ensure_ascii=False)),
        "metadata_json": np.asarray(json.dumps(dict(evidence.metadata), ensure_ascii=False, sort_keys=True)),
        "track_names_json": np.asarray(json.dumps(track_names)),
    }
    for index, name in enumerate(track_names):
        payload[f"track_{index:03d}"] = np.asarray(evidence.tracks[name], dtype=np.float64)
    handle = tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".npz", dir=target.parent, delete=False)
    temp_path = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temp_path, **payload)
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def load_acoustic_evidence_npz(
    artifact: Path,
    *,
    expected_wav_sha256: str | None = None,
) -> AcousticEvidence:
    source = artifact.resolve()
    with np.load(source, allow_pickle=False) as payload:
        if int(payload["artifact_schema_version"]) != 1:
            raise ValueError("지원하지 않는 acoustic evidence artifact schema입니다")
        wav_sha256 = str(payload["wav_sha256"])
        if expected_wav_sha256 is not None and wav_sha256 != expected_wav_sha256:
            raise ValueError("acoustic evidence artifact의 WAV SHA-256이 다릅니다")
        frame_times = np.asarray(payload["frame_times_ms"], dtype=np.float64)
        track_names = json.loads(str(payload["track_names_json"]))
        if not isinstance(track_names, list) or any(not isinstance(name, str) for name in track_names):
            raise ValueError("acoustic evidence artifact track_names가 유효하지 않습니다")
        tracks = {
            name: np.asarray(payload[f"track_{index:03d}"], dtype=np.float64)
            for index, name in enumerate(track_names)
        }
        if any(values.shape != frame_times.shape for values in tracks.values()):
            raise ValueError("acoustic evidence artifact track 길이가 timebase와 다릅니다")
        if not np.all(np.isfinite(frame_times)) or any(not np.all(np.isfinite(values)) for values in tracks.values()):
            raise ValueError("acoustic evidence artifact에 non-finite 값이 있습니다")
        metadata = json.loads(str(payload["metadata_json"]))
        warnings = json.loads(str(payload["warnings_json"]))
        if not isinstance(metadata, Mapping) or not isinstance(warnings, list):
            raise ValueError("acoustic evidence artifact metadata가 유효하지 않습니다")
        return AcousticEvidence(
            wav_path=Path(str(payload["wav_path"])),
            wav_sha256=wav_sha256,
            source_sample_rate=int(payload["source_sample_rate"]),
            analysis_sample_rate=int(payload["analysis_sample_rate"]),
            duration_ms=float(payload["duration_ms"]),
            frame_times_ms=frame_times,
            tracks=tracks,
            active_start_ms=float(payload["active_start_ms"]),
            active_end_ms=float(payload["active_end_ms"]),
            feature_config_sha256=str(payload["feature_config_sha256"]),
            warnings=tuple(str(item) for item in warnings),
            metadata=dict(metadata),
        )
def _frame_analysis(
    path: Path,
    *,
    internal_sample_rate: int,
    convert_to_mono: bool,
    resampler: str,
    frame_ms: float,
    hop_ms: float,
    canonicalize_digital_padding: bool = False,
) -> _FrameAnalysis:
    audio, source_sample_rate = sf.read(path, dtype="float32", always_2d=True)
    first = 0
    if canonicalize_digital_padding:
        nonzero = np.flatnonzero(np.any(audio != 0.0, axis=1))
        if len(nonzero):
            first = int(nonzero[0])
    analysis = _frame_analysis_samples(
        audio[first:],
        source_sample_rate,
        internal_sample_rate=internal_sample_rate,
        convert_to_mono=convert_to_mono,
        resampler=resampler,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        time_offset_ms=first * 1000.0 / source_sample_rate,
        source_label=str(path),
    )
    return replace(analysis, duration_ms=len(audio) * 1000.0 / source_sample_rate)


def _frame_analysis_segment(
    path: Path,
    *,
    start_ms: float,
    end_ms: float,
    internal_sample_rate: int,
    convert_to_mono: bool,
    resampler: str,
    frame_ms: float,
    hop_ms: float,
) -> _FrameAnalysis:
    """Analyze a seekable WAV span while retaining original absolute frame times."""
    info = sf.info(path)
    source_sample_rate = int(info.samplerate)
    first = max(0, min(int(info.frames), int(round(float(start_ms) * source_sample_rate / 1000.0))))
    last = max(first + 1, min(int(info.frames), int(round(float(end_ms) * source_sample_rate / 1000.0))))
    with sf.SoundFile(path) as handle:
        handle.seek(first)
        audio = handle.read(last - first, dtype="float32", always_2d=True)
    return _frame_analysis_samples(
        audio,
        source_sample_rate,
        internal_sample_rate=internal_sample_rate,
        convert_to_mono=convert_to_mono,
        resampler=resampler,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        time_offset_ms=1000.0 * first / source_sample_rate,
        source_label=str(path),
    )


def _frame_analysis_samples(
    audio: np.ndarray,
    source_sample_rate: int,
    *,
    internal_sample_rate: int,
    convert_to_mono: bool,
    resampler: str,
    frame_ms: float,
    hop_ms: float,
    time_offset_ms: float,
    source_label: str,
) -> _FrameAnalysis:
    if audio.size == 0:
        raise ValueError(f"빈 WAV입니다: {source_label}")
    if audio.shape[1] > 1 and not convert_to_mono:
        raise ValueError(f"다채널 WAV에는 audio.convert_to_mono=true가 필요합니다: {source_label}")
    mono = np.mean(audio, axis=1, dtype=np.float64)
    mono = np.nan_to_num(mono, copy=False)
    duration_ms = 1000.0 * len(mono) / float(source_sample_rate)
    analysis_sample_rate = int(internal_sample_rate)
    normalized_resampler = resampler.strip().casefold()
    if analysis_sample_rate != source_sample_rate:
        if normalized_resampler not in {"soxr_hq", "scipy_polyphase", "polyphase"}:
            raise ValueError(f"지원하지 않는 내부 resampler입니다: {resampler}")
        divisor = math.gcd(source_sample_rate, analysis_sample_rate)
        mono = resample_poly(mono, analysis_sample_rate // divisor, source_sample_rate // divisor)
        resample_method = "scipy.signal.resample_poly"
    else:
        resample_method = "identity"

    frame_length = max(128, int(round(analysis_sample_rate * frame_ms / 1000.0)))
    hop_length = max(32, int(round(analysis_sample_rate * hop_ms / 1000.0)))
    if len(mono) < frame_length:
        mono = np.pad(mono, (0, frame_length - len(mono)))
    remainder = (len(mono) - frame_length) % hop_length
    if remainder:
        mono = np.pad(mono, (0, hop_length - remainder))
    frames = np.lib.stride_tricks.sliding_window_view(mono, frame_length)[::hop_length]
    window = np.hanning(frame_length).astype(np.float64)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-10
    normalized_magnitude = magnitude / np.maximum(np.sum(magnitude, axis=1, keepdims=True), 1e-10)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-8)
    frame_times_ms = (
        np.arange(len(frames), dtype=np.float64) * hop_length + frame_length / 2.0
    ) * 1000.0 / analysis_sample_rate + float(time_offset_ms)
    return _FrameAnalysis(
        mono=mono,
        source_sample_rate=int(source_sample_rate),
        analysis_sample_rate=analysis_sample_rate,
        duration_ms=float(duration_ms),
        frames=frames,
        frame_times_ms=frame_times_ms,
        rms_db=rms_db,
        magnitude=magnitude,
        normalized_magnitude=normalized_magnitude,
        frequencies=np.fft.rfftfreq(frame_length, d=1.0 / analysis_sample_rate),
        frame_ms=float(frame_length) * 1000.0 / analysis_sample_rate,
        hop_ms=float(hop_length) * 1000.0 / analysis_sample_rate,
        resample_method=resample_method,
        channel_mix_policy="mean" if audio.shape[1] > 1 else "mono_identity",
    )


def _legacy_tracks(analysis: _FrameAnalysis) -> dict[str, np.ndarray]:
    magnitude = analysis.magnitude
    normalized = analysis.normalized_magnitude
    positive_delta = np.maximum(normalized[1:] - normalized[:-1], 0.0)
    flux = np.concatenate(([0.0], np.sqrt(np.sum(positive_delta * positive_delta, axis=1))))
    flatness = np.exp(np.mean(np.log(magnitude), axis=1)) / np.maximum(np.mean(magnitude, axis=1), 1e-9)
    periodicity = np.clip(1.0 - flatness, 0.0, 1.0)
    high_mask = analysis.frequencies >= min(3000.0, analysis.analysis_sample_rate * 0.35)
    high_band = np.sum(normalized[:, high_mask], axis=1) if np.any(high_mask) else np.zeros(len(analysis.frames))
    zcr = np.mean(np.abs(np.diff(np.signbit(analysis.frames), axis=1)), axis=1)
    energy = _robust_normalize(analysis.rms_db)
    flux_norm = _robust_normalize(flux)
    periodicity_norm = _robust_normalize(periodicity)
    high_norm = _robust_normalize(high_band)
    zcr_norm = _robust_normalize(zcr)
    energy_rise = _robust_normalize(np.maximum(np.gradient(energy), 0.0))
    periodicity_rise = _robust_normalize(np.maximum(np.gradient(periodicity_norm), 0.0))
    boundary = _smooth(0.50 * flux_norm + 0.30 * energy_rise + 0.20 * periodicity_rise, 3)
    transition = _smooth(
        0.38 * periodicity_rise + 0.27 * energy_rise + 0.20 * periodicity_norm + 0.15 * flux_norm,
        3,
    )
    aperiodic = np.clip(1.0 - periodicity_norm, 0.0, 1.0)
    low_energy = np.clip(1.0 - energy, 0.0, 1.0)
    stability = np.clip(1.0 - _smooth(flux_norm, 5), 0.0, 1.0)
    vowel_like = np.clip(0.45 * periodicity_norm + 0.30 * energy + 0.25 * stability, 0.0, 1.0)
    consonant_like = np.clip(0.35 * flux_norm + 0.30 * aperiodic + 0.20 * high_norm + 0.15 * energy, 0.0, 1.0)
    return {
        "energy": energy,
        "low_energy": low_energy,
        "active": np.clip(0.75 * energy + 0.25 * np.maximum(periodicity_norm, aperiodic), 0.0, 1.0),
        "flux": flux_norm,
        "periodicity": periodicity_norm,
        "aperiodic": aperiodic,
        "energy_rise": energy_rise,
        "periodicity_rise": periodicity_rise,
        "high_band_energy": high_norm,
        "zcr": zcr_norm,
        "spectral_stability": stability,
        "boundary": boundary,
        "transition": transition,
        "vowel_like": vowel_like,
        "consonant_like": consonant_like,
    }


def _multiband_pitch_tracks(analysis: _FrameAnalysis) -> dict[str, np.ndarray]:
    normalized = analysis.normalized_magnitude
    frequencies = analysis.frequencies
    energy = _robust_normalize(analysis.rms_db)
    low_energy = np.clip(1.0 - energy, 0.0, 1.0)
    spectral_delta = np.zeros(len(normalized), dtype=np.float64)
    if len(normalized) > 1:
        spectral_delta[1:] = np.sum(np.abs(normalized[1:] - normalized[:-1]), axis=1)
    positive_delta = np.maximum(normalized[1:] - normalized[:-1], 0.0)
    flux = np.concatenate(([0.0], np.sqrt(np.sum(positive_delta * positive_delta, axis=1))))
    flux_norm = _robust_normalize(flux)
    shape_delta = _robust_normalize(spectral_delta)

    flatness = np.exp(np.mean(np.log(analysis.magnitude), axis=1)) / np.maximum(
        np.mean(analysis.magnitude, axis=1), 1e-10
    )
    flatness_norm = np.clip(flatness, 0.0, 1.0)
    entropy = -np.sum(normalized * np.log(normalized + 1e-12), axis=1) / math.log(normalized.shape[1])
    entropy = np.clip(entropy, 0.0, 1.0)
    centroid_hz = np.sum(normalized * frequencies[None, :], axis=1)
    centroid = np.clip(centroid_hz / max(1.0, analysis.analysis_sample_rate / 2.0), 0.0, 1.0)
    cumulative = np.cumsum(normalized, axis=1)
    rolloff_indices = np.argmax(cumulative >= 0.85, axis=1)
    rolloff = np.clip(frequencies[rolloff_indices] / max(1.0, analysis.analysis_sample_rate / 2.0), 0.0, 1.0)
    low_band = _band_ratio(normalized, frequencies, 0.0, 500.0)
    mid_band = _band_ratio(normalized, frequencies, 500.0, 3000.0)
    high_band = _band_ratio(normalized, frequencies, 3000.0, analysis.analysis_sample_rate / 2.0 + 1.0)
    low_band_norm = _robust_normalize(low_band)
    mid_band_norm = _robust_normalize(mid_band)
    high_band_norm = _robust_normalize(high_band)
    zcr = _robust_normalize(np.mean(np.abs(np.diff(np.signbit(analysis.frames), axis=1)), axis=1))

    f0_hz, harmonicity = _autocorrelation_pitch(
        analysis.frames,
        analysis.analysis_sample_rate,
        floor_hz=55.0,
        ceiling_hz=700.0,
    )
    periodicity = np.clip((harmonicity - 0.12) / 0.68, 0.0, 1.0)
    periodicity *= np.clip(0.25 + 0.75 * energy, 0.0, 1.0)
    hnr_db = 10.0 * np.log10(np.clip(harmonicity, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - harmonicity, 1e-5, 1.0))
    hnr = np.clip((hnr_db + 5.0) / 30.0, 0.0, 1.0)
    voiced_mask = (periodicity >= 0.25) & (energy >= 0.08)
    f0_hz = np.where(voiced_mask, f0_hz, 0.0)
    log_f0 = np.zeros_like(f0_hz)
    log_f0[voiced_mask] = np.log2(np.maximum(f0_hz[voiced_mask], 1.0))
    f0_delta = np.zeros_like(log_f0)
    if len(log_f0) > 1:
        valid_pairs = voiced_mask[1:] & voiced_mask[:-1]
        f0_delta[1:][valid_pairs] = np.abs(log_f0[1:][valid_pairs] - log_f0[:-1][valid_pairs])
    f0_stability = np.where(voiced_mask, np.clip(1.0 - _robust_normalize(f0_delta), 0.0, 1.0), 0.0)
    voicing = np.clip(
        0.48 * periodicity + 0.22 * hnr + 0.18 * (1.0 - flatness_norm) + 0.12 * energy,
        0.0,
        1.0,
    )
    aperiodic = np.clip(0.48 * (1.0 - periodicity) + 0.27 * flatness_norm + 0.25 * entropy, 0.0, 1.0)
    primitives = {
        "energy": energy,
        "low_energy": low_energy,
        "flux": flux_norm,
        "periodicity": periodicity,
        "aperiodic": aperiodic,
        "hnr": hnr,
        "voicing": voicing,
        "f0_hz": f0_hz,
        "f0_stability": f0_stability,
        "spectral_flatness": flatness_norm,
        "spectral_entropy": entropy,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "spectral_shape_delta": shape_delta,
        "low_band_ratio": low_band_norm,
        "mid_band_ratio": mid_band_norm,
        "high_band_energy": high_band_norm,
        "zcr": zcr,
    }
    return _derived_tracks(primitives)


def _multiband_pitch_v3_tracks(base_tracks: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add confidence-gated pitch and local before/after evidence to v2 tracks.

    This intentionally extends rather than mutates v2.  It keeps v2 artifacts
    replayable while allowing the HSMM profile to distinguish a reliable voiced
    onset from a high-energy, weakly periodic consonant or breath.
    """
    tracks = {name: np.asarray(values, dtype=np.float64).copy() for name, values in base_tracks.items()}
    energy = tracks["energy"]
    periodicity = tracks["periodicity"]
    hnr = tracks["hnr"]
    voicing = tracks["voicing"]
    f0_present = (tracks["f0_hz"] > 0.0).astype(np.float64)
    pitch_confidence = np.clip(
        (0.43 * periodicity + 0.27 * hnr + 0.20 * voicing + 0.10 * energy)
        * (0.20 + 0.80 * f0_present),
        0.0,
        1.0,
    )
    energy_rise_local = _local_rise_contrast(energy)
    periodicity_rise_local = _local_rise_contrast(periodicity)
    voicing_rise_local = _local_rise_contrast(voicing)
    aperiodicity_fall_local = _local_fall_contrast(tracks["aperiodic"])
    spectral_change_local = _local_peak_contrast(tracks["spectral_shape_delta"])
    high_band_rise_local = _local_rise_contrast(tracks["high_band_energy"])
    pitch_confidence_rise_local = _local_rise_contrast(pitch_confidence)
    vowel_onset = np.clip(
        0.27 * periodicity_rise_local
        + 0.22 * voicing_rise_local
        + 0.18 * pitch_confidence_rise_local
        + 0.14 * aperiodicity_fall_local
        + 0.11 * energy_rise_local
        + 0.08 * spectral_change_local,
        0.0,
        1.0,
    )
    boundary_local = np.clip(
        0.31 * spectral_change_local
        + 0.22 * energy_rise_local
        + 0.19 * periodicity_rise_local
        + 0.16 * voicing_rise_local
        + 0.12 * high_band_rise_local,
        0.0,
        1.0,
    )
    burst_local = np.clip(
        0.36 * spectral_change_local
        + 0.29 * energy_rise_local
        + 0.21 * high_band_rise_local
        + 0.14 * tracks["flux"],
        0.0,
        1.0,
    )
    tracks.update(
        {
            "pitch_confidence": pitch_confidence,
            "pitch_confidence_rise_local": pitch_confidence_rise_local,
            "energy_rise_local": energy_rise_local,
            "periodicity_rise_local": periodicity_rise_local,
            "voicing_rise_local": voicing_rise_local,
            "aperiodicity_fall_local": aperiodicity_fall_local,
            "spectral_change_local": spectral_change_local,
            "high_band_rise_local": high_band_rise_local,
            "vowel_onset_likelihood": vowel_onset,
            "boundary_local_contrast": boundary_local,
            "burst_local_contrast": burst_local,
        }
    )
    return tracks


def _local_rise_contrast(values: np.ndarray, radius: int = 4) -> np.ndarray:
    return _local_step_contrast(values, radius=radius, direction="rise")


def _local_fall_contrast(values: np.ndarray, radius: int = 4) -> np.ndarray:
    return _local_step_contrast(values, radius=radius, direction="fall")


def _local_peak_contrast(values: np.ndarray, radius: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 2:
        return np.zeros_like(values)
    baseline = np.empty_like(values)
    for index in range(len(values)):
        left = values[max(0, index - radius) : index]
        right = values[index + 1 : min(len(values), index + radius + 1)]
        neighbors = np.concatenate((left, right))
        baseline[index] = float(np.median(neighbors)) if len(neighbors) else values[index]
    return _robust_normalize(np.maximum(values - baseline, 0.0))


def _local_step_contrast(values: np.ndarray, *, radius: int, direction: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 2:
        return np.zeros_like(values)
    contrast = np.empty_like(values)
    for index in range(len(values)):
        left = values[max(0, index - radius) : index]
        right = values[index + 1 : min(len(values), index + radius + 1)]
        before = float(np.median(left)) if len(left) else float(values[index])
        after = float(np.median(right)) if len(right) else float(values[index])
        contrast[index] = after - before if direction == "rise" else before - after
    return _robust_normalize(np.maximum(contrast, 0.0))


def _derived_tracks(primitives: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    tracks = {name: np.asarray(values, dtype=np.float64) for name, values in primitives.items()}
    energy = tracks["energy"]
    periodicity = tracks["periodicity"]
    voicing = tracks["voicing"]
    aperiodic = tracks["aperiodic"]
    flux = tracks["flux"]
    shape_delta = tracks["spectral_shape_delta"]
    high_band = tracks["high_band_energy"]
    low_band = tracks["low_band_ratio"]
    mid_band = tracks["mid_band_ratio"]
    centroid = tracks["spectral_centroid"]
    stability = np.clip(1.0 - _smooth(shape_delta, 5), 0.0, 1.0)
    energy_gradient = np.gradient(energy) if len(energy) > 1 else np.zeros_like(energy)
    periodicity_gradient = np.gradient(periodicity) if len(periodicity) > 1 else np.zeros_like(periodicity)
    voicing_gradient = np.gradient(voicing) if len(voicing) > 1 else np.zeros_like(voicing)
    band_balance = low_band - mid_band
    band_delta = np.abs(np.gradient(band_balance)) if len(band_balance) > 1 else np.zeros_like(band_balance)
    energy_rise = _robust_normalize(np.maximum(energy_gradient, 0.0))
    energy_fall = _robust_normalize(np.maximum(-energy_gradient, 0.0))
    periodicity_rise = _robust_normalize(np.maximum(periodicity_gradient, 0.0))
    periodicity_fall = _robust_normalize(np.maximum(-periodicity_gradient, 0.0))
    voicing_change = _robust_normalize(np.abs(voicing_gradient))
    active = np.clip(0.62 * energy + 0.23 * np.maximum(voicing, aperiodic) + 0.15 * (1.0 - tracks["low_energy"]), 0.0, 1.0)
    onset = _smooth(0.34 * flux + 0.28 * energy_rise + 0.23 * shape_delta + 0.15 * high_band, 3)
    boundary = _smooth(0.31 * shape_delta + 0.25 * flux + 0.22 * voicing_change + 0.12 * energy_rise + 0.10 * energy_fall, 3)
    transition = _smooth(0.28 * periodicity_rise + 0.24 * voicing_change + 0.20 * shape_delta + 0.16 * energy_rise + 0.12 * flux, 3)
    vowel_like = np.clip(
        0.27 * periodicity + 0.20 * voicing + 0.16 * tracks["hnr"] + 0.15 * mid_band + 0.12 * stability + 0.10 * energy,
        0.0,
        1.0,
    )
    consonant_like = np.clip(
        0.24 * aperiodic + 0.20 * shape_delta + 0.17 * high_band + 0.15 * flux + 0.12 * tracks["zcr"] + 0.12 * active,
        0.0,
        1.0,
    )
    breath = np.clip(active * (0.35 * aperiodic + 0.27 * high_band + 0.20 * tracks["spectral_entropy"] + 0.18 * (1.0 - periodicity)), 0.0, 1.0)
    closure = np.clip(0.42 * tracks["low_energy"] + 0.22 * energy_fall + 0.18 * stability + 0.10 * consonant_like + 0.08 * (1.0 - voicing), 0.0, 1.0)
    burst = np.clip(0.40 * flux + 0.25 * energy_rise + 0.20 * high_band + 0.15 * shape_delta, 0.0, 1.0)
    nasal_murmur = np.clip(low_band * voicing * (1.0 - centroid) * (0.45 + 0.55 * stability), 0.0, 1.0)
    sonorant_boundary = np.clip(_robust_normalize(band_delta) * voicing * (0.55 + 0.45 * shape_delta), 0.0, 1.0)
    nucleus = np.clip(vowel_like * (0.35 + 0.35 * stability + 0.20 * energy + 0.10 * tracks["f0_stability"]), 0.0, 1.0)
    vowel_boundary = np.clip(0.42 * transition + 0.23 * boundary + 0.20 * voicing_change + 0.15 * vowel_like, 0.0, 1.0)
    tracks.update(
        {
            "active": active,
            "energy_rise": energy_rise,
            "energy_fall": energy_fall,
            "periodicity_rise": periodicity_rise,
            "periodicity_fall": periodicity_fall,
            "voicing_change": voicing_change,
            "spectral_stability": stability,
            "onset_strength": onset,
            "boundary": boundary,
            "transition": transition,
            "vowel_like": vowel_like,
            "consonant_like": consonant_like,
            "breath_likelihood": breath,
            "closure_likelihood": closure,
            "burst_likelihood": burst,
            "nasal_murmur_likelihood": nasal_murmur,
            "sonorant_boundary_likelihood": sonorant_boundary,
            "nucleus_likelihood": nucleus,
            "vowel_boundary_likelihood": vowel_boundary,
        }
    )
    return tracks


def _fuse_world_tracks(
    analysis: _FrameAnalysis,
    base_tracks: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    try:
        import pyworld as pw
    except ImportError as error:
        raise RuntimeError(
            "world_fusion_v2에는 pyworld가 필요합니다. `pip install universal-auto-oto[world]`를 실행하세요."
        ) from error
    wav = np.asarray(analysis.mono, dtype=np.float64)
    sample_rate = int(analysis.analysis_sample_rate)
    raw_f0, time_axis = pw.dio(wav, sample_rate, frame_period=analysis.hop_ms, f0_floor=55.0, f0_ceil=700.0)
    f0 = pw.stonemask(wav, raw_f0, time_axis, sample_rate)
    spectral_envelope = pw.cheaptrick(wav, f0, time_axis, sample_rate)
    aperiodicity = pw.d4c(wav, f0, time_axis, sample_rate)
    world_times_ms = np.asarray(time_axis, dtype=np.float64) * 1000.0
    world_f0 = _interp_track(world_times_ms, np.asarray(f0, dtype=np.float64), analysis.frame_times_ms)
    world_aperiodic = _interp_track(
        world_times_ms,
        np.mean(np.asarray(aperiodicity, dtype=np.float64), axis=1),
        analysis.frame_times_ms,
    )
    world_periodicity = np.clip(1.0 - world_aperiodic, 0.0, 1.0)
    world_voicing = np.clip((world_f0 > 40.0).astype(np.float64) * (0.35 + 0.65 * world_periodicity), 0.0, 1.0)
    log_sp = np.log1p(np.asarray(spectral_envelope, dtype=np.float64))
    sp_delta = np.zeros(len(log_sp), dtype=np.float64)
    if len(log_sp) > 1:
        sp_delta[1:] = np.mean(np.abs(log_sp[1:] - log_sp[:-1]), axis=1)
    world_stability = np.clip(
        1.0 - _interp_track(world_times_ms, _robust_normalize(sp_delta), analysis.frame_times_ms),
        0.0,
        1.0,
    )
    primitives = dict(base_tracks)
    primitives["f0_hz"] = np.where(world_voicing >= 0.25, world_f0, 0.0)
    primitives["periodicity"] = np.clip(0.42 * primitives["periodicity"] + 0.58 * world_periodicity, 0.0, 1.0)
    primitives["aperiodic"] = np.clip(0.42 * primitives["aperiodic"] + 0.58 * world_aperiodic, 0.0, 1.0)
    primitives["voicing"] = np.clip(0.38 * primitives["voicing"] + 0.62 * world_voicing, 0.0, 1.0)
    primitives["spectral_shape_delta"] = np.clip(
        0.60 * primitives["spectral_shape_delta"] + 0.40 * (1.0 - world_stability),
        0.0,
        1.0,
    )
    fused = _derived_tracks(primitives)
    fused.update(
        {
            "world_f0_hz": world_f0,
            "world_periodicity": world_periodicity,
            "world_aperiodic": world_aperiodic,
            "world_voicing": world_voicing,
            "world_spectral_stability": world_stability,
            "world_nucleus": np.clip(
                0.34 * world_voicing + 0.28 * world_periodicity + 0.22 * world_stability + 0.16 * fused["energy"],
                0.0,
                1.0,
            ),
        }
    )
    try:
        version = importlib.metadata.version("pyworld")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return fused, {
        "world_provider": "pyworld",
        "world_provider_version": version,
        "world_frame_period_ms": analysis.hop_ms,
        "world_fusion_weights": {"numpy": 0.42, "world": 0.58},
    }


DENSE_SPEECH_RANGE_DB = 20.0


def _activity_region(
    analysis: _FrameAnalysis,
    tracks: Mapping[str, np.ndarray],
    warnings: list[str],
    *,
    expected_regions: int = 1,
    dense_speech_floor: bool = False,
) -> tuple[float, float, dict[str, float]]:
    peak_db = float(np.percentile(analysis.rms_db, 99.0))
    noise_db = float(np.percentile(analysis.rms_db, 15.0))
    dense_floor_db = None
    if dense_speech_floor and noise_db > peak_db - DENSE_SPEECH_RANGE_DB:
        # A recording that is sung almost end to end has speech in its 15th percentile, which
        # puts the threshold inside the singing and cuts its first syllables off. Use the
        # quietest 100 ms instead (the edges before/after the singing).
        window = max(1, int(round(100.0 / analysis.hop_ms)))
        smoothed = np.convolve(analysis.rms_db, np.ones(window) / window, mode="valid")
        if len(smoothed) and float(smoothed.min()) < noise_db:
            dense_floor_db = float(smoothed.min())
            noise_db = dense_floor_db
    sequence_aware = int(expected_regions) >= 4
    noise_margin_db = 6.0 if sequence_aware else 8.0
    peak_range_db = 38.0 if sequence_aware else 35.0
    active_threshold_db = max(noise_db + noise_margin_db, peak_db - peak_range_db)
    active_mask = analysis.rms_db >= active_threshold_db
    if "active" in tracks and dense_floor_db is None:
        # The "active" track is normalised between the file's own 10th/95th percentiles, so in a
        # densely sung file it marks the softer syllables inactive as well; the floor replaces it.
        active_mask &= np.asarray(tracks["active"]) >= 0.12
    active_mask = np.convolve(active_mask.astype(np.int8), np.ones(5, dtype=np.int8), mode="same") >= 2
    if np.any(active_mask):
        indices = _select_main_activity_region(
            active_mask,
            analysis.rms_db,
            active_threshold_db,
            max_gap_frames=max(1, int(round(120.0 / analysis.hop_ms))),
            expected_regions=max(1, int(expected_regions)),
        )
        active_start_ms = max(0.0, float(analysis.frame_times_ms[indices[0]]) - 40.0)
        active_end_ms = min(analysis.duration_ms, float(analysis.frame_times_ms[indices[-1]]) + 60.0)
    else:
        active_start_ms = 0.0
        active_end_ms = analysis.duration_ms
        warnings.append("no_reliable_activity")
    periodicity = np.asarray(tracks.get("periodicity", np.zeros(len(analysis.frames))))
    voiced_fraction = float(np.mean(periodicity >= 0.35)) if len(periodicity) else 0.0
    f0 = np.asarray(tracks.get("f0_hz", np.zeros(len(analysis.frames))))
    voiced_f0 = f0[f0 > 0.0]
    return active_start_ms, active_end_ms, {
        "peak_db": peak_db,
        "noise_db": noise_db,
        "active_threshold_db": active_threshold_db,
        "activity_noise_margin_db": noise_margin_db,
        "activity_peak_range_db": peak_range_db,
        **({"dense_speech_floor_db": dense_floor_db} if dense_floor_db is not None else {}),
        "expected_activity_regions": float(max(1, int(expected_regions))),
        "voiced_frame_fraction": voiced_fraction,
        "median_f0_hz": float(np.median(voiced_f0)) if len(voiced_f0) else 0.0,
    }


def _autocorrelation_pitch(
    frames: np.ndarray,
    sample_rate: int,
    *,
    floor_hz: float,
    ceiling_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count, frame_length = frames.shape
    fft_size = 1 << int(math.ceil(math.log2(max(2, frame_length * 2 - 1))))
    min_lag = max(1, int(math.floor(sample_rate / ceiling_hz)))
    max_lag = min(frame_length - 2, int(math.ceil(sample_rate / floor_hz)))
    if max_lag <= min_lag:
        return np.zeros(len(frames)), np.zeros(len(frames))
    f0 = np.zeros(frame_count, dtype=np.float64)
    harmonicity = np.zeros(frame_count, dtype=np.float64)
    window = np.hanning(frame_length)[None, :]
    # Long reclist recordings can contain tens of thousands of frames. Chunking
    # keeps the FFT/autocorrelation working set bounded without changing output.
    for chunk_start in range(0, frame_count, 2048):
        chunk = np.asarray(frames[chunk_start : chunk_start + 2048], dtype=np.float64)
        centered = (chunk - np.mean(chunk, axis=1, keepdims=True)) * window
        spectrum = np.fft.rfft(centered, fft_size, axis=1)
        autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size, axis=1)[:, :frame_length]
        autocorr /= np.maximum(autocorr[:, :1], 1e-12)
        search = autocorr[:, min_lag : max_lag + 1]
        relative = np.argmax(search, axis=1)
        lags = relative.astype(np.float64) + float(min_lag)
        rows = np.arange(len(chunk))
        peaks = search[rows, relative]
        left = autocorr[rows, np.maximum(min_lag, lags.astype(np.int64) - 1)]
        right = autocorr[rows, np.minimum(max_lag, lags.astype(np.int64) + 1)]
        denominator = left - 2.0 * peaks + right
        correction = np.zeros_like(denominator)
        np.divide(
            0.5 * (left - right),
            denominator,
            out=correction,
            where=np.abs(denominator) > 1e-8,
        )
        refined_lags = np.clip(lags + correction, float(min_lag), float(max_lag))
        chunk_end = chunk_start + len(chunk)
        f0[chunk_start:chunk_end] = float(sample_rate) / np.maximum(refined_lags, 1.0)
        harmonicity[chunk_start:chunk_end] = np.clip(peaks, 0.0, 1.0)
    return f0, harmonicity


def _band_ratio(
    normalized_magnitude: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(mask):
        return np.zeros(len(normalized_magnitude), dtype=np.float64)
    return np.sum(normalized_magnitude[:, mask], axis=1)


def _interp_track(source_times_ms: np.ndarray, values: np.ndarray, target_times_ms: np.ndarray) -> np.ndarray:
    if len(source_times_ms) == 0 or len(values) == 0:
        return np.zeros(len(target_times_ms), dtype=np.float64)
    return np.interp(target_times_ms, source_times_ms, values, left=float(values[0]), right=float(values[-1]))


__all__ = [
    "AcousticEvidence",
    "FEATURE_BACKEND_ID",
    "LEGACY_FEATURE_BACKEND_ID",
    "MULTIBAND_V3_FEATURE_BACKEND_ID",
    "NEURAL_FEATURE_BACKEND_ID",
    "SUPPORTED_FEATURE_BACKENDS",
    "WORLD_FEATURE_BACKEND_ID",
    "extract_acoustic_evidence",
    "load_acoustic_evidence_npz",
    "write_acoustic_evidence_npz",
]
