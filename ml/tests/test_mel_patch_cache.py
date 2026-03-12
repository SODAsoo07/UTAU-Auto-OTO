import numpy as np

from core.oto_ml.features.mel_patches import (
    DEFAULT_FRAME_HOP_MS,
    DEFAULT_MEL_BINS,
    DEFAULT_ONSET_WINDOW_MS,
    compute_log_mel,
    extract_patch_from_mel,
    make_mel_patch_key,
    resolve_mel_patch_anchors,
)


def test_mel_patch_key_stability():
    key = make_mel_patch_key(
        language="Korean",
        format_type="CVVC",
        voicebank_id="BankA",
        wav_norm="a.wav",
        alias_norm="a k",
        occurrence_index=2,
        row_index_in_wav=5,
    )
    assert key == "korean|cvvc|banka|a.wav|a k|2|5"


def test_resolve_mel_patch_anchors():
    feat = {
        "mel_offset_candidate_ms": 120.0,
        "mel_cutoff_candidate_ms": 500.0,
        "base_offset": 10.0,
        "base_cutoff_abs": 400.0,
    }
    onset, tail, source = resolve_mel_patch_anchors(feat)
    assert onset == 120.0
    assert tail == 500.0
    assert source == "mel_candidate"

    feat = {
        "mel_offset_candidate_ms": 0.0,
        "mel_cutoff_candidate_ms": 0.0,
        "base_offset": 10.0,
        "base_cutoff_abs": 400.0,
    }
    onset, tail, source = resolve_mel_patch_anchors(feat)
    assert onset == 10.0
    assert tail == 400.0
    assert source == "base_fallback"


def test_extract_patch_shape():
    sr = 16000
    audio = np.zeros(int(sr * 0.5), dtype=np.float32)
    mel, hop_ms = compute_log_mel(
        audio,
        sr,
        mel_bins=DEFAULT_MEL_BINS,
        frame_hop_ms=DEFAULT_FRAME_HOP_MS,
        win_ms=25.0,
    )
    onset_patch = extract_patch_from_mel(
        mel,
        anchor_ms=120.0,
        window_ms=DEFAULT_ONSET_WINDOW_MS,
        hop_ms=DEFAULT_FRAME_HOP_MS,
        mel_bins=DEFAULT_MEL_BINS,
    )
    expected_frames = int(round((DEFAULT_ONSET_WINDOW_MS[1] - DEFAULT_ONSET_WINDOW_MS[0]) / DEFAULT_FRAME_HOP_MS)) + 1
    assert onset_patch.shape == (expected_frames, DEFAULT_MEL_BINS)
