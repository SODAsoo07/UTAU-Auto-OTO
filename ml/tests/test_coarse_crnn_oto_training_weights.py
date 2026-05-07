from __future__ import annotations


def test_context_loss_multiplier_targets_cvvc_vc_multi():
    from core.coarse_crnn.oto_training import OtoTrainConfig, _context_loss_multiplier

    cfg = OtoTrainConfig(cvvc_vc_multi_loss_weight=2.75)
    row = {
        "language": "korean",
        "format_type": "cvvc",
        "alias_type": "vc",
        "transition_type": "multi",
        "alias_role": "vc",
    }

    assert _context_loss_multiplier(row, cfg) == 2.75


def test_context_loss_multiplier_targets_korean_vcv_head():
    from core.coarse_crnn.oto_training import OtoTrainConfig, _context_loss_multiplier

    cfg = OtoTrainConfig(korean_vcv_head_loss_weight=1.9)
    row = {
        "language": "korean",
        "format_type": "vcv",
        "alias_type": "cv_head",
        "transition_type": "multi",
        "alias_role": "-cv",
    }

    assert _context_loss_multiplier(row, cfg) == 1.9


def test_cpu_feature_cache_reuses_loaded_features(monkeypatch):
    import numpy as np

    from core.coarse_crnn.oto_model import OtoCrnnConfig
    from core.coarse_crnn.oto_training import OtoAnchorDataset, OtoTrainConfig

    calls = {"wav": 0, "mel": 0}

    def _fake_load_wav_mono(_path, target_sr):
        calls["wav"] += 1
        return np.zeros((3200,), dtype=np.float32), int(target_sr), 0.2

    def _fake_log_mel_spectrogram(_samples, _sr, *, n_mels, frame_ms, hop_ms):
        calls["mel"] += 1
        _ = frame_ms
        _ = hop_ms
        return np.zeros((20, int(n_mels)), dtype=np.float32), 0.01

    monkeypatch.setattr("core.coarse_crnn.oto_training.load_wav_mono", _fake_load_wav_mono)
    monkeypatch.setattr("core.coarse_crnn.oto_training.log_mel_spectrogram", _fake_log_mel_spectrogram)

    row = {
        "audio": "dummy.wav",
        "language": "korean",
        "format_type": "cv",
        "alias_type": "cv",
        "transition_type": "cv",
        "alias_role": "cv",
        "anchor_offset_ms": 20.0,
        "anchor_overlap_ms": 30.0,
        "anchor_preutterance_ms": 40.0,
        "anchor_consonant_ms": 50.0,
        "anchor_cutoff_ms": 60.0,
    }
    ds = OtoAnchorDataset(
        [row],
        OtoCrnnConfig(),
        train_config=OtoTrainConfig(cpu_feature_cache_size=1),
        train=False,
    )
    _ = ds[0]
    _ = ds[0]
    assert calls["wav"] == 1
    assert calls["mel"] == 1


def test_iter_device_batches_cpu_path_keeps_batch_count():
    torch = __import__("torch")
    from core.coarse_crnn.oto_training import _iter_device_batches

    loader = [
        (torch.zeros((1, 2), dtype=torch.float32), torch.ones((1,), dtype=torch.float32)),
        (torch.ones((1, 2), dtype=torch.float32), torch.zeros((1,), dtype=torch.float32)),
    ]
    batches = list(_iter_device_batches(loader, torch.device("cpu"), torch=torch, prefetch_batches=2))
    assert len(batches) == 2
    assert all(batch[0].device.type == "cpu" for batch in batches)
