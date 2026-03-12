from __future__ import annotations

import os
import unittest
from unittest import mock

from core.oto_ml.coupled.training import _estimate_rawmel_patch_cache_mb, _resolve_rawmel_prefetch_mode


class _FakeDevice:
    def __init__(self, device_type: str):
        self.type = device_type


class CoupledRawmelPrefetchTests(unittest.TestCase):
    def test_estimate_patch_cache_mb_positive(self):
        value = _estimate_rawmel_patch_cache_mb(100, onset_frames=49, tail_frames=49, mel_bins=80)
        self.assertGreater(value, 0.0)

    def test_auto_prefetch_uses_gpu_when_small_and_cuda(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            result = _resolve_rawmel_prefetch_mode(
                "auto",
                run_device=_FakeDevice("cuda"),
                train_count=100,
                valid_count=20,
                onset_frames=49,
                tail_frames=49,
                mel_bins=80,
            )
        self.assertEqual(result["mode"], "gpu")

    def test_auto_prefetch_falls_back_to_train_when_large(self):
        with mock.patch.dict(os.environ, {"UTOA_ML_RAWMEL_GPU_CACHE_MAX_MB": "64"}, clear=False):
            result = _resolve_rawmel_prefetch_mode(
                "auto",
                run_device=_FakeDevice("cuda"),
                train_count=20000,
                valid_count=5000,
                onset_frames=49,
                tail_frames=49,
                mel_bins=80,
            )
        self.assertEqual(result["mode"], "train")

    def test_gpu_prefetch_on_cpu_device_becomes_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            result = _resolve_rawmel_prefetch_mode(
                "gpu",
                run_device=_FakeDevice("cpu"),
                train_count=100,
                valid_count=20,
                onset_frames=49,
                tail_frames=49,
                mel_bins=80,
            )
        self.assertEqual(result["mode"], "none")


if __name__ == "__main__":
    unittest.main()
