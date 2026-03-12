import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_generator import _guard_kr_blank_region_span
from core.oto_validator import _should_warn_blank_only_span


def _build_mel_ctx(active_start_ms=60.0):
    times = np.arange(0.0, 121.0, 10.0, dtype=np.float64)
    active_mask = times >= active_start_ms
    return {
        "times_ms": times,
        "energy": np.where(active_mask, 0.28, 0.02).astype(np.float64),
        "db_db": np.where(active_mask, -18.0, -52.0).astype(np.float64),
        "db_silence_th": -42.0,
        "cls_silence_sparse": np.where(active_mask, 0.0, 1.0).astype(np.float64),
        "cls_voiced_formant": np.where(active_mask, 1.0, 0.0).astype(np.float64),
    }


class KoreanBlankRegionGuardTests(unittest.TestCase):
    def test_cv_reverts_when_candidate_span_is_blank_only(self):
        mel_ctx = _build_mel_ctx(active_start_ms=60.0)
        params, reverted = _guard_kr_blank_region_span(
            0.0,
            18.0,
            -38.0,
            16.0,
            8.0,
            base_offset=60.0,
            base_consonant=96.0,
            base_cutoff=-38.0,
            base_pre=24.0,
            base_ovl=12.0,
            alias_type="cv",
            mel_ctx=mel_ctx,
        )
        self.assertTrue(reverted)
        self.assertAlmostEqual(params[0], 60.0)

    def test_vc_keeps_candidate_when_span_is_not_fully_blank(self):
        mel_ctx = _build_mel_ctx(active_start_ms=60.0)
        params, reverted = _guard_kr_blank_region_span(
            10.0,
            42.0,
            -58.0,
            24.0,
            10.0,
            base_offset=60.0,
            base_consonant=96.0,
            base_cutoff=-38.0,
            base_pre=24.0,
            base_ovl=12.0,
            alias_type="vc",
            mel_ctx=mel_ctx,
        )
        self.assertFalse(reverted)
        self.assertAlmostEqual(params[0], 10.0)

    def test_vv_reverts_on_blank_onset_region(self):
        mel_ctx = _build_mel_ctx(active_start_ms=80.0)
        params, reverted = _guard_kr_blank_region_span(
            0.0,
            40.0,
            -70.0,
            20.0,
            10.0,
            base_offset=80.0,
            base_consonant=116.0,
            base_cutoff=-52.0,
            base_pre=20.0,
            base_ovl=10.0,
            alias_type="vv",
            mel_ctx=mel_ctx,
        )
        self.assertTrue(reverted)
        self.assertAlmostEqual(params[0], 80.0)

    def test_validator_warn_policy_is_stricter_for_cv_than_vc(self):
        stats = {
            "frame_count": 8,
            "blank_ratio": 0.95,
            "active_ratio": 0.05,
            "onset_blank_ratio": 1.0,
        }
        self.assertTrue(_should_warn_blank_only_span("cv", stats))
        self.assertFalse(_should_warn_blank_only_span("vc", stats))


if __name__ == "__main__":
    unittest.main()
