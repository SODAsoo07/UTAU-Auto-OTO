import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_generator import (
    _apply_soft_mel_offset_cutoff_guard,
    _stabilize_params_to_phone_activity,
)


class OtoGeneratorStabilizeGuardTests(unittest.TestCase):
    def _phones(self):
        return [
            SimpleNamespace(minTime=0.00, maxTime=0.10),
            SimpleNamespace(minTime=0.20, maxTime=0.30),
        ]

    def test_stabilize_uses_blend_for_large_delta(self):
        phones = self._phones()
        # pre_abs=150ms, target(next_start-4)=196ms -> delta=46ms (>30)
        o, c, ct, p, ov = _stabilize_params_to_phone_activity(
            100.0, 170.0, -260.0, 50.0, 20.0, phones, alias_type="cv"
        )
        self.assertAlmostEqual(o, 120.7, places=1)
        self.assertGreaterEqual(c, p)
        self.assertGreater(abs(ct), c)

    def test_stabilize_keeps_hard_snap_for_small_delta(self):
        phones = self._phones()
        # pre_abs=180ms, target(next_start-4)=196ms -> delta=16ms (<=30)
        o, c, ct, p, ov = _stabilize_params_to_phone_activity(
            120.0, 190.0, -280.0, 60.0, 18.0, phones, alias_type="cv"
        )
        self.assertAlmostEqual(o, 136.0, places=1)
        self.assertGreaterEqual(c, p)
        self.assertGreater(abs(ct), c)

    def test_soft_mel_guard_prefers_onset_candidate_when_available(self):
        t_ms = np.arange(10, dtype=np.float64) * 10.0
        db = np.array([-20.0] * 10, dtype=np.float64)
        f0 = np.zeros(10, dtype=np.float64)

        # No onset candidate path: fallback to earliest sound_start_idx.
        en_no_onset = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.145, 0.08, 0.145, 0.08], dtype=np.float64)
        mel_no_onset = {
            "times_ms": t_ms,
            "energy": en_no_onset,
            "db_db": db,
            "f0_voicing": f0,
            "db_silence_th": -42.0,
        }
        o1, *_ = _apply_soft_mel_offset_cutoff_guard(
            0.0, 60.0, -140.0, 80.0, 20.0, "cv", mel_no_onset, alias_text="sa", file_format="vcv"
        )

        # Onset candidate path: onset index appears later and should be preferred.
        en_onset = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.145, 0.08, 0.08, 0.145, 0.30], dtype=np.float64)
        mel_onset = {
            "times_ms": t_ms,
            "energy": en_onset,
            "db_db": db,
            "f0_voicing": f0,
            "db_silence_th": -42.0,
        }
        o2, *_ = _apply_soft_mel_offset_cutoff_guard(
            0.0, 60.0, -140.0, 80.0, 20.0, "cv", mel_onset, alias_text="sa", file_format="vcv"
        )

        # onset candidate is later than fallback sound-start in this setup,
        # so offset should move forward more.
        self.assertGreater(o2, o1 + 2.0)


if __name__ == "__main__":
    unittest.main()
