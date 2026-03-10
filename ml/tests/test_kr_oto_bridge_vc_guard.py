import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_oto_bridge import _compute_vc_from_adjacent_cv


class KrOtoBridgeVcGuardTests(unittest.TestCase):
    def test_stop_like_vc_is_clamped_near_next_onset(self):
        prev_cv = {
            "pre": 86.0,
            "cons_gap": 60.0,
            "vowel_len": 118.0,
            "vowel_end_abs": 520.0,
        }
        next_cv = {
            "pre": 90.0,
            "pre_abs": 702.0,
            "cons_abs": 764.0,
            "onset_abs": 668.0,
            "cons_gap": 72.0,
        }
        offset, consonant, cutoff, pre, _ovl = _compute_vc_from_adjacent_cv(
            prev_cv,
            next_cv,
            alias_type="vc",
            is_plosive_sibilant=True,
        )
        next_onset_rel = float(next_cv["onset_abs"]) - float(offset)
        self.assertLessEqual(consonant, next_onset_rel - 5.0)
        self.assertLessEqual(abs(cutoff), next_onset_rel + 1.0)
        self.assertGreaterEqual(abs(cutoff), consonant + 8.0)
        self.assertGreaterEqual(consonant, pre + 10.0)


if __name__ == "__main__":
    unittest.main()

