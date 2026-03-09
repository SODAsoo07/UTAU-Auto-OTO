import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_timing_v2 import (
    build_realized_cv_anchor,
    extract_vcv_anchor_points,
    prepare_vcv_syllable_timing,
)


def _validate(offset, consonant, cutoff, pre, ovl):
    pre = max(float(pre), 0.0)
    ovl = max(0.0, min(float(ovl), pre))
    consonant = max(float(consonant), pre + 1.0)
    cutoff_abs = max(abs(float(cutoff)), consonant + 1.0)
    return float(offset), float(consonant), -float(cutoff_abs), float(pre), float(ovl)


class KrTimingV2Tests(unittest.TestCase):
    def test_build_realized_cv_anchor_shape(self):
        anchor = build_realized_cv_anchor(
            90.0,
            170.0,
            -240.0,
            70.0,
            35.0,
            onset_abs=100.0,
            vowel_start_abs=130.0,
            vowel_end_abs=260.0,
            c_end_abs=125.0,
        )
        self.assertEqual(anchor["pre_abs"], 160.0)
        self.assertEqual(anchor["cons_abs"], 260.0)
        self.assertEqual(anchor["c_end_abs"], 125.0)
        self.assertGreater(anchor["vowel_len"], 100.0)

    def test_extract_vcv_anchor_points_prefers_last_phone_for_multi_phone_syllable(self):
        syllables = [
            {
                "phones": [
                    SimpleNamespace(minTime=0.10, maxTime=0.13),
                    SimpleNamespace(minTime=0.14, maxTime=0.28),
                ]
            }
        ]
        anchor, next_onset, next_vowel = extract_vcv_anchor_points(syllables, 0)
        self.assertEqual(anchor, 140.0)
        self.assertEqual(next_onset, 140.0)
        self.assertEqual(next_vowel, 280.0)

    def test_prepare_vcv_syllable_timing_returns_valid_tuple(self):
        syllables = [{"phones": [SimpleNamespace(mark="g")]}, {"phones": [SimpleNamespace(mark="a")]}]

        def _prepare_bounds(_syllables, idx):
            data = {
                0: (0, [SimpleNamespace(mark="g")], 100.0, 130.0, 140.0, 240.0),
                1: (1, [SimpleNamespace(mark="d")], 260.0, 300.0, 310.0, 420.0),
            }
            return data[idx]

        result = prepare_vcv_syllable_timing(
            syllables,
            current_w_idx=0,
            cv_seq_idx=0,
            diphthong_cv_consonant_ratio=0.6,
            forced_w_idx=1,
            prepare_cv_bounds_fn=_prepare_bounds,
            adaptive_overlap_fn=lambda pre, c_hint, mode="vcv": min(pre * 0.5, 60.0),
            validate_fn=_validate,
        )
        current_w_idx, cv_seq_idx, offset, consonant, cutoff, pre, ovl = result
        self.assertEqual(current_w_idx, 1)
        self.assertEqual(cv_seq_idx, 2)
        self.assertGreaterEqual(offset, 0.0)
        self.assertGreater(consonant, pre)
        self.assertLess(cutoff, 0.0)
        self.assertLessEqual(ovl, pre)


if __name__ == "__main__":
    unittest.main()
