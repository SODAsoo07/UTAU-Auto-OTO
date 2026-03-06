import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.timing_anchor_profiles import get_anchor_profile
from core.timing_anchor_runtime import AnchorTimingContext, apply_anchor_lock


class TimingAnchorRuntimeTests(unittest.TestCase):
    def test_anchor_lock_clamps_pre_abs_window(self):
        profile = get_anchor_profile("japanese", "vcv", "vcv")
        self.assertIsNotNone(profile)
        ctx = AnchorTimingContext(
            file_duration_ms=4000.0,
            timeline_start_ms=0.0,
            timeline_end_ms=3600.0,
            anchor_abs_ms=1000.0,
            next_onset_abs_ms=1100.0,
            alias_type="vcv",
            language="japanese",
            format_type="vcv",
            mapping_confidence=1.0,
        )
        # pre_abs starts too early and should be pulled into [anchor-8, anchor+4].
        out = apply_anchor_lock((860.0, 120.0, -220.0, 40.0, 10.0), ctx, profile)
        pre_abs = out.offset + out.pre
        self.assertGreaterEqual(pre_abs, 992.0)
        self.assertLessEqual(pre_abs, 1004.0)

    def test_anchor_lock_preserves_parameter_order(self):
        profile = get_anchor_profile("japanese", "cvvc", "vc")
        self.assertIsNotNone(profile)
        ctx = AnchorTimingContext(
            file_duration_ms=5000.0,
            timeline_start_ms=0.0,
            timeline_end_ms=4200.0,
            anchor_abs_ms=2100.0,
            next_onset_abs_ms=2140.0,
            next_vowel_abs_ms=2180.0,
            alias_type="vc",
            language="japanese",
            format_type="cvvc",
            mapping_confidence=1.0,
        )
        out = apply_anchor_lock((1970.0, 55.0, -70.0, 28.0, 30.0), ctx, profile)
        self.assertGreaterEqual(out.pre, out.ovl)
        self.assertGreater(out.consonant, out.pre)
        self.assertGreater(abs(out.cutoff), out.consonant)

    def test_anchor_shift_is_limited(self):
        profile = get_anchor_profile("japanese", "vcv", "vcv")
        self.assertIsNotNone(profile)
        ctx = AnchorTimingContext(
            file_duration_ms=1000.0,
            timeline_start_ms=0.0,
            timeline_end_ms=950.0,
            anchor_abs_ms=900.0,
            alias_type="vcv",
            language="japanese",
            format_type="vcv",
            mapping_confidence=1.0,
        )
        out = apply_anchor_lock((10.0, 120.0, -220.0, 40.0, 12.0), ctx, profile)
        # max shift for 1000ms file with ratio 0.06 is clamped to at least 45ms.
        self.assertLessEqual(abs(out.anchor_shift_ms), 45.5)


if __name__ == "__main__":
    unittest.main()

