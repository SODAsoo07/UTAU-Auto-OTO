import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.timing_anchor_profiles import get_anchor_profile, is_anchor_lock_enabled


class TimingAnchorProfileTests(unittest.TestCase):
    def test_get_anchor_profile_japanese_vcv_exists(self):
        profile = get_anchor_profile("japanese", "vcv", "vcv")
        self.assertIsNotNone(profile)
        self.assertGreater(profile.pre_floor_ms, 0.0)
        self.assertGreater(profile.cons_gap_target_ms, profile.cons_gap_min_ms - 1.0)

    def test_get_anchor_profile_unknown_returns_none(self):
        profile = get_anchor_profile("japanese", "vcv", "unknown_alias")
        self.assertIsNone(profile)

    def test_get_anchor_profile_korean_vcv_exists(self):
        profile = get_anchor_profile("korean", "vcv", "vcv")
        self.assertIsNotNone(profile)
        self.assertGreater(profile.cut_gap_target_ms, 0.0)

    def test_get_anchor_profile_korean_cvc_uses_cvc_specific_bridge_profile(self):
        profile = get_anchor_profile("korean", "cvc", "vc")
        self.assertIsNotNone(profile)
        self.assertGreater(profile.cons_gap_target_ms, 40.0)
        self.assertGreater(profile.cut_gap_target_ms, 30.0)

    def test_enable_gate_returns_boolean(self):
        self.assertIsInstance(is_anchor_lock_enabled("japanese", "vcv"), bool)
        self.assertIsInstance(is_anchor_lock_enabled("japanese", "cvvc"), bool)
        self.assertIsInstance(is_anchor_lock_enabled("korean", "cvvc"), bool)
        self.assertIsInstance(is_anchor_lock_enabled("korean", "vcv"), bool)


if __name__ == "__main__":
    unittest.main()
