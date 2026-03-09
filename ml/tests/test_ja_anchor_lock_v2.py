import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_anchor_lock_v2 import (
    build_ja_anchor_lock_log_record,
    build_ja_anchor_lock_stats_delta,
    retune_ja_anchor_profile,
)
from core.timing_anchor_profiles import AnchorTimingProfile


class JaAnchorLockV2Tests(unittest.TestCase):
    def test_retune_ja_anchor_profile_relaxes_vcv_cut_clamp(self):
        base = AnchorTimingProfile(
            pre_window_before_ms=8.0,
            pre_window_after_ms=4.0,
            pre_floor_ms=42.0,
            ovl_gap_min_ms=4.0,
            ovl_gap_max_ms=10.0,
            ovl_gap_target_ms=7.0,
            cons_gap_min_ms=50.0,
            cons_gap_max_ms=100.0,
            cons_gap_target_ms=70.0,
            cut_gap_min_ms=20.0,
            cut_gap_max_ms=50.0,
            cut_gap_target_ms=30.0,
            cut_to_next_onset_allow_ms=6.0,
            cut_to_next_vowel_allow_ms=4.0,
        )
        tuned = retune_ja_anchor_profile(base, format_type="vcv", alias_key="vcv")
        self.assertIsNone(tuned.cut_to_next_onset_allow_ms)
        self.assertIsNone(tuned.cut_to_next_vowel_allow_ms)
        self.assertGreater(tuned.pre_window_after_ms, base.pre_window_after_ms)

    def test_build_ja_anchor_lock_stats_delta_marks_vc_guard(self):
        delta = build_ja_anchor_lock_stats_delta("vc", {"pre_anchor_lock", "cutoff_next_onset_clamp"})
        self.assertEqual(delta["anchor_locked_count"], 1)
        self.assertEqual(delta["cutoff_clamped_count"], 1)
        self.assertEqual(delta["vc_cutoff_leak_guard_count"], 1)

    def test_build_ja_anchor_lock_log_record_returns_payload(self):
        result = SimpleNamespace(
            offset=10.0,
            consonant=80.0,
            cutoff=-120.0,
            pre=40.0,
            ovl=24.0,
            anchor_shift_ms=5.0,
            applied_rules=["pre_anchor_lock"],
        )
        record = build_ja_anchor_lock_log_record(
            fname="a.wav",
            alias_text="a ka",
            format_type="vcv",
            alias_type="vcv",
            lite=False,
            before=(12.0, 84.0, -125.0, 44.0, 22.0),
            result=result,
        )
        self.assertEqual(record["language"], "japanese")
        self.assertEqual(record["after"]["offset"], 10.0)
        self.assertIn("pre_anchor_lock", record["rules"])


if __name__ == "__main__":
    unittest.main()
