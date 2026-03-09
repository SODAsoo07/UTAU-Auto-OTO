import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_anchor_lock_v2 import (
    build_kr_anchor_lock_log_record,
    build_kr_anchor_lock_stats_delta,
    resolve_kr_anchor_targets,
)


class KrAnchorLockV2Tests(unittest.TestCase):
    def test_resolve_kr_anchor_targets_uses_next_onset_for_vc(self):
        anchor_abs, next_onset_abs, next_vowel_abs = resolve_kr_anchor_targets("vc", 140.0, 220.0, 320.0)
        self.assertEqual(anchor_abs, 220.0)
        self.assertEqual(next_onset_abs, 220.0)
        self.assertEqual(next_vowel_abs, 320.0)

    def test_build_kr_anchor_lock_stats_delta_marks_vc_cutoff_guard(self):
        delta = build_kr_anchor_lock_stats_delta("vc", {"pre_anchor_lock", "cutoff_next_onset_clamp"})
        self.assertEqual(delta["anchor_locked_count"], 1)
        self.assertEqual(delta["cutoff_clamped_count"], 1)
        self.assertEqual(delta["vc_cutoff_leak_guard_count"], 1)

    def test_build_kr_anchor_lock_log_record_returns_none_without_rules(self):
        result = SimpleNamespace(
            offset=1.0,
            consonant=2.0,
            cutoff=-3.0,
            pre=1.0,
            ovl=0.5,
            anchor_shift_ms=0.0,
            applied_rules=[],
        )
        record = build_kr_anchor_lock_log_record(
            fname="a.wav",
            alias_text="ga",
            format_type="cvvc",
            alias_type="cv",
            lite=False,
            before=(1.0, 2.0, -3.0, 1.0, 0.5),
            result=result,
        )
        self.assertIsNone(record)

    def test_build_kr_anchor_lock_log_record_serializes_result(self):
        result = SimpleNamespace(
            offset=10.0,
            consonant=80.0,
            cutoff=-120.0,
            pre=40.0,
            ovl=25.0,
            anchor_shift_ms=6.0,
            applied_rules=["pre_anchor_lock", "cutoff_next_vowel_clamp"],
        )
        record = build_kr_anchor_lock_log_record(
            fname="b.wav",
            alias_text="a g",
            format_type="cvvc",
            alias_type="vc",
            lite=True,
            before=(12.0, 85.0, -130.0, 44.0, 20.0),
            result=result,
        )
        self.assertEqual(record["language"], "korean")
        self.assertEqual(record["lite"], True)
        self.assertIn("cutoff_next_vowel_clamp", record["rules"])
        self.assertEqual(record["after"]["offset"], 10.0)


if __name__ == "__main__":
    unittest.main()
