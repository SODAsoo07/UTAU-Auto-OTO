import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.anchor_lock_adapter_v2 import apply_language_anchor_lock
from core.timing_anchor_profiles import AnchorTimingProfile


class AnchorLockAdapterV2Tests(unittest.TestCase):
    def test_apply_language_anchor_lock_returns_before_when_disabled(self):
        before = (1.0, 2.0, -3.0, 1.0, 0.5)
        result = apply_language_anchor_lock(
            language="korean",
            format_type="cvvc",
            alias_type="cv",
            before=before,
            file_duration_ms=1000.0,
            timeline_start_ms=0.0,
            timeline_end_ms=1000.0,
            anchor_abs_ms=200.0,
            validate_fn=lambda *args: args,
            lite=False,
            is_enabled_fn=lambda lang, fmt: False,
            get_profile_fn=lambda lang, fmt, alias: object(),
        )
        self.assertEqual(result, before)

    def test_apply_language_anchor_lock_applies_stats_and_log_hooks(self):
        captured = {"stats": None, "log": None}
        profile = AnchorTimingProfile(
            pre_window_before_ms=8.0,
            pre_window_after_ms=8.0,
            pre_floor_ms=30.0,
            ovl_gap_min_ms=4.0,
            ovl_gap_max_ms=12.0,
            ovl_gap_target_ms=8.0,
            cons_gap_min_ms=30.0,
            cons_gap_max_ms=80.0,
            cons_gap_target_ms=44.0,
            cut_gap_min_ms=20.0,
            cut_gap_max_ms=60.0,
            cut_gap_target_ms=30.0,
        )

        def _validate(offset, consonant, cutoff, pre, ovl):
            return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)

        result = apply_language_anchor_lock(
            language="japanese",
            format_type="vcv",
            alias_type="vcv",
            before=(10.0, 80.0, -120.0, 40.0, 20.0),
            file_duration_ms=1000.0,
            timeline_start_ms=0.0,
            timeline_end_ms=900.0,
            anchor_abs_ms=210.0,
            next_onset_abs_ms=220.0,
            next_vowel_abs_ms=300.0,
            mapping_confidence=1.0,
            validate_fn=_validate,
            lite=True,
            is_enabled_fn=lambda lang, fmt: True,
            get_profile_fn=lambda lang, fmt, alias: profile,
            retune_profile_fn=lambda profile: profile,
            apply_stats_delta_fn=lambda alias, rules: captured.update({"stats": (alias, tuple(rules))}),
            build_log_record_fn=lambda fmt, alias, before, result: {
                "fmt": fmt,
                "alias": alias,
                "rules": list(getattr(result, "applied_rules", ())),
            },
            append_log_fn=lambda path, record: captured.update({"log": (path, record)}),
            log_path="dummy.jsonl",
        )
        self.assertEqual(len(result), 5)
        self.assertIsNotNone(captured["stats"])
        self.assertEqual(captured["stats"][0], "vcv")
        self.assertEqual(captured["log"][0], "dummy.jsonl")
        self.assertEqual(captured["log"][1]["fmt"], "vcv")


if __name__ == "__main__":
    unittest.main()
