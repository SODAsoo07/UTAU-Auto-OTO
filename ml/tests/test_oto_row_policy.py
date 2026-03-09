import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_row_policy import (
    apply_row_confidence_penalty,
    build_mapping_trace_record,
    should_trace_mapping_decision,
)


class OtoRowPolicyTests(unittest.TestCase):
    def test_apply_row_confidence_penalty_clamps_at_floor(self):
        self.assertAlmostEqual(apply_row_confidence_penalty(0.5, 0.2), 0.3)
        self.assertAlmostEqual(apply_row_confidence_penalty(0.1, 0.3), 0.0)
        self.assertAlmostEqual(apply_row_confidence_penalty(0.1, 0.3, floor=0.05), 0.05)

    def test_should_trace_mapping_decision_when_low_or_shifted(self):
        self.assertTrue(
            should_trace_mapping_decision(
                mapping_tier="low",
                expected_idx=1,
                mapped_idx=1,
                target_token="go",
                mapped_token="go",
            )
        )
        self.assertTrue(
            should_trace_mapping_decision(
                mapping_tier="mid",
                expected_idx=1,
                mapped_idx=2,
                target_token="go",
                mapped_token="go",
            )
        )
        self.assertFalse(
            should_trace_mapping_decision(
                mapping_tier="high",
                expected_idx=1,
                mapped_idx=1,
                target_token="go",
                mapped_token="go",
            )
        )

    def test_build_mapping_trace_record_preserves_core_fields(self):
        record = build_mapping_trace_record(
            event="ja_mapping_decision",
            fname="a.wav",
            alias="go",
            alias_type="cv",
            format_type="cvvc",
            target_token="go",
            expected_idx=1,
            mapped_idx=2,
            expected_token="go",
            mapped_token="gyo",
            mapping_tier="mid",
            mapping_reason_code="alias_recover",
            mapping_confidence=0.72,
            extra={"note": "test"},
        )
        self.assertEqual(record["delta"], 1)
        self.assertEqual(record["event"], "ja_mapping_decision")
        self.assertEqual(record["mapped_token"], "gyo")
        self.assertEqual(record["note"], "test")


if __name__ == "__main__":
    unittest.main()
