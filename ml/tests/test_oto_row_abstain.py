import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_row_abstain import decide_cv_row_abstain


class OtoRowAbstainTests(unittest.TestCase):
    def test_row_abstain_skips_invalid_index(self):
        result = decide_cv_row_abstain(
            alias_type="cv",
            format_type="cvvc",
            candidate_idx=5,
            candidate_count=2,
            candidate_active=True,
            active_only_formats={"cvvc", "cv"},
        )
        self.assertTrue(result["should_skip"])
        self.assertEqual(result["reason"], "row_invalid_candidate_index")

    def test_row_abstain_skips_inactive_cv_in_active_only_format(self):
        result = decide_cv_row_abstain(
            alias_type="cv_head",
            format_type="cvvc",
            candidate_idx=1,
            candidate_count=3,
            candidate_active=False,
            active_only_formats={"cvvc", "cv"},
        )
        self.assertTrue(result["should_skip"])
        self.assertEqual(result["reason"], "row_inactive_candidate")

    def test_row_abstain_keeps_non_cv_alias(self):
        result = decide_cv_row_abstain(
            alias_type="vv",
            format_type="cvvc",
            candidate_idx=1,
            candidate_count=3,
            candidate_active=False,
            active_only_formats={"cvvc", "cv"},
        )
        self.assertFalse(result["should_skip"])

    def test_row_abstain_skips_low_margin_candidate(self):
        result = decide_cv_row_abstain(
            alias_type="cv",
            format_type="cvvc",
            candidate_idx=1,
            candidate_count=3,
            candidate_active=True,
            confidence_margin=2.5,
            min_confidence_margin=6.0,
            active_only_formats={"cvvc", "cv"},
            margin_formats={"cvvc", "cv"},
        )
        self.assertTrue(result["should_skip"])
        self.assertEqual(result["reason"], "row_low_margin_candidate")

    def test_row_abstain_keeps_cvc_cv_when_cvc_not_gated(self):
        result = decide_cv_row_abstain(
            alias_type="cv",
            format_type="cvc",
            candidate_idx=1,
            candidate_count=3,
            candidate_active=False,
            confidence_margin=-4.0,
            min_confidence_margin=6.0,
            active_only_formats={"cvvc", "cv"},
            margin_formats={"cvvc", "cv"},
        )
        self.assertFalse(result["should_skip"])
        self.assertEqual(result["reason"], "")


if __name__ == "__main__":
    unittest.main()
