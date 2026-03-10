import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core.oto_ml_features as oto_ml_features


class KoreanBreathBranchingTests(unittest.TestCase):
    def test_classify_kr_breath_alias_kind(self):
        self.assertEqual(oto_ml_features._classify_kr_breath_alias_kind("a R", "vc"), "tail_breath")
        self.assertEqual(oto_ml_features._classify_kr_breath_alias_kind("oH", "vc"), "tail_breath")
        self.assertEqual(oto_ml_features._classify_kr_breath_alias_kind("br", "br"), "standalone_breath")
        self.assertEqual(oto_ml_features._classify_kr_breath_alias_kind("bre", "br"), "standalone_breath")
        self.assertEqual(oto_ml_features._classify_kr_breath_alias_kind("a r", "vc"), "")

    def test_derive_alias_group_distinguishes_tail_and_standalone_breath(self):
        vc_feat = {"alias_type": "vc", "coda_type": "other", "is_diphthong": 0.0, "onset_class": ""}
        br_feat = {"alias_type": "br", "coda_type": "none", "is_diphthong": 0.0, "onset_class": ""}
        self.assertEqual(oto_ml_features._derive_alias_group("korean", vc_feat, alias="a R"), "vc_tail_breath")
        self.assertEqual(oto_ml_features._derive_alias_group("korean", vc_feat, alias="ang"), "vc_other")
        self.assertEqual(oto_ml_features._derive_alias_group("korean", br_feat, alias="br"), "br_standalone")

    def test_training_quality_uses_relaxed_limits_for_tail_breath(self):
        row_ok = {
            "alias_type": "vc",
            "alias_group": "vc_tail_breath",
            "coda_type": "other",
            "is_diphthong": 0.0,
            "delta_offset": 100.0,
            "delta_cons": 30.0,
            "delta_cutoff": 320.0,
            "delta_pre": 180.0,
            "delta_ovl": 140.0,
            "base_offset_to_expected_ms": 0.0,
        }
        keep, reasons, _score = oto_ml_features._evaluate_training_row_quality("korean", row_ok)
        self.assertEqual(keep, 1)
        self.assertEqual(reasons, "")

        row_bad = dict(row_ok)
        row_bad["delta_cutoff"] = 500.0
        keep_bad, reasons_bad, _ = oto_ml_features._evaluate_training_row_quality("korean", row_bad)
        self.assertEqual(keep_bad, 0)
        self.assertIn("gross_outlier", reasons_bad)


if __name__ == "__main__":
    unittest.main()
