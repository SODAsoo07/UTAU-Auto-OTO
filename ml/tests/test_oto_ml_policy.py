import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_policy import (
    alias_family_to_alias_types,
    default_training_filters,
    infer_alias_family,
    selector_enabled_by_default,
)


class OtoMlPolicyTests(unittest.TestCase):
    def test_default_training_filters_are_more_conservative_for_cvvc_and_cvc(self):
        kr_cv = default_training_filters("korean", "cv")
        kr_cvc = default_training_filters("korean", "cvc")
        kr_cvvc = default_training_filters("korean", "cvvc")
        jp_cvvc = default_training_filters("japanese", "cvvc")
        self.assertGreater(kr_cvc["min_mapping_confidence"], kr_cv["min_mapping_confidence"])
        self.assertGreater(kr_cvvc["min_mapping_confidence"], kr_cv["min_mapping_confidence"])
        self.assertFalse(kr_cv["use_pseudo_labels"])
        self.assertFalse(jp_cvvc["use_pseudo_labels"])

    def test_alias_family_mapping(self):
        self.assertEqual(alias_family_to_alias_types("cv"), ["cv", "cv_head"])
        self.assertEqual(alias_family_to_alias_types("vowel"), ["mono"])
        self.assertEqual(alias_family_to_alias_types("bridge"), ["vc", "vv", "vcv"])

    def test_infer_alias_family(self):
        self.assertEqual(infer_alias_family("korean", {"format_type": "cvc", "alias_type": "cv"}), "cv")
        self.assertEqual(infer_alias_family("korean", {"format_type": "cvc", "alias_type": "mono"}), "vowel")
        self.assertEqual(infer_alias_family("japanese", {"format_type": "cvvc", "alias_type": "vc"}), "bridge")

    def test_selector_default_policy(self):
        self.assertTrue(selector_enabled_by_default("korean", "cv", "cv"))
        self.assertTrue(selector_enabled_by_default("korean", "cv", "vowel"))
        self.assertFalse(selector_enabled_by_default("korean", "cvvc", "cv"))
        self.assertFalse(selector_enabled_by_default("korean", "cvc", "cv"))
        self.assertFalse(selector_enabled_by_default("japanese", "vcv", "bridge"))


if __name__ == "__main__":
    unittest.main()
