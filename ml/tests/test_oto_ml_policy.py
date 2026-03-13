import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_policy import (
    alias_family_to_alias_types,
    default_training_filters,
    delta_enabled_by_default,
    infer_alias_family,
    recommended_alias_family_splits,
    selector_enabled_by_default,
    should_split_alias_families,
)


class OtoMlPolicyTests(unittest.TestCase):
    def test_default_training_filters_are_more_conservative_for_cvvc_and_cvc(self):
        kr_cv = default_training_filters("korean", "cv")
        kr_cvc = default_training_filters("korean", "cvc")
        kr_cvvc = default_training_filters("korean", "cvvc")
        jp_cvvc = default_training_filters("japanese", "cvvc")
        self.assertGreater(kr_cvc["min_mapping_confidence"], kr_cv["min_mapping_confidence"])
        self.assertGreater(kr_cvvc["min_mapping_confidence"], kr_cv["min_mapping_confidence"])

    def test_bridge_family_filters_drop_train_keep_for_cvc_and_cvvc(self):
        kr_cvc_vc = default_training_filters("korean", "cvc", "vc")
        kr_cvvc_vc = default_training_filters("korean", "cvvc", "vc")
        jp_cvvc_vc = default_training_filters("japanese", "cvvc", "vc")
        kr_cv_vc = default_training_filters("korean", "cv", "vc")
        self.assertFalse(kr_cvc_vc["require_train_keep"])
        self.assertFalse(kr_cvvc_vc["require_train_keep"])
        self.assertFalse(jp_cvvc_vc["require_train_keep"])
        self.assertTrue(kr_cv_vc["require_train_keep"])

    def test_alias_family_mapping(self):
        self.assertEqual(alias_family_to_alias_types("cv"), ["cv", "cv_head"])
        self.assertEqual(alias_family_to_alias_types("vowel"), ["mono"])
        self.assertEqual(alias_family_to_alias_types("vc"), ["vc", "vv"])
        self.assertEqual(alias_family_to_alias_types("vcv"), ["vcv"])

    def test_infer_alias_family(self):
        self.assertEqual(infer_alias_family("korean", {"format_type": "cvc", "alias_type": "cv"}), "cv")
        self.assertEqual(infer_alias_family("korean", {"format_type": "cvc", "alias_type": "mono"}), "vowel")
        self.assertEqual(infer_alias_family("japanese", {"format_type": "cvvc", "alias_type": "vc"}), "vc")
        self.assertEqual(infer_alias_family("japanese", {"format_type": "cvvc", "alias_type": "vcv"}), "vcv")

    def test_selector_default_policy(self):
        self.assertTrue(selector_enabled_by_default("korean", "cv", "cv"))
        self.assertTrue(selector_enabled_by_default("korean", "cv", "vowel"))
        self.assertTrue(selector_enabled_by_default("korean", "cvc", "cv"))
        self.assertTrue(selector_enabled_by_default("korean", "cvc", "vc"))
        self.assertTrue(selector_enabled_by_default("korean", "cvc", "vcv"))
        self.assertTrue(selector_enabled_by_default("japanese", "cvvc", "cv"))
        self.assertTrue(selector_enabled_by_default("japanese", "cvvc", "vc"))
        self.assertTrue(selector_enabled_by_default("japanese", "cvvc", "vcv"))
        self.assertFalse(selector_enabled_by_default("korean", "cvvc", "cv"))
        self.assertFalse(selector_enabled_by_default("korean", "vcv", "vcv"))
        self.assertFalse(selector_enabled_by_default("japanese", "cv", "cv"))
        self.assertFalse(selector_enabled_by_default("japanese", "vcv", "vcv"))


    def test_recommended_family_split_policy(self):
        self.assertFalse(should_split_alias_families("korean", "cv"))
        self.assertFalse(should_split_alias_families("japanese", "cv"))
        self.assertEqual(recommended_alias_family_splits("korean", "cvc"), ["cv", "vc", "vcv"])
        self.assertEqual(recommended_alias_family_splits("korean", "cvvc"), ["cv", "vc", "vcv"])
        self.assertEqual(recommended_alias_family_splits("japanese", "cvvc"), ["cv", "vc", "vcv"])

    def test_delta_default_policy(self):
        self.assertTrue(delta_enabled_by_default("korean", "cv", "cv"))
        self.assertTrue(delta_enabled_by_default("korean", "cvc", "vc"))
        self.assertFalse(delta_enabled_by_default("japanese", "cvvc", "cv"))
        self.assertTrue(delta_enabled_by_default("japanese", "cvvc", "vc"))

if __name__ == "__main__":
    unittest.main()
