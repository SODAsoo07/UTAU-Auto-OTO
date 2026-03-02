import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_refiner import (
    _apply_language_specific_delta_policy,
    _route_format_for_feature,
    apply_oto_ml_to_oto_file,
)


class MlFallbackTests(unittest.TestCase):
    def test_japanese_vcv_routes_bridge_and_head_aliases(self):
        routed_vc = _route_format_for_feature(
            "japanese",
            {"format_type": "vcv", "alias_type": "vc", "coda_type": "none"},
        )
        routed_vv = _route_format_for_feature(
            "japanese",
            {"format_type": "vcv", "alias_type": "vv", "coda_type": "none"},
        )
        routed_vcv = _route_format_for_feature(
            "japanese",
            {"format_type": "vcv", "alias_type": "vcv", "coda_type": "none"},
        )
        routed_cv = _route_format_for_feature(
            "japanese",
            {"format_type": "vcv", "alias_type": "cv", "coda_type": "none"},
        )
        routed_head = _route_format_for_feature(
            "japanese",
            {"format_type": "vcv", "alias_type": "cv_head", "coda_type": "none"},
        )
        self.assertEqual(routed_vc, "vcv")
        self.assertEqual(routed_vv, "vcv")
        self.assertEqual(routed_vcv, "vcv")
        self.assertEqual(routed_head, "vcv")
        self.assertIsNone(routed_cv)

    def test_japanese_vcv_format_override_blocks_cvvc_route(self):
        routed = _route_format_for_feature(
            "japanese",
            {"format_type": "cvvc", "alias_type": "cv", "coda_type": "none"},
            format_override="vcv",
        )
        self.assertIsNone(routed)

        routed_bridge = _route_format_for_feature(
            "japanese",
            {"format_type": "cvvc", "alias_type": "vcv", "coda_type": "none"},
            format_override="vcv",
        )
        self.assertEqual(routed_bridge, "vcv")

    def test_korean_vcv_routes_vv_to_cvvc(self):
        routed = _route_format_for_feature(
            "korean",
            {"format_type": "vcv", "alias_type": "vv", "coda_type": "none"},
        )
        self.assertEqual(routed, "cvvc")

    def test_korean_vcv_routes_coda_vc_to_cvc(self):
        routed = _route_format_for_feature(
            "korean",
            {"format_type": "vcv", "alias_type": "vc", "coda_type": "stop"},
        )
        self.assertEqual(routed, "cvc")

    def test_korean_vcv_routes_other_aliases_to_general(self):
        routed_cv = _route_format_for_feature(
            "korean",
            {"format_type": "vcv", "alias_type": "cv", "coda_type": "none"},
        )
        routed_vc_other = _route_format_for_feature(
            "korean",
            {"format_type": "vcv", "alias_type": "vc", "coda_type": "none"},
        )
        self.assertEqual(routed_cv, "general")
        self.assertEqual(routed_vc_other, "general")

    def test_korean_format_override_keeps_cvvc_route(self):
        routed = _route_format_for_feature(
            "korean",
            {"format_type": "cvc", "alias_type": "vc", "coda_type": "stop"},
            format_override="cvvc",
        )
        self.assertEqual(routed, "cvvc")

    def test_korean_cvvc_routes_only_bridge_aliases(self):
        routed_vv = _route_format_for_feature(
            "korean",
            {"format_type": "cvvc", "alias_type": "vv", "coda_type": "none"},
        )
        routed_vc = _route_format_for_feature(
            "korean",
            {"format_type": "cvvc", "alias_type": "vc", "coda_type": "stop"},
        )
        routed_cv = _route_format_for_feature(
            "korean",
            {"format_type": "cvvc", "alias_type": "cv", "coda_type": "none"},
        )
        routed_head = _route_format_for_feature(
            "korean",
            {"format_type": "cvvc", "alias_type": "cv_head", "coda_type": "none"},
        )
        self.assertEqual(routed_vv, "cvvc")
        self.assertEqual(routed_vc, "cvvc")
        self.assertIsNone(routed_cv)
        self.assertIsNone(routed_head)

    def test_japanese_vcv_n_bridge_voiced_deltas_are_heavily_damped(self):
        adjusted = _apply_language_specific_delta_policy(
            "japanese",
            {"format_type": "vcv", "alias_type": "vcv", "alias_text": "n じょ"},
            {
                "delta_offset": -40.0,
                "delta_pre": -30.0,
                "delta_cons": 20.0,
                "delta_cutoff": 50.0,
                "delta_ovl": 12.0,
            },
        )
        self.assertGreater(adjusted["delta_offset"], -10.0)
        self.assertGreater(adjusted["delta_pre"], -10.0)
        self.assertLess(adjusted["delta_cutoff"], 30.0)

    def test_japanese_cvvc_sibilant_does_not_pull_pre_and_cutoff_too_far_forward(self):
        adjusted = _apply_language_specific_delta_policy(
            "japanese",
            {"format_type": "cvvc", "alias_type": "cv", "alias_text": "し"},
            {
                "delta_offset": -24.0,
                "delta_pre": -20.0,
                "delta_cons": -12.0,
                "delta_cutoff": 60.0,
                "delta_ovl": -10.0,
            },
        )
        self.assertEqual(adjusted["delta_offset"], 0.0)
        self.assertGreater(adjusted["delta_pre"], -8.0)
        self.assertLess(adjusted["delta_cutoff"], 20.0)

    def test_japanese_cvvc_plosive_cv_freezes_offset_delta(self):
        adjusted = _apply_language_specific_delta_policy(
            "japanese",
            {"format_type": "cvvc", "alias_type": "cv_head", "alias_text": "- た"},
            {
                "delta_offset": -18.0,
                "delta_pre": -12.0,
                "delta_cons": 8.0,
                "delta_cutoff": 40.0,
                "delta_ovl": 4.0,
            },
        )
        self.assertEqual(adjusted["delta_offset"], 0.0)
        self.assertLess(adjusted["delta_cutoff"], 12.0)

    def test_korean_cvc_routes_only_coda_vc(self):
        routed_vc = _route_format_for_feature(
            "korean",
            {"format_type": "cvc", "alias_type": "vc", "coda_type": "stop"},
        )
        routed_cv = _route_format_for_feature(
            "korean",
            {"format_type": "cvc", "alias_type": "cv", "coda_type": "none"},
        )
        routed_head = _route_format_for_feature(
            "korean",
            {"format_type": "cvc", "alias_type": "cv_head", "coda_type": "none"},
        )
        self.assertEqual(routed_vc, "cvc")
        self.assertIsNone(routed_cv)
        self.assertIsNone(routed_head)

    def test_refiner_skips_when_model_missing(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            original = "test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n"
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write(original)
            os.environ["UTOA_DISABLE_OTO_ML"] = "1"
            try:
                changed = apply_oto_ml_to_oto_file("korean", oto_path, tg_dir=td, wav_dir=td)
            finally:
                os.environ.pop("UTOA_DISABLE_OTO_ML", None)
            self.assertEqual(changed, 0)
            with open(oto_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), original)

    def test_refiner_skips_when_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            original = "test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n"
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write(original)
            changed = apply_oto_ml_to_oto_file("korean", oto_path, tg_dir=td, wav_dir=td, enabled=False)
            self.assertEqual(changed, 0)
            with open(oto_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), original)


if __name__ == "__main__":
    unittest.main()
