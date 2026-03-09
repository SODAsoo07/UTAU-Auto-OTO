import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_refiner import (
    _apply_japanese_cvvc_cv_post_guard,
    _apply_korean_bridge_post_guard,
    _apply_language_specific_delta_policy,
    _route_format_for_feature,
    apply_oto_ml_selector_candidate,
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

    def test_korean_vcv_routes_coda_vc_to_cv_family(self):
        routed = _route_format_for_feature(
            "korean",
            {"format_type": "vcv", "alias_type": "vc", "coda_type": "stop"},
        )
        self.assertEqual(routed, "cv")

    def test_korean_vcv_routes_non_bridge_aliases_to_vcv(self):
        routed_cv = _route_format_for_feature(
            "korean",
            {"format_type": "vcv", "alias_type": "cv", "coda_type": "none"},
        )
        routed_vc_other = _route_format_for_feature(
            "korean",
            {"format_type": "vcv", "alias_type": "vc", "coda_type": "none"},
        )
        self.assertEqual(routed_cv, "vcv")
        self.assertEqual(routed_vc_other, "vcv")

    def test_korean_format_override_keeps_cvvc_route(self):
        routed = _route_format_for_feature(
            "korean",
            {"format_type": "cvc", "alias_type": "vc", "coda_type": "stop"},
            format_override="cvvc",
        )
        self.assertEqual(routed, "cvvc")

    def test_japanese_cv_routes_only_cv_like_aliases(self):
        routed_cv = _route_format_for_feature(
            "japanese",
            {"format_type": "cv", "alias_type": "cv", "mapping_confidence": 0.9},
        )
        routed_mono = _route_format_for_feature(
            "japanese",
            {"format_type": "cv", "alias_type": "mono", "mapping_confidence": 0.9},
        )
        routed_vc = _route_format_for_feature(
            "japanese",
            {"format_type": "cv", "alias_type": "vc", "mapping_confidence": 0.9},
        )
        self.assertEqual(routed_cv, "cv")
        self.assertEqual(routed_mono, "cv")
        self.assertIsNone(routed_vc)

    def test_korean_cv_routes_only_cv_like_aliases(self):
        routed_cv = _route_format_for_feature(
            "korean",
            {"format_type": "cv", "alias_type": "cv", "mapping_confidence": 0.9},
        )
        routed_mono = _route_format_for_feature(
            "korean",
            {"format_type": "cv", "alias_type": "mono", "mapping_confidence": 0.9},
        )
        routed_vc = _route_format_for_feature(
            "korean",
            {"format_type": "cv", "alias_type": "vc", "mapping_confidence": 0.9, "coda_type": "stop"},
        )
        self.assertEqual(routed_cv, "cv")
        self.assertEqual(routed_mono, "cv")
        self.assertEqual(routed_vc, "cv")

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

    def test_japanese_cvvc_cv_blocks_positive_cutoff_extension_even_for_non_sensitive_onset(self):
        adjusted = _apply_language_specific_delta_policy(
            "japanese",
            {"format_type": "cvvc", "alias_type": "cv", "alias_text": "あ"},
            {
                "delta_offset": 6.0,
                "delta_pre": 4.0,
                "delta_cons": 8.0,
                "delta_cutoff": 24.0,
                "delta_ovl": 3.0,
            },
        )
        self.assertLessEqual(adjusted["delta_cutoff"], 0.0)

    def test_korean_cvvc_vc_stop_damps_aggressive_deltas(self):
        adjusted = _apply_language_specific_delta_policy(
            "korean",
            {"format_type": "cvvc", "alias_type": "vc", "coda_type": "stop", "mapping_confidence": 0.9},
            {
                "delta_offset": -40.0,
                "delta_pre": -30.0,
                "delta_cons": 20.0,
                "delta_cutoff": 50.0,
                "delta_ovl": -10.0,
            },
        )
        self.assertGreater(adjusted["delta_offset"], -20.0)
        self.assertGreater(adjusted["delta_pre"], -18.0)
        self.assertLess(adjusted["delta_cutoff"], 20.0)

    def test_korean_vv_damps_offset_and_overlap_shift(self):
        adjusted = _apply_language_specific_delta_policy(
            "korean",
            {"format_type": "cvvc", "alias_type": "vv", "mapping_confidence": 0.95},
            {
                "delta_offset": -25.0,
                "delta_pre": 20.0,
                "delta_cons": 16.0,
                "delta_cutoff": 36.0,
                "delta_ovl": 18.0,
            },
        )
        self.assertGreater(adjusted["delta_offset"], -16.0)
        self.assertLess(adjusted["delta_ovl"], 12.0)

    def test_japanese_cvvc_cv_post_guard_blocks_too_early_preutterance(self):
        def _validate(offset, consonant, cutoff, pre, ovl):
            return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)

        out = _apply_japanese_cvvc_cv_post_guard(
            {
                "format_type": "cvvc",
                "alias_type": "cv",
                "curr_phone_start_ms": 120.0,
                "curr_phone_len_ms": 90.0,
                "curr_vowel_start_ms": 170.0,
                "expected_anchor_ms": 170.0,
                "base_offset": 140.0,
            },
            (90.0, 115.0, -160.0, 25.0, 10.0),
            _validate,
        )
        offset, _cons, _cutoff, pre, _ovl = out
        self.assertGreaterEqual(offset + pre, 146.0)
        self.assertGreaterEqual(offset, 112.0)

    def test_selector_only_path_for_japanese_cvvc_cv_uses_candidate_without_delta_pull(self):
        out = apply_oto_ml_selector_candidate(
            "japanese",
            {
                "format_type": "cvvc",
                "alias_type": "cv",
                "curr_phone_start_ms": 120.0,
                "curr_phone_len_ms": 90.0,
                "curr_vowel_start_ms": 170.0,
                "expected_anchor_ms": 170.0,
                "base_offset": 140.0,
            },
            (100.0, 118.0, 165.0, 30.0, 12.0),
        )
        offset, _cons, _cutoff, pre, _ovl = out
        self.assertGreaterEqual(offset + pre, 146.0)

    def test_korean_vc_post_guard_limits_cutoff_extension(self):
        def _validate(offset, consonant, cutoff, pre, ovl):
            return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)

        out = _apply_korean_bridge_post_guard(
            {
                "alias_type": "vc",
                "coda_type": "stop",
                "curr_phone_end_ms": 1000.0,
                "next_phone_gap_ms": 120.0,
            },
            (840.0, 220.0, -420.0, 160.0, 40.0),
            _validate,
        )
        _off, cons, cutoff, pre, ovl = out
        self.assertLessEqual(abs(cutoff), 287.0)
        self.assertGreaterEqual(pre - ovl, 9.0)
        self.assertGreaterEqual(cons - pre, 13.0)

    def test_korean_cvc_routes_to_cvc_family(self):
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
        self.assertEqual(routed_cv, "cvc")
        self.assertEqual(routed_head, "cvc")

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

    def test_refiner_policy_off_reports_standard_code(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            original = "test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n"
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write(original)
            report = {}
            changed = apply_oto_ml_to_oto_file(
                "korean",
                oto_path,
                tg_dir=td,
                wav_dir=td,
                policy="off",
                report=report,
            )
            self.assertEqual(changed, 0)
            self.assertEqual(report["code"], "ML_POLICY_OFF")
            self.assertEqual(report["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
