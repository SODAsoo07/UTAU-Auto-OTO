import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_features import (
    FEATURE_NAMES,
    _normalize_alias_for_match,
    canonicalize_feature_row,
    extract_feature_rows,
    get_feature_schema,
)
from core.ja_oto_generator import (
    _adaptive_ja_overlap,
    _clamp_ja_bridge_overlap,
    _recenter_ja_params_around_pre,
    _select_ja_cv_syllable_index,
    _select_vcv_syllable_index,
    detect_ja_alias_format,
)
from core.ja_lab_generator import parse_ja_filename
from core.oto_generator import _recenter_kr_params_around_pre


class FeatureExtractionTests(unittest.TestCase):
    def test_schema_contains_feature_names(self):
        schema = get_feature_schema()
        self.assertEqual(schema["feature_names"], FEATURE_NAMES)

    def test_canonicalize_feature_row_defaults(self):
        row = canonicalize_feature_row({"language": "korean", "base_offset": 123.4})
        self.assertEqual(row["language"], "korean")
        self.assertAlmostEqual(row["base_offset"], 123.4)
        self.assertIn("alias_type", row)
        self.assertIn("energy_mean", row)

    def test_extract_feature_rows_without_tg_or_wav(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n")
            rows = extract_feature_rows("korean", oto_path, tg_dir=td, wav_dir=td)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["wav"], "test.wav")
            self.assertEqual(rows[0]["language"], "korean")

    def test_alias_match_normalization_strips_pitch_suffix(self):
        self.assertEqual(_normalize_alias_for_match("a k_C4", language="japanese"), "a k")
        self.assertEqual(_normalize_alias_for_match("- ka-D4", language="japanese"), "- ka")
        self.assertEqual(_normalize_alias_for_match("ri F4", language="japanese"), "ri")

    def test_alias_match_normalization_strips_style_suffix(self):
        self.assertEqual(_normalize_alias_for_match("ga_soft", language="japanese"), "ga")
        self.assertEqual(_normalize_alias_for_match("a か連", language="japanese"), "a か")
        self.assertEqual(_normalize_alias_for_match("lyeo부드러움", language="korean"), "lyeo")

    def test_japanese_format_prefers_cvvc_or_cv(self):
        self.assertEqual(detect_ja_alias_format(["- か", "か", "き"]), "cv")
        self.assertEqual(detect_ja_alias_format(["a k", "ka", "a i"]), "cvvc")
        self.assertEqual(detect_ja_alias_format([f"a {'か'}", f"i {'き'}"]), "vcv")

    def test_parse_ja_filename_keeps_small_vowel_combo_as_single_mora(self):
        self.assertEqual(
            parse_ja_filename("すぃすぃすすぃせすぃそ.wav"),
            ["si", "si", "su", "si", "se", "si", "so"],
        )

    def test_parse_ja_filename_keeps_yoon_combo_as_single_mora(self):
        self.assertEqual(
            parse_ja_filename("じぇじぇじょじぇんじぇじゃ.wav"),
            ["je", "je", "jo", "je", "n", "je", "ja"],
        )

    def test_japanese_cv_selector_prefers_real_se_over_previous_si(self):
        syllables = [{"word": s} for s in ["si", "si", "su", "si", "se", "si", "so"]]
        self.assertEqual(_select_ja_cv_syllable_index("せ", 3, syllables, alias_type="cv"), 4)

    def test_japanese_vcv_selector_prefers_real_je_combo(self):
        syllables = [{"word": s} for s in ["je", "je", "jo", "je", "n", "je", "ja"]]
        self.assertEqual(_select_vcv_syllable_index("a じぇ", 2, syllables), 3)

    def test_japanese_bridge_overlap_moves_close_to_pre(self):
        vc_hard = _clamp_ja_bridge_overlap(120.0, 20.0, "k", mode="vc")
        vc_soft = _clamp_ja_bridge_overlap(120.0, 20.0, "m", mode="vc")
        vv = _clamp_ja_bridge_overlap(120.0, 20.0, "", mode="vv")
        self.assertGreater(vc_hard, 95.0)
        self.assertGreater(vc_soft, vc_hard)
        self.assertGreater(vv, vc_soft)

    def test_adaptive_japanese_overlap_prefers_vv_then_sonorant_then_hard_vc(self):
        vc_hard = _adaptive_ja_overlap(120.0, "k", mode="vc")
        vc_sonorant = _adaptive_ja_overlap(120.0, "m", mode="vc")
        vv = _adaptive_ja_overlap(120.0, "", mode="vv")
        self.assertGreater(vc_sonorant, vc_hard)
        self.assertGreater(vv, vc_sonorant)

    def test_recenter_japanese_params_around_pre_reduces_bridge_gap(self):
        _o, c, ct, p, ov = _recenter_ja_params_around_pre(
            0.0, 220.0, -360.0, 120.0, 20.0, alias_type="vc", alias_text="a k"
        )
        self.assertLessEqual(p - ov, 20.0)
        self.assertGreaterEqual(c - p, 20.0)
        self.assertGreaterEqual(abs(ct) - c, 12.0)

    def test_recenter_korean_params_around_pre_reduces_bridge_gap(self):
        _o, c, ct, p, ov = _recenter_kr_params_around_pre(
            0.0, 220.0, -360.0, 120.0, 20.0, alias_type="vc", alias_text="a T"
        )
        self.assertLessEqual(p - ov, 20.0)
        self.assertGreaterEqual(c - p, 18.0)
        self.assertGreaterEqual(abs(ct) - c, 12.0)


if __name__ == "__main__":
    unittest.main()
