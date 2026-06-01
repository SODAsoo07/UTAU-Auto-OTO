import unittest

from core.alias_annotation_utils import strip_alias_annotation_suffixes
from core.ja_oto_mapping import classify_ja_alias, is_breath as is_ja_breath
from core.kr_oto_rules import classify_alias as classify_kr_alias
from core.kr_oto_rules import is_breath as is_kr_breath


class SpecialAliasSupportTests(unittest.TestCase):
    def test_japanese_middle_dot_alias_as_edge(self):
        self.assertEqual(classify_ja_alias("a \u30fb"), "vc")
        self.assertEqual(classify_ja_alias("\u30fb ka"), "cv_head")
        self.assertEqual(classify_ja_alias("a\u00b7"), "vc")

    def test_korean_middle_dot_alias_as_edge(self):
        self.assertEqual(classify_kr_alias("a \u30fb"), "vc")
        self.assertEqual(classify_kr_alias("\u30fb a"), "cv_head")

    def test_korean_vocal_fry_hint_alias(self):
        self.assertEqual(classify_kr_alias("a fry"), "vc")
        self.assertEqual(classify_kr_alias("- vf"), "cv_head")

    def test_english_breath_aliases(self):
        self.assertTrue(is_ja_breath("inhale"))
        self.assertTrue(is_kr_breath("exhale"))
        self.assertEqual(classify_ja_alias("breathy"), "br")
        self.assertEqual(classify_kr_alias("aspirate"), "br")

    def test_tail_breath_aliases_are_vc_not_br(self):
        self.assertEqual(classify_kr_alias("a R"), "vc")
        self.assertEqual(classify_kr_alias("i 吸"), "vc")
        self.assertFalse(is_kr_breath("a R"))
        self.assertFalse(is_kr_breath("i 吸"))

        self.assertEqual(classify_ja_alias("あ 吸い"), "vc")
        self.assertEqual(classify_ja_alias("a H"), "vc")
        self.assertFalse(is_ja_breath("あ 吸い"))
        self.assertFalse(is_ja_breath("a H"))

    def test_korean_custom_map_lowercase_liquids_are_phonetic(self):
        self.assertEqual(classify_kr_alias("ra", {"ra": "r"}), "cv")
        self.assertEqual(classify_kr_alias("la", {"la": "l"}), "cv")
        self.assertEqual(classify_kr_alias("ha", {"ha": "h"}), "cv")
        self.assertEqual(classify_kr_alias("release", {"release": "R"}), "br")

    def test_style_suffix_stripping(self):
        stripped = strip_alias_annotation_suffixes("ga_eng01", classifier=classify_kr_alias)
        self.assertEqual(stripped, "ga")
        fry_kept = strip_alias_annotation_suffixes("ga_fry01", classifier=classify_kr_alias)
        self.assertEqual(fry_kept, "ga_fry")
        untouched_edge = strip_alias_annotation_suffixes("a \u30fb", classifier=classify_kr_alias)
        self.assertEqual(untouched_edge, "a \u30fb")

    def test_pitch_suffix_stripping(self):
        self.assertEqual(strip_alias_annotation_suffixes("ga_C4", classifier=classify_kr_alias), "ga")
        self.assertEqual(strip_alias_annotation_suffixes("ga C#4", classifier=classify_kr_alias), "ga")
        self.assertEqual(strip_alias_annotation_suffixes("ga_note_D4", classifier=classify_kr_alias), "ga")


if __name__ == "__main__":
    unittest.main()
