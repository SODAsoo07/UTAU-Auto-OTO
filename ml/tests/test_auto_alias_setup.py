import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_auto_alias_setup import build_ja_auto_file_groups
from core.kr_auto_alias_setup import build_kr_auto_file_groups


class AutoAliasSetupTests(unittest.TestCase):
    def test_build_ja_auto_file_groups_builds_cvvc_lines(self):
        logs = []
        groups = build_ja_auto_file_groups(
            tg_entries=[{"path": "a.TextGrid", "real_name": "a-ka.wav"}],
            auto_gen_format="cvvc",
            log_fn=logs.append,
            load_textgrid_fn=lambda path: object(),
            parse_filename_fn=lambda base: ["a", "ka"],
            split_syllable_fn=lambda syl: ("k" if syl == "ka" else "", "a"),
            is_vowel_chain_fn=lambda syllables: False,
        )
        self.assertIn("a-ka.wav", groups)
        self.assertEqual(groups["a-ka.wav"], [
            "a-ka.wav=a,0,0,0,0,0",
            "a-ka.wav=ka,0,0,0,0,0",
            "a-ka.wav=a k,0,0,0,0,0",
        ])

    def test_build_kr_auto_file_groups_builds_cvvc_lines(self):
        class FakeTier(list):
            def __init__(self, name, rows):
                super().__init__(rows)
                self.name = name

        class FakeGrid(list):
            pass

        grid = FakeGrid([
            FakeTier("words", [SimpleNamespace(mark="가"), SimpleNamespace(mark="나")])
        ])
        groups = build_kr_auto_file_groups(
            tg_entries=[{"path": "kr.TextGrid", "real_name": "ga-na.wav"}],
            auto_gen_format="cvvc",
            log_fn=lambda msg: None,
            load_textgrid_fn=lambda path: grid,
            decompose_hangul_to_roman_fn=lambda ch: ["ga"] if ch == "가" else ["na"],
            split_syllable_parts_fn=lambda roman: ("g", "a", "") if roman == "ga" else ("n", "a", ""),
            kr_vowels={"a", "i", "u", "e", "o"},
            kr_consonants={"g", "n", "k", "t", "p", "m", "l", "r", "s", "h", "j", "c"},
        )
        self.assertIn("ga-na.wav", groups)
        self.assertEqual(groups["ga-na.wav"], [
            "ga-na.wav=ga,0,0,0,0,0",
            "ga-na.wav=na,0,0,0,0,0",
            "ga-na.wav=a n,0,0,0,0,0",
        ])


if __name__ == "__main__":
    unittest.main()
