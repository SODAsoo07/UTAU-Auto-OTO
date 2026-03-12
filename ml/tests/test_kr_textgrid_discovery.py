import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_generator import (
    _build_wav_index,
    _iter_textgrid_files,
    _resolve_real_wav_name_for_textgrid,
    _should_keep_template_alias_set_exact,
    normalize_key,
)
from core.oto_validator import _build_textgrid_index


class KoreanTextGridDiscoveryTests(unittest.TestCase):
    def test_generator_discovers_nested_textgrids(self):
        with tempfile.TemporaryDirectory() as td:
            nested = os.path.join(td, "textgrids")
            os.makedirs(nested, exist_ok=True)
            tg_path = os.path.join(nested, "ga_gi_gu_ge_go_geu_geo_ga.TextGrid")
            open(tg_path, "w", encoding="utf-8").close()

            found = sorted(os.path.join(d, n) for d, n in _iter_textgrid_files(td))
            self.assertEqual(found, [tg_path])

    def test_validator_index_matches_template_style_name(self):
        with tempfile.TemporaryDirectory() as td:
            nested = os.path.join(td, "textgrids")
            os.makedirs(nested, exist_ok=True)
            tg_path = os.path.join(nested, "ga_gi_gu_ge_go_geu_geo_ga.TextGrid")
            open(tg_path, "w", encoding="utf-8").close()

            idx = _build_textgrid_index(td)
            key = normalize_key("ga'gi'gu'ge'go'geu'geo'ga.wav")
            self.assertEqual(idx.get(key), tg_path)

    def test_generator_resolves_real_wav_name_from_textgrid_variant(self):
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "ga_gi_gu_ge_go_geu_geo_ga.wav")
            open(wav_path, "wb").close()

            wav_index = _build_wav_index(td)
            real_name = _resolve_real_wav_name_for_textgrid("ga'gi'gu'ge'go'geu'geo'ga.TextGrid", td, wav_index)

            self.assertEqual(real_name, "ga_gi_gu_ge_go_geu_geo_ga.wav")

    def test_template_mode_keeps_exact_alias_set_only_when_both_options_are_off(self):
        self.assertTrue(
            _should_keep_template_alias_set_exact(
                use_template=True,
                generate_openutau=False,
                gen_missing_vowels=False,
            )
        )
        self.assertFalse(
            _should_keep_template_alias_set_exact(
                use_template=True,
                generate_openutau=True,
                gen_missing_vowels=False,
            )
        )
        self.assertFalse(
            _should_keep_template_alias_set_exact(
                use_template=True,
                generate_openutau=False,
                gen_missing_vowels=True,
            )
        )
        self.assertFalse(
            _should_keep_template_alias_set_exact(
                use_template=False,
                generate_openutau=False,
                gen_missing_vowels=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
