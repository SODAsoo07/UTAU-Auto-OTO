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
    generate_korean_phonemizer_aliases,
    normalize_key,
    replace_oto_line_wav_name,
)
from core.file_prepare import prepare_file_context
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

    def test_prepare_file_context_keeps_logical_name_and_uses_actual_wav_name(self):
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "ga_gi_gu_ge_go_geu_geo_ga.wav")
            open(wav_path, "wb").close()
            wav_index = _build_wav_index(td)

            ctx = prepare_file_context(
                fname="ga'gi'gu'ge'go'geu'geo'ga.wav",
                lines=[],
                resolve_tg_info_fn=lambda _fname: {
                    "path": os.path.join(td, "ga'gi'gu'ge'go'geu'geo'ga.TextGrid"),
                    "real_name": "ga'gi'gu'ge'go'geu'geo'ga.wav",
                    "output_name": "ga_gi_gu_ge_go_geu_geo_ga.wav",
                },
                wav_root_for_signal=td,
                wav_index_for_signal=wav_index,
                mel_cache_for_signal={},
                find_wav_path_fn=lambda wav_name, wav_root, index: os.path.join(wav_root, index[normalize_key(wav_name)].split(os.sep)[-1]),
                read_wav_fn=lambda _path: ([], 0),
                mel_envelope_fn=lambda _audio, _sr: {},
            )

            self.assertEqual(ctx.real_wav_name, "ga'gi'gu'ge'go'geu'geo'ga.wav")
            self.assertEqual(ctx.output_wav_name, "ga_gi_gu_ge_go_geu_geo_ga.wav")
            self.assertEqual(os.path.basename(ctx.wav_path_for_signal), "ga_gi_gu_ge_go_geu_geo_ga.wav")

    def test_korean_phonemizer_aliases_are_targeted(self):
        self.assertEqual(
            generate_korean_phonemizer_aliases("ga", alias_type="cv", format_type="cvc"),
            ["ga", "- ga", "-ga"],
        )
        self.assertEqual(
            generate_korean_phonemizer_aliases("a n", alias_type="vc", format_type="cvvc"),
            ["a n", "an", "a N", "aN"],
        )

    def test_replace_oto_line_wav_name_only_rewrites_left_side(self):
        line = "ga'gi'gu'ge'go'geu'geo'ga.wav=ga,0,0,0,0,0"
        self.assertEqual(
            replace_oto_line_wav_name(line, "ga_gi_gu_ge_go_geu_geo_ga.wav"),
            "ga_gi_gu_ge_go_geu_geo_ga.wav=ga,0,0,0,0,0",
        )

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
