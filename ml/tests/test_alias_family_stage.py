import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_alias_family_stage import (
    build_ja_alias_family_state,
    try_handle_ja_br_alias,
    try_handle_ja_tail_breath_alias,
)
from core.kr_alias_family_stage import (
    build_kr_alias_family_state,
    try_handle_kr_glottal_alias,
)


class AliasFamilyStageTests(unittest.TestCase):
    def test_build_kr_alias_family_state_preserves_cv_vc_family_flags(self):
        state = build_kr_alias_family_state(
            alias="a R",
            alias_type="vcv",
            uses_vc_context_fn=lambda value: value in {"vc", "vv"},
            is_diphthong_fn=lambda alias: alias == "gwa",
            extract_cv_alias_token_fn=lambda alias: alias.split()[0],
            detect_glottal_kind_fn=lambda alias: "tail" if alias.endswith("ʔ") else "",
            kr_vowels={"a", "i", "u", "e", "o"},
        )
        self.assertEqual(state.alias_type, "vcv")
        self.assertFalse(state.is_vc)
        self.assertTrue(state.is_vcv)
        self.assertFalse(state.is_cv_head)
        self.assertFalse(state.is_diph)
        self.assertEqual(state.target_clean, "a")
        self.assertEqual(state.breath_tail, "R")

    def test_try_handle_kr_glottal_alias_appends_row_without_touching_cv_branch(self):
        final_lines = []
        vowel_phone = SimpleNamespace(minTime=0.10, maxTime=0.24)
        handled, current_w_idx, cv_seq_idx = try_handle_kr_glottal_alias(
            alias="a ʔ",
            state=SimpleNamespace(glottal_kind="tail"),
            current_w_idx=0,
            cv_seq_idx=0,
            syllables_info=[{"phones": [vowel_phone], "end_time": 0.36}],
            ph_intervals=[vowel_phone, SimpleNamespace(minTime=0.24, maxTime=0.36)],
            real_wav_name="a.wav",
            alias_suffix="",
            final_lines=final_lines,
            validate_fn=lambda o, c, ct, p, ov: (o, c, ct, p, ov),
            apply_alias_suffix_fn=lambda alias, suffix: alias,
            find_vowel_phone_fn=lambda phones: ("a", phones[-1]),
        )
        self.assertTrue(handled)
        self.assertEqual(current_w_idx, 0)
        self.assertEqual(cv_seq_idx, 0)
        self.assertEqual(len(final_lines), 1)
        self.assertIn("a.wav=a ʔ,", final_lines[0])

    def test_build_ja_alias_family_state_keeps_vv_separate_from_cv(self):
        state = build_ja_alias_family_state(
            alias="a i",
            alias_type="vcv",
            format_type="cvvc",
            is_vowel_token_fn=lambda token: token in {"a", "i", "u", "e", "o"},
            ja_consonants={"k", "s", "t"},
        )
        self.assertEqual(state.alias_type, "vv")
        self.assertTrue(state.is_vc)
        self.assertFalse(state.is_vcv)
        self.assertFalse(state.is_cv_head)
        self.assertEqual(state.tail_breath, "")

    def test_try_handle_ja_br_alias_appends_openutau_rows(self):
        final_lines = []
        handled = try_handle_ja_br_alias(
            alias="br",
            alias_type="br",
            ph_intervals=[SimpleNamespace(minTime=0.05, maxTime=0.22)],
            real_wav_name="breath.wav",
            final_lines=final_lines,
            post_ctx=SimpleNamespace(post_adjust=lambda *args, **kwargs: (10.0, 20.0, -30.0, 0.0, 0.0)),
            alias_out_fn=lambda alias: f"out_{alias}",
            generate_openutau=True,
            generate_openutau_aliases_fn=lambda alias: [alias, alias + "2"],
        )
        self.assertTrue(handled)
        self.assertEqual(
            final_lines,
            [
                "breath.wav=out_br,10.00,20.00,-30.00,0.00,0.00",
                "breath.wav=out_br2,10.00,20.00,-30.00,0.00,0.00",
            ],
        )

    def test_try_handle_ja_tail_breath_alias_uses_mono_params_only(self):
        final_lines = []
        handled, current_w_idx = try_handle_ja_tail_breath_alias(
            alias="a R",
            state=SimpleNamespace(tail_breath="R"),
            current_w_idx=0,
            syllables_info=[{"phones": [SimpleNamespace(mark="a", minTime=0.12, maxTime=0.30)]}],
            real_wav_name="tail.wav",
            final_lines=final_lines,
            post_ctx=SimpleNamespace(post_adjust=lambda *args, **kwargs: (40.0, 55.0, -66.0, 77.0, 22.0)),
            alias_out_fn=lambda alias: alias,
            generate_openutau=False,
            generate_openutau_aliases_fn=lambda alias: [alias],
        )
        self.assertTrue(handled)
        self.assertEqual(current_w_idx, 0)
        self.assertEqual(final_lines, ["tail.wav=a R,40.00,55.00,-66.00,77.00,22.00"])


if __name__ == "__main__":
    unittest.main()
