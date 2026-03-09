import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_loop_prep import prepare_ja_loop_state
from core.kr_loop_prep import prepare_kr_loop_state


class LoopPrepTests(unittest.TestCase):
    def test_prepare_kr_loop_state_detects_format_and_targets(self):
        phone_tier = [SimpleNamespace(mark="g", minTime=0.0, maxTime=0.1), SimpleNamespace(mark="a", minTime=0.1, maxTime=0.2)]
        word_tier = [SimpleNamespace(mark="가", minTime=0.0, maxTime=0.2)]
        logs = []
        result = prepare_kr_loop_state(
            lines=["a.wav=ga,0,0,0,0,0"],
            real_wav_name="a.wav",
            phone_tier=phone_tier,
            word_tier=word_tier,
            wav_duration_ms=0.0,
            custom_map=None,
            kr_mapping_words_fallback_enabled=True,
            kr_mapping_spn_ratio_threshold=0.35,
            kr_mapping_min_vowel_phone_ratio=0.5,
            kr_mapping_confidence_threshold=None,
            debug_log_fn=logs.append,
            debug_reason_logging=True,
            decompose_hangul_to_roman_fn=lambda ch: ["ga"],
            synthesize_word_phones_fn=lambda *args: [],
            detect_alias_format_fn=lambda aliases, custom_map=None: "cv",
            extract_cv_targets_from_lines_fn=lambda lines, custom_map=None: ["ga"],
            extract_cv_targets_from_filename_fn=lambda name: ["ga"],
            collect_phone_quality_fn=lambda *args, **kwargs: {"low_confidence_reasons": [], "spn_ratio_in_phone_tier": 0.0},
            resolve_mapping_conf_threshold_fn=lambda fmt, override_threshold=None: 0.5,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.file_format, "cv")
        self.assertEqual(result.targets_for_build, ["ga"])
        self.assertTrue(any("형식 감지" in line for line in logs))

    def test_prepare_kr_loop_state_keeps_cvc_bank_format_and_locks_filename_sequence(self):
        phone_tier = [SimpleNamespace(mark="g", minTime=0.0, maxTime=0.1), SimpleNamespace(mark="a", minTime=0.1, maxTime=0.2)]
        word_tier = [SimpleNamespace(mark="가", minTime=0.0, maxTime=0.2)]
        result = prepare_kr_loop_state(
            lines=["a.wav=ga,0,0,0,0,0"],
            real_wav_name="a.wav",
            phone_tier=phone_tier,
            word_tier=word_tier,
            wav_duration_ms=0.0,
            custom_map=None,
            kr_mapping_words_fallback_enabled=True,
            kr_mapping_spn_ratio_threshold=0.35,
            kr_mapping_min_vowel_phone_ratio=0.5,
            kr_mapping_confidence_threshold=None,
            debug_log_fn=lambda msg: None,
            debug_reason_logging=True,
            decompose_hangul_to_roman_fn=lambda ch: ["ga"],
            synthesize_word_phones_fn=lambda *args: [],
            detect_alias_format_fn=lambda aliases, custom_map=None: "cv",
            extract_cv_targets_from_lines_fn=lambda lines, custom_map=None: ["ga"],
            extract_cv_targets_from_filename_fn=lambda name: ["ga", "ga"],
            collect_phone_quality_fn=lambda *args, **kwargs: {
                "low_confidence_reasons": ["insufficient_vowel_phones"],
                "spn_ratio_in_phone_tier": 0.0,
                "phones_vs_expected_syllables_ratio": 0.5,
                "known_vowel_phone_count": 1.0,
                "expected_syllables": 2,
            },
            resolve_mapping_conf_threshold_fn=lambda fmt, override_threshold=None: 0.5,
            preferred_format="cvc",
        )
        self.assertEqual(result.detected_format, "cv")
        self.assertEqual(result.file_format, "cvc")
        self.assertIn(result.textgrid_trust_tier, {"mid", "low"})
        self.assertTrue(result.prefer_filename_sequence)
        self.assertEqual(result.targets_for_build, ["ga", "ga"])

    def test_prepare_ja_loop_state_marks_empty_intervals(self):
        ph_tier = [SimpleNamespace(mark="spn", minTime=0.0, maxTime=0.1)]
        word_tier = []
        result = prepare_ja_loop_state(
            fname="a.wav",
            lines=["a.wav=ka,0,0,0,0,0"],
            real_wav_name="a.wav",
            ph_tier=ph_tier,
            word_tier=word_tier,
            custom_map=None,
            forced_format=None,
            ja_mapping_words_fallback_enabled=True,
            ja_mapping_spn_ratio_threshold=0.35,
            ja_mapping_min_vowel_phone_ratio=0.5,
            ja_mapping_confidence_threshold=None,
            debug_reason_logging=True,
            log_fn=lambda msg: None,
            parse_filename_fn=lambda base: ["ka"],
            is_vowel_chain_fn=lambda syllables: False,
            extract_cv_targets_fn=lambda lines, custom_map=None: ["ka"],
            detect_alias_format_fn=lambda aliases, custom_map=None: "cvvc",
            get_profile_fn=lambda fmt: {"preset": fmt},
            collect_phone_quality_fn=lambda *args, **kwargs: {"low_confidence_reasons": ["insufficient_phones"], "spn_ratio_in_phone_tier": 0.5},
            build_words_synth_phones_fn=lambda words, syllables: [],
            resolve_conf_threshold_fn=lambda fmt, override_threshold=None: 0.64,
        )
        self.assertEqual(result.status, "empty_intervals")
        self.assertIn("phone_quality", result.meta)

    def test_prepare_ja_loop_state_marks_low_trust_cvvc_for_filename_sequence(self):
        ph_tier = [
            SimpleNamespace(mark="k", minTime=0.0, maxTime=0.05),
            SimpleNamespace(mark="a", minTime=0.05, maxTime=0.10),
            SimpleNamespace(mark="spn", minTime=0.10, maxTime=0.12),
        ]
        word_tier = [SimpleNamespace(mark="か", minTime=0.0, maxTime=0.12)]
        result = prepare_ja_loop_state(
            fname="a.wav",
            lines=["a.wav=ka,0,0,0,0,0", "a.wav=ki,0,0,0,0,0"],
            real_wav_name="a.wav",
            ph_tier=ph_tier,
            word_tier=word_tier,
            custom_map=None,
            forced_format="cvvc",
            ja_mapping_words_fallback_enabled=True,
            ja_mapping_spn_ratio_threshold=0.35,
            ja_mapping_min_vowel_phone_ratio=0.5,
            ja_mapping_confidence_threshold=None,
            debug_reason_logging=True,
            log_fn=lambda msg: None,
            parse_filename_fn=lambda base: ["ka", "ki"],
            is_vowel_chain_fn=lambda syllables: False,
            extract_cv_targets_fn=lambda lines, custom_map=None: ["ka", "ki"],
            detect_alias_format_fn=lambda aliases, custom_map=None: "cv",
            get_profile_fn=lambda fmt: {"preset": fmt},
            collect_phone_quality_fn=lambda *args, **kwargs: {
                "low_confidence_reasons": ["insufficient_vowel_phones"],
                "spn_ratio_in_phone_tier": 0.40,
                "phones_vs_expected_syllables_ratio": 0.50,
                "known_vowel_phone_count": 1.0,
                "expected_syllables": 2,
            },
            build_words_synth_phones_fn=lambda words, syllables: [],
            resolve_conf_threshold_fn=lambda fmt, override_threshold=None: 0.64,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.format_type, "cvvc")
        self.assertIn(result.textgrid_trust_tier, {"mid", "low"})
        self.assertTrue(result.prefer_filename_sequence)


if __name__ == "__main__":
    unittest.main()
