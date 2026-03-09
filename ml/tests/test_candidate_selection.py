import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_candidate_selection_v2 import select_kr_syllable_source
from core.oto_candidate_selection import maybe_promote_alias_candidate, select_primary_mapping_candidate


class CandidateSelectionTests(unittest.TestCase):
    def test_select_primary_mapping_candidate_prefers_filename_lock_for_forced_words(self):
        candidates = [
            {"name": "alias_phone_fallback", "objective": 80.0},
            {"name": "filename_token", "objective": 70.0},
        ]
        selected, reason, by_name = select_primary_mapping_candidate(
            candidates,
            format_type="cvvc",
            forced_words_mapping=True,
        )
        self.assertEqual(reason, "filename_words_lock")
        self.assertIs(selected, by_name["filename_token"])

    def test_maybe_promote_alias_candidate_requires_gain(self):
        selected = {"name": "filename_token", "objective": 75.0, "score": 74.0, "mean_syll_conf": 0.70, "lock_order": False}
        alias = {"name": "alias_phone_fallback", "objective": 84.0, "score": 80.0, "mean_syll_conf": 0.80}
        promoted, changed = maybe_promote_alias_candidate(
            selected_candidate=selected,
            alias_candidate=alias,
            provisional_conf=0.8,
            conf_threshold=0.6,
            format_type="vcv",
        )
        self.assertTrue(changed)
        self.assertIs(promoted, alias)

    def test_select_kr_syllable_source_forces_alias_on_order_locked_mismatch(self):
        words = [{"phones": [1], "roman_cv": "go"}]
        alias = [{"phones": [1], "roman_cv": "go"}, {"phones": [1], "roman_cv": "gyo"}]
        picked = select_kr_syllable_source(
            file_format="cvvc",
            prefer_filename_sequence=False,
            low_phone_quality=False,
            used_words_based=True,
            words_syllables=words,
            alias_syllables=alias,
            targets_for_build=["go", "gyo"],
            score_mapping_fn=lambda infos, targets: 80.0 if len(infos) == 2 else 50.0,
            should_prefer_alias_fn=lambda *args, **kwargs: False,
            compute_glide_mismatch_fn=lambda infos, targets: 0.0,
            is_order_locked_format_fn=lambda fmt: fmt == "cvvc",
        )
        self.assertEqual(picked["mapping_reason_code"], "order_locked_length_mismatch")
        self.assertTrue(picked["used_alias_based"])

    def test_select_kr_syllable_source_keeps_words_on_low_quality_flag(self):
        words = [{"phones": [1], "roman_cv": "go"}]
        alias = [{"phones": [1], "roman_cv": "go"}]
        picked = select_kr_syllable_source(
            file_format="vcv",
            prefer_filename_sequence=False,
            low_phone_quality=True,
            used_words_based=True,
            words_syllables=words,
            alias_syllables=alias,
            targets_for_build=["go"],
            score_mapping_fn=lambda infos, targets: 70.0,
            should_prefer_alias_fn=lambda *args, **kwargs: False,
            compute_glide_mismatch_fn=lambda infos, targets: 0.0,
            is_order_locked_format_fn=lambda fmt: False,
        )
        self.assertEqual(picked["mapping_reason_code"], "words_low_phone_quality")
        self.assertTrue(picked["used_words_based"])


if __name__ == "__main__":
    unittest.main()
