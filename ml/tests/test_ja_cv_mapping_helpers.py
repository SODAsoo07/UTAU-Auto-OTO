import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_oto_generator import (
    _build_ja_mapping_trace_record,
    _find_ja_cv_vowel_match_index,
    _should_allow_ja_soft_forward_shift,
)
from core.ja_oto_mapping import (
    _ja_soft_cv_match_level,
    _select_ja_cv_syllable_index,
)


class JaCvMappingHelpersTests(unittest.TestCase):
    def test_soft_cv_match_level_distinguishes_inserted_vowel(self):
        self.assertGreaterEqual(_ja_soft_cv_match_level("kya", "kiya"), 2)
        self.assertLessEqual(_ja_soft_cv_match_level("ka", "kya"), 1)

    def test_should_allow_soft_forward_shift_only_when_mapped_is_better(self):
        self.assertTrue(_should_allow_ja_soft_forward_shift("kya", "ka", "kiya"))
        self.assertFalse(_should_allow_ja_soft_forward_shift("ka", "ka", "kya"))

    def test_find_cv_vowel_match_index_prefers_inserted_vowel_candidate(self):
        syllables_info = [
            {"roman": "ka", "phones": [1]},
            {"roman": "kiya", "phones": [1]},
        ]
        self.assertEqual(
            _find_ja_cv_vowel_match_index("kya", 0, syllables_info, search_back=0, search_fwd=1),
            1,
        )

    def test_select_cv_syllable_index_allows_one_step_soft_forward_match(self):
        syllables_info = [
            {"roman": "ka", "phones": [1]},
            {"roman": "kiya", "phones": [1]},
        ]
        self.assertEqual(_select_ja_cv_syllable_index("kya", 0, syllables_info, alias_type="cv"), 1)
        self.assertEqual(_select_ja_cv_syllable_index("- kya", 0, syllables_info, alias_type="cv_head"), 1)

    def test_build_mapping_trace_record_captures_match_levels(self):
        row = _build_ja_mapping_trace_record(
            fname="a.wav",
            alias="- kya",
            alias_type="cv_head",
            format_type="cvvc",
            target_tok="kya",
            expected_idx=0,
            mapped_idx=1,
            expected_tok="ka",
            mapped_tok="kiya",
            mapping_tier="mid",
            mapping_reason_code="filename_low_conf",
            mapping_confidence=0.64,
            filename_order_locked=True,
            local_conf=0.42,
        )
        self.assertEqual(row["delta"], 1)
        self.assertEqual(row["expected_match_level"], 1)
        self.assertGreaterEqual(row["mapped_match_level"], 2)


if __name__ == "__main__":
    unittest.main()
