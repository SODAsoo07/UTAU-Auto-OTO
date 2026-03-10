import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_mapping_scoring_v2 import (
    clamp_ja_cv_index_to_order,
    find_ja_cv_vowel_match_index,
    find_ja_exact_target_index,
    prefer_vcv_candidate_index,
    should_allow_ja_soft_forward_shift,
)


class JaMappingScoringV2Tests(unittest.TestCase):
    def test_soft_forward_shift_requires_better_match(self):
        self.assertTrue(should_allow_ja_soft_forward_shift("kya", "ka", "kiya"))
        self.assertFalse(should_allow_ja_soft_forward_shift("ka", "ka", "kya"))
        self.assertFalse(should_allow_ja_soft_forward_shift("n", "na", "n"))

    def test_find_ja_cv_vowel_match_index_prefers_inserted_vowel_candidate(self):
        syllables_info = [
            {"roman": "ka", "phones": [1]},
            {"roman": "kiya", "phones": [1]},
        ]
        self.assertEqual(
            find_ja_cv_vowel_match_index("kya", 0, syllables_info, search_back=0, search_fwd=1),
            1,
        )

    def test_find_ja_exact_target_index_prefers_nearest_forward_match(self):
        syllables_info = [
            {"roman": "ka"},
            {"roman": "ki"},
            {"roman": "ka"},
        ]
        self.assertEqual(find_ja_exact_target_index("ka", 1, syllables_info, search_back=1, search_fwd=1), 2)

    def test_prefer_vcv_candidate_index_only_allows_small_exact_fix(self):
        syllables_info = [
            {"roman": "ka"},
            {"roman": "ki"},
            {"roman": "ku"},
        ]
        self.assertEqual(prefer_vcv_candidate_index(0, 1, "ki", syllables_info, max_delta=1), 1)
        self.assertEqual(prefer_vcv_candidate_index(0, 2, "ku", syllables_info, max_delta=1), 0)

    def test_clamp_ja_cv_index_to_order_blocks_large_forward_jump_in_cvvc(self):
        syllables_info = [
            {"roman": "ka"},
            {"roman": "ki"},
            {"roman": "ku"},
        ]
        self.assertEqual(
            clamp_ja_cv_index_to_order(
                "ku",
                0,
                2,
                syllables_info,
                format_type="cvvc",
                filename_order_locked=True,
                mapping_tier="mid",
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
