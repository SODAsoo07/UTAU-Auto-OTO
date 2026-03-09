import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_oto_generator import (
    _build_ja_planned_cv_indices,
    _build_ja_mapping_trace_record,
    _clamp_ja_cv_index_to_order,
    _compute_ja_youon_mismatch_ratio,
    _find_ja_cv_vowel_match_index,
    _is_ja_cv_syllable_active,
    _remap_ja_forced_cv_index,
    _remap_ja_cvvc_inactive_cv_index,
    _should_force_alias_for_ja_cvvc,
    _should_allow_ja_soft_forward_shift,
)
from core.ja_oto_mapping import (
    _ja_soft_cv_match_level,
    _select_ja_cv_syllable_index,
)


class JaCvMappingHelpersTests(unittest.TestCase):
    @staticmethod
    def _p(mark, start, end):
        return type("P", (), {"mark": mark, "minTime": float(start), "maxTime": float(end)})()

    def test_soft_cv_match_level_distinguishes_inserted_vowel(self):
        self.assertGreaterEqual(_ja_soft_cv_match_level("kya", "kiya"), 2)
        self.assertLessEqual(_ja_soft_cv_match_level("ka", "kya"), 1)

    def test_should_allow_soft_forward_shift_only_when_mapped_is_better(self):
        self.assertTrue(_should_allow_ja_soft_forward_shift("kya", "ka", "kiya"))
        self.assertFalse(_should_allow_ja_soft_forward_shift("ka", "ka", "kya"))
        self.assertFalse(_should_allow_ja_soft_forward_shift("n", "na", "n"))

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

    def test_cvvc_clamp_blocks_wrong_forward_shift(self):
        syllables_info = [
            {"roman": "ka", "phones": [1]},
            {"roman": "kya", "phones": [1]},
        ]
        self.assertEqual(
            _clamp_ja_cv_index_to_order(
                "ka",
                0,
                1,
                syllables_info,
                format_type="cvvc",
                filename_order_locked=False,
                mapping_tier="high",
            ),
            0,
        )

    def test_cvvc_clamp_keeps_inserted_vowel_forward_fix(self):
        syllables_info = [
            {"roman": "ka", "phones": [1]},
            {"roman": "kiya", "phones": [1]},
        ]
        self.assertEqual(
            _clamp_ja_cv_index_to_order(
                "kya",
                0,
                1,
                syllables_info,
                format_type="cvvc",
                filename_order_locked=True,
                mapping_tier="mid",
            ),
            1,
        )

    def test_compute_ja_youon_mismatch_ratio_detects_cv_to_youon_drift(self):
        candidate_infos = [
            {"roman": "kya", "phones": [1]},
            {"roman": "ka", "phones": [1]},
            {"roman": "kya", "phones": [1]},
        ]
        ratio = _compute_ja_youon_mismatch_ratio(candidate_infos, ["ka", "ka", "ka"])
        self.assertAlmostEqual(ratio, 2.0 / 3.0, places=6)

    def test_should_force_alias_for_ja_cvvc_when_words_candidate_is_unstable(self):
        selected = {
            "name": "words_tier",
            "use_filename": False,
            "order_preserving": False,
            "infos": [{"roman": "kya"}, {"roman": "kya"}, {"roman": "ka"}],
        }
        alias = {
            "name": "alias_phone_fallback",
            "use_alias": True,
            "infos": [{"roman": "ka"}, {"roman": "ka"}, {"roman": "ka"}],
        }
        force, reason, ratio = _should_force_alias_for_ja_cvvc(
            selected_candidate=selected,
            alias_candidate=alias,
            cv_targets=["ka", "ka", "ka"],
            low_phone_quality=False,
        )
        self.assertTrue(force)
        self.assertEqual(reason, "alias_cvvc_youon_mismatch")
        self.assertGreater(ratio, 0.3)

    def test_remap_ja_forced_cv_index_prefers_nearby_exact_match(self):
        syllables_info = [
            {"roman": "ka", "phones": [1]},
            {"roman": "kya", "phones": [1]},
            {"roman": "ka", "phones": [1]},
        ]
        self.assertEqual(_remap_ja_forced_cv_index("ka", 1, syllables_info), 2)

    def test_remap_ja_forced_cv_index_keeps_expected_when_it_is_already_exact(self):
        syllables_info = [
            {"roman": "ka", "phones": [1]},
            {"roman": "kya", "phones": [1]},
        ]
        self.assertEqual(_remap_ja_forced_cv_index("ka", 0, syllables_info), 0)

    def test_is_ja_cv_syllable_active_requires_vowel_for_cv(self):
        syl = {
            "roman": "ka",
            "phones": [
                self._p("k", 0.00, 0.03),
                self._p("cl", 0.03, 0.05),
            ],
        }
        self.assertFalse(_is_ja_cv_syllable_active(syl, require_vowel=True))

    def test_remap_ja_cvvc_inactive_cv_index_moves_to_active_neighbor(self):
        syllables_info = [
            {"roman": "ka", "phones": [self._p("sil", 0.00, 0.08)]},
            {"roman": "ka", "phones": [self._p("k", 0.08, 0.10), self._p("a", 0.10, 0.18)]},
            {"roman": "ki", "phones": [self._p("k", 0.18, 0.20), self._p("i", 0.20, 0.28)]},
        ]
        out = _remap_ja_cvvc_inactive_cv_index(
            "ka",
            0,
            0,
            syllables_info,
            alias_type="cv",
        )
        self.assertEqual(out, 1)

    def test_build_ja_planned_cv_indices_prefers_active_exact_sequence(self):
        syllables_info = [
            {"roman": "ka", "phones": [self._p("sil", 0.00, 0.06)]},
            {"roman": "ka", "phones": [self._p("k", 0.06, 0.09), self._p("a", 0.09, 0.16)]},
            {"roman": "kya", "phones": [self._p("ky", 0.16, 0.19), self._p("a", 0.19, 0.27)]},
        ]
        plan = _build_ja_planned_cv_indices(["ka", "kya"], syllables_info)
        self.assertEqual(plan, [1, 2])


if __name__ == "__main__":
    unittest.main()
