import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_oto_mapping import _compute_kr_glide_mismatch_ratio
from core.oto_generator import (
    _build_kr_planned_cv_indices,
    _is_kr_cv_syllable_active,
    _remap_kr_forced_cv_index,
)
from core.kr_mapping_v2 import build_kr_cv_anchor_plan


class KrCvvcMappingGuardTests(unittest.TestCase):
    @staticmethod
    def _p(mark, start, end):
        return type("P", (), {"mark": mark, "minTime": float(start), "maxTime": float(end)})()

    def test_compute_kr_glide_mismatch_ratio_detects_cv_to_diphthong_drift(self):
        words = [
            {"roman_cv": "go"},
            {"roman_cv": "gyo"},
            {"roman_cv": "go"},
        ]
        targets = ["go", "go", "go"]

        ratio = _compute_kr_glide_mismatch_ratio(words, targets)
        self.assertAlmostEqual(ratio, 1.0 / 3.0, places=6)

    def test_remap_forced_cv_index_falls_back_to_exact_token_when_out_of_range(self):
        romaji = ["go", "go", "gyo", "gyo"]
        # forced occurrence index(예: filename 기준 6번째)이 words 기반 음절 길이를 넘어간 상황
        remapped = _remap_kr_forced_cv_index("go", romaji, expected_idx=2)
        self.assertEqual(remapped, 1)

    def test_remap_forced_cv_index_returns_none_when_token_absent(self):
        romaji = ["gyo", "gyo"]
        remapped = _remap_kr_forced_cv_index("go", romaji, expected_idx=0)
        self.assertIsNone(remapped)

    def test_is_kr_cv_syllable_active_requires_vowel(self):
        syl = {
            "roman_cv": "go",
            "phones": [
                self._p("g", 0.00, 0.03),
                self._p("cl", 0.03, 0.05),
            ],
        }
        self.assertFalse(_is_kr_cv_syllable_active(syl, require_vowel=True))

    def test_build_kr_planned_cv_indices_prefers_active_exact_sequence(self):
        syllables_info = [
            {"roman_cv": "go", "phones": [self._p("sil", 0.00, 0.05)]},
            {"roman_cv": "go", "phones": [self._p("g", 0.05, 0.08), self._p("o", 0.08, 0.16)]},
            {"roman_cv": "gyo", "phones": [self._p("gy", 0.16, 0.20), self._p("o", 0.20, 0.29)]},
        ]
        plan = _build_kr_planned_cv_indices(["go", "gyo"], syllables_info)
        self.assertEqual(plan, [1, 2])

    def test_build_kr_cv_anchor_plan_can_use_mel_scores(self):
        syllables_info = [
            {
                "roman_cv": "ga",
                "phones": [self._p("g", 0.00, 0.03), self._p("a", 0.03, 0.12)],
                "mel_voiced_formant_conf": 0.05,
                "mel_silence_sparse_conf": 0.50,
                "blank_confidence": 0.45,
            },
            {
                "roman_cv": "ga",
                "phones": [self._p("g", 0.12, 0.15), self._p("a", 0.15, 0.28)],
                "mel_voiced_formant_conf": 0.95,
                "mel_silence_sparse_conf": 0.05,
                "blank_confidence": 0.05,
            },
        ]
        plan = build_kr_cv_anchor_plan(["ga"], syllables_info, use_mel=True)
        self.assertEqual(plan.get("indices"), [1])


if __name__ == "__main__":
    unittest.main()
