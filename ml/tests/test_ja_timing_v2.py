import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_timing_v2 import (
    build_realized_cv_anchor,
    compute_ja_vc_from_adjacent_cv,
    compute_vcv_params_from_virtual_split,
    enforce_cv_pre_anchor_guard,
    enforce_vcv_cv_entry_guard,
    estimate_ja_cv_anchor,
)


def _validate(offset, consonant, cutoff, pre, ovl):
    pre = max(float(pre), 0.0)
    ovl = max(0.0, min(float(ovl), pre))
    consonant = max(float(consonant), pre + 1.0)
    cutoff_abs = max(abs(float(cutoff)), consonant + 1.0)
    return float(offset), float(consonant), -float(cutoff_abs), float(pre), float(ovl)


class JaTimingV2Tests(unittest.TestCase):
    def test_build_realized_cv_anchor_returns_expected_shape(self):
        anchor = build_realized_cv_anchor(
            100.0,
            180.0,
            -260.0,
            80.0,
            50.0,
            onset_abs=120.0,
            vowel_start_abs=150.0,
            c_end_abs=148.0,
            vowel_end_abs=280.0,
        )
        self.assertEqual(anchor["pre_abs"], 180.0)
        self.assertEqual(anchor["cons_abs"], 280.0)
        self.assertGreater(anchor["vowel_len"], 100.0)
        self.assertGreater(anchor["cons_gap"], 10.0)
        self.assertGreater(anchor["cut_gap"], 16.0)

    def test_compute_ja_vc_from_adjacent_cv_returns_valid_tuple(self):
        prev_cv = {
            "pre": 80.0,
            "cons_gap": 60.0,
            "vowel_len": 90.0,
            "vowel_end_abs": 420.0,
        }
        next_cv = {
            "pre": 84.0,
            "pre_abs": 520.0,
            "cons_abs": 590.0,
            "onset_abs": 500.0,
            "cons_gap": 70.0,
        }
        result = compute_ja_vc_from_adjacent_cv(
            prev_cv,
            next_cv,
            alias_type="vc",
            c_char="k",
            bridge_profile={},
            plosive_consonants={"k", "t"},
            validate_fn=_validate,
        )
        self.assertIsNotNone(result)
        offset, consonant, cutoff, pre, ovl = result
        self.assertGreater(pre, 0.0)
        self.assertLessEqual(ovl, pre)
        self.assertGreater(consonant, pre)
        self.assertLess(cutoff, 0.0)

    def test_compute_vcv_params_from_virtual_split_preserves_param_order(self):
        result = compute_vcv_params_from_virtual_split(
            "a ka",
            100.0,
            220.0,
            260.0,
            420.0,
            base_shape={"offset": 8.0, "consonant": 18.0, "cutoff": 24.0, "preutterance": 10.0, "overlap": 4.0},
            extract_target_syllable_fn=lambda alias: alias.split()[-1],
            split_syllable_fn=lambda tok: (tok[:-1], tok[-1:]),
            get_bridge_profile_fn=lambda onset: {"cut_to_next_allow": 28.0},
            is_n_bridge_fn=lambda alias, mode: False,
            onset_class_fn=lambda onset: "plain",
            validate_fn=_validate,
            apply_base_shape_blend_fn=lambda offset, consonant, cutoff, pre, ovl, base_shape, alias_type="vcv": (
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
            ),
        )
        offset, consonant, cutoff, pre, ovl = result
        self.assertGreaterEqual(offset, 0.0)
        self.assertGreater(consonant, pre)
        self.assertLess(cutoff, 0.0)
        self.assertLessEqual(ovl, pre)

    def test_enforce_cv_pre_anchor_guard_clamps_pre_abs_near_c_end(self):
        offset, consonant, cutoff, pre, ovl = enforce_cv_pre_anchor_guard(
            100.0,
            170.0,
            -240.0,
            80.0,
            70.0,
            c_end_abs=150.0,
            alias_type="cv",
            validate_fn=_validate,
        )
        self.assertLessEqual(abs((offset + pre) - 150.0), 7.0)
        self.assertLessEqual(ovl, pre)

    def test_enforce_vcv_cv_entry_guard_preserves_basic_order(self):
        result = enforce_vcv_cv_entry_guard(
            120.0,
            220.0,
            -320.0,
            90.0,
            40.0,
            c_boundary=230.0,
            n_end=360.0,
            alias_text="a a",
            is_n_bridge_fn=lambda alias, mode: False,
            extract_target_syllable_fn=lambda alias: "a",
            split_syllable_fn=lambda tok: ("", "a"),
            validate_fn=_validate,
        )
        offset, consonant, cutoff, pre, ovl = result
        self.assertGreaterEqual(offset + pre, 226.0)
        self.assertGreater(consonant, pre)
        self.assertLessEqual(ovl, pre)
        self.assertLess(cutoff, 0.0)

    def test_estimate_ja_cv_anchor_builds_anchor_record(self):
        syllables = [
            {
                "roman_cv": "ka",
                "phones": [SimpleNamespace(mark="k"), SimpleNamespace(mark="a")],
            }
        ]
        anchor = estimate_ja_cv_anchor(
            0,
            syllables,
            extract_cv_bounds_fn=lambda phones, alias_text="", alias_type="cv": (120.0, 150.0, 155.0, 300.0),
            cv_offset_and_pre_fn=lambda c_start, c_end, alias, **kwargs: (90.0, 60.0),
            adaptive_overlap_fn=lambda pre, c_hint, mode="cv": 35.0,
            validate_fn=_validate,
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["onset_abs"], 120.0)
        self.assertEqual(anchor["c_end_abs"], 150.0)
        self.assertEqual(anchor["vowel_start_abs"], 155.0)
        self.assertEqual(anchor["vowel_end_abs"], 300.0)
        self.assertGreater(anchor["cons"], anchor["pre"])


if __name__ == "__main__":
    unittest.main()
