import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_features import get_delta_clip_limits
from core.oto_ml_refiner import _apply_korean_cv_destination_guard, _apply_korean_delta_policy


def _validate(offset, consonant, cutoff, pre, ovl, alias_type=""):
    offset = max(float(offset), 0.0)
    pre = max(float(pre), 0.0)
    ovl = max(0.0, min(float(ovl), max(pre - 1.0, 0.0)))
    min_cons_gap = 10.0 if str(alias_type or "").strip().lower() == "vc" else 14.0
    consonant = max(float(consonant), pre + min_cons_gap)
    cutoff_abs = max(abs(float(cutoff)), consonant + 8.0)
    return offset, consonant, -cutoff_abs, pre, ovl


class OtoMlCvGuardTests(unittest.TestCase):
    def test_korean_cv_clip_limits_are_tightened(self):
        cv_limits = get_delta_clip_limits("korean", alias_type="cv")
        cv_head_limits = get_delta_clip_limits("korean", alias_type="cv_head")

        self.assertEqual(cv_limits["delta_offset"], [-36.0, 40.0])
        self.assertEqual(cv_limits["delta_pre"], [-28.0, 34.0])
        self.assertEqual(cv_head_limits["delta_offset"], [-42.0, 48.0])
        self.assertEqual(cv_head_limits["delta_pre"], [-32.0, 38.0])

    def test_destination_guard_reverts_blankward_cv_shift(self):
        row_context = {
            "format_type": "cvvc",
            "alias_type": "cv",
            "base_offset": 108.0,
            "base_cons": 162.0,
            "base_cutoff_abs": 248.0,
            "base_pre": 54.0,
            "base_ovl": 26.0,
            "curr_phone_start_ms": 106.0,
            "curr_phone_len_ms": 72.0,
            "curr_vowel_start_ms": 160.0,
            "expected_anchor_ms": 160.0,
            "mapping_confidence": 0.86,
            "blank_span_confidence": 0.66,
            "syllable_blank_confidence": 0.61,
            "syllable_mel_silence_conf": 0.64,
            "syllable_mel_voiced_conf": 0.12,
            "db_silence_ratio": 0.69,
            "mel_window_silence_ratio": 0.67,
            "mel_offset_candidate_ms": 110.0,
        }
        base_params = _validate(
            row_context["base_offset"],
            row_context["base_cons"],
            -(row_context["base_cutoff_abs"] - row_context["base_offset"]),
            row_context["base_pre"],
            row_context["base_ovl"],
            alias_type="cv",
        )

        # Predicted params move the CV earlier into blank-ish space.
        pred_params = (92.0, 150.0, -120.0, 34.0, 12.0)
        guarded = _apply_korean_cv_destination_guard(row_context, pred_params, _validate)

        self.assertEqual(guarded[0], base_params[0])
        self.assertEqual(guarded[3], base_params[3])

    def test_destination_guard_locks_cv_location_but_keeps_tail_adjustment(self):
        row_context = {
            "format_type": "cvc",
            "alias_type": "cv",
            "base_offset": 120.0,
            "base_cons": 174.0,
            "base_cutoff_abs": 254.0,
            "base_pre": 58.0,
            "base_ovl": 29.0,
            "curr_phone_start_ms": 118.0,
            "curr_phone_len_ms": 70.0,
            "curr_vowel_start_ms": 176.0,
            "expected_anchor_ms": 176.0,
            "mapping_confidence": 0.93,
            "blank_span_confidence": 0.18,
            "syllable_blank_confidence": 0.14,
            "syllable_mel_silence_conf": 0.16,
            "syllable_mel_voiced_conf": 0.72,
            "db_silence_ratio": 0.21,
            "mel_window_silence_ratio": 0.24,
            "mel_offset_candidate_ms": 122.0,
        }

        pred_params = (124.0, 190.0, -150.0, 56.0, 27.0)
        guarded = _apply_korean_cv_destination_guard(row_context, pred_params, _validate)
        base_params = _validate(
            row_context["base_offset"],
            row_context["base_cons"],
            -(row_context["base_cutoff_abs"] - row_context["base_offset"]),
            row_context["base_pre"],
            row_context["base_ovl"],
            alias_type="cv",
        )

        self.assertEqual(guarded[0], base_params[0])
        self.assertEqual(guarded[3], base_params[3])
        self.assertGreaterEqual(guarded[1], base_params[1])

    def test_cvvc_cv_policy_locks_offset_and_pre_deltas(self):
        row_context = {
            "format_type": "cvvc",
            "alias_type": "cv",
            "mapping_confidence": 0.95,
            "blank_span_confidence": 0.10,
            "syllable_blank_confidence": 0.08,
            "syllable_mel_silence_conf": 0.07,
            "syllable_mel_voiced_conf": 0.80,
            "db_silence_ratio": 0.15,
            "mel_window_silence_ratio": 0.18,
        }
        deltas = _apply_korean_delta_policy(
            row_context,
            {
                "delta_offset": -22.0,
                "delta_pre": -18.0,
                "delta_cons": 14.0,
                "delta_cutoff": 20.0,
                "delta_ovl": 5.0,
            },
        )

        self.assertEqual(deltas["delta_offset"], 0.0)
        self.assertEqual(deltas["delta_pre"], 0.0)


if __name__ == "__main__":
    unittest.main()
