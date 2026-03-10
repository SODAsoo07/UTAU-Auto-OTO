import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_mapping_select_v2 import select_kr_general_cv_index, select_kr_vcv_index
from core.kr_oto_rules import _cv_match_score, _split_kr_syllable_parts


def _noop_log(_msg):
    return None


def _resolve_stub(*_args, **_kwargs):
    return 2, 3, {
        "jump_blocked": 0,
        "raw_chosen_idx": 2,
        "chosen_idx": 2,
        "best_score": 74.0,
        "expected_score": 66.0,
    }


def _resolve_vcv_stub(*_args, **_kwargs):
    return 2, 3, {
        "jump_blocked": 0,
        "raw_chosen_idx": 2,
        "chosen_idx": 2,
        "best_score": 74.0,
        "expected_score": 66.0,
    }


def _penalty(conf, amount):
    return float(conf) - float(amount)


class KrMappingSelectV2Tests(unittest.TestCase):
    def setUp(self):
        self.romaji = ["ma", "mya", "mwa"]
        self.blank = [0.70, 0.12, 0.80]
        self.syllables = [
            {
                "blank_confidence": 0.70,
                "mel_voiced_formant_conf": 0.10,
                "mel_silence_sparse_conf": 0.70,
                "mel_unvoiced_diffuse_conf": 0.20,
                "mel_breath_like_conf": 0.12,
            },
            {
                "blank_confidence": 0.12,
                "mel_voiced_formant_conf": 0.95,
                "mel_silence_sparse_conf": 0.08,
                "mel_unvoiced_diffuse_conf": 0.22,
                "mel_breath_like_conf": 0.06,
            },
            {
                "blank_confidence": 0.80,
                "mel_voiced_formant_conf": 0.05,
                "mel_silence_sparse_conf": 0.82,
                "mel_unvoiced_diffuse_conf": 0.18,
                "mel_breath_like_conf": 0.30,
            },
        ]

    def test_general_cv_selection_uses_mel_guided_remap_for_cvvc_low_conf(self):
        out = select_kr_general_cv_index(
            alias="mya",
            alias_type="cv",
            fname="x.wav",
            file_format="cvvc",
            target_clean="mya",
            current_w_idx=0,
            cv_seq_idx=0,
            row_mapping_confidence=0.42,
            row_jump_default=1,
            row_jump_high_conf=2,
            file_mapping_conf_th=0.55,
            file_mapping_low_conf=True,
            romaji_syllables=self.romaji,
            forced_vv_idx=None,
            planned_vv_idx=None,
            planned_cv_idx=None,
            forced_cvvc_idx=None,
            remap_forced_cv_index_fn=lambda *_a, **_k: None,
            cv_match_score_fn=_cv_match_score,
            split_syllable_parts_fn=_split_kr_syllable_parts,
            apply_row_confidence_penalty_fn=_penalty,
            resolve_cv_syllable_index_fn=_resolve_stub,
            should_allow_exact_vowel_fix_fn=lambda *_a, **_k: False,
            find_cv_vowel_match_index_fn=lambda *_a, **_k: None,
            clamp_cv_index_to_order_fn=lambda _fmt, _t, _r, _e, m: m,
            log_fn=_noop_log,
            debug_logging=False,
            syllable_blank_confidences=self.blank,
            syllables_info=self.syllables,
        )
        self.assertEqual(out["selected_w_idx"], 1)

    def test_vcv_selection_uses_mel_guided_remap_for_cvvc_low_conf(self):
        idx, _seq, _conf = select_kr_vcv_index(
            target_clean="mya",
            cv_seq_idx=0,
            current_w_idx=0,
            romaji_syllables=self.romaji,
            syllables_info=self.syllables,
            file_format="cvvc",
            row_mapping_confidence=0.45,
            row_jump_default=1,
            row_jump_high_conf=2,
            file_mapping_conf_th=0.55,
            kr_planned_cv_indices=None,
            resolve_planned_cv_index_fn=lambda *_a, **_k: None,
            resolve_cv_syllable_index_fn=_resolve_vcv_stub,
            clamp_cv_index_to_order_fn=lambda _fmt, _t, _r, _e, m: m,
            split_syllable_parts_fn=_split_kr_syllable_parts,
            find_cv_vowel_match_index_fn=lambda *_a, **_k: None,
            cv_match_score_fn=_cv_match_score,
            apply_row_confidence_penalty_fn=_penalty,
            log_fn=_noop_log,
            debug_logging=False,
            fname="x.wav",
            alias="a mya",
            syllable_blank_confidences=self.blank,
        )
        self.assertEqual(idx, 1)


if __name__ == "__main__":
    unittest.main()

