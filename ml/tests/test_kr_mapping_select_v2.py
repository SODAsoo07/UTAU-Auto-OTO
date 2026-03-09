import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_mapping_select_v2 import (
    resolve_kr_cv_head_forced_index,
    select_kr_vcv_index,
)


class KrMappingSelectV2Tests(unittest.TestCase):
    def test_select_kr_vcv_index_prefers_planned_index(self):
        idx, seq_idx, conf = select_kr_vcv_index(
            target_clean="ga",
            cv_seq_idx=0,
            current_w_idx=0,
            romaji_syllables=["ga", "na"],
            syllables_info=[{}, {}],
            file_format="vcv",
            row_mapping_confidence=0.9,
            row_jump_default=1,
            row_jump_high_conf=2,
            file_mapping_conf_th=0.5,
            kr_planned_cv_indices=[1],
            resolve_planned_cv_index_fn=lambda planned, expected, target, syllables, alias_type="vcv": 1,
            resolve_cv_syllable_index_fn=lambda *args, **kwargs: (0, 1, {"jump_blocked": 0}),
            clamp_cv_index_to_order_fn=lambda *args: args[-1],
            split_syllable_parts_fn=lambda token: ("g", token[-1], ""),
            find_cv_vowel_match_index_fn=lambda *args, **kwargs: None,
            cv_match_score_fn=lambda a, b: 100.0 if a == b else 20.0,
            apply_row_confidence_penalty_fn=lambda conf, penalty: conf - penalty,
            log_fn=lambda msg: None,
            debug_logging=False,
            fname="a.wav",
            alias="ga",
        )
        self.assertEqual((idx, seq_idx), (1, 2))
        self.assertEqual(conf, 0.9)

    def test_resolve_kr_cv_head_forced_index_remaps_out_of_range(self):
        idx = resolve_kr_cv_head_forced_index(
            alias="- ga",
            alias_type="cv_head",
            cv_seq_idx=0,
            target_clean="ga",
            romaji_syllables=["ga"],
            syllables_info=[{}],
            kr_planned_cv_indices=[],
            kr_cvvc_occurrence_map={},
            kr_cvvc_occurrence_state={},
            resolve_planned_cv_index_fn=lambda *args, **kwargs: None,
            resolve_cvvc_occurrence_index_fn=lambda *args, **kwargs: 3,
            remap_forced_cv_index_fn=lambda target, syllables, expected: 0,
            log_fn=lambda msg: None,
            debug_logging=False,
            fname="a.wav",
        )
        self.assertEqual(idx, 0)


if __name__ == "__main__":
    unittest.main()
