import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_mapping_select_v2 import resolve_ja_forced_cv_index, select_ja_vcv_mapping


class JaMappingSelectV2Tests(unittest.TestCase):
    def test_resolve_ja_forced_cv_index_prefers_planned_index(self):
        idx = resolve_ja_forced_cv_index(
            alias="ka",
            alias_type="cv",
            target_tok="ka",
            expected_seq_idx=0,
            expected_idx=0,
            syllables_info=[{}, {}],
            planned_indices=[1],
            occurrence_map={},
            occurrence_state={},
            resolve_planned_cv_index_fn=lambda planned, expected, target, syllables, alias_type="cv": 1,
            resolve_cvvc_occurrence_index_fn=lambda *args, **kwargs: None,
            remap_forced_cv_index_fn=lambda *args, **kwargs: None,
            log_fn=lambda msg: None,
            debug_logging=False,
            fname="a.wav",
        )
        self.assertEqual(idx, 1)

    def test_resolve_ja_forced_cv_index_remaps_out_of_range(self):
        idx = resolve_ja_forced_cv_index(
            alias="- ka",
            alias_type="cv_head",
            target_tok="ka",
            expected_seq_idx=0,
            expected_idx=0,
            syllables_info=[{}],
            planned_indices=[],
            occurrence_map={},
            occurrence_state={},
            resolve_planned_cv_index_fn=lambda *args, **kwargs: None,
            resolve_cvvc_occurrence_index_fn=lambda *args, **kwargs: 4,
            remap_forced_cv_index_fn=lambda *args, **kwargs: 0,
            log_fn=lambda msg: None,
            debug_logging=False,
            fname="a.wav",
        )
        self.assertEqual(idx, 0)

    def test_select_ja_vcv_mapping_blocks_backward_resync_when_expected_already_matches(self):
        traces = []
        result = select_ja_vcv_mapping(
            alias="a ka",
            fname="a.wav",
            format_type="vcv",
            stable_vcv_seq_idx=1,
            cv_seq_idx=0,
            syllables_info=[{"tok": "ka"}, {"tok": "ka"}, {"tok": "ki"}],
            ja_planned_cv_indices=[],
            last_vcv_mapped_idx=-1,
            mapping_tier="mid",
            mapping_reason_code="test",
            mapping_confidence_base=0.8,
            filename_order_locked=False,
            syllable_confidence_by_idx=[0.9, 0.8, 0.7],
            log_fn=lambda msg: None,
            debug_logging=False,
            resolve_planned_cv_index_fn=lambda *args, **kwargs: None,
            select_vcv_syllable_index_fn=lambda *args, **kwargs: 1,
            extract_target_syllable_fn=lambda alias: "ka",
            find_exact_target_index_fn=lambda *args, **kwargs: 0,
            normalize_syllable_token_fn=lambda tok: str(tok or "").strip(),
            syllable_info_token_fn=lambda syl: syl.get("tok", ""),
            split_syllable_fn=lambda tok: ("k", tok[-1:] if tok else ""),
            ja_vowels={"a", "i", "u", "e", "o"},
            find_vowel_match_index_fn=lambda *args, **kwargs: None,
            prefer_vcv_candidate_index_fn=lambda expected, mapped, *_args, **_kwargs: mapped,
            should_trace_mapping_decision_fn=lambda **kwargs: True,
            build_mapping_trace_record_fn=lambda **kwargs: kwargs,
            append_mapping_trace_fn=lambda rec: traces.append(rec),
            decide_cv_row_abstain_fn=lambda **kwargs: {"should_skip": False, "reason": ""},
            is_cv_syllable_active_fn=lambda *_args, **_kwargs: True,
        )
        self.assertEqual(result["expected_idx"], 1)
        self.assertEqual(result["mapped_idx"], 1)
        self.assertFalse(result["row_abstain"]["should_skip"])
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["mapped_idx"], 1)

    def test_select_ja_vcv_mapping_clamps_large_forward_jump_and_traces(self):
        traces = []
        result = select_ja_vcv_mapping(
            alias="a ko",
            fname="a.wav",
            format_type="vcv",
            stable_vcv_seq_idx=1,
            cv_seq_idx=0,
            syllables_info=[
                {"tok": "a"},
                {"tok": "ka"},
                {"tok": "ki"},
                {"tok": "ku"},
                {"tok": "ko"},
            ],
            ja_planned_cv_indices=[],
            last_vcv_mapped_idx=1,
            mapping_tier="low",
            mapping_reason_code="test",
            mapping_confidence_base=0.6,
            filename_order_locked=False,
            syllable_confidence_by_idx=[0.9, 0.8, 0.7, 0.6, 0.5],
            log_fn=lambda msg: None,
            debug_logging=False,
            resolve_planned_cv_index_fn=lambda *args, **kwargs: None,
            select_vcv_syllable_index_fn=lambda *args, **kwargs: 4,
            extract_target_syllable_fn=lambda alias: "ko",
            find_exact_target_index_fn=lambda *args, **kwargs: None,
            normalize_syllable_token_fn=lambda tok: str(tok or "").strip(),
            syllable_info_token_fn=lambda syl: syl.get("tok", ""),
            split_syllable_fn=lambda tok: ("k", tok[-1:] if tok else ""),
            ja_vowels={"a", "i", "u", "e", "o"},
            find_vowel_match_index_fn=lambda *args, **kwargs: None,
            prefer_vcv_candidate_index_fn=lambda expected, mapped, *_args, **_kwargs: mapped,
            should_trace_mapping_decision_fn=lambda **kwargs: True,
            build_mapping_trace_record_fn=lambda **kwargs: kwargs,
            append_mapping_trace_fn=lambda rec: traces.append(rec),
            decide_cv_row_abstain_fn=lambda **kwargs: {"should_skip": False, "reason": ""},
            is_cv_syllable_active_fn=lambda *_args, **_kwargs: True,
        )
        self.assertEqual(result["expected_idx"], 1)
        self.assertEqual(result["mapped_idx"], 1)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["mapped_idx"], 1)


if __name__ == "__main__":
    unittest.main()
