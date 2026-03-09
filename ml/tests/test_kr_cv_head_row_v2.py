import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_cv_head_row_v2 import run_kr_cv_head_row


class KrCvHeadRowV2Tests(unittest.TestCase):
    def test_run_kr_cv_head_row_executes_finalize(self):
        calls = {"finalize": 0}
        current_w_idx, cv_seq_idx = run_kr_cv_head_row(
            syllables_info=[{"phones": []}],
            current_w_idx=0,
            cv_seq_idx=0,
            alias="- ga",
            forced_w_idx=0,
            file_format="cvvc",
            real_wav_name="a.wav",
            final_lines=[],
            generate_openutau=False,
            alias_suffix="",
            wav_duration_ms=1000.0,
            timeline_start_ms=0.0,
            timeline_end_ms=1000.0,
            row_mapping_confidence=0.9,
            mel_ctx_for_file=None,
            base_shape={},
            ph_intervals=[],
            fname="a.wav",
            log_fn=lambda msg: None,
            validate_fn=lambda *args: args,
            prepare_cv_head_syllable_timing_fn=lambda *args, **kwargs: (0, 1, 10.0, 50.0, -90.0, 30.0, 20.0, 120.0, 240.0),
            apply_post_timing_pipeline_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0, 1.0, 2.0, 3.0),
            guard_cv_head_offset_to_current_onset_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 4.0),
            ensure_cv_head_min_vowel_coverage_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 5.0),
            guard_cv_cutoff_to_next_onset_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 6.0),
            prepare_cv_head_anchor_context_fn=lambda *args, **kwargs: {
                "anchor_abs": 130.0,
                "onset_abs": 100.0,
                "vowel_start_abs": 120.0,
                "vowel_end_abs": 240.0,
                "c_end_abs": 110.0,
            },
            prepare_cv_bounds_fn=lambda *args, **kwargs: None,
            apply_anchor_lock_fn=lambda **kwargs: (11.0, 51.0, -91.0, 31.0, 21.0),
            build_anchor_record_fn=lambda idx, **kwargs: (idx, {"ok": True}),
            finalize_row_fn=lambda **kwargs: calls.update({"finalize": calls["finalize"] + 1}),
            row_builder_fn=lambda *args, **kwargs: [],
            build_guard_messages_fn=lambda *args, **kwargs: ["m1"],
            log_post_timing_events_fn=lambda *args, **kwargs: None,
            anchor_store={},
        )
        self.assertEqual((current_w_idx, cv_seq_idx), (0, 1))
        self.assertEqual(calls["finalize"], 1)


if __name__ == "__main__":
    unittest.main()
