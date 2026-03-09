import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_vcv_row_v2 import run_kr_vcv_row


class KrVcvRowV2Tests(unittest.TestCase):
    def test_run_kr_vcv_row_executes_timing_anchor_and_finalize(self):
        calls = {"finalize": 0}
        current_w_idx, cv_seq_idx = run_kr_vcv_row(
            syllables_info=[{"phones": []}],
            current_w_idx=0,
            cv_seq_idx=0,
            forced_w_idx=0,
            diphthong_cv_consonant_ratio=0.6,
            alias="ga",
            file_format="vcv",
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
            kr_post_ctx=object(),
            fname="a.wav",
            log_fn=lambda msg: None,
            validate_fn=lambda *args: args,
            prepare_vcv_syllable_timing_fn=lambda *args, **kwargs: (0, 1, 10.0, 50.0, -90.0, 30.0, 20.0),
            apply_post_timing_pipeline_fn=lambda *args, **kwargs: (10.0, 50.0, -90.0, 30.0, 20.0, 1.0, 2.0, 3.0),
            extract_vcv_anchor_points_fn=lambda syllables, idx: (120.0, 120.0, 220.0),
            apply_anchor_lock_fn=lambda **kwargs: (11.0, 51.0, -91.0, 31.0, 21.0),
            finalize_row_fn=lambda **kwargs: calls.update({"finalize": calls["finalize"] + 1}),
            row_builder_fn=lambda *args, **kwargs: [],
            log_post_timing_events_fn=lambda *args, **kwargs: None,
        )
        self.assertEqual(current_w_idx, 0)
        self.assertEqual(cv_seq_idx, 1)
        self.assertEqual(calls["finalize"], 1)


if __name__ == "__main__":
    unittest.main()
