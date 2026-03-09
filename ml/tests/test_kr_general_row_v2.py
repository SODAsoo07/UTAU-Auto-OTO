import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_general_row_v2 import run_kr_general_row


class KrGeneralRowV2Tests(unittest.TestCase):
    def test_run_kr_general_row_executes_finalize(self):
        calls = {"finalize": 0}
        run_kr_general_row(
            final_lines=[],
            real_wav_name="a.wav",
            alias="ga",
            alias_type="cv",
            file_format="cvvc",
            offset=10.0,
            consonant=50.0,
            cutoff=-90.0,
            pre=30.0,
            ovl=20.0,
            soft_off_shift=1.0,
            soft_cut_shift=2.0,
            cutoff_reduced=3.0,
            selected_w_idx=0,
            current_w_idx=0,
            c_end=120.0,
            n_start=130.0,
            n_end=240.0,
            c_start=100.0,
            row_mapping_confidence=0.9,
            timeline_start_ms=0.0,
            timeline_end_ms=1000.0,
            wav_duration_ms=1000.0,
            generate_openutau=False,
            alias_suffix="",
            fname="a.wav",
            log_fn=lambda msg: None,
            validate_fn=lambda *args: args,
            bridge_pair={},
            realized_cv_anchor_by_idx={},
            cv_anchor_by_idx={},
            refine_bridge_fn=lambda *args, **kwargs: args[:5],
            resolve_anchor_targets_fn=lambda alias_type, c_end, n_start, n_end: (c_end, n_start, n_end),
            apply_anchor_lock_fn=lambda **kwargs: (11.0, 51.0, -91.0, 31.0, 21.0),
            build_anchor_record_fn=lambda idx, **kwargs: (idx, {"ok": True}),
            build_bridge_message_fn=lambda *args, **kwargs: None,
            finalize_row_fn=lambda **kwargs: calls.update({"finalize": calls["finalize"] + 1}),
            row_builder_fn=lambda *args, **kwargs: [],
            log_post_timing_events_fn=lambda *args, **kwargs: None,
        )
        self.assertEqual(calls["finalize"], 1)


if __name__ == "__main__":
    unittest.main()
