import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_row_finalize_v2 import finalize_kr_row


class KrRowFinalizeV2Tests(unittest.TestCase):
    def test_finalize_kr_row_logs_post_events_and_commits_rows(self):
        lines = []
        logs = []
        anchor_store = {}

        def _row_builder(*args, **kwargs):
            return ["a.wav=ka,1,2,3,4,5"]

        def _log_post(log_fn, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced):
            log_fn(f"post:{fname}:{alias}:{soft_off_shift}:{soft_cut_shift}:{cutoff_reduced}")

        finalize_kr_row(
            final_lines=lines,
            row_builder_fn=_row_builder,
            real_wav_name="a.wav",
            alias="ka",
            offset=1.0,
            consonant=2.0,
            cutoff=-3.0,
            pre=4.0,
            ovl=5.0,
            generate_openutau=False,
            alias_suffix="",
            alias_type="cv",
            wav_duration_ms=100.0,
            validate_fn=lambda *args: args,
            log_post_timing_events_fn=_log_post,
            log_fn=logs.append,
            fname="a.wav",
            soft_off_shift=1.0,
            soft_cut_shift=2.0,
            cutoff_reduced=3.0,
            anchor_store=anchor_store,
            anchor_record=(0, {"ok": True}),
            messages=["m1"],
        )
        self.assertEqual(lines, ["a.wav=ka,1,2,3,4,5"])
        self.assertEqual(anchor_store[0], {"ok": True})
        self.assertEqual(logs[0], "post:a.wav:ka:1.0:2.0:3.0")
        self.assertEqual(logs[1], "m1")


if __name__ == "__main__":
    unittest.main()
