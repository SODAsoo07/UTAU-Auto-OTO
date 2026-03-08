import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.generator_finish import (
    GeneratorFinishContext,
    cleanup_timing_anchor_jsonl_files,
    finalize_generator_finish,
    write_jsonl_records,
    write_oto_lines,
)


class GeneratorFinishTests(unittest.TestCase):
    def test_write_oto_lines_writes_utf8_text(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "oto.ini")
            write_oto_lines(out_path, ["a.wav=a,0,0,0,0,0", "b.wav=ba,1,2,3,4,5"])
            with open(out_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read().splitlines(), ["a.wav=a,0,0,0,0,0", "b.wav=ba,1,2,3,4,5"])

    def test_cleanup_timing_anchor_jsonl_files_removes_matching_files(self):
        with tempfile.TemporaryDirectory() as td:
            keep_path = os.path.join(td, "keep.jsonl")
            remove_path = os.path.join(td, "timing_anchor_kr_test.jsonl")
            open(keep_path, "w", encoding="utf-8").close()
            open(remove_path, "w", encoding="utf-8").close()
            removed, failed = cleanup_timing_anchor_jsonl_files(td, "timing_anchor_kr_")
            self.assertEqual((removed, failed), (1, 0))
            self.assertTrue(os.path.exists(keep_path))
            self.assertFalse(os.path.exists(remove_path))

    def test_finalize_generator_finish_logs_summary_and_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            trace_path = os.path.join(td, "timing_anchor_ja_sample.jsonl")
            open(trace_path, "w", encoding="utf-8").close()
            logs = []
            ctx = GeneratorFinishContext(
                log_fn=logs.append,
                anchor_stats={
                    "anchor_locked_count": 2,
                    "cutoff_clamped_count": 1,
                    "vc_cutoff_leak_guard_count": 0,
                },
                anchor_log_path=os.path.join(td, "anchor.log"),
                anchor_log_dir=td,
                cleanup_timing_jsonl=True,
                timing_jsonl_prefix="timing_anchor_ja_",
            )
            finalize_generator_finish(ctx)
            self.assertTrue(any("anchor_locked_count=2" in line for line in logs))
            self.assertTrue(any("timing jsonl cleanup: 1 files" in line for line in logs))

    def test_write_jsonl_records_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "nested", "trace.jsonl")
            count = write_jsonl_records(out_path, [{"x": 1}, {"x": 2}])
            self.assertEqual(count, 2)
            with open(out_path, "r", encoding="utf-8") as f:
                self.assertEqual(len(f.read().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
