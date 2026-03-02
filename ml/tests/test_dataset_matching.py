import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_dataset import build_oto_ml_dataset


class DatasetMatchingTests(unittest.TestCase):
    def test_occurrence_matching_builds_single_row(self):
        with tempfile.TemporaryDirectory() as td:
            auto_path = os.path.join(td, "auto.ini")
            manual_path = os.path.join(td, "manual.ini")
            line = "test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n"
            with open(auto_path, "w", encoding="utf-8") as f:
                f.write(line)
            with open(manual_path, "w", encoding="utf-8") as f:
                f.write("test.wav=ga,110.0,210.0,-310.0,130.0,45.0\n")
            rows, stats = build_oto_ml_dataset("korean", auto_path, manual_path, tg_dir=td, wav_dir=td)
            self.assertEqual(stats["matched_rows"], 1)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["delta_offset"], 10.0)
            self.assertEqual(rows[0]["alias_group"], "cv")
            self.assertEqual(rows[0]["train_keep_default"], 1)
            self.assertEqual(rows[0]["train_skip_reason"], "")


if __name__ == "__main__":
    unittest.main()
