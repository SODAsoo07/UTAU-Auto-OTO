import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_file_utils import (
    build_occurrence_map,
    extract_base_timing_shape,
    parse_oto_line,
    read_oto_rows_for_profile,
    read_text_with_fallback,
)


class OtoFileUtilsTests(unittest.TestCase):
    def test_parse_oto_line_reads_standard_row(self):
        row = parse_oto_line("a.wav=ka,10,80,-150,40,12")
        self.assertIsNotNone(row)
        self.assertEqual(row["wav"], "a.wav")
        self.assertEqual(row["alias"], "ka")
        self.assertAlmostEqual(row["pre"], 40.0)

    def test_extract_base_timing_shape_skips_zero_row(self):
        self.assertIsNone(extract_base_timing_shape("a.wav=ka,0,0,0,0,0"))
        shape = extract_base_timing_shape("a.wav=ka,10,80,-150,40,12")
        self.assertIsNotNone(shape)
        self.assertAlmostEqual(shape["cons_gap"], 40.0)

    def test_read_rows_and_occurrence_map_keep_duplicate_rows_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=ka,10,80,-150,40,12\n")
                f.write("a.wav=ka,14,84,-154,44,14\n")
            text = read_text_with_fallback(oto_path)
            self.assertIn("a.wav=ka", text)
            rows = read_oto_rows_for_profile(
                oto_path,
                wav_normalizer=str.lower,
                alias_normalizer=str.lower,
            )
            mapped = build_occurrence_map(rows)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(mapped), 2)
            self.assertIn(("a.wav", "ka", 0), mapped)
            self.assertIn(("a.wav", "ka", 1), mapped)


if __name__ == "__main__":
    unittest.main()
