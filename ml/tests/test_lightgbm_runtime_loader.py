import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import _make_label_gain, _normalize_model_file_for_runtime


class LightGBMRuntimeLoaderTests(unittest.TestCase):
    def test_make_label_gain_expands_to_required_max_label(self):
        self.assertEqual(_make_label_gain(0), [0])
        self.assertEqual(_make_label_gain(3), [0, 1, 3, 7])
        self.assertEqual(len(_make_label_gain(7)), 8)

    def test_normalize_model_file_keeps_lf_file_as_is(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model_lf.txt")
            with open(model_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("tree\nversion=v4\n")
            out_path = _normalize_model_file_for_runtime(model_path)
            self.assertEqual(os.path.abspath(out_path), os.path.abspath(model_path))

    def test_normalize_model_file_creates_lf_copy_for_crlf(self):
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model_crlf.txt")
            with open(model_path, "w", encoding="utf-8", newline="\r\n") as f:
                f.write("tree\nversion=v4\n")
            out_path = _normalize_model_file_for_runtime(model_path)
            self.assertNotEqual(os.path.abspath(out_path), os.path.abspath(model_path))
            with open(out_path, "rb") as f:
                content = f.read()
            self.assertNotIn(b"\r", content)
            self.assertIn(b"\n", content)


if __name__ == "__main__":
    unittest.main()
