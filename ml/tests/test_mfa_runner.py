import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.mfa_runner import (
    _contains_non_ascii,
    _prepare_ascii_safe_alignment_workspace,
)


class MfaRunnerTests(unittest.TestCase):
    def test_contains_non_ascii(self):
        self.assertFalse(_contains_non_ascii(r"C:\tmp\abc"))
        self.assertTrue(_contains_non_ascii(r"C:\tmp\波音リツEve"))

    def test_prepare_ascii_safe_alignment_workspace_uses_ascii_paths(self):
        with tempfile.TemporaryDirectory() as td:
            source = os.path.join(td, "波音リツEve")
            out = os.path.join(td, "출력")
            os.makedirs(source, exist_ok=True)
            with open(os.path.join(source, "a.wav"), "wb") as f:
                f.write(b"RIFF")
            with open(os.path.join(source, "a.lab"), "w", encoding="utf-8") as f:
                f.write("0 1 a\n")
            dict_path = os.path.join(source, "사전.txt")
            with open(dict_path, "w", encoding="utf-8") as f:
                f.write("a a\n")

            ws = _prepare_ascii_safe_alignment_workspace(source, dict_path, out)
            self.assertTrue(os.path.isdir(ws["corpus_dir"]))
            self.assertTrue(os.path.isfile(os.path.join(ws["corpus_dir"], "a.wav")))
            self.assertTrue(os.path.isfile(os.path.join(ws["corpus_dir"], "a.lab")))
            self.assertTrue(os.path.isfile(ws["dict_path"]))
            self.assertFalse(_contains_non_ascii(ws["base"]))
            self.assertFalse(_contains_non_ascii(ws["corpus_dir"]))
            self.assertFalse(_contains_non_ascii(ws["dict_path"]))
            self.assertFalse(_contains_non_ascii(ws["output_dir"]))


if __name__ == "__main__":
    unittest.main()
