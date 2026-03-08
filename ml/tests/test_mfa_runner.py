import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.mfa_runner import (
    _contains_non_ascii,
    _decode_subprocess_output,
    _prepare_ascii_safe_alignment_workspace,
    mfa_python_version_requires_downgrade,
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

    def test_decode_subprocess_output_falls_back_from_utf8_mode(self):
        expected = [0xD55C, 0xAE00, 0x20, 0xACBD, 0xB85C, 0x20, 0xCD9C, 0xB825]
        text = "".join(chr(v) for v in expected)
        raw = text.encode("cp949")
        self.assertEqual([ord(ch) for ch in _decode_subprocess_output(raw)], expected)

    def test_python_313_requires_downgrade(self):
        self.assertTrue(mfa_python_version_requires_downgrade("3.13.0"))
        self.assertTrue(mfa_python_version_requires_downgrade("3.14"))

    def test_python_312_or_lower_does_not_require_downgrade(self):
        self.assertFalse(mfa_python_version_requires_downgrade("3.12.9"))
        self.assertFalse(mfa_python_version_requires_downgrade("3.10.11"))
        self.assertFalse(mfa_python_version_requires_downgrade(""))


if __name__ == "__main__":
    unittest.main()
