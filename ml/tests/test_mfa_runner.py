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
    ensure_korean_support,
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

    def test_ensure_korean_support_repairs_pkg_resources_before_eunjeon_check(self):
        import core.mfa_runner as mfa_runner

        calls = []

        class Result:
            def __init__(self, returncode=0, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        with tempfile.TemporaryDirectory() as td:
            env_dir = os.path.join(td, ".env")
            scripts_dir = os.path.join(env_dir, "Scripts")
            os.makedirs(scripts_dir, exist_ok=True)
            python_exe = os.path.join(env_dir, "python.exe")
            pip_exe = os.path.join(scripts_dir, "pip.exe")
            mfa_exe = os.path.join(scripts_dir, "mfa.exe")
            for p in (python_exe, pip_exe, mfa_exe):
                with open(p, "wb") as f:
                    f.write(b"")

            real_run = mfa_runner._run_subprocess_text
            real_get_env = mfa_runner._get_conda_env
            real_patch = mfa_runner.patch_mfa_korean_support
            real_which = mfa_runner.shutil.which

            state = {"pkg_ok": False, "imports_ok": False}

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                cmd_str = " ".join(cmd)
                if "import pkg_resources" in cmd_str:
                    if state["pkg_ok"]:
                        return Result(0)
                    return Result(1, stderr="ModuleNotFoundError: No module named 'pkg_resources'")
                if "install --upgrade setuptools<81" in cmd_str:
                    state["pkg_ok"] = True
                    return Result(0)
                if "import eunjeon; import jamo" in cmd_str:
                    if state["pkg_ok"]:
                        state["imports_ok"] = True
                        return Result(0)
                    return Result(1, stderr="pkg_resources missing")
                return Result(0)

            try:
                mfa_runner._run_subprocess_text = fake_run
                mfa_runner._get_conda_env = lambda _: {}
                mfa_runner.patch_mfa_korean_support = lambda *args, **kwargs: True
                mfa_runner.shutil.which = lambda _: None
                self.assertTrue(ensure_korean_support(mfa_exe))
            finally:
                mfa_runner._run_subprocess_text = real_run
                mfa_runner._get_conda_env = real_get_env
                mfa_runner.patch_mfa_korean_support = real_patch
                mfa_runner.shutil.which = real_which

        joined = "\n".join(" ".join(cmd) for cmd in calls)
        self.assertIn("import pkg_resources", joined)
        self.assertIn("install --upgrade setuptools<81", joined)
        self.assertIn("import eunjeon; import jamo", joined)


if __name__ == "__main__":
    unittest.main()
