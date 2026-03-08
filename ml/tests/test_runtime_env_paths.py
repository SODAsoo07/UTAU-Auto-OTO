import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.mfa_runner import find_mfa_executable, get_default_mfa_env_dir
from core.sofa_runner import _resolve_sofa_repo_dir, get_sofa_env_python
from scripts.verify_runtime_portability import probe_runtime_portability


class RuntimeEnvPathTests(unittest.TestCase):
    def test_find_mfa_executable_prefers_shared_env(self):
        with mock.patch.dict(os.environ, {"PUBLIC": r"C:\Users\Public"}, clear=False):
            shared_mfa = os.path.join(get_default_mfa_env_dir(), "Scripts", "mfa.exe")

            def fake_exists(path):
                return os.path.normcase(os.path.normpath(path)) == os.path.normcase(os.path.normpath(shared_mfa))

            with mock.patch("core.mfa_runner.os.path.exists", side_effect=fake_exists):
                with mock.patch("core.mfa_runner.shutil.which", return_value=""):
                    found = find_mfa_executable()
        self.assertEqual(
            os.path.normcase(os.path.normpath(found)),
            os.path.normcase(os.path.normpath(shared_mfa)),
        )

    def test_get_sofa_env_python_uses_shared_env_if_present(self):
        shared_python = os.path.join(
            os.environ.get("LOCALAPPDATA", r"C:\Users\Test\AppData\Local"),
            "UTAU_Auto_OTO_v3",
            ".env_sofa",
            "Scripts",
            "python.exe",
        )

        def fake_exists(path):
            return os.path.normcase(os.path.normpath(path)) == os.path.normcase(os.path.normpath(shared_python))

        with mock.patch("core.sofa_runner.os.path.exists", side_effect=fake_exists):
            found = get_sofa_env_python()
        self.assertEqual(
            os.path.normcase(os.path.normpath(found)),
            os.path.normcase(os.path.normpath(shared_python)),
        )

    def test_resolve_sofa_repo_dir_uses_legacy_repo_if_present(self):
        legacy_repo = os.path.join(ROOT, ".sofa", "SOFA")
        legacy_infer = os.path.join(legacy_repo, "infer.py")

        def fake_exists(path):
            return os.path.normcase(os.path.normpath(path)) == os.path.normcase(os.path.normpath(legacy_infer))

        with mock.patch("core.sofa_runner.os.path.exists", side_effect=fake_exists):
            resolved = _resolve_sofa_repo_dir("")
        self.assertEqual(
            os.path.normcase(os.path.normpath(resolved)),
            os.path.normcase(os.path.normpath(legacy_repo)),
        )

    def test_portable_probe_prefers_shared_env_even_for_moved_exe(self):
        fake_exe = os.path.join(ROOT, "dist", "Moved", "UTAU_Auto_OTO.exe")
        shared_mfa = os.path.join(get_default_mfa_env_dir(), "Scripts", "mfa.exe")
        shared_sofa_python = os.path.join(
            os.environ.get("LOCALAPPDATA", r"C:\Users\Test\AppData\Local"),
            "UTAU_Auto_OTO_v3",
            ".env_sofa",
            "Scripts",
            "python.exe",
        )
        shared_sofa_repo = os.path.join(
            os.environ.get("LOCALAPPDATA", r"C:\Users\Test\AppData\Local"),
            "UTAU_Auto_OTO_v3",
            ".sofa",
            "SOFA",
        )
        shared_sofa_infer = os.path.join(shared_sofa_repo, "infer.py")

        def fake_exists(path):
            normalized = os.path.normcase(os.path.normpath(path))
            return normalized in {
                os.path.normcase(os.path.normpath(shared_mfa)),
                os.path.normcase(os.path.normpath(shared_sofa_python)),
                os.path.normcase(os.path.normpath(shared_sofa_infer)),
            }

        def fake_diag(_mfa_path="", language="korean", callback=None):
            return {"ready": True, "issues": [], "checks": {"mfa_executable": True}}

        with mock.patch("core.mfa_runner.os.path.exists", side_effect=fake_exists):
            with mock.patch("core.mfa_runner.shutil.which", return_value=""):
                with mock.patch("scripts.verify_runtime_portability.diagnose_mfa_runtime", side_effect=fake_diag):
                    with mock.patch("core.sofa_runner.os.path.exists", side_effect=fake_exists):
                        report = probe_runtime_portability(fake_exe, language="korean")

        self.assertTrue(report["mfa_uses_shared_root"])
        self.assertTrue(report["sofa_python_exists"])
        self.assertTrue(report["sofa_repo_exists"])


if __name__ == "__main__":
    unittest.main()
