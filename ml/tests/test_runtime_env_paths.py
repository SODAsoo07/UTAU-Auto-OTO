import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.mfa_runner import find_mfa_executable, get_default_mfa_env_dir
from core.oto_ml_refiner import _resolve_model_dir
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

    def test_portable_probe_prefers_shared_env_even_for_moved_exe(self):
        fake_exe = os.path.join(ROOT, "dist", "Moved", "UTAU_Auto_OTO.exe")
        shared_mfa = os.path.join(get_default_mfa_env_dir(), "Scripts", "mfa.exe")

        def fake_exists(path):
            normalized = os.path.normcase(os.path.normpath(path))
            return normalized == os.path.normcase(os.path.normpath(shared_mfa))

        def fake_diag(_mfa_path="", language="korean", callback=None):
            return {"ready": True, "issues": [], "checks": {"mfa_executable": True}}

        with mock.patch("core.mfa_runner.os.path.exists", side_effect=fake_exists):
            with mock.patch("core.mfa_runner.shutil.which", return_value=""):
                with mock.patch("scripts.verify_runtime_portability.diagnose_mfa_runtime", side_effect=fake_diag):
                    report = probe_runtime_portability(fake_exe, language="korean")

        self.assertTrue(report["mfa_uses_shared_root"])

    def test_oto_ml_resolve_prefers_workspace_models_before_installed_and_assets(self):
        workspace_model = os.path.join(
            ROOT, "logs", "ml_workspace", "models", "korean", "cvvc", "families", "cv", "v1", "model_meta.json"
        )
        installed_model = os.path.join(
            ROOT, "models_installed", "oto_ml", "korean", "cvvc", "families", "cv", "v1", "model_meta.json"
        )
        asset_model = os.path.join(
            ROOT, "assets", "models", "oto_ml", "korean", "cvvc", "families", "cv", "v1", "model_meta.json"
        )

        def fake_isfile(path):
            normalized = os.path.normcase(os.path.normpath(path))
            return normalized in {
                os.path.normcase(os.path.normpath(workspace_model)),
                os.path.normcase(os.path.normpath(installed_model)),
                os.path.normcase(os.path.normpath(asset_model)),
            }

        with mock.patch("core.oto_ml_refiner.os.path.isfile", side_effect=fake_isfile):
            resolved = _resolve_model_dir("korean", "cvvc", alias_family="cv")

        self.assertEqual(
            os.path.normcase(os.path.normpath(resolved)),
            os.path.normcase(os.path.normpath(os.path.dirname(workspace_model))),
        )

    def test_oto_ml_resolve_supports_legacy_export_model_root(self):
        export_model = os.path.join(ROOT, "ML_models", "korean_cvvc_profile_run_new", "model_meta.json")

        def fake_isfile(path):
            normalized = os.path.normcase(os.path.normpath(path))
            return normalized == os.path.normcase(os.path.normpath(export_model))

        with mock.patch("core.oto_ml_refiner.os.path.isfile", side_effect=fake_isfile):
            resolved = _resolve_model_dir("korean", "cvvc", alias_family="")

        self.assertEqual(
            os.path.normcase(os.path.normpath(resolved)),
            os.path.normcase(os.path.normpath(os.path.dirname(export_model))),
        )


if __name__ == "__main__":
    unittest.main()
