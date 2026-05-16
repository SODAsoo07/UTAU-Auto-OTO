import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml.oto_ml_bundle_install import install_exported_bundle
from core.oto_ml_export import export_model_bundle


class BundleInstallTests(unittest.TestCase):
    def _make_bundle(self, root: str) -> str:
        model_dir = os.path.join(root, "assets", "models", "oto_ml", "japanese", "cvvc", "v1")
        os.makedirs(model_dir, exist_ok=True)
        files = {
            "feature_schema.json": "{}",
            "eval_summary.json": "{}",
            "model_offset.txt": "x",
            "model_cons.txt": "x",
            "model_cutoff.txt": "x",
            "model_pre.txt": "x",
            "model_ovl.txt": "x",
        }
        for name, content in files.items():
            with open(os.path.join(model_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
        with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            f.write('{"language":"japanese","format_type":"cvvc","backend":"lightgbm","model_version":"v1","feature_version":"v1"}')
        return model_dir

    def _make_coupled_bundle(self, root: str) -> str:
        model_dir = os.path.join(root, "assets", "models", "oto_ml", "korean", "vcv", "v1")
        os.makedirs(model_dir, exist_ok=True)
        files = {
            "feature_schema.json": "{}",
            "eval_summary.json": "{}",
        }
        for name, content in files.items():
            with open(os.path.join(model_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
        with open(os.path.join(model_dir, "coupled_model.pt"), "wb") as f:
            f.write(b"pt")
        with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            f.write(
                '{"language":"korean","format_type":"vcv","backend":"coupled_nn_v1","model_version":"v2","feature_version":"v7","head_mode":"split","anchor_targets":["delta_offset","delta_pre","delta_cutoff"],"delta_targets":["delta_cons","delta_ovl"]}'
            )
        return model_dir

    def _make_ensemble_bundle(self, root: str) -> str:
        model_dir = os.path.join(root, "assets", "models", "oto_ml", "korean", "cvc", "v1")
        os.makedirs(model_dir, exist_ok=True)
        for name, content in {
            "feature_schema.json": "{}",
            "eval_summary.json": "{}",
        }.items():
            with open(os.path.join(model_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
        with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            f.write(
                '{"language":"korean","format_type":"cvc","backend":"ensemble_v1","model_version":"v1","feature_version":"v11","meta_enabled":true}'
            )
        for subdir, files in {
            "lightgbm": {
                "feature_schema.json": "{}",
                "eval_summary.json": "{}",
                "model_offset.txt": "x",
                "model_cons.txt": "x",
                "model_cutoff.txt": "x",
                "model_pre.txt": "x",
                "model_ovl.txt": "x",
                "model_meta.json": '{"language":"korean","format_type":"cvc","backend":"lightgbm","model_version":"v1","feature_version":"v11"}',
            },
            "coupled": {
                "feature_schema.json": "{}",
                "eval_summary.json": "{}",
                "model_meta.json": '{"language":"korean","format_type":"cvc","backend":"coupled_nn_v2_rawmel","model_version":"v2","feature_version":"v11","head_mode":"split","anchor_targets":["delta_offset","delta_pre","delta_cutoff"],"delta_targets":["delta_cons","delta_ovl"]}',
            },
            "meta": {
                "feature_schema.json": "{}",
                "eval_summary.json": "{}",
                "model_offset.txt": "x",
                "model_cons.txt": "x",
                "model_cutoff.txt": "x",
                "model_pre.txt": "x",
                "model_ovl.txt": "x",
                "model_meta.json": '{"language":"korean","format_type":"cvc","backend":"lightgbm","model_version":"v1","feature_version":"ensemble_v1_meta"}',
            },
        }.items():
            subdir_path = os.path.join(model_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            for name, content in files.items():
                path = os.path.join(subdir_path, name)
                if name == "coupled_model.pt":
                    with open(path, "wb") as f:
                        f.write(b"pt")
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
            if subdir == "coupled":
                with open(os.path.join(subdir_path, "coupled_model.pt"), "wb") as f:
                    f.write(b"pt")
        return model_dir

    def test_install_exported_zip_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_bundle(td)
            export_root = os.path.join(td, "exports")
            manifest = export_model_bundle(model_dir, export_root, create_zip=True)
            result = install_exported_bundle(manifest["zip_path"], install_root=os.path.join(td, "installed"))
            self.assertTrue(os.path.isdir(result["installed_dir"]))
            self.assertTrue(os.path.isfile(os.path.join(result["installed_dir"], "model_meta.json")))

    def test_install_family_bundle_preserves_family_path(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = os.path.join(td, "assets", "models", "oto_ml", "japanese", "cvvc", "families", "bridge", "v1")
            os.makedirs(model_dir, exist_ok=True)
            files = {
                "feature_schema.json": "{}",
                "eval_summary.json": "{}",
                "model_offset.txt": "x",
                "model_cons.txt": "x",
                "model_cutoff.txt": "x",
                "model_pre.txt": "x",
                "model_ovl.txt": "x",
            }
            for name, content in files.items():
                with open(os.path.join(model_dir, name), "w", encoding="utf-8") as f:
                    f.write(content)
            with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
                f.write(
                    '{"language":"japanese","format_type":"cvvc","backend":"lightgbm","model_version":"v1","feature_version":"v1","filters":{"alias_family":"bridge"}}'
                )
            export_root = os.path.join(td, "exports")
            manifest = export_model_bundle(model_dir, export_root)
            result = install_exported_bundle(manifest["export_dir"], install_root=os.path.join(td, "installed"))
            self.assertTrue(result["installed_dir"].endswith(os.path.join("japanese", "cvvc", "families", "vc", "v1")))
            self.assertEqual(result["alias_family"], "vc")
            self.assertTrue(os.path.isfile(os.path.join(result["installed_dir"], "model_meta.json")))

    def test_install_exported_coupled_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_coupled_bundle(td)
            export_root = os.path.join(td, "exports")
            with self.assertRaisesRegex(ValueError, "no longer supported"):
                export_model_bundle(model_dir, export_root, create_zip=True)

    def test_install_exported_ensemble_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_ensemble_bundle(td)
            export_root = os.path.join(td, "exports")
            manifest = export_model_bundle(model_dir, export_root, create_zip=True)
            result = install_exported_bundle(manifest["zip_path"], install_root=os.path.join(td, "installed"))
            self.assertTrue(os.path.isdir(result["installed_dir"]))
            self.assertTrue(os.path.isfile(os.path.join(result["installed_dir"], "lightgbm", "model_offset.txt")))
            self.assertTrue(os.path.isfile(os.path.join(result["installed_dir"], "coupled", "coupled_model.pt")))
            self.assertTrue(os.path.isfile(os.path.join(result["installed_dir"], "meta", "model_offset.txt")))
            self.assertTrue(result["installed_dir"].endswith(os.path.join("korean", "cvc", "v1")))


if __name__ == "__main__":
    unittest.main()
