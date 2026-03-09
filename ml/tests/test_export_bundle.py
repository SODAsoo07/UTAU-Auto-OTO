import json
import os
import sys
import tempfile
import unittest
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_export import export_model_bundle, validate_bundle_dir


class ExportBundleTests(unittest.TestCase):
    def _make_bundle(self, root: str) -> str:
        model_dir = os.path.join(root, "japanese", "cvvc", "v1")
        os.makedirs(model_dir, exist_ok=True)
        for name in [
            "feature_schema.json",
            "model_meta.json",
            "eval_summary.json",
            "model_offset.txt",
            "model_cons.txt",
            "model_cutoff.txt",
            "model_pre.txt",
            "model_ovl.txt",
        ]:
            with open(os.path.join(model_dir, name), "w", encoding="utf-8") as f:
                if name == "model_meta.json":
                    json.dump({
                        "language": "japanese",
                        "format_type": "cvvc",
                        "backend": "lightgbm",
                        "model_version": "v1",
                        "feature_version": "v1",
                        "train_rows": 10,
                        "voicebank_count": 2,
                    }, f)
                else:
                    f.write("{}")
        return model_dir

    def test_validate_bundle_dir(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_bundle(td)
            self.assertEqual(validate_bundle_dir(model_dir), [])

    def test_export_model_bundle_with_zip(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_bundle(td)
            export_root = os.path.join(td, "exports")
            manifest = export_model_bundle(model_dir, export_root, create_zip=True)
            self.assertTrue(os.path.isdir(manifest["export_dir"]))
            self.assertTrue(os.path.isfile(os.path.join(manifest["export_dir"], "bundle_manifest.json")))
            self.assertTrue(os.path.isfile(manifest["zip_path"]))
            with zipfile.ZipFile(manifest["zip_path"], "r") as zf:
                names = set(zf.namelist())
            self.assertIn(f"{manifest['bundle_slug']}/model_meta.json", names)

    def test_export_model_bundle_keeps_family_routing(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = os.path.join(td, "japanese", "cvvc", "families", "cv", "v1")
            os.makedirs(model_dir, exist_ok=True)
            for name in [
                "feature_schema.json",
                "eval_summary.json",
                "model_offset.txt",
                "model_cons.txt",
                "model_cutoff.txt",
                "model_pre.txt",
                "model_ovl.txt",
            ]:
                with open(os.path.join(model_dir, name), "w", encoding="utf-8") as f:
                    f.write("{}")
            with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "language": "japanese",
                    "format_type": "cvvc",
                    "backend": "lightgbm",
                    "model_version": "v1",
                    "feature_version": "v1",
                    "filters": {"alias_family": "cv"},
                }, f)
            manifest = export_model_bundle(model_dir, os.path.join(td, "exports"))
            self.assertEqual(manifest["alias_family"], "cv")
            self.assertEqual(manifest["bundle_slug"], "japanese_cvvc_cv_v1")
            self.assertEqual(manifest["install_subdir"], "japanese/cvvc/families/cv/v1")


if __name__ == "__main__":
    unittest.main()
