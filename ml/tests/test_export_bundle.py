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

    def _make_coupled_bundle(self, root: str) -> str:
        model_dir = os.path.join(root, "korean", "vcv", "v1")
        os.makedirs(model_dir, exist_ok=True)
        for name in [
            "feature_schema.json",
            "model_meta.json",
            "eval_summary.json",
            "coupled_model.pt",
        ]:
            path = os.path.join(model_dir, name)
            if name == "model_meta.json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "language": "korean",
                            "format_type": "vcv",
                            "backend": "coupled_nn_v1",
                            "model_version": "v2",
                            "feature_version": "v7",
                            "head_mode": "split",
                            "anchor_targets": ["delta_offset", "delta_pre", "delta_cutoff"],
                            "delta_targets": ["delta_cons", "delta_ovl"],
                        },
                        f,
                    )
            elif name == "coupled_model.pt":
                with open(path, "wb") as f:
                    f.write(b"pt")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")
        return model_dir

    def _make_ensemble_bundle(self, root: str) -> str:
        model_dir = os.path.join(root, "korean", "cvc", "v1")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(model_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(model_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "language": "korean",
                    "format_type": "cvc",
                    "backend": "ensemble_v1",
                    "model_version": "v1",
                    "feature_version": "v11",
                    "meta_enabled": True,
                },
                f,
            )
        for subdir, files in {
            "lightgbm": [
                "feature_schema.json",
                "model_meta.json",
                "eval_summary.json",
                "model_offset.txt",
                "model_cons.txt",
                "model_cutoff.txt",
                "model_pre.txt",
                "model_ovl.txt",
            ],
            "coupled": [
                "feature_schema.json",
                "model_meta.json",
                "eval_summary.json",
                "coupled_model.pt",
            ],
            "meta": [
                "feature_schema.json",
                "model_meta.json",
                "eval_summary.json",
                "model_offset.txt",
                "model_cons.txt",
                "model_cutoff.txt",
                "model_pre.txt",
                "model_ovl.txt",
            ],
        }.items():
            subdir_path = os.path.join(model_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            for name in files:
                path = os.path.join(subdir_path, name)
                if name == "model_meta.json":
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "language": "korean",
                                "format_type": "cvc",
                                "backend": "coupled_nn_v2_rawmel" if subdir == "coupled" else "lightgbm",
                                "model_version": "v2",
                                "feature_version": "v11",
                                "head_mode": "split",
                                "anchor_targets": ["delta_offset", "delta_pre", "delta_cutoff"],
                                "delta_targets": ["delta_cons", "delta_ovl"],
                            },
                            f,
                        )
                elif name.endswith(".pt"):
                    with open(path, "wb") as f:
                        f.write(b"pt")
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write("{}")
        return model_dir

    def test_validate_bundle_dir(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_bundle(td)
            self.assertEqual(validate_bundle_dir(model_dir), [])

    def test_validate_bundle_dir_coupled(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_coupled_bundle(td)
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

    def test_export_model_bundle_coupled_uses_coupled_required_files(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_coupled_bundle(td)
            manifest = export_model_bundle(model_dir, os.path.join(td, "exports"))
            self.assertEqual(manifest["backend"], "coupled_nn_v1")
            self.assertIn("coupled_model.pt", manifest["required_files"])
            names = {entry["name"] for entry in manifest["files"]}
            self.assertIn("coupled_model.pt", names)
            self.assertNotIn("model_offset.txt", names)

    def test_export_model_bundle_ensemble_keeps_nested_files(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_ensemble_bundle(td)
            manifest = export_model_bundle(model_dir, os.path.join(td, "exports"))
            self.assertEqual(manifest["backend"], "ensemble_v1")
            names = {entry["name"] for entry in manifest["files"]}
            self.assertIn("lightgbm/model_offset.txt", names)
            self.assertIn("coupled/coupled_model.pt", names)
            self.assertIn("meta/model_offset.txt", names)


if __name__ == "__main__":
    unittest.main()
