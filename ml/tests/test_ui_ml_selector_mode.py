import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.app_mixins import AppRuntimeMixin


class _DummyRuntime(AppRuntimeMixin):
    _is_closing = False

    def _append_log(self, _msg):
        pass


class UiMlSelectorModeTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("UTOA_FORCE_OTO_SELECTOR", None)
        os.environ.pop("UTOA_DISABLE_OTO_SELECTOR", None)
        self.runtime = _DummyRuntime()

    def tearDown(self):
        os.environ.pop("UTOA_FORCE_OTO_SELECTOR", None)
        os.environ.pop("UTOA_DISABLE_OTO_SELECTOR", None)

    def test_selector_mode_policy_clears_overrides(self):
        os.environ["UTOA_FORCE_OTO_SELECTOR"] = "1"
        self.assertEqual(self.runtime._apply_ml_selector_runtime_mode("기본 정책"), "policy")
        self.assertNotIn("UTOA_FORCE_OTO_SELECTOR", os.environ)
        self.assertNotIn("UTOA_DISABLE_OTO_SELECTOR", os.environ)

    def test_selector_mode_delta_sets_disable_flag(self):
        self.assertEqual(self.runtime._apply_ml_selector_runtime_mode("델타만"), "delta")
        self.assertEqual(os.environ.get("UTOA_DISABLE_OTO_SELECTOR"), "1")
        self.assertNotIn("UTOA_FORCE_OTO_SELECTOR", os.environ)

    def test_selector_mode_selector_sets_force_flag(self):
        self.assertEqual(self.runtime._apply_ml_selector_runtime_mode("델타+셀렉터"), "selector")
        self.assertEqual(os.environ.get("UTOA_FORCE_OTO_SELECTOR"), "1")
        self.assertNotIn("UTOA_DISABLE_OTO_SELECTOR", os.environ)

    def test_cleanup_keeps_lab_and_textgrid_files(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "oto.ini")
            lab_path = os.path.join(td, "sample.lab")
            tg_path = os.path.join(td, "sample.TextGrid")
            tg_dir = os.path.join(td, "textgrids")
            os.makedirs(tg_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("oto")
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write("a")
            with open(tg_path, "w", encoding="utf-8") as f:
                f.write("tg")
            with open(os.path.join(tg_dir, "inner.TextGrid"), "w", encoding="utf-8") as f:
                f.write("inner")
            self.runtime._cleanup_generated_output_artifacts(out_path, snapshot=None)
            self.assertTrue(os.path.exists(lab_path))
            self.assertTrue(os.path.exists(tg_path))
            self.assertTrue(os.path.isdir(tg_dir))

    def test_cleanup_removes_only_new_generated_oto_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "oto_out.ini")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("main")

            existing_sub = os.path.join(td, "C4")
            os.makedirs(existing_sub, exist_ok=True)
            existing_oto = os.path.join(existing_sub, "oto.ini")
            with open(existing_oto, "w", encoding="utf-8") as f:
                f.write("existing")

            snapshot = self.runtime._snapshot_output_tree_for_cleanup(out_path)

            byproduct_dir = os.path.join(td, "tmp_gen")
            os.makedirs(byproduct_dir, exist_ok=True)
            byproduct_oto = os.path.join(byproduct_dir, "oto.ini")
            with open(byproduct_oto, "w", encoding="utf-8") as f:
                f.write("tmp")

            byproduct_ml = os.path.join(td, "oto_auto_ml.ini")
            with open(byproduct_ml, "w", encoding="utf-8") as f:
                f.write("tmp_ml")

            non_oto_file = os.path.join(td, "new_file.txt")
            with open(non_oto_file, "w", encoding="utf-8") as f:
                f.write("keep")

            self.runtime._cleanup_generated_output_artifacts(out_path, snapshot=snapshot)

            self.assertTrue(os.path.exists(out_path))
            self.assertTrue(os.path.exists(existing_oto))
            self.assertFalse(os.path.exists(byproduct_oto))
            self.assertFalse(os.path.exists(byproduct_ml))
            self.assertTrue(os.path.exists(non_oto_file))


if __name__ == "__main__":
    unittest.main()
