import os
import sys
import tempfile
import json
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.app_mixins import FileDialogMixin


class _DummyVar:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _DummyUi(FileDialogMixin):
    def __init__(self, app_dir, language="korean", auto_format="자동 감지 (권장)"):
        self.app_dir = app_dir
        self._language = language
        self.auto_format_var = _DummyVar(auto_format)
        self.ml_model_root_kr_var = _DummyVar("")
        self.ml_model_root_ja_var = _DummyVar("")
        self.ml_coupled_backend_var = _DummyVar("auto")

    def _get_language(self):
        return self._language


class UiMlModelDefaultTests(unittest.TestCase):
    def test_preferred_browse_dir_prefers_current_format_ensemble(self):
        with tempfile.TemporaryDirectory() as td:
            ensemble_dir = os.path.join(td, "ML_models", "korean", "cvc", "v1_ensemble")
            os.makedirs(ensemble_dir, exist_ok=True)
            ui = _DummyUi(td, language="korean", auto_format="CVC (한국어 전용)")

            preferred = ui._preferred_ml_model_browse_dir(ui.ml_model_root_kr_var)

        self.assertEqual(
            os.path.normcase(os.path.normpath(preferred)),
            os.path.normcase(os.path.normpath(ensemble_dir)),
        )

    def test_apply_recommended_defaults_sets_language_root_and_ensemble_backend(self):
        with tempfile.TemporaryDirectory() as td:
            ensemble_dir = os.path.join(td, "ML_models", "korean", "cvc", "v1_ensemble")
            os.makedirs(ensemble_dir, exist_ok=True)
            with open(os.path.join(ensemble_dir, "model_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"backend": "ensemble_v1"}, f)
            ui = _DummyUi(td, language="korean", auto_format="자동 감지 (권장)")

            ui._apply_recommended_ml_model_defaults()

        self.assertEqual(
            os.path.normcase(os.path.normpath(ui.ml_model_root_kr_var.get())),
            os.path.normcase(os.path.normpath(os.path.join(td, "ML_models", "korean"))),
        )
        self.assertEqual(ui.ml_coupled_backend_var.get(), "ensemble")


if __name__ == "__main__":
    unittest.main()
