from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_PATH = os.path.join(ROOT, "ml", "scripts", "evaluate_oto_ml_batch.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_oto_ml_batch", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class EvaluateOtoMlBatchTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_render_table_outputs_headers(self):
        text = self.mod._render_table(
            [{"lang": "korean", "format": "cvc", "rows": 10}],
            ["lang", "format", "rows"],
        )
        self.assertIn("lang", text)
        self.assertIn("korean", text)

    def test_run_batch_evaluation_discovers_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = os.path.join(td, "workspace")
            reports = os.path.join(td, "reports")
            os.makedirs(os.path.join(workspace, "datasets", "korean"), exist_ok=True)
            os.makedirs(os.path.join(workspace, "models", "korean_cvc_bundle"), exist_ok=True)
            dataset = os.path.join(workspace, "datasets", "korean", "dataset_korean_cvc.csv")
            selector_dataset = os.path.join(workspace, "datasets", "korean", "selector_dataset_korean_cvc.csv")
            with open(dataset, "w", encoding="utf-8") as f:
                f.write("dummy\n")
            with open(os.path.join(workspace, "models", "korean_cvc_bundle", "selector_model.txt"), "w", encoding="utf-8") as f:
                f.write("x")
            with open(os.path.join(workspace, "models", "korean_cvc_bundle", "selector_meta.json"), "w", encoding="utf-8") as f:
                json.dump({}, f)

            delta_summary = {
                "rows": 12,
                "targets": {
                    "delta_offset": {"baseline_mae": 10, "model_mae": 8},
                    "delta_cons": {"baseline_mae": 5, "model_mae": 4},
                    "delta_cutoff": {"baseline_mae": 9, "model_mae": 7},
                    "delta_pre": {"baseline_mae": 3, "model_mae": 2},
                    "delta_ovl": {"baseline_mae": 4, "model_mae": 1},
                },
            }
            selector_summary = {
                "rows": 20,
                "groups": 6,
                "objective": "pointwise",
                "top1_baseline": 0.2,
                "top1_model": 0.5,
                "score_mae": 0.1,
            }

            def _fake_build_selector(**kwargs):
                with open(selector_dataset, "w", encoding="utf-8") as f:
                    f.write("dummy\n")
                return {"rows": 1, "groups": 1, "out_csv": selector_dataset}

            with mock.patch.object(self.mod, "evaluate_lightgbm_bundle", return_value=delta_summary), \
                 mock.patch.object(self.mod, "evaluate_lightgbm_selector_bundle", return_value=selector_summary), \
                 mock.patch.object(self.mod, "build_selector_dataset_csv_from_delta_dataset", side_effect=_fake_build_selector):
                result = self.mod.run_batch_evaluation(workspace, reports)

            self.assertEqual(len(result["delta_table"]), 1)
            self.assertEqual(result["delta_table"][0]["format"], "cvc")
            self.assertEqual(len(result["selector_table"]), 1)
            self.assertTrue(os.path.exists(os.path.join(reports, "delta_eval_korean_cvc.json")))
            self.assertTrue(os.path.exists(os.path.join(reports, "selector_eval_korean_cvc.json")))


if __name__ == "__main__":
    unittest.main()
