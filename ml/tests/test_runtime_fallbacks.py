import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.alignment_pipeline import run_alignment_with_fallback
from core.mfa_runner import check_mfa_ready
from core.oto_ml_refiner import apply_oto_ml_to_oto_file, check_oto_ml_ready
from core.oto_ml_runtime import OtoDeltaResult, OtoModelBundle


class RuntimeFallbackTests(unittest.TestCase):
    def test_check_mfa_ready_reports_executable_missing(self):
        with mock.patch("core.mfa_runner.find_mfa_executable", return_value=""):
            report = check_mfa_ready(language="korean", mfa_path="")
        self.assertEqual(report["code"], "ALIGN_EXEC_MISSING")
        self.assertFalse(report["ready"])

    def test_run_alignment_with_fallback_reports_output_empty(self):
        with mock.patch("core.alignment_pipeline.has_textgrid_files", side_effect=[False, False]), mock.patch(
            "core.alignment_pipeline.check_mfa_ready",
            return_value={"code": "OK", "mfa_path": "C:/fake/mfa.exe", "ready": True},
        ), mock.patch("core.alignment_pipeline.run_mfa_align", return_value=(True, "")):
            report = run_alignment_with_fallback(
                language="korean",
                wav_folder="wav",
                dictionary_path="dict",
                output_folder="out",
                primary_aligner="mfa",
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["code"], "ALIGN_OUTPUT_EMPTY")

    def test_check_oto_ml_ready_reports_model_missing(self):
        with mock.patch("core.oto_ml_refiner._resolve_model_dir", return_value=None):
            report = check_oto_ml_ready("korean", "cvvc")
        self.assertEqual(report["code"], "ML_MODEL_MISSING")
        self.assertFalse(report["ready"])

    def test_apply_oto_ml_to_oto_file_falls_back_when_model_missing_under_on_policy(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n")
            report = {}
            with mock.patch(
                "core.oto_ml_refiner.parse_oto_rows",
                return_value=[
                    {
                        "line_index": 0,
                        "wav": "test.wav",
                        "alias": "ga",
                        "offset": 100.0,
                        "cons": 200.0,
                        "cutoff": -300.0,
                        "pre": 120.0,
                        "ovl": 40.0,
                    }
                ],
            ), mock.patch(
                "core.oto_ml_refiner.extract_feature_rows",
                return_value=[{"line_index": 0, "format_type": "cvvc", "alias_type": "vc", "coda_type": "stop"}],
            ), mock.patch("core.oto_ml_refiner._resolve_coupled_model_dir", return_value=None), mock.patch(
                "core.oto_ml_refiner._resolve_lightgbm_model_dir", return_value=None
            ):
                changed = apply_oto_ml_to_oto_file(
                    "korean",
                    oto_path,
                    tg_dir=td,
                    wav_dir=td,
                    policy="on",
                    report=report,
                )
        self.assertEqual(changed, 0)
        self.assertEqual(report["code"], "ML_MODEL_MISSING")
        self.assertEqual(report["status"], "fallback")
        self.assertTrue(report["fallback_used"])

    def test_apply_oto_ml_to_oto_file_falls_back_from_coupled_low_confidence_to_lightgbm(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n")

            rows = [
                {
                    "line_index": 0,
                    "wav": "test.wav",
                    "alias": "ga",
                    "offset": 100.0,
                    "cons": 200.0,
                    "cutoff": -300.0,
                    "pre": 120.0,
                    "ovl": 40.0,
                }
            ]
            feats = [
                {
                    "line_index": 0,
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "coda_type": "stop",
                    "blank_span_confidence": 0.82,
                }
            ]
            coupled_bundle = OtoModelBundle(
                backend="coupled_nn_v1",
                model_dir="C:/models/coupled",
                meta={},
                feature_schema={},
                payload={},
            )
            lightgbm_bundle = OtoModelBundle(
                backend="lightgbm",
                model_dir="C:/models/lightgbm",
                meta={},
                feature_schema={},
                payload={},
            )

            def _predict(bundle, _row):
                if bundle.backend == "coupled_nn_v1":
                    return OtoDeltaResult(
                        deltas={
                            "delta_offset": 0.0,
                            "delta_cons": 0.0,
                            "delta_cutoff": 0.0,
                            "delta_pre": 0.0,
                            "delta_ovl": 0.0,
                        },
                        backend="coupled_nn_v1",
                        applied_model="C:/models/coupled",
                        confidence=0.2,
                    )
                return OtoDeltaResult(
                    deltas={
                        "delta_offset": 2.0,
                        "delta_cons": 0.0,
                        "delta_cutoff": 0.0,
                        "delta_pre": 0.0,
                        "delta_ovl": 0.0,
                    },
                    backend="lightgbm",
                    applied_model="C:/models/lightgbm",
                    confidence=None,
                )

            report = {}
            with mock.patch("core.oto_ml_refiner.parse_oto_rows", return_value=rows), mock.patch(
                "core.oto_ml_refiner.extract_feature_rows", return_value=feats
            ), mock.patch(
                "core.oto_ml_refiner._resolve_coupled_model_dir", return_value="C:/models/coupled"
            ), mock.patch(
                "core.oto_ml_refiner._resolve_lightgbm_model_dir", return_value="C:/models/lightgbm"
            ), mock.patch(
                "core.oto_ml_refiner._read_bundle_backend",
                side_effect=lambda path: "coupled_nn_v1" if "coupled" in str(path) else "lightgbm",
            ), mock.patch(
                "core.oto_ml_refiner.load_oto_model_bundle",
                side_effect=lambda path: coupled_bundle if "coupled" in str(path) else lightgbm_bundle,
            ), mock.patch(
                "core.oto_ml_refiner.predict_oto_deltas",
                side_effect=_predict,
            ), mock.patch(
                "core.oto_ml_refiner.delta_enabled_by_default", return_value=True
            ), mock.patch(
                "core.oto_ml_refiner.selector_enabled_by_default", return_value=False
            ), mock.patch(
                "core.oto_ml_refiner.apply_oto_ml_delta",
                side_effect=lambda _lang, _feat, _bundle, **_kw: (102.0, 200.0, -300.0, 120.0, 40.0),
            ), mock.patch(
                "core.oto_ml_refiner._gated_ensemble_enabled", return_value=False
            ):
                changed = apply_oto_ml_to_oto_file(
                    "korean",
                    oto_path,
                    tg_dir=td,
                    wav_dir=td,
                    policy="on",
                    report=report,
                )

        self.assertEqual(changed, 1)
        self.assertEqual(report["status"], "applied")
        self.assertEqual(report["route"], "coupled_nn_v1->lightgbm")
        self.assertGreaterEqual(float(report["blank_confidence"]), 0.8)
        self.assertIn("coupled_low_confidence", str(report.get("fallback_reason", "")))
        self.assertTrue(report["fallback_used"])

    def test_apply_oto_ml_to_oto_file_uses_lightgbm_when_coupled_load_fails(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("test.wav=ga,100.0,200.0,-300.0,120.0,40.0\n")

            rows = [
                {
                    "line_index": 0,
                    "wav": "test.wav",
                    "alias": "ga",
                    "offset": 100.0,
                    "cons": 200.0,
                    "cutoff": -300.0,
                    "pre": 120.0,
                    "ovl": 40.0,
                }
            ]
            feats = [
                {
                    "line_index": 0,
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "coda_type": "stop",
                    "blank_span_confidence": 0.35,
                }
            ]
            lightgbm_bundle = OtoModelBundle(
                backend="lightgbm",
                model_dir="C:/models/lightgbm",
                meta={},
                feature_schema={},
                payload={},
            )

            report = {}
            with mock.patch("core.oto_ml_refiner.parse_oto_rows", return_value=rows), mock.patch(
                "core.oto_ml_refiner.extract_feature_rows", return_value=feats
            ), mock.patch(
                "core.oto_ml_refiner._resolve_coupled_model_dir", return_value="C:/models/coupled"
            ), mock.patch(
                "core.oto_ml_refiner._resolve_lightgbm_model_dir", return_value="C:/models/lightgbm"
            ), mock.patch(
                "core.oto_ml_refiner._read_bundle_backend",
                side_effect=lambda path: "coupled_nn_v1" if "coupled" in str(path) else "lightgbm",
            ), mock.patch(
                "core.oto_ml_refiner.load_oto_model_bundle",
                side_effect=lambda path: None if "coupled" in str(path) else lightgbm_bundle,
            ), mock.patch(
                "core.oto_ml_refiner.delta_enabled_by_default", return_value=True
            ), mock.patch(
                "core.oto_ml_refiner.selector_enabled_by_default", return_value=False
            ), mock.patch(
                "core.oto_ml_refiner.apply_oto_ml_delta",
                side_effect=lambda _lang, _feat, _bundle, **_kw: (101.0, 200.0, -300.0, 120.0, 40.0),
            ):
                changed = apply_oto_ml_to_oto_file(
                    "korean",
                    oto_path,
                    tg_dir=td,
                    wav_dir=td,
                    policy="on",
                    report=report,
                )

        self.assertEqual(changed, 1)
        self.assertEqual(report["status"], "applied")
        self.assertEqual(report["route"], "coupled_nn_v1->lightgbm_load_fallback")
        self.assertIn("coupled_load_failed", str(report.get("fallback_reason", "")))


if __name__ == "__main__":
    unittest.main()
