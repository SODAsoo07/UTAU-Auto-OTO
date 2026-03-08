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
            ), mock.patch("core.oto_ml_refiner._resolve_model_dir", return_value=None):
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


if __name__ == "__main__":
    unittest.main()
