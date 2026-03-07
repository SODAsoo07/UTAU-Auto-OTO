import csv
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.tune_alignment_thresholds import (
    _build_runtime_batch_config,
    _build_env_overrides,
    _format_env_suffix,
    _parse_float_grid,
    _rank_key,
    _write_csv,
)


class TuneAlignmentThresholdsTests(unittest.TestCase):
    def test_parse_float_grid_supports_none_and_dedup(self):
        vals = _parse_float_grid("0.60, 0.60, none, 0.70")
        self.assertEqual(vals, [0.6, None, 0.7])

    def test_build_env_overrides_skips_none(self):
        env = _build_env_overrides(0.68, None, None, 0.35)
        self.assertEqual(env["UTOA_JA_MAPPING_CONF_THRESHOLD"], "0.6800")
        self.assertEqual(env["UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD"], "0.3500")
        self.assertNotIn("UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD", env)

    def test_rank_key_prefers_no_regression(self):
        good = {
            "run_failed": False,
            "regression_count": 0,
            "errors_delta_sum": -1,
            "quality_case_count": 1,
            "quality_suspicious_ratio": 0.10,
            "quality_mae_pre_abs": 40.0,
            "quality_mae_offset": 30.0,
            "warnings_delta_sum": 0,
            "improvement_count": 1,
            "status_changed_cases": 0,
        }
        bad = {
            "run_failed": False,
            "regression_count": 1,
            "errors_delta_sum": -5,
            "quality_case_count": 1,
            "quality_suspicious_ratio": 0.05,
            "quality_mae_pre_abs": 10.0,
            "quality_mae_offset": 10.0,
            "warnings_delta_sum": -3,
            "improvement_count": 2,
            "status_changed_cases": 0,
        }
        self.assertLess(_rank_key(good), _rank_key(bad))

    def test_rank_key_uses_quality_metrics_as_tie_breaker(self):
        better_quality = {
            "run_failed": False,
            "regression_count": 0,
            "errors_delta_sum": 0,
            "quality_total_case_count": 2,
            "quality_case_count": 1,
            "quality_compared_case_count": 1,
            "quality_suspicious_ratio": 0.10,
            "quality_cv_like_order_jump_ratio": 0.05,
            "quality_cv_like_order_jump_severity": 3.0,
            "quality_cv_like_suspicious_ratio": 0.10,
            "quality_weighted_mae_pre_abs": 45.0,
            "quality_p90_pre_abs": 70.0,
            "quality_mae_pre_abs": 50.0,
            "quality_weighted_mae_offset": 25.0,
            "quality_p90_offset": 40.0,
            "quality_mae_offset": 30.0,
            "warnings_delta_sum": 0,
            "improvement_count": 0,
            "status_changed_cases": 0,
        }
        worse_quality = dict(better_quality)
        worse_quality["quality_cv_like_order_jump_ratio"] = 0.20
        self.assertLess(_rank_key(better_quality), _rank_key(worse_quality))

    def test_rank_key_uses_p90_after_weighted_mae(self):
        better_tail = {
            "run_failed": False,
            "regression_count": 0,
            "errors_delta_sum": 0,
            "quality_total_case_count": 1,
            "quality_case_count": 1,
            "quality_compared_case_count": 1,
            "quality_cv_like_order_jump_ratio": 0.0,
            "quality_cv_like_order_jump_severity": 0.0,
            "quality_cv_like_suspicious_ratio": 0.0,
            "quality_suspicious_ratio": 0.0,
            "quality_weighted_mae_pre_abs": 10.0,
            "quality_p90_pre_abs": 30.0,
            "quality_weighted_mae_offset": 8.0,
            "quality_p90_offset": 20.0,
            "warnings_delta_sum": 0,
            "improvement_count": 0,
            "status_changed_cases": 0,
        }
        worse_tail = dict(better_tail)
        worse_tail["quality_p90_pre_abs"] = 80.0
        self.assertLess(_rank_key(better_tail), _rank_key(worse_tail))

    def test_format_env_suffix_default(self):
        self.assertEqual(_format_env_suffix({}), "default")

    def test_write_csv_allows_compare_path_and_quality_fields(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "report.csv")
            _write_csv(
                csv_path,
                [
                    {
                        "rank": 1,
                        "run_tag": "run1",
                        "env_suffix": "default",
                        "run_failed": False,
                        "returncode": 0,
                        "regression_count": 0,
                        "improvement_count": 0,
                        "errors_delta_sum": 0,
                        "warnings_delta_sum": 0,
                        "status_changed_cases": 0,
                        "summary_path": "summary.json",
                        "log_path": "run.log",
                        "runtime_config_path": "runtime.yaml",
                        "UTOA_JA_MAPPING_CONF_THRESHOLD": "0.6400",
                        "UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD": "0.3000",
                        "UTOA_KR_MAPPING_CONF_THRESHOLD": "",
                        "UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD": "",
                        "quality_case_count": 1,
                        "quality_compared_case_count": 1,
                        "quality_shared_rows": 10,
                        "quality_suspicious_rows": 2,
                        "quality_suspicious_ratio": 0.2,
                        "quality_mae_pre_abs": 12.5,
                        "quality_mae_offset": 11.0,
                        "compare_path": "compare.json",
                    }
                ],
            )
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["compare_path"], "compare.json")

    def test_build_runtime_batch_config_forces_safe_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "cases.yaml")
            voicebank_dir = os.path.join(td, "voicebank")
            os.makedirs(os.path.join(voicebank_dir, "textgrids"), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    "defaults:\n"
                    "  replace_oto_ini: true\n"
                    "  output_name_template: \"oto.ini\"\n"
                    "cases:\n"
                    "  - name: sample\n"
                    "    language: japanese\n"
                    f"    voicebank_dir: \"{voicebank_dir.replace(os.sep, '/')}\"\n"
                    "    textgrid_dir: textgrids\n"
                    "    replace_oto_ini: true\n"
                    "    output_name: oto.ini\n"
                )

            payload = _build_runtime_batch_config(
                config_path=config_path,
                runtime_dir=os.path.join(td, "runtime"),
                run_tag="run123",
            )
            runtime_doc = payload["config"]
            runtime_case = runtime_doc["cases"][0]

            self.assertFalse(runtime_doc["defaults"]["replace_oto_ini"])
            self.assertFalse(runtime_case["replace_oto_ini"])
            self.assertNotEqual(os.path.normpath(runtime_case["out_dir"]), os.path.normpath(voicebank_dir))
            self.assertEqual(runtime_case["output_name"], "oto.run123.sample.ini")
            self.assertTrue(runtime_case["textgrid_dir"].endswith(os.path.join("voicebank", "textgrids")))

    def test_build_runtime_batch_config_rejects_sanitized_name_collision(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "cases.yaml")
            voicebank_dir = os.path.join(td, "voicebank")
            os.makedirs(os.path.join(voicebank_dir, "textgrids"), exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    "defaults:\n"
                    "  replace_oto_ini: false\n"
                    "cases:\n"
                    "  - name: sample/a\n"
                    "    language: japanese\n"
                    f"    voicebank_dir: \"{voicebank_dir.replace(os.sep, '/')}\"\n"
                    "    textgrid_dir: textgrids\n"
                    "  - name: sample:a\n"
                    "    language: japanese\n"
                    f"    voicebank_dir: \"{voicebank_dir.replace(os.sep, '/')}\"\n"
                    "    textgrid_dir: textgrids\n"
                )

            with self.assertRaisesRegex(ValueError, "generated output collision"):
                _build_runtime_batch_config(
                    config_path=config_path,
                    runtime_dir=os.path.join(td, "runtime"),
                    run_tag="run123",
                )


if __name__ == "__main__":
    unittest.main()
