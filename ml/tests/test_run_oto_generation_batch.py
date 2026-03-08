import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.run_oto_generation_batch import (
    _collect_preflight_issues,
    _resolve_case_path,
    _resolve_case_settings,
    _write_preflight_report,
    _write_summary_text,
)


class RunOtoGenerationBatchTests(unittest.TestCase):
    def test_resolve_case_path_prefers_voicebank_candidate_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            config_dir = os.path.join(td, "config")
            voicebank_dir = os.path.join(td, "voicebank")
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(voicebank_dir, exist_ok=True)
            resolved = _resolve_case_path(config_dir, voicebank_dir, "textgrids")
            self.assertEqual(resolved, os.path.join(voicebank_dir, "textgrids"))

    def test_collect_preflight_issues_detects_output_collision(self):
        case_infos = [
            {
                "name": "a",
                "enabled": True,
                "language": "japanese",
                "voicebank_dir": __file__,
                "textgrid_dir": __file__,
                "tpl_path": "",
                "custom_phonemes_path": "",
                "output_oto": "same.ini",
                "replace_oto_ini": False,
                "replace_target_oto": "",
            },
            {
                "name": "b",
                "enabled": True,
                "language": "japanese",
                "voicebank_dir": __file__,
                "textgrid_dir": __file__,
                "tpl_path": "",
                "custom_phonemes_path": "",
                "output_oto": "same.ini",
                "replace_oto_ini": False,
                "replace_target_oto": "",
            },
        ]
        issues = _collect_preflight_issues(case_infos)
        self.assertTrue(any("output collision" in msg for msg in issues["errors"]))

    def test_resolve_case_settings_keeps_voicebank_relative_textgrid(self):
        with tempfile.TemporaryDirectory() as td:
            voicebank_dir = os.path.join(td, "voicebank")
            os.makedirs(voicebank_dir, exist_ok=True)
            case = {
                "name": "sample",
                "language": "japanese",
                "voicebank_dir": voicebank_dir,
                "textgrid_dir": "textgrids",
            }
            info = _resolve_case_settings(
                case=case,
                defaults={},
                config_dir=td,
                run_tag="run1",
                case_index=1,
            )
            self.assertEqual(info["textgrid_dir"], os.path.join(voicebank_dir, "textgrids"))

    def test_collect_preflight_issues_allows_missing_textgrid_when_runtime_alignment_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            voicebank_dir = os.path.join(td, "voicebank")
            os.makedirs(voicebank_dir, exist_ok=True)
            case_infos = [
                {
                    "name": "a",
                    "enabled": True,
                    "language": "japanese",
                    "voicebank_dir": voicebank_dir,
                    "textgrid_dir": os.path.join(voicebank_dir, "textgrids"),
                    "tpl_path": "",
                    "custom_phonemes_path": "",
                    "output_oto": os.path.join(voicebank_dir, "oto.test.ini"),
                    "replace_oto_ini": False,
                    "replace_target_oto": "",
                    "align_if_missing": True,
                }
            ]
            issues = _collect_preflight_issues(case_infos)
            self.assertFalse(issues["errors"])
            self.assertTrue(any("align_if_missing=true" in msg for msg in issues["warnings"]))

    def test_write_preflight_report_and_summary_text(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run")
            os.makedirs(run_dir, exist_ok=True)
            preflight_path = _write_preflight_report(
                run_dir=run_dir,
                config_path=os.path.join(td, "config.yaml"),
                case_infos=[{"name": "sample"}],
                issues={"errors": ["x"], "warnings": []},
            )
            with open(preflight_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["error_count"], 1)

            summary_txt = _write_summary_text(
                os.path.join(run_dir, "summary.json"),
                {
                    "config": "cfg",
                    "run_tag": "tag",
                    "run_dir": "dir",
                    "total_cases": 1,
                    "ok_cases": 1,
                    "error_cases": 0,
                    "skipped_cases": 0,
                    "results": [
                        {
                            "name": "sample",
                            "status": "ok",
                            "reason": "",
                            "failure_code": "OK",
                            "output_oto": "oto.ini",
                            "alignment": {"code": "OK", "used_engine": "sofa", "fallback_used": True},
                            "ml": {"code": "ML_MODEL_MISSING", "status": "fallback", "fallback_used": True},
                            "fallback_path": ["align:mfa->sofa", "ml:on->base"],
                            "validation": {"checked_files": 3, "warnings": 1, "errors": 0},
                        }
                    ],
                },
            )
            with open(summary_txt, "r", encoding="utf-8") as f:
                text = f.read()
            self.assertIn("[ok] sample", text)
            self.assertIn("output_oto=oto.ini", text)
            self.assertIn("failure_code=OK", text)
            self.assertIn("alignment=code=OK used_engine=sofa fallback_used=True", text)
            self.assertIn("ml=code=ML_MODEL_MISSING status=fallback fallback_used=True", text)
            self.assertIn("fallback_path=align:mfa->sofa,ml:on->base", text)


if __name__ == "__main__":
    unittest.main()
