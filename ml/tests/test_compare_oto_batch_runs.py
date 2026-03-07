import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.compare_oto_batch_runs import compare_runs


def _write_summary(path: str, rows):
    payload = {
        "run_tag": "test",
        "results": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class CompareBatchRunsTests(unittest.TestCase):
    def test_compare_detects_regression_and_improvement(self):
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "base_summary.json")
            new = os.path.join(td, "new_summary.json")

            _write_summary(
                base,
                [
                    {
                        "name": "ja_cvvc",
                        "status": "ok",
                        "validation": {"warnings": 2, "errors": 0, "checked_files": 100, "total_lines": 1200},
                    },
                    {
                        "name": "kr_cvvc",
                        "status": "error",
                        "validation": {"warnings": 3, "errors": 4, "checked_files": 80, "total_lines": 900},
                    },
                ],
            )
            _write_summary(
                new,
                [
                    {
                        "name": "ja_cvvc",
                        "status": "error",
                        "validation": {"warnings": 4, "errors": 1, "checked_files": 100, "total_lines": 1200},
                    },
                    {
                        "name": "kr_cvvc",
                        "status": "ok",
                        "validation": {"warnings": 1, "errors": 0, "checked_files": 80, "total_lines": 900},
                    },
                ],
            )

            report = compare_runs(base_summary=base, new_summary=new)
            agg = report["aggregate"]

            self.assertEqual(int(agg["cases"]), 2)
            self.assertEqual(int(agg["status_changed_cases"]), 2)
            self.assertEqual(int(agg["errors_delta_sum"]), -3)
            self.assertIn("ja_cvvc", agg["regression_cases"])
            self.assertIn("kr_cvvc", agg["improvement_cases"])

    def test_compare_handles_missing_case_in_new_run(self):
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "base_summary.json")
            new = os.path.join(td, "new_summary.json")

            _write_summary(
                base,
                [
                    {
                        "name": "only_in_base",
                        "status": "ok",
                        "validation": {"warnings": 0, "errors": 0},
                    }
                ],
            )
            _write_summary(new, [])

            report = compare_runs(base_summary=base, new_summary=new)
            case = report["case_diffs"]["only_in_base"]
            self.assertEqual(case["status_new"], "missing")
            self.assertTrue(case["status_changed"])
            self.assertIn("only_in_base", report["aggregate"]["regression_cases"])

    def test_compare_rejects_duplicate_case_name(self):
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "base_summary.json")
            new = os.path.join(td, "new_summary.json")

            _write_summary(
                base,
                [
                    {"name": "dup", "status": "ok", "validation": {"warnings": 0, "errors": 0}},
                    {"name": "dup", "status": "ok", "validation": {"warnings": 0, "errors": 0}},
                ],
            )
            _write_summary(new, [])

            with self.assertRaisesRegex(ValueError, "duplicate case name"):
                compare_runs(base_summary=base, new_summary=new)

    def test_compare_rejects_duplicate_output_oto(self):
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "base_summary.json")
            new = os.path.join(td, "new_summary.json")

            _write_summary(
                base,
                [
                    {
                        "name": "case_a",
                        "status": "ok",
                        "output_oto": os.path.join(td, "same.ini"),
                        "validation": {"warnings": 0, "errors": 0},
                    },
                    {
                        "name": "case_b",
                        "status": "ok",
                        "output_oto": os.path.join(td, "same.ini"),
                        "validation": {"warnings": 0, "errors": 0},
                    },
                ],
            )
            _write_summary(new, [])

            with self.assertRaisesRegex(ValueError, "duplicate output_oto path"):
                compare_runs(base_summary=base, new_summary=new)


if __name__ == "__main__":
    unittest.main()
