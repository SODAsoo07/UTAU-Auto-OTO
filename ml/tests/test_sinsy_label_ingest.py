import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.sinsy_label_ingest import (
    SinsyLabelEntry,
    build_sinsy_guided_anchor_plan,
    load_sinsy_label_entries,
)


class SinsyLabelIngestTests(unittest.TestCase):
    def test_load_sinsy_label_entries_scales_hts_time_units_to_ms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.lab")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("0 500000 go\n")
                handle.write("500000 1000000 da\n")

            entries = load_sinsy_label_entries(explicit_path=path)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].label, "go")
        self.assertAlmostEqual(entries[0].start_ms, 0.0)
        self.assertAlmostEqual(entries[0].end_ms, 50.0)
        self.assertAlmostEqual(entries[1].start_ms, 50.0)
        self.assertAlmostEqual(entries[1].end_ms, 100.0)

    def test_build_sinsy_guided_anchor_plan_maps_labels_to_real_syllables(self):
        syllables_info = [
            {"start_time": 0.00, "end_time": 0.05},
            {"start_time": 0.05, "end_time": 0.10},
            {"start_time": 0.10, "end_time": 0.15},
        ]
        labels = [
            SinsyLabelEntry(start_ms=0.0, end_ms=48.0, label="go"),
            SinsyLabelEntry(start_ms=49.0, end_ms=98.0, label="da"),
            SinsyLabelEntry(start_ms=101.0, end_ms=148.0, label="ra"),
        ]

        plan = build_sinsy_guided_anchor_plan(
            expected_tokens=["go", "da", "ra"],
            syllables_info=syllables_info,
            label_entries=labels,
            normalize_expected_fn=lambda value: str(value or "").strip().lower(),
            normalize_label_fn=lambda value: str(value or "").strip().lower(),
            label_match_score_fn=lambda target, label: 32.0 if target == label else -16.0,
        )

        self.assertEqual(plan["indices"], [0, 1, 2])
        self.assertEqual(plan["source"], "sinsy_labels")
        self.assertTrue(plan["meta"]["available"])
        self.assertGreater(plan["meta"]["margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
