import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_oto_file_consistency import (
    apply_file_level_validation,
    enforce_adjacent_continuity,
    smooth_abrupt_changes,
)


def _validate(offset, consonant, cutoff, pre, ovl, alias_type=""):
    offset = max(float(offset), 0.0)
    pre = max(float(pre), 0.0)
    ovl = max(0.0, min(float(ovl), max(pre - 1.0, 0.0)))
    min_cons_gap = 10.0 if str(alias_type or "").strip().lower() == "vc" else 14.0
    consonant = max(float(consonant), pre + min_cons_gap)
    cutoff_abs = max(abs(float(cutoff)), consonant + 8.0)
    return offset, consonant, -cutoff_abs, pre, ovl


class OtoFileConsistencyTests(unittest.TestCase):
    def test_enforce_adjacent_continuity_clamps_excess_overlap(self):
        rows = [
            {"alias": "ga", "offset": 100.0, "cons": 150.0, "cutoff": -320.0, "pre": 80.0, "ovl": 30.0},
            {"alias": "da", "offset": 350.0, "cons": 120.0, "cutoff": -200.0, "pre": 70.0, "ovl": 20.0},
        ]
        row_types = ["cv", "cv"]
        changed = enforce_adjacent_continuity(rows, row_types, _validate)
        self.assertGreaterEqual(changed, 1)
        overlap = (rows[0]["offset"] + abs(rows[0]["cutoff"])) - rows[1]["offset"]
        self.assertLessEqual(overlap, 50.0 + 1e-6)

    def test_enforce_adjacent_continuity_clamps_large_gap(self):
        rows = [
            {"alias": "ga", "offset": 100.0, "cons": 140.0, "cutoff": -180.0, "pre": 70.0, "ovl": 20.0},
            {"alias": "da", "offset": 400.0, "cons": 120.0, "cutoff": -210.0, "pre": 65.0, "ovl": 18.0},
        ]
        row_types = ["cv", "cv"]
        changed = enforce_adjacent_continuity(rows, row_types, _validate)
        self.assertGreaterEqual(changed, 1)
        overlap = (rows[0]["offset"] + abs(rows[0]["cutoff"])) - rows[1]["offset"]
        self.assertGreaterEqual(overlap, -80.0 - 1e-6)

    def test_smooth_abrupt_changes_smooths_cutoff_and_keeps_group_rule(self):
        rows = [
            {"alias": "ga", "offset": 100.0, "cons": 95.0, "cutoff": -160.0, "pre": 60.0, "ovl": 32.0},
            {"alias": "ga2", "offset": 100.0, "cons": 220.0, "cutoff": -360.0, "pre": 150.0, "ovl": 95.0},
        ]
        row_types = ["cv", "cv"]
        changed = smooth_abrupt_changes(rows, row_types, _validate)
        self.assertGreaterEqual(changed, 1)
        self.assertLess(abs(rows[1]["cutoff"]), 360.0)
        self.assertLess(rows[1]["pre"], 150.0)

        rows_other_group = [
            {"alias": "ga", "offset": 100.0, "cons": 95.0, "cutoff": -160.0, "pre": 60.0, "ovl": 32.0},
            {"alias": "ma", "offset": 100.0, "cons": 220.0, "cutoff": -360.0, "pre": 150.0, "ovl": 95.0},
        ]
        changed_other = smooth_abrupt_changes(rows_other_group, row_types, _validate)
        self.assertEqual(changed_other, 0)

    def test_apply_file_level_validation_enforces_order(self):
        rows = [
            {"alias": "ga", "offset": -5.0, "cons": 40.0, "cutoff": -30.0, "pre": 50.0, "ovl": 60.0},
        ]
        row_types = ["cv"]
        changed = apply_file_level_validation(rows, row_types, _validate)
        self.assertEqual(changed, 1)
        row = rows[0]
        self.assertGreaterEqual(row["offset"], 0.0)
        self.assertLess(row["ovl"], row["pre"] + 1e-6)
        self.assertGreaterEqual(row["cons"], row["pre"])
        self.assertGreater(abs(row["cutoff"]), row["cons"])


if __name__ == "__main__":
    unittest.main()
