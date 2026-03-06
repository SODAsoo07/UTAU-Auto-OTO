import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.tune_alignment_thresholds import (
    _build_env_overrides,
    _format_env_suffix,
    _parse_float_grid,
    _rank_key,
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
            "warnings_delta_sum": 0,
            "improvement_count": 1,
            "status_changed_cases": 0,
        }
        bad = {
            "run_failed": False,
            "regression_count": 1,
            "errors_delta_sum": -5,
            "warnings_delta_sum": -3,
            "improvement_count": 2,
            "status_changed_cases": 0,
        }
        self.assertLess(_rank_key(good), _rank_key(bad))

    def test_format_env_suffix_default(self):
        self.assertEqual(_format_env_suffix({}), "default")


if __name__ == "__main__":
    unittest.main()
