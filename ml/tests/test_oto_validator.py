import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_validator import _should_skip_validation_row


class OtoValidatorTests(unittest.TestCase):
    def test_skips_blank_alias_rows(self):
        self.assertTrue(
            _should_skip_validation_row(
                {
                    "alias": "",
                    "cons": 120.0,
                    "cutoff": -220.0,
                    "pre": 90.0,
                    "ovl": 50.0,
                }
            )
        )

    def test_skips_zeroed_tail_placeholder_rows(self):
        self.assertTrue(
            _should_skip_validation_row(
                {
                    "alias": "a b",
                    "cons": 6.0,
                    "cutoff": -4.0,
                    "pre": 0.0,
                    "ovl": 0.0,
                }
            )
        )

    def test_keeps_regular_alias_rows(self):
        self.assertFalse(
            _should_skip_validation_row(
                {
                    "alias": "a b",
                    "cons": 138.0,
                    "cutoff": -146.0,
                    "pre": 92.0,
                    "ovl": 70.0,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
