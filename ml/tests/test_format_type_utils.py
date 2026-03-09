import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.format_type_utils import (
    normalize_auto_format_value,
    normalize_format_type,
    normalize_language_name,
)


class FormatTypeUtilsTests(unittest.TestCase):
    def test_normalize_language_name_maps_short_codes(self):
        self.assertEqual(normalize_language_name("jp"), "japanese")
        self.assertEqual(normalize_language_name("ja"), "japanese")
        self.assertEqual(normalize_language_name("kr"), "korean")
        self.assertEqual(normalize_language_name("ko"), "korean")

    def test_normalize_format_type_keeps_korean_cvc_distinct(self):
        self.assertEqual(normalize_format_type("korean", "mono"), "cv")
        self.assertEqual(normalize_format_type("korean", "cv_simple"), "cv")
        self.assertEqual(normalize_format_type("korean", "CV_and_연단음"), "cv")
        self.assertEqual(normalize_format_type("korean", "cvc"), "cvc")
        self.assertEqual(normalize_format_type("korean", "CVC/연단음"), "cvc")
        self.assertEqual(normalize_format_type("japanese", "cvc"), "cv")
        self.assertEqual(normalize_format_type("japanese", "CV_and_Rentan"), "cv")

    def test_normalize_auto_format_value_keeps_korean_cvc_bucket(self):
        self.assertEqual(normalize_auto_format_value("korean", "cvc"), "cvc")
        self.assertEqual(normalize_auto_format_value("korean", "CVC (한국어 전용)"), "cvc")
        self.assertEqual(normalize_auto_format_value("korean", "CV (단독음)"), "cv")
        self.assertEqual(normalize_auto_format_value("korean", "CV/연단음"), "cv")
        self.assertEqual(normalize_auto_format_value("korean", "CVC"), "cvc")
        self.assertEqual(normalize_auto_format_value("korean", "CVC/연단음"), "cvc")
        self.assertEqual(normalize_auto_format_value("japanese", "CV (단독음)"), "cv")
        self.assertEqual(normalize_auto_format_value("japanese", "CV/연단음"), "cv")
        self.assertEqual(normalize_auto_format_value("japanese", "CVC"), "cv")
        self.assertEqual(normalize_auto_format_value("japanese", "CVC/연단음"), "cv")


if __name__ == "__main__":
    unittest.main()
