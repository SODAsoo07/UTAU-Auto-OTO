import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.layout_mixin import LayoutMixin


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _DummyLayout(LayoutMixin):
    def __init__(self, value):
        self.lang_var = _DummyVar(value)


class UiLanguageDetectionTests(unittest.TestCase):
    def test_detects_japanese_from_english_label(self):
        self.assertEqual(_DummyLayout("Japanese (日本語)")._get_language(), "japanese")

    def test_detects_japanese_from_japanese_label(self):
        self.assertEqual(_DummyLayout("日本語")._get_language(), "japanese")

    def test_detects_korean_from_english_label(self):
        self.assertEqual(_DummyLayout("Korean (한국어)")._get_language(), "korean")


if __name__ == "__main__":
    unittest.main()
