import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.app_mixins import AppRuntimeMixin


class _DummyRuntime(AppRuntimeMixin):
    _is_closing = False


class UiMlSelectorModeTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("UTOA_FORCE_OTO_SELECTOR", None)
        os.environ.pop("UTOA_DISABLE_OTO_SELECTOR", None)
        self.runtime = _DummyRuntime()

    def tearDown(self):
        os.environ.pop("UTOA_FORCE_OTO_SELECTOR", None)
        os.environ.pop("UTOA_DISABLE_OTO_SELECTOR", None)

    def test_selector_mode_policy_clears_overrides(self):
        os.environ["UTOA_FORCE_OTO_SELECTOR"] = "1"
        self.assertEqual(self.runtime._apply_ml_selector_runtime_mode("기본 정책"), "policy")
        self.assertNotIn("UTOA_FORCE_OTO_SELECTOR", os.environ)
        self.assertNotIn("UTOA_DISABLE_OTO_SELECTOR", os.environ)

    def test_selector_mode_delta_sets_disable_flag(self):
        self.assertEqual(self.runtime._apply_ml_selector_runtime_mode("델타만"), "delta")
        self.assertEqual(os.environ.get("UTOA_DISABLE_OTO_SELECTOR"), "1")
        self.assertNotIn("UTOA_FORCE_OTO_SELECTOR", os.environ)

    def test_selector_mode_selector_sets_force_flag(self):
        self.assertEqual(self.runtime._apply_ml_selector_runtime_mode("델타+셀렉터"), "selector")
        self.assertEqual(os.environ.get("UTOA_FORCE_OTO_SELECTOR"), "1")
        self.assertNotIn("UTOA_DISABLE_OTO_SELECTOR", os.environ)


if __name__ == "__main__":
    unittest.main()
