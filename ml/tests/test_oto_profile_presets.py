import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_profile_presets import get_kr_profile_preset
from core.oto_generator import _default_kr_profile_cache_path


class OtoProfilePresetTests(unittest.TestCase):
    def test_get_kr_profile_preset_returns_cvc_specific_profile(self):
        preset = get_kr_profile_preset("cvc")
        self.assertEqual(preset.get("format_type"), "cvc")
        self.assertEqual(preset.get("preset_name"), "kr_cvc_reference_v3")
        self.assertIn("vc", preset.get("buckets", {}))

    def test_kr_profile_cache_path_is_format_specific(self):
        cvc_path = _default_kr_profile_cache_path("cvc")
        cvvc_path = _default_kr_profile_cache_path("cvvc")
        self.assertNotEqual(cvc_path, cvvc_path)
        self.assertTrue(cvc_path.endswith("kr_oto_reference_profile_cvc.json"))
        self.assertTrue(cvvc_path.endswith("kr_oto_reference_profile_cvvc.json"))


if __name__ == "__main__":
    unittest.main()
