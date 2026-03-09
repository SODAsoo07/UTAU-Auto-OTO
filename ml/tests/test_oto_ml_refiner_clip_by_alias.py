import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_refiner import _clip_delta


class OtoMlRefinerClipByAliasTests(unittest.TestCase):
    def test_clip_delta_uses_alias_type_limits_for_korean(self):
        # cv offset clip is narrower than the generic KR clip
        v = _clip_delta("korean", "delta_offset", 200.0, alias_type="cv")
        self.assertEqual(v, 160.0)

        # vc offset clip still allows 200
        v_vc = _clip_delta("korean", "delta_offset", 200.0, alias_type="vc")
        self.assertEqual(v_vc, 200.0)

    def test_clip_delta_falls_back_to_generic_when_alias_type_missing(self):
        v = _clip_delta("korean", "delta_offset", 200.0)
        self.assertEqual(v, 200.0)

    def test_clip_delta_keeps_japanese_generic_limits(self):
        # Japanese clip table does not split by alias type.
        v = _clip_delta("japanese", "delta_offset", 300.0, alias_type="cv")
        self.assertEqual(v, 260.0)


if __name__ == "__main__":
    unittest.main()
