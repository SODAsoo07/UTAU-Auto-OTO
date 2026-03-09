import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_row_output_v2 import prepare_oto_alias_rows


class OtoRowOutputV2Tests(unittest.TestCase):
    def test_prepare_oto_alias_rows_generates_multiple_alias_rows(self):
        params, rows = prepare_oto_alias_rows(
            "a.wav",
            "ka",
            10.0,
            50.0,
            -90.0,
            30.0,
            20.0,
            generate_openutau=True,
            generate_aliases_fn=lambda alias: [alias, f"- {alias}"],
            alias_transform_fn=lambda alias: alias.upper(),
        )
        self.assertEqual(params, (10.0, 50.0, -90.0, 30.0, 20.0))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].startswith("a.wav=KA,10.00"))

    def test_prepare_oto_alias_rows_applies_pre_write_adjust(self):
        params, rows = prepare_oto_alias_rows(
            "b.wav",
            "ga",
            10.0,
            60.0,
            -100.0,
            35.0,
            18.0,
            generate_openutau=False,
            generate_aliases_fn=lambda alias: [alias],
            alias_transform_fn=lambda alias: alias,
            pre_write_adjust_fn=lambda offset, consonant, cutoff, pre, ovl: (
                offset + 5.0,
                consonant + 10.0,
                cutoff - 20.0,
                pre + 2.0,
                ovl + 1.0,
            ),
        )
        self.assertEqual(params, (15.0, 70.0, -120.0, 37.0, 19.0))
        self.assertEqual(rows, ["b.wav=ga,15.00,70.00,-120.00,37.00,19.00"])


if __name__ == "__main__":
    unittest.main()
