import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_oto_file_finalize import apply_kr_mel_refine_to_oto_file


def _validate(offset, consonant, cutoff, pre, ovl):
    return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)


class KrOtoFileFinalizeTests(unittest.TestCase):
    def test_apply_kr_mel_refine_returns_zero_without_valid_wav_dir(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=ga,100,180,-260,90,30\n")
            changed = apply_kr_mel_refine_to_oto_file(
                oto_path,
                os.path.join(td, "missing"),
                validate_fn=_validate,
                normalize_key_fn=str.lower,
            )
            self.assertEqual(changed, 0)


if __name__ == "__main__":
    unittest.main()
