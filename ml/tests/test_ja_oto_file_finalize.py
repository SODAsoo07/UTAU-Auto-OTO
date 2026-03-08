import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_oto_file_finalize import apply_ja_mel_refine_to_oto_file


def _validate(offset, consonant, cutoff, pre, ovl):
    return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)


def _classify(alias, _custom_map=None):
    return "cv"


class JaOtoFileFinalizeTests(unittest.TestCase):
    def test_apply_ja_mel_refine_returns_zero_without_valid_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=ka,100,180,-260,90,30\n")
            changed = apply_ja_mel_refine_to_oto_file(
                oto_path,
                os.path.join(td, "missing_wavs"),
                validate_fn=_validate,
                classify_alias_fn=_classify,
            )
            self.assertEqual(changed, 0)


if __name__ == "__main__":
    unittest.main()
