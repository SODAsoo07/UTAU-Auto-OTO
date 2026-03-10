import os
import sys
import tempfile
import unittest
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.kr_oto_file_finalize import (
    apply_kr_mel_refine_to_oto_file,
    apply_kr_wav_duration_safety_to_oto_file,
)


def _validate(offset, consonant, cutoff, pre, ovl):
    return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)


def _write_silence_wav(path, sr=44100, duration_ms=1000):
    n = int(sr * (duration_ms / 1000.0))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n)


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

    def test_apply_kr_wav_duration_safety_clamps_invalid_rows(self):
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "a.wav")
            oto_path = os.path.join(td, "oto.ini")
            _write_silence_wav(wav_path, duration_ms=1000)
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=a my,1200,440,-595,360,265\n")

            changed = apply_kr_wav_duration_safety_to_oto_file(
                oto_path,
                td,
                custom_map=None,
                validate_fn=_validate,
                normalize_key_fn=str.lower,
            )
            self.assertGreater(changed, 0)

            with open(oto_path, "r", encoding="utf-8") as rf:
                line = rf.read().strip()
            rhs = line.split("=", 1)[1]
            parts = rhs.split(",")
            offset = float(parts[1])
            cons = float(parts[2])
            cutoff = float(parts[3])
            pre = float(parts[4])
            ovl = float(parts[5])
            active_len = 1000.0 - offset - abs(cutoff)

            self.assertLessEqual(offset, 1000.0 - 16.0 + 1e-6)
            self.assertGreaterEqual(active_len, 3.5)
            self.assertLessEqual(pre, cons + 1e-6)
            self.assertLessEqual(ovl, pre + 1e-6)


if __name__ == "__main__":
    unittest.main()
