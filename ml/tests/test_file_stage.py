import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_file_stage import try_handle_ja_single_vowel_file
from core.kr_file_stage import try_handle_kr_single_vowel_file


class FileStageTests(unittest.TestCase):
    def test_try_handle_kr_single_vowel_file_appends_rows(self):
        final_lines = []
        recorded = []

        def _append(final_lines, wav, alias, offset, consonant, cutoff, pre, ovl, **kwargs):
            final_lines.append(f"{wav}={alias},{offset:.1f},{consonant:.1f},{cutoff:.1f},{pre:.1f},{ovl:.1f}")

        handled = try_handle_kr_single_vowel_file(
            fname="a.wav",
            lines=["a.wav=a,0,0,0,0,0"],
            real_wav_name="a.wav",
            ph_intervals=[SimpleNamespace(minTime=0.1, maxTime=0.3)],
            wd_intervals=[SimpleNamespace(minTime=0.1, maxTime=0.3)],
            final_lines=final_lines,
            log_fn=lambda msg: None,
            record_unset_fn=lambda *args: recorded.append(args),
            validate_fn=lambda o, c, ct, p, ov: (o, c, ct, p, ov),
            append_alias_rows_fn=_append,
            apply_suffix_to_oto_line_fn=lambda line, suffix: line,
            generate_openutau=False,
            alias_suffix="",
        )
        self.assertTrue(handled)
        self.assertEqual(recorded, [])
        self.assertEqual(len(final_lines), 1)
        self.assertTrue(final_lines[0].startswith("a.wav=a,"))
        parts = final_lines[0].split("=", 1)[1].split(",")
        offset = float(parts[1])
        consonant = float(parts[2])
        cutoff = float(parts[3])
        pre = float(parts[4])
        ovl = float(parts[5])
        self.assertLess(offset, 100.0)
        self.assertGreater(pre, 0.0)
        self.assertGreater(ovl, 0.0)
        self.assertLess(ovl, pre)
        self.assertGreater(abs(cutoff), consonant)

    def test_try_handle_ja_single_vowel_file_appends_rows(self):
        final_lines = []
        recorded = []
        post_ctx = SimpleNamespace(post_adjust=lambda *args, **kwargs: (100.0, 50.0, -160.0, 0.0, 0.0))
        handled = try_handle_ja_single_vowel_file(
            fname="a.wav",
            lines=["a.wav=ka,0,0,0,0,0"],
            real_wav_name="a.wav",
            ph_intervals=[SimpleNamespace(minTime=0.1, maxTime=0.3)],
            final_lines=final_lines,
            log_fn=lambda msg: None,
            record_unset_fn=lambda *args: recorded.append(args),
            post_ctx=post_ctx,
            alias_out_fn=lambda a: f"out_{a}",
            apply_suffix_to_oto_line_fn=lambda line, suffix: f"{line}{suffix}",
            generate_openutau=True,
            generate_openutau_aliases_fn=lambda alias: [alias, alias + "2"],
            alias_suffix="_C4",
        )
        self.assertTrue(handled)
        self.assertEqual(recorded, [])
        self.assertEqual(final_lines, [
            "a.wav=out_ka,100.00,50.00,-160.00,0.00,0.00",
            "a.wav=out_ka2,100.00,50.00,-160.00,0.00,0.00",
        ])


if __name__ == "__main__":
    unittest.main()
