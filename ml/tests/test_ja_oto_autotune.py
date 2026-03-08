import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.ja_oto_autotune import (
    JaAutotuneRefreshContext,
    _profile_path_for_out,
    apply_ja_autotune_delta,
    apply_ja_autotune_profile_to_oto,
    refresh_ja_autotune_profile_after_generation,
    save_ja_autotune_profile,
)


def _validate(offset, consonant, cutoff, pre, ovl):
    return float(offset), float(consonant), float(cutoff), float(pre), float(ovl)


class JaOtoAutotuneTests(unittest.TestCase):
    def test_profile_path_for_out_places_hidden_profile_in_output_dir(self):
        path = _profile_path_for_out(r"C:\tmp\voice\oto.ini")
        self.assertTrue(path.endswith(r"voice\.ja_oto_autotune_profile.json"))

    def test_apply_ja_autotune_delta_uses_bucket_values(self):
        profile = {
            "buckets": {
                "cv": {
                    "n": 12,
                    "offset": 20.0,
                    "cons": 10.0,
                    "cutoff": -15.0,
                    "pre": 8.0,
                    "ovl": 5.0,
                }
            }
        }
        out = apply_ja_autotune_delta(
            "cv",
            100.0,
            180.0,
            -260.0,
            90.0,
            30.0,
            profile,
            validate_fn=_validate,
        )
        self.assertGreater(out[0], 100.0)
        self.assertGreater(out[1], 180.0)
        self.assertLess(out[2], -260.0)

    def test_apply_ja_autotune_profile_to_oto_accepts_saved_profile(self):
        with tempfile.TemporaryDirectory() as td:
            oto_path = os.path.join(td, "oto.ini")
            profile_path = os.path.join(td, "profile.json")
            with open(oto_path, "w", encoding="utf-8") as f:
                f.write("a.wav=ka,100,180,-260,90,30\n")
            profile = {"buckets": {"cv": {"n": 12, "offset": 10.0, "cons": 5.0, "cutoff": -8.0, "pre": 4.0, "ovl": 3.0}}}
            self.assertTrue(save_ja_autotune_profile(profile_path, profile))
            changed = apply_ja_autotune_profile_to_oto(
                oto_path,
                profile_path,
                validate_fn=_validate,
            )
            self.assertEqual(changed, 1)

    def test_refresh_ja_autotune_profile_after_generation_returns_none_without_reference(self):
        logs = []
        with tempfile.TemporaryDirectory() as td:
            ctx = JaAutotuneRefreshContext(
                out_path=os.path.join(td, "oto.ini"),
                profile_path=os.path.join(td, "profile.json"),
                had_profile_on_start=True,
                custom_map=None,
                log_fn=logs.append,
                validate_fn=_validate,
            )
            result = refresh_ja_autotune_profile_after_generation(ctx)
        self.assertIsNone(result)
        self.assertEqual(logs, [])

    def test_refresh_ja_autotune_profile_after_generation_bootstraps_profile(self):
        logs = []
        trained = {"buckets": {"cv": {"n": 8}}, "matched_pairs": 12}
        with tempfile.TemporaryDirectory() as td:
            ctx = JaAutotuneRefreshContext(
                out_path=os.path.join(td, "oto.ini"),
                profile_path=os.path.join(td, "profile.json"),
                had_profile_on_start=False,
                custom_map=None,
                log_fn=logs.append,
                validate_fn=_validate,
            )
            with mock.patch("core.ja_oto_autotune._find_reference_oto", return_value=os.path.join(td, "ref.ini")), \
                mock.patch("core.ja_oto_autotune._train_ja_autotune_profile", return_value=trained), \
                mock.patch("core.ja_oto_autotune.save_ja_autotune_profile", return_value=True), \
                mock.patch("core.ja_oto_autotune._apply_profile_to_oto_file", return_value=3) as apply_mock:
                result = refresh_ja_autotune_profile_after_generation(ctx)
        self.assertIs(result, trained)
        apply_mock.assert_called_once()
        self.assertTrue(any("profile updated" in line for line in logs))
        self.assertTrue(any("bootstrap profile applied: 3 lines adjusted" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
