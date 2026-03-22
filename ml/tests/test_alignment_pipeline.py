import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.alignment_pipeline import run_alignment_with_fallback
from core.pipeline_status import ALIGN_SKIPPED, normalize_aligner_name, resolve_aligner_chain


class AlignmentPipelineTests(unittest.TestCase):
    def test_normalize_aligner_name_accepts_none_aliases(self):
        for raw in ("none", "off", "skip", "disabled", "no_align", "nomfa", "no_mfa", "No-MFA (Experimental)"):
            self.assertEqual(normalize_aligner_name(raw, default="mfa"), "none")

    def test_resolve_aligner_chain_keeps_none(self):
        self.assertEqual(resolve_aligner_chain("none", ""), ["none"])
        self.assertEqual(resolve_aligner_chain("none", "mfa"), ["none", "mfa"])

    def test_run_alignment_none_bypasses_mfa(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("core.alignment_pipeline.check_mfa_ready") as mocked_ready:
                with mock.patch("core.alignment_pipeline.run_mfa_align") as mocked_align:
                    result = run_alignment_with_fallback(
                        language="korean",
                        wav_folder=td,
                        dictionary_path=os.path.join(td, "dictionary.txt"),
                        output_folder=os.path.join(td, "textgrids"),
                        primary_aligner="none",
                        fallback_aligner="mfa",
                        mfa_path="",
                        mfa_align_profile="default",
                        callback=None,
                    )
        mocked_ready.assert_not_called()
        mocked_align.assert_not_called()
        self.assertTrue(bool(result.get("ok", False)))
        self.assertEqual(str(result.get("code", "")), ALIGN_SKIPPED)
        self.assertEqual(str(result.get("used_engine", "")), "none")

    def test_unknown_primary_still_falls_back_to_mfa(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch(
                "core.alignment_pipeline.check_mfa_ready",
                return_value={"code": "ALIGN_NOT_READY", "message": "not ready", "mfa_path": ""},
            ) as mocked_ready:
                result = run_alignment_with_fallback(
                    language="korean",
                    wav_folder=td,
                    dictionary_path=os.path.join(td, "dictionary.txt"),
                    output_folder=os.path.join(td, "textgrids"),
                    primary_aligner="unknown_engine",
                    fallback_aligner="",
                    mfa_path="",
                    mfa_align_profile="default",
                    callback=None,
                )
        mocked_ready.assert_called_once()
        self.assertFalse(bool(result.get("ok", True)))
        self.assertEqual(str(result.get("primary_engine", "")), "mfa")


if __name__ == "__main__":
    unittest.main()
