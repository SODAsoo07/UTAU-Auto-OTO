import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.alignment_pipeline import run_alignment_with_fallback
from core.pipeline_status import normalize_aligner_name, resolve_aligner_chain


class AlignmentPipelineTests(unittest.TestCase):
    def test_normalize_aligner_name_rejects_none_aliases(self):
        for raw in ("none", "off", "skip", "disabled", "no_align", "nomfa", "no_mfa", "No-MFA (Experimental)"):
            self.assertEqual(normalize_aligner_name(raw, default="mfa"), "mfa")

    def test_resolve_aligner_chain_converts_none_to_mfa(self):
        self.assertEqual(resolve_aligner_chain("none", ""), ["mfa"])
        self.assertEqual(resolve_aligner_chain("none", "mfa"), ["mfa"])

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
