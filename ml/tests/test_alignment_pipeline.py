import os
import sys
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import alignment_pipeline
from core.alignment_pipeline import run_alignment_with_fallback
from core.pipeline_status import ALIGN_EXEC_MISSING, ALIGN_NOT_READY, OK, normalize_aligner_name, resolve_aligner_chain


class AlignmentPipelineTests(unittest.TestCase):
    def test_normalize_aligner_name_accepts_none_aliases(self):
        for raw in ("none", "off", "skip", "nomfa", "no_mfa", "No-MFA (Experimental)"):
            self.assertEqual(normalize_aligner_name(raw, default="mfa"), "none")

    def test_normalize_aligner_name_keeps_unknown_as_default(self):
        for raw in ("disabled", "no_align"):
            self.assertEqual(normalize_aligner_name(raw, default="mfa"), "mfa")

    def test_normalize_aligner_name_accepts_domino_aliases(self):
        for raw in ("domino", "pydomino", "Domino (JP)", "jp-domino"):
            self.assertEqual(normalize_aligner_name(raw, default="mfa"), "domino")

    def test_resolve_aligner_chain_keeps_none_then_optional_fallback(self):
        self.assertEqual(resolve_aligner_chain("none", ""), ["none"])
        self.assertEqual(resolve_aligner_chain("none", "mfa"), ["none", "mfa"])

    def test_resolve_aligner_chain_keeps_domino_then_mfa(self):
        self.assertEqual(resolve_aligner_chain("domino", "mfa"), ["domino", "mfa"])

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

    def test_domino_primary_falls_back_to_mfa_when_domino_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = os.path.join(td, "textgrids")

            def _fake_mfa_align(_mfa_path, _wav_folder, _dict_path, out_dir, **_kwargs):
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "sample.TextGrid"), "w", encoding="utf-8") as handle:
                    handle.write('File type = "ooTextFile"\n')
                return True, ""

            with mock.patch(
                "core.alignment_pipeline.check_domino_ready",
                return_value={"code": ALIGN_NOT_READY, "message": "domino missing", "domino_onnx_path": ""},
            ) as mocked_domino_ready, mock.patch(
                "core.alignment_pipeline.check_mfa_ready",
                return_value={"code": OK, "message": "ready", "mfa_path": "fake_mfa"},
            ) as mocked_mfa_ready, mock.patch(
                "core.alignment_pipeline.run_mfa_align",
                side_effect=_fake_mfa_align,
            ) as mocked_mfa_align:
                result = run_alignment_with_fallback(
                    language="japanese",
                    wav_folder=td,
                    dictionary_path=os.path.join(td, "dictionary.txt"),
                    output_folder=output_dir,
                    primary_aligner="domino",
                    fallback_aligner="mfa",
                    mfa_path="",
                    mfa_align_profile="default",
                    callback=None,
                )

        mocked_domino_ready.assert_called_once()
        mocked_mfa_ready.assert_called_once()
        mocked_mfa_align.assert_called_once()
        self.assertTrue(bool(result.get("ok", False)))
        self.assertEqual(str(result.get("used_engine", "")), "mfa")
        self.assertTrue(bool(result.get("fallback_used", False)))
        self.assertIn("engine:domino->mfa", str(result.get("fallback_path", "")))

    def test_domino_primary_success_without_mfa(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = os.path.join(td, "textgrids")

            def _fake_domino_align(_onnx, _wav, _dict, out_dir, **_kwargs):
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "sample.TextGrid"), "w", encoding="utf-8") as handle:
                    handle.write('File type = "ooTextFile"\n')
                return True, "ok"

            with mock.patch(
                "core.alignment_pipeline.check_domino_ready",
                return_value={"code": OK, "message": "ready", "domino_onnx_path": "x.onnx"},
            ) as mocked_domino_ready, mock.patch(
                "core.alignment_pipeline.run_domino_align",
                side_effect=_fake_domino_align,
            ) as mocked_domino_align, mock.patch(
                "core.alignment_pipeline.check_mfa_ready"
            ) as mocked_mfa_ready:
                result = run_alignment_with_fallback(
                    language="japanese",
                    wav_folder=td,
                    dictionary_path=os.path.join(td, "dictionary.txt"),
                    output_folder=output_dir,
                    primary_aligner="domino",
                    fallback_aligner="mfa",
                    mfa_path="",
                    mfa_align_profile="default",
                    callback=None,
                )

        mocked_domino_ready.assert_called_once()
        mocked_domino_align.assert_called_once()
        mocked_mfa_ready.assert_not_called()
        self.assertTrue(bool(result.get("ok", False)))
        self.assertEqual(str(result.get("used_engine", "")), "domino")

    def test_mfa_precheck_not_ready_still_attempts_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            dict_path = os.path.join(td, "dictionary.txt")
            with open(dict_path, "w", encoding="utf-8") as handle:
                handle.write("a a\n")
            output_dir = os.path.join(td, "textgrids")

            def _fake_mfa_align(_mfa_path, _wav_folder, _dict_path, out_dir, **_kwargs):
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "sample.TextGrid"), "w", encoding="utf-8") as handle:
                    handle.write('File type = "ooTextFile"\n')
                return True, ""

            with mock.patch(
                "core.alignment_pipeline.check_mfa_ready",
                return_value={"code": ALIGN_NOT_READY, "message": "model probe failed", "mfa_path": "fake_mfa"},
            ) as mocked_ready, mock.patch(
                "core.alignment_pipeline.run_mfa_align",
                side_effect=_fake_mfa_align,
            ) as mocked_mfa_align:
                result = run_alignment_with_fallback(
                    language="korean",
                    wav_folder=td,
                    dictionary_path=dict_path,
                    output_folder=output_dir,
                    primary_aligner="mfa",
                    fallback_aligner="",
                    mfa_path="",
                    mfa_align_profile="default",
                    callback=None,
                )
        mocked_ready.assert_called_once()
        mocked_mfa_align.assert_called_once()
        self.assertTrue(bool(result.get("ok", False)))

    def test_mfa_precheck_exec_missing_still_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            dict_path = os.path.join(td, "dictionary.txt")
            with open(dict_path, "w", encoding="utf-8") as handle:
                handle.write("a a\n")
            result = run_alignment_with_fallback(
                language="korean",
                wav_folder=td,
                dictionary_path=dict_path,
                output_folder=os.path.join(td, "textgrids"),
                primary_aligner="mfa",
                fallback_aligner="",
                mfa_path="",
                mfa_align_profile="default",
                callback=None,
            )
        self.assertFalse(bool(result.get("ok", True)))
        self.assertEqual(str(result.get("code", "")), ALIGN_EXEC_MISSING)

    def test_mfa_precheck_soft_gate_limit_escalates_to_block(self):
        with tempfile.TemporaryDirectory() as td:
            dict_path = os.path.join(td, "dictionary.txt")
            with open(dict_path, "w", encoding="utf-8") as handle:
                handle.write("a a\n")
            output_dir = os.path.join(td, "textgrids")

            with patch.dict(os.environ, {"UTOA_MFA_SOFT_GATE_RETRY_LIMIT": "2"}, clear=False):
                alignment_pipeline._MFA_SOFT_GATE_FAIL_CACHE.clear()
                with mock.patch(
                    "core.alignment_pipeline.check_mfa_ready",
                    return_value={"code": ALIGN_NOT_READY, "message": "precheck unstable", "mfa_path": "fake_mfa"},
                ), mock.patch(
                    "core.alignment_pipeline.run_mfa_align",
                    return_value=(False, "MFA alignment failed"),
                ) as mocked_mfa_align:
                    for _ in range(3):
                        run_alignment_with_fallback(
                            language="korean",
                            wav_folder=td,
                            dictionary_path=dict_path,
                            output_folder=output_dir,
                            primary_aligner="mfa",
                            fallback_aligner="",
                            mfa_path="",
                            mfa_align_profile="default",
                            callback=None,
                        )
            self.assertEqual(mocked_mfa_align.call_count, 2)

if __name__ == "__main__":
    unittest.main()
