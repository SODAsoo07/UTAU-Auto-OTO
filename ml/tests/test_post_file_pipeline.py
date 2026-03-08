import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.post_file_pipeline import resolve_wav_dir_from_tg_folder, run_ml_post_stage


class PostFilePipelineTests(unittest.TestCase):
    def test_resolve_wav_dir_from_tg_folder_returns_parent_dir(self):
        got = resolve_wav_dir_from_tg_folder(r"C:\tmp\voicebank\textgrids")
        self.assertEqual(got, os.path.abspath(r"C:\tmp\voicebank"))

    def test_run_ml_post_stage_records_runtime_report(self):
        runtime_report = {}
        logs = []

        def _fake_apply(*args, **kwargs):
            report = kwargs["report"]
            report.update({"stage": "ml", "status": "applied", "code": "ML_APPLIED"})
            return 2

        with mock.patch("core.oto_ml_refiner.apply_oto_ml_to_oto_file", side_effect=_fake_apply):
            changed = run_ml_post_stage(
                language="korean",
                out_path="out.ini",
                tg_folder=r"C:\tmp\voicebank\textgrids",
                custom_phonemes_path="",
                enable_ml_correction=True,
                format_override="cvc",
                ml_policy="auto",
                runtime_report=runtime_report,
                log_fn=logs.append,
            )

        self.assertEqual(changed, 2)
        self.assertEqual(runtime_report["ml"]["code"], "ML_APPLIED")
        self.assertTrue(any("ML refine changed" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
