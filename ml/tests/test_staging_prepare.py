import os
import sys
import tempfile
import unittest
import json
from unittest import mock
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import core.oto_ml_batch_prepare as batch_prepare_mod
from core.oto_ml_batch_prepare import _discover_work_items, prepare_staged_auto_pairs
from core.oto_ml_collection import _discover_oto_files


class StagingPrepareTests(unittest.TestCase):
    def test_empty_top_level_oto_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "dataset", "japanese", "vcv", "Voice")
            sub = os.path.join(root, "C4")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(root, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(sub, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(sub, "a.wav"), "wb") as f:
                f.write(b"RIFF")

            otos = _discover_oto_files(root)
            self.assertEqual(len(otos), 1)
            self.assertTrue(otos[0].endswith(os.path.join("C4", "oto.ini")))

            items = _discover_work_items(os.path.join(td, "dataset"))
            self.assertEqual(len(items), 1)
            self.assertTrue(items[0].manual_oto.endswith(os.path.join("C4", "oto.ini")))

    def test_prepare_reuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            work = os.path.join(td, "dataset", "korean", "cvc", "Voice")
            os.makedirs(work, exist_ok=True)
            with open(os.path.join(work, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "oto_auto_ml.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "a.wav"), "wb") as f:
                f.write(b"RIFF")
            with open(os.path.join(work, "a.TextGrid"), "w", encoding="utf-8") as f:
                f.write("dummy")

            result = prepare_staged_auto_pairs(os.path.join(td, "dataset"), dry_run=False)
            self.assertEqual(result["summary"]["prepared"], 1)
            self.assertEqual(result["items"][0].status, "prepared_existing")
            self.assertEqual(result["items"][0].tg_dir, work)

    def test_prepare_retries_mfa_with_fallback_profile_on_native_crash(self):
        with tempfile.TemporaryDirectory() as td:
            work = os.path.join(td, "dataset", "korean", "cvc", "Voice")
            os.makedirs(work, exist_ok=True)
            with open(os.path.join(work, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "a.wav"), "wb") as f:
                f.write(b"RIFF")

            calls = []

            def fake_run_mfa_align(**kwargs):
                calls.append(kwargs["align_profile"])
                if kwargs["align_profile"] == "default":
                    return False, "MFA alignment failed (code: 3221225477) | tail: INFO     Compiling training graphs..."
                return True, ""

            with mock.patch.object(batch_prepare_mod, "find_mfa_executable", return_value="mfa.exe"), \
                 mock.patch.object(batch_prepare_mod, "_prepare_lab_and_dict"), \
                 mock.patch.object(batch_prepare_mod, "_generate_auto_oto"), \
                 mock.patch.object(batch_prepare_mod, "run_mfa_align", side_effect=fake_run_mfa_align):
                result = prepare_staged_auto_pairs(os.path.join(td, "dataset"), dry_run=False)

            self.assertEqual(calls, ["default", "accurate"])
            self.assertEqual(result["summary"]["prepared"], 1)
            self.assertEqual(result["items"][0].status, "prepared")

    def test_prepare_does_not_retry_when_mfa_dependency_is_missing(self):
        with tempfile.TemporaryDirectory() as td:
            work = os.path.join(td, "dataset", "korean", "cvc", "Voice")
            os.makedirs(work, exist_ok=True)
            with open(os.path.join(work, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "a.wav"), "wb") as f:
                f.write(b"RIFF")

            calls = []

            def fake_run_mfa_align(**kwargs):
                calls.append(kwargs["align_profile"])
                return False, "Missing Korean tokenizer dependencies (eunjeon, jamo)."

            with mock.patch.object(batch_prepare_mod, "find_mfa_executable", return_value="mfa.exe"), \
                 mock.patch.object(batch_prepare_mod, "_prepare_lab_and_dict"), \
                 mock.patch.object(batch_prepare_mod, "_generate_auto_oto"), \
                 mock.patch.object(batch_prepare_mod, "run_mfa_align", side_effect=fake_run_mfa_align):
                result = prepare_staged_auto_pairs(os.path.join(td, "dataset"), dry_run=False)

            self.assertEqual(calls, ["default"])
            self.assertEqual(result["summary"]["skipped"], 1)
            self.assertTrue(result["items"][0].reason.startswith("align_failed:Missing Korean tokenizer dependencies"))

    def test_prepare_lab_and_dict_uses_ascii_safe_korean_labs(self):
        import core.oto_ml_prepare_steps as steps_mod

        item = SimpleNamespace(language="korean", work_dir="C:\\tmp\\voice", dict_path="")
        logs = []

        with mock.patch.object(steps_mod, "generate_labs") as mock_labs, \
             mock.patch.object(steps_mod, "generate_dictionary") as mock_dict:
            steps_mod._prepare_lab_and_dict(item, logs)

        mock_labs.assert_called_once()
        self.assertEqual(mock_labs.call_args.kwargs.get("convert_to_hangul"), False)
        mock_dict.assert_called_once()

    def test_prepare_resume_skips_checkpointed_prepared_items(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, "dataset")
            work_done = os.path.join(dataset_root, "korean", "cvc", "DoneVoice")
            work_pending = os.path.join(dataset_root, "korean", "cvc", "PendingVoice")
            os.makedirs(work_done, exist_ok=True)
            os.makedirs(work_pending, exist_ok=True)

            for work in (work_done, work_pending):
                with open(os.path.join(work, "oto.ini"), "w", encoding="utf-8") as f:
                    f.write("a.wav=ga,0,1,-2,0,0\n")
                with open(os.path.join(work, "a.wav"), "wb") as f:
                    f.write(b"RIFF")

            with open(os.path.join(work_done, "oto_auto_ml.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work_done, "a.TextGrid"), "w", encoding="utf-8") as f:
                f.write("dummy")

            report_path = os.path.join(dataset_root, "_manifest", "prepared_auto_pairs.json")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": {"total_items": 2, "prepared": 1, "skipped": 0},
                        "items": [
                            {
                                "language": "korean",
                                "format_type": "cvc",
                                "stage_root": work_done,
                                "work_dir": work_done,
                                "manual_oto": os.path.join(work_done, "oto.ini"),
                                "auto_oto": os.path.join(work_done, "oto_auto_ml.ini"),
                                "tg_dir": work_done,
                                "dict_path": os.path.join(work_done, "dictionary_auto.txt"),
                                "mfa_path": "mfa.exe",
                                "status": "prepared",
                                "reason": "",
                            }
                        ],
                        "logs": {
                            os.path.relpath(work_done, dataset_root): ["old-log"],
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            prepared_calls = []

            def fake_prepare(item, logs):
                prepared_calls.append(item.work_dir)

            with mock.patch.object(batch_prepare_mod, "find_mfa_executable", return_value="mfa.exe"), \
                 mock.patch.object(batch_prepare_mod, "_prepare_lab_and_dict", side_effect=fake_prepare), \
                 mock.patch.object(batch_prepare_mod, "_generate_auto_oto"), \
                 mock.patch.object(batch_prepare_mod, "run_mfa_align", return_value=(True, "")):
                result = prepare_staged_auto_pairs(
                    dataset_root,
                    dry_run=False,
                    report_path=report_path,
                    resume=True,
                )

            self.assertEqual(prepared_calls, [work_pending])
            self.assertEqual(result["summary"]["prepared"], 2)
            self.assertEqual(result["summary"]["resume_skipped"], 1)
            by_work = {item.work_dir: item for item in result["items"]}
            self.assertEqual(by_work[work_done].status, "prepared")
            self.assertEqual(by_work[work_pending].status, "prepared")

    def test_prepare_resume_retry_failed_reprocesses_skipped_item(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, "dataset")
            work = os.path.join(dataset_root, "korean", "cvc", "RetryVoice")
            os.makedirs(work, exist_ok=True)
            with open(os.path.join(work, "oto.ini"), "w", encoding="utf-8") as f:
                f.write("a.wav=ga,0,1,-2,0,0\n")
            with open(os.path.join(work, "a.wav"), "wb") as f:
                f.write(b"RIFF")

            report_path = os.path.join(dataset_root, "_manifest", "prepared_auto_pairs.json")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": {"total_items": 1, "prepared": 0, "skipped": 1},
                        "items": [
                            {
                                "language": "korean",
                                "format_type": "cvc",
                                "stage_root": work,
                                "work_dir": work,
                                "manual_oto": os.path.join(work, "oto.ini"),
                                "auto_oto": os.path.join(work, "oto_auto_ml.ini"),
                                "tg_dir": work,
                                "dict_path": os.path.join(work, "dictionary_auto.txt"),
                                "mfa_path": "mfa.exe",
                                "status": "skip",
                                "reason": "align_failed:test",
                            }
                        ],
                        "logs": {},
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            with mock.patch.object(batch_prepare_mod, "find_mfa_executable", return_value="mfa.exe"), \
                 mock.patch.object(batch_prepare_mod, "_prepare_lab_and_dict"), \
                 mock.patch.object(batch_prepare_mod, "_generate_auto_oto"), \
                 mock.patch.object(batch_prepare_mod, "run_mfa_align", return_value=(True, "")):
                result = prepare_staged_auto_pairs(
                    dataset_root,
                    dry_run=False,
                    report_path=report_path,
                    resume=True,
                    retry_failed=True,
                )

            self.assertEqual(result["summary"]["prepared"], 1)
            self.assertEqual(result["summary"]["skipped"], 0)
            self.assertEqual(result["items"][0].status, "prepared")

    def test_prepare_writes_checkpoint_before_interrupt(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, "dataset")
            work_a = os.path.join(dataset_root, "korean", "cvc", "VoiceA")
            work_b = os.path.join(dataset_root, "korean", "cvc", "VoiceB")
            os.makedirs(work_a, exist_ok=True)
            os.makedirs(work_b, exist_ok=True)
            for work in (work_a, work_b):
                with open(os.path.join(work, "oto.ini"), "w", encoding="utf-8") as f:
                    f.write("a.wav=ga,0,1,-2,0,0\n")
                with open(os.path.join(work, "a.wav"), "wb") as f:
                    f.write(b"RIFF")

            report_path = os.path.join(dataset_root, "_manifest", "prepared_auto_pairs.json")
            call_count = {"prepare": 0}

            def fake_prepare(item, logs):
                call_count["prepare"] += 1
                if call_count["prepare"] >= 2:
                    raise KeyboardInterrupt()

            with mock.patch.object(batch_prepare_mod, "find_mfa_executable", return_value="mfa.exe"), \
                 mock.patch.object(batch_prepare_mod, "_prepare_lab_and_dict", side_effect=fake_prepare), \
                 mock.patch.object(batch_prepare_mod, "_generate_auto_oto"), \
                 mock.patch.object(batch_prepare_mod, "run_mfa_align", return_value=(True, "")):
                with self.assertRaises(KeyboardInterrupt):
                    prepare_staged_auto_pairs(
                        dataset_root,
                        dry_run=False,
                        report_path=report_path,
                    )

            self.assertTrue(os.path.isfile(report_path))
            with open(report_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(len(payload["items"]), 1)
            self.assertEqual(payload["summary"]["prepared"], 1)
            self.assertEqual(payload["summary"]["pending"], 1)


if __name__ == "__main__":
    unittest.main()
