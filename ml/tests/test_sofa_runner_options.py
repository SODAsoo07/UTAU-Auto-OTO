import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.sofa_runner import SofaAlignOptions, run_sofa_align


class SofaRunnerOptionTests(unittest.TestCase):
    def test_sofa_align_options_roundtrip(self):
        options = SofaAlignOptions.from_dict(
            {
                "repo_dir": "repo",
                "ckpt_path": "a.ckpt",
                "dictionary_path": "dict.txt",
                "mode": "match",
                "g2p": "Phoneme",
                "ap_detector": "None",
                "ap_detector_config": "ap.json",
                "save_confidence": True,
                "out_formats": "TextGrid,htk",
                "extra_infer_args": ["--foo", "bar"],
            }
        )
        data = options.to_dict()
        self.assertEqual(data["repo_dir"], "repo")
        self.assertEqual(data["mode"], "match")
        self.assertEqual(data["ap_detector_config"], "ap.json")
        self.assertTrue(data["save_confidence"])
        self.assertEqual(data["out_formats"], ["TextGrid", "htk"])

    def test_run_sofa_align_two_pass_selects_better_confidence(self):
        with tempfile.TemporaryDirectory() as td:
            wav_dir = os.path.join(td, "wavs")
            out_dir = os.path.join(td, "out")
            repo_dir = os.path.join(td, "repo")
            os.makedirs(wav_dir, exist_ok=True)
            os.makedirs(repo_dir, exist_ok=True)

            ckpt = os.path.join(td, "model.ckpt")
            with open(ckpt, "w", encoding="utf-8") as f:
                f.write("dummy")
            dictionary = os.path.join(td, "dict.txt")
            with open(dictionary, "w", encoding="utf-8") as f:
                f.write("a\ta\n")

            for name in ("a", "b"):
                with open(os.path.join(wav_dir, f"{name}.wav"), "wb") as f:
                    f.write(b"RIFF")
                with open(os.path.join(wav_dir, f"{name}.lab"), "w", encoding="utf-8") as f:
                    f.write("a")

            dummy_infer = os.path.join(repo_dir, "infer.py")
            with open(dummy_infer, "w", encoding="utf-8") as f:
                f.write("# dummy")

            def fake_run_sofa_infer(
                py,
                infer_py,
                infer_is_wrapper,
                ckpt_path,
                seg_root,
                dictionary_path,
                out_formats,
                mode,
                g2p,
                ap_detector,
                ap_detector_config,
                save_confidence,
                extra_infer_args,
                sofa_root,
                run_env,
                callback=None,
            ):
                singer = os.path.join(seg_root, "singer1")
                tg_dir = os.path.join(singer, "TextGrid")
                conf_dir = os.path.join(singer, "confidence")
                os.makedirs(tg_dir, exist_ok=True)
                os.makedirs(conf_dir, exist_ok=True)

                names = [
                    os.path.splitext(name)[0]
                    for name in os.listdir(singer)
                    if name.lower().endswith(".wav")
                ]
                for name in names:
                    with open(os.path.join(tg_dir, f"{name}.TextGrid"), "w", encoding="utf-8") as tf:
                        tf.write(f"mode:{mode}")

                with open(os.path.join(conf_dir, "confidence.csv"), "w", encoding="utf-8", newline="") as cf:
                    writer = csv.DictWriter(cf, fieldnames=["name", "confidence"])
                    writer.writeheader()
                    for name in names:
                        if mode == "force":
                            conf = 0.2 if name == "a" else 0.9
                        else:
                            conf = 0.8 if name == "a" else 0.7
                        writer.writerow({"name": name, "confidence": conf})
                return True, ""

            with mock.patch("core.sofa_runner.ensure_sofa_support", return_value=(True, "")), mock.patch(
                "core.sofa_runner._resolve_python_exe", return_value=sys.executable
            ), mock.patch("core.sofa_runner._ensure_runtime_ffmpeg_bin", return_value=""), mock.patch(
                "core.sofa_runner._prepend_ffmpeg_to_env_path", side_effect=lambda env, callback=None, extra_dirs=None: env
            ), mock.patch("core.sofa_runner._resolve_infer_entry", return_value=(dummy_infer, False)), mock.patch(
                "core.sofa_runner._run_sofa_infer", side_effect=fake_run_sofa_infer
            ):
                ok, err = run_sofa_align(
                    wav_folder=wav_dir,
                    output_folder=out_dir,
                    ckpt_path=ckpt,
                    dictionary_path=dictionary,
                    sofa_repo_dir=repo_dir,
                    mode="force",
                    save_confidence=True,
                    two_pass_retry=True,
                    confidence_threshold=0.5,
                )

            self.assertTrue(ok, msg=err)
            with open(os.path.join(out_dir, "a.TextGrid"), "r", encoding="utf-8") as f:
                self.assertIn("mode:match", f.read())
            with open(os.path.join(out_dir, "b.TextGrid"), "r", encoding="utf-8") as f:
                self.assertIn("mode:force", f.read())

            summary_json = os.path.join(out_dir, "sofa_confidence_summary.json")
            self.assertTrue(os.path.exists(summary_json))
            with open(summary_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            per_name = {row["name"]: row for row in data}
            self.assertEqual(per_name["a"]["selected_pass"], "pass2")
            self.assertEqual(per_name["b"]["selected_pass"], "pass1")


if __name__ == "__main__":
    unittest.main()
