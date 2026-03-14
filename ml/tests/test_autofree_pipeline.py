import json
import os
import sys
import tempfile
import unittest
import wave
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml.autofree.data import build_audio_only_rows, resolve_manual_cutoff_abs
from core.oto_ml_export import required_bundle_files_for_backend
from core.oto_ml_runtime import OtoModelBundle, load_oto_model_bundle
from core.oto_ml_autofree_runtime import validate_autofree_bundle_contract


def _write_silence_wav(path: str, sr: int = 16000, duration_ms: int = 250):
    nframes = int(sr * (duration_ms / 1000.0))
    silence = (b"\x00\x00") * nframes
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(silence)


class AutofreePipelineTests(unittest.TestCase):
    def test_resolve_manual_cutoff_abs_negative(self):
        cutoff_abs, mode = resolve_manual_cutoff_abs(
            manual_offset=120.0,
            manual_cutoff=-80.0,
            hint_cutoff_abs=220.0,
            wav_duration_ms=500.0,
        )
        self.assertEqual(mode, "negative_relative")
        self.assertAlmostEqual(cutoff_abs, 200.0)

    def test_resolve_manual_cutoff_abs_positive(self):
        cutoff_abs, mode = resolve_manual_cutoff_abs(
            manual_offset=100.0,
            manual_cutoff=60.0,
            hint_cutoff_abs=165.0,
            wav_duration_ms=800.0,
        )
        self.assertIn(mode, {"positive_relative", "positive_absolute", "positive_from_wav_end"})
        self.assertIsNotNone(cutoff_abs)
        self.assertGreater(float(cutoff_abs), 102.0)

    def test_audio_only_requires_tokens_when_filename_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            wav_dir = os.path.join(td, "wav")
            os.makedirs(wav_dir, exist_ok=True)
            _write_silence_wav(os.path.join(wav_dir, "a.wav"))
            rows, stats = build_audio_only_rows(
                language="korean",
                wav_dir=wav_dir,
                allow_filename_tokens=False,
            )
            self.assertEqual(rows, [])
            self.assertGreaterEqual(int(stats.get("skip_token_missing", 0)), 1)

    def test_validate_autofree_bundle_contract_rejects_legacy(self):
        bundle = OtoModelBundle(
            backend="lightgbm",
            model_dir="C:/tmp/model",
            meta={"backend": "lightgbm"},
            feature_schema={"feature_names": []},
            payload={"dummy": True},
        )
        ok, reason = validate_autofree_bundle_contract(bundle)
        self.assertFalse(ok)
        self.assertEqual(reason, "backend_mismatch")

    def test_runtime_loader_dispatches_autofree_backend(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "model_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "backend": "autofree_v1",
                        "feature_schema_version": "af_v1",
                        "target_schema_version": "af_v1_abs",
                        "incompatible_with_legacy": True,
                    },
                    f,
                )
            with mock.patch("core.oto_ml_autofree.load_autofree_bundle", return_value={"ok": True}) as mocked:
                bundle = load_oto_model_bundle(td)
            self.assertIsNotNone(bundle)
            self.assertEqual(bundle.backend, "autofree_v1")
            self.assertEqual(bundle.payload, {"ok": True})
            mocked.assert_called_once()

    def test_autofree_required_bundle_files(self):
        required = required_bundle_files_for_backend("autofree_v1")
        self.assertIn("target_schema.json", required)
        self.assertIn("model_target_offset.txt", required)
        self.assertIn("model_target_cutoff_abs.txt", required)
        self.assertIn("model_confidence.txt", required)


if __name__ == "__main__":
    unittest.main()
