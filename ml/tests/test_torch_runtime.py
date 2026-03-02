import csv
import json
import math
import os
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_runtime import load_oto_model_bundle, predict_oto_deltas
from core.oto_torch_dataset import build_torch_dataset
from core.oto_torch_trainer import train_torch_bundle


def _write_wav(path, freq=220.0, sr=44100, duration=0.35):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = (0.2 * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    pcm = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class TorchRuntimeTests(unittest.TestCase):
    def test_runtime_load_and_predict(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb_dir = os.path.join(dataset_root, 'korean', 'cvvc', 'KB1')
            _write_wav(os.path.join(vb_dir, 'x.wav'), 220.0)
            _write_wav(os.path.join(vb_dir, 'y.wav'), 240.0)
            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {"voicebank_id": "KB1", "wav": "x.wav", "alias": "a i", "line_index": "0", "language": "korean", "format_type": "cvvc", "alias_type": "vv", "alias_group": "vv", "coda_type": "none", "mapping_confidence": "0.95", "train_keep_default": "1", "base_offset": "100", "base_pre": "55", "expected_anchor_ms": "155", "delta_offset": "1", "delta_cons": "2", "delta_cutoff": "1", "delta_pre": "0.5", "delta_ovl": "0.2"},
                {"voicebank_id": "KB1", "wav": "y.wav", "alias": "a ng", "line_index": "1", "language": "korean", "format_type": "cvvc", "alias_type": "vc", "alias_group": "vc_sonorant", "coda_type": "nasal", "mapping_confidence": "0.95", "train_keep_default": "1", "base_offset": "90", "base_pre": "48", "expected_anchor_ms": "138", "delta_offset": "-1", "delta_cons": "1", "delta_cutoff": "2", "delta_pre": "1", "delta_ovl": "0.4"},
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)
            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "KB1", "language": "korean", "format_type": "cvvc", "wav_dir": vb_dir},
                ], f, ensure_ascii=False, indent=2)
            ds_dir = os.path.join(td, 'torch_dataset')
            build_torch_dataset('korean_bridge_cvvc', csv_path, ds_dir, dataset_root=dataset_root, build_manifest_path=build_manifest)
            model_dir = os.path.join(td, 'model')
            train_torch_bundle(ds_dir, model_dir, language='korean', task_name='korean_bridge_cvvc', epochs=1, batch_size=1, use_amp=False, device_override='cpu')
            bundle = load_oto_model_bundle(model_dir)
            self.assertIsNotNone(bundle)
            self.assertEqual(bundle.backend, 'pytorch')
            result = predict_oto_deltas(bundle, {
                "language": "korean",
                "format_type": "cvvc",
                "alias_type": "vv",
                "alias_group": "vv",
                "voicebank_id": "KB1",
                "base_offset": 100.0,
                "base_cons": 150.0,
                "base_cutoff_abs": 240.0,
                "base_pre": 55.0,
                "base_ovl": 20.0,
                "mapping_confidence": 0.9,
                "mel_window": np.zeros((80, 160), dtype=np.float32),
            })
            self.assertEqual(result.backend, 'pytorch')
            self.assertEqual(set(result.deltas.keys()), {"delta_offset", "delta_cons", "delta_cutoff", "delta_pre", "delta_ovl"})


if __name__ == '__main__':
    unittest.main()
