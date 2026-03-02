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

from core.oto_torch_dataset import build_torch_dataset
from core.oto_torch_export import export_torch_bundle
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


class TorchBundleTests(unittest.TestCase):
    def test_train_and_export_torch_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            wav_specs = [
                ('VB1', 'a.wav', 220.0, 'vv', 'vv'),
                ('VB1', 'c.wav', 230.0, 'vc', 'vc_other'),
                ('VB2', 'b.wav', 260.0, 'vv', 'vv'),
                ('VB2', 'd.wav', 280.0, 'vcv', 'vcv'),
            ]
            for vb, wav, freq, _atype, _agroup in wav_specs:
                _write_wav(os.path.join(dataset_root, 'japanese', 'vcv', vb, wav), freq)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = []
            for idx, (vb, wav, _freq, alias_type, alias_group) in enumerate(wav_specs):
                rows.append({
                    "voicebank_id": vb, "wav": wav, "alias": f"alias{idx}", "line_index": str(idx),
                    "language": "japanese", "format_type": "vcv", "alias_type": alias_type,
                    "alias_group": alias_group, "coda_type": "none", "mapping_confidence": "0.9",
                    "train_keep_default": "1", "base_offset": str(100 + idx * 10), "base_pre": "50",
                    "expected_anchor_ms": str(150 + idx * 10), "delta_offset": str(idx), "delta_cons": "1",
                    "delta_cutoff": "2", "delta_pre": "1", "delta_ovl": "0.5"
                })
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VB1", "language": "japanese", "format_type": "vcv", "wav_dir": os.path.join(dataset_root, 'japanese', 'vcv', 'VB1')},
                    {"status": "built", "voicebank_id": "VB2", "language": "japanese", "format_type": "vcv", "wav_dir": os.path.join(dataset_root, 'japanese', 'vcv', 'VB2')},
                ], f, ensure_ascii=False, indent=2)

            ds_dir = os.path.join(td, 'torch_dataset')
            build_torch_dataset('japanese_bridge_vcv', csv_path, ds_dir, dataset_root=dataset_root, build_manifest_path=build_manifest)
            model_dir = os.path.join(td, 'model')
            meta = train_torch_bundle(ds_dir, model_dir, language='japanese', task_name='japanese_bridge_vcv', epochs=1, batch_size=2, use_amp=False, device_override='cpu')
            self.assertEqual(meta['backend'], 'pytorch')
            self.assertTrue(os.path.isfile(os.path.join(model_dir, 'model.pt')))
            export_dir = os.path.join(td, 'export')
            manifest = export_torch_bundle(model_dir, export_dir, create_zip=True)
            self.assertTrue(os.path.isfile(os.path.join(manifest['export_dir'], 'bundle_manifest.json')))
            self.assertTrue(os.path.isfile(manifest['zip_path']))


if __name__ == '__main__':
    unittest.main()
