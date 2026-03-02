import csv
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_torch_dataset import build_torch_dataset, load_torch_dataset_index


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


class TorchDatasetTests(unittest.TestCase):
    def test_build_torch_dataset_creates_shards(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb1 = os.path.join(dataset_root, 'japanese', 'vcv', 'VB1')
            vb2 = os.path.join(dataset_root, 'japanese', 'vcv', 'VB2')
            _write_wav(os.path.join(vb1, 'a.wav'), 220.0)
            _write_wav(os.path.join(vb2, 'b.wav'), 260.0)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {"voicebank_id": "VB1", "wav": "a.wav", "alias": "a i", "line_index": "0", "language": "japanese", "format_type": "vcv", "alias_type": "vv", "alias_group": "vv", "coda_type": "none", "mapping_confidence": "0.9", "train_keep_default": "1", "base_offset": "100", "base_pre": "50", "expected_anchor_ms": "150", "delta_offset": "1", "delta_cons": "2", "delta_cutoff": "3", "delta_pre": "4", "delta_ovl": "5"},
                {"voicebank_id": "VB2", "wav": "b.wav", "alias": "n jo", "line_index": "1", "language": "japanese", "format_type": "vcv", "alias_type": "vcv", "alias_group": "vcv", "coda_type": "none", "mapping_confidence": "0.8", "train_keep_default": "1", "base_offset": "120", "base_pre": "45", "expected_anchor_ms": "165", "delta_offset": "-1", "delta_cons": "1", "delta_cutoff": "2", "delta_pre": "1", "delta_ovl": "0.5"},
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VB1", "language": "japanese", "format_type": "vcv", "wav_dir": vb1},
                    {"status": "built", "voicebank_id": "VB2", "language": "japanese", "format_type": "vcv", "wav_dir": vb2},
                ], f, ensure_ascii=False, indent=2)

            out_dir = os.path.join(td, 'torch_dataset')
            summary = build_torch_dataset(
                task_name='japanese_bridge_vcv',
                dataset_csv=csv_path,
                out_dir=out_dir,
                dataset_root=dataset_root,
                build_manifest_path=build_manifest,
            )
            self.assertEqual(summary.row_count, 2)
            index = load_torch_dataset_index(summary.index_path)
            self.assertEqual(index['summary']['row_count'], 2)
            self.assertEqual(len(index['shards']), 2)
            with np.load(index['shards'][0]['path']) as shard:
                self.assertEqual(shard['mel_windows'].shape[1:], (80, 160))

    def test_build_torch_dataset_allows_legacy_rows_without_v4_confidence_columns(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb1 = os.path.join(dataset_root, 'korean', 'cvvc', 'VBK')
            _write_wav(os.path.join(vb1, 'a.wav'), 220.0)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a l",
                    "line_index": "0",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_liquid",
                    "coda_type": "liquid",
                    "train_keep_default": "1",
                    "base_offset": "100",
                    "base_pre": "50",
                    "expected_anchor_ms": "150",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VBK", "language": "korean", "format_type": "cvvc", "wav_dir": vb1},
                ], f, ensure_ascii=False, indent=2)

            out_dir = os.path.join(td, 'torch_dataset')
            summary = build_torch_dataset(
                task_name='korean_bridge_cvvc',
                dataset_csv=csv_path,
                out_dir=out_dir,
                dataset_root=dataset_root,
                build_manifest_path=build_manifest,
            )
            self.assertEqual(summary.row_count, 1)

    def test_korean_bridge_task_does_not_require_train_keep_default(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb1 = os.path.join(dataset_root, 'korean', 'cvvc', 'VBK')
            _write_wav(os.path.join(vb1, 'a.wav'), 220.0)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a l",
                    "line_index": "0",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_liquid",
                    "coda_type": "liquid",
                    "mapping_confidence": "0.35",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "base_offset": "100",
                    "base_pre": "50",
                    "expected_anchor_ms": "150",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VBK", "language": "korean", "format_type": "cvvc", "wav_dir": vb1},
                ], f, ensure_ascii=False, indent=2)

            out_dir = os.path.join(td, 'torch_dataset')
            summary = build_torch_dataset(
                task_name='korean_bridge_cvvc',
                dataset_csv=csv_path,
                out_dir=out_dir,
                dataset_root=dataset_root,
                build_manifest_path=build_manifest,
            )
            self.assertEqual(summary.row_count, 1)

    def test_korean_bridge_cvvc_vv_requires_alias_occurrence_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb1 = os.path.join(dataset_root, 'korean', 'cvvc', 'VBK')
            _write_wav(os.path.join(vb1, 'a.wav'), 220.0)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a i",
                    "line_index": "0",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vv",
                    "alias_group": "vv",
                    "coda_type": "none",
                    "mapping_confidence": "0.80",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "used_alias_occurrence_mapping": "0",
                    "base_offset": "100",
                    "base_pre": "50",
                    "expected_anchor_ms": "150",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a u",
                    "line_index": "1",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vv",
                    "alias_group": "vv",
                    "coda_type": "none",
                    "mapping_confidence": "0.82",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "used_alias_occurrence_mapping": "1",
                    "base_offset": "110",
                    "base_pre": "52",
                    "expected_anchor_ms": "162",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VBK", "language": "korean", "format_type": "cvvc", "wav_dir": vb1},
                ], f, ensure_ascii=False, indent=2)

            out_dir = os.path.join(td, 'torch_dataset')
            summary = build_torch_dataset(
                task_name='korean_bridge_cvvc_vv',
                dataset_csv=csv_path,
                out_dir=out_dir,
                dataset_root=dataset_root,
                build_manifest_path=build_manifest,
            )
            self.assertEqual(summary.row_count, 1)

    def test_korean_bridge_cvvc_vc_filters_low_confidence_rows(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb1 = os.path.join(dataset_root, 'korean', 'cvvc', 'VBK')
            _write_wav(os.path.join(vb1, 'a.wav'), 220.0)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a l",
                    "line_index": "0",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_liquid",
                    "coda_type": "liquid",
                    "mapping_confidence": "0.40",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "base_offset": "100",
                    "base_pre": "50",
                    "expected_anchor_ms": "150",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a n",
                    "line_index": "1",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_nasal",
                    "coda_type": "nasal",
                    "mapping_confidence": "0.63",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "base_offset": "110",
                    "base_pre": "52",
                    "expected_anchor_ms": "162",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VBK", "language": "korean", "format_type": "cvvc", "wav_dir": vb1},
                ], f, ensure_ascii=False, indent=2)

            out_dir = os.path.join(td, 'torch_dataset')
            summary = build_torch_dataset(
                task_name='korean_bridge_cvvc_vc',
                dataset_csv=csv_path,
                out_dir=out_dir,
                dataset_root=dataset_root,
                build_manifest_path=build_manifest,
            )
            self.assertEqual(summary.row_count, 1)

    def test_korean_bridge_cvvc_vc_sonorant_keeps_only_nasal_liquid(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb1 = os.path.join(dataset_root, 'korean', 'cvvc', 'VBK')
            _write_wav(os.path.join(vb1, 'a.wav'), 220.0)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a k",
                    "line_index": "0",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_stop",
                    "coda_type": "stop",
                    "mapping_confidence": "0.91",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "base_offset": "100",
                    "base_pre": "50",
                    "expected_anchor_ms": "150",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a n",
                    "line_index": "1",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_sonorant",
                    "coda_type": "nasal",
                    "mapping_confidence": "0.58",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "base_offset": "110",
                    "base_pre": "52",
                    "expected_anchor_ms": "162",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VBK", "language": "korean", "format_type": "cvvc", "wav_dir": vb1},
                ], f, ensure_ascii=False, indent=2)

            out_dir = os.path.join(td, 'torch_dataset')
            summary = build_torch_dataset(
                task_name='korean_bridge_cvvc_vc_sonorant',
                dataset_csv=csv_path,
                out_dir=out_dir,
                dataset_root=dataset_root,
                build_manifest_path=build_manifest,
            )
            self.assertEqual(summary.row_count, 1)
            index = load_torch_dataset_index(summary.index_path)
            with np.load(index['shards'][0]['path']) as shard:
                self.assertEqual(shard['alias'][0], 'a n')

    def test_korean_bridge_cvvc_vc_stop_requires_higher_confidence(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb1 = os.path.join(dataset_root, 'korean', 'cvvc', 'VBK')
            _write_wav(os.path.join(vb1, 'a.wav'), 220.0)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a k",
                    "line_index": "0",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_stop",
                    "coda_type": "stop",
                    "mapping_confidence": "0.58",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "base_offset": "100",
                    "base_pre": "50",
                    "expected_anchor_ms": "150",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a t",
                    "line_index": "1",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_stop",
                    "coda_type": "stop",
                    "mapping_confidence": "0.63",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "base_offset": "110",
                    "base_pre": "52",
                    "expected_anchor_ms": "162",
                    "delta_offset": "1",
                    "delta_cons": "2",
                    "delta_cutoff": "3",
                    "delta_pre": "4",
                    "delta_ovl": "5",
                },
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VBK", "language": "korean", "format_type": "cvvc", "wav_dir": vb1},
                ], f, ensure_ascii=False, indent=2)

            out_dir = os.path.join(td, 'torch_dataset')
            summary = build_torch_dataset(
                task_name='korean_bridge_cvvc_vc_stop',
                dataset_csv=csv_path,
                out_dir=out_dir,
                dataset_root=dataset_root,
                build_manifest_path=build_manifest,
            )
            self.assertEqual(summary.row_count, 1)
            index = load_torch_dataset_index(summary.index_path)
            with np.load(index['shards'][0]['path']) as shard:
                self.assertEqual(shard['alias'][0], 'a t')

    def test_korean_bridge_cvvc_vc_stop_filters_extreme_target_outliers(self):
        with tempfile.TemporaryDirectory() as td:
            dataset_root = os.path.join(td, 'dataset')
            vb1 = os.path.join(dataset_root, 'korean', 'cvvc', 'VBK')
            _write_wav(os.path.join(vb1, 'a.wav'), 220.0)

            csv_path = os.path.join(td, 'dataset.csv')
            rows = [
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a k",
                    "line_index": "0",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_stop",
                    "coda_type": "stop",
                    "mapping_confidence": "0.83",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "delta_offset": "2000",
                    "delta_cons": "20",
                    "delta_cutoff": "1400",
                    "delta_pre": "30",
                    "delta_ovl": "15",
                    "base_offset": "100",
                    "base_pre": "50",
                    "expected_anchor_ms": "150",
                },
                {
                    "voicebank_id": "VBK",
                    "wav": "a.wav",
                    "alias": "a t",
                    "line_index": "1",
                    "language": "korean",
                    "format_type": "cvvc",
                    "alias_type": "vc",
                    "alias_group": "vc_stop",
                    "coda_type": "stop",
                    "mapping_confidence": "0.83",
                    "train_keep_default": "0",
                    "used_nuclei_fallback": "0",
                    "delta_offset": "900",
                    "delta_cons": "20",
                    "delta_cutoff": "1200",
                    "delta_pre": "30",
                    "delta_ovl": "15",
                    "base_offset": "100",
                    "base_pre": "50",
                    "expected_anchor_ms": "150",
                },
            ]
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
                writer.writeheader()
                writer.writerows(rows)

            build_manifest = os.path.join(td, 'build.json')
            with open(build_manifest, 'w', encoding='utf-8') as f:
                json.dump([
                    {"status": "built", "voicebank_id": "VBK", "language": "korean", "format_type": "cvvc", "wav_dir": vb1},
                ], f, ensure_ascii=False, indent=2)

            out_dir = os.path.join(td, 'torch_dataset')
            summary = build_torch_dataset(
                task_name='korean_bridge_cvvc_vc_stop',
                dataset_csv=csv_path,
                out_dir=out_dir,
                dataset_root=dataset_root,
                build_manifest_path=build_manifest,
            )
            self.assertEqual(summary.row_count, 1)
            index = load_torch_dataset_index(summary.index_path)
            with np.load(index['shards'][0]['path']) as shard:
                self.assertEqual(shard['alias'][0], 'a t')


if __name__ == '__main__':
    unittest.main()
