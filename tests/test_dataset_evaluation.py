import json
from pathlib import Path

import numpy as np
import soundfile as sf

from universal_auto_oto.dataset_evaluation import evaluate_dataset_manifest


def test_evaluate_dataset_manifest_aggregates_manual_pairs(tmp_path: Path):
    bank = tmp_path / "bank"
    bank.mkdir()
    sf.write(bank / "a.wav", np.zeros(16000, dtype=np.float32), 16000)
    manual = bank / "oto.ini"
    candidate = bank / "candidate.ini"
    manual.write_text("a.wav=a,100,200,-300,80,40\na.wav=unset,0,0,0,0,0\n", encoding="utf-8")
    candidate.write_text("a.wav=a,110,210,-300,85,45\na.wav=unset,10,20,-30,5,2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "language": "test",
                        "format_type": "cv",
                        "work_dir": str(bank),
                        "manual_oto": str(manual),
                        "candidate_oto": str(candidate),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_dataset_manifest(manifest, candidate_field="candidate_oto")

    assert result["evaluated_items"] == 1
    assert result["matched_rows"] == 1
    assert result["reference_rows_parsed"] == 2
    assert result["reference_unset_rows"] == 1
    assert result["metrics_by_group"]["overall"]["pre_abs"]["mae"] == 15.0
    assert len(result["recording_results"]) == 1
    recording = result["recording_results"][0]
    assert recording["matched_rows"] == 1
    assert recording["pooled_anchor_mae_ms"] == 14.0
    assert recording["speaker_metadata_status"] == "missing"
    assert Path(recording["wav_path"]) == (bank / "a.wav").resolve()
    assert result["evaluation_policy"]["gold_timing_used_for_generation"] is False
