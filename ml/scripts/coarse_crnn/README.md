# coarse_crnn workflow

This workflow is intentionally separate from the existing `sequence` aligner.

The same audio/CRNN code also has an OTO-anchor mode. For UTAU voicebanks this
is the preferred direction: train directly against `oto.ini` parameters instead
of using coarse phone boundaries as the final target.

Build a Korean/Japanese manifest:

```powershell
python -m ml.scripts.coarse_crnn.build_manifest `
  --dataset-staged dataset_staged `
  --public-root "C:\Users\oyh57\SODAsoo1\VocalSynth\Data\PublicData" `
  --out ml_workspace\coarse_crnn\manifest.jsonl
```

Train a small CPU-friendly CRNN:

```powershell
python -m ml.scripts.coarse_crnn.train `
  --manifest ml_workspace\coarse_crnn\manifest.jsonl `
  --out ml_workspace\models\coarse_crnn\coarse_crnn.pt `
  --device cpu
```

For actual local training, prefer the existing `.venv` and CUDA when available:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ml.scripts.coarse_crnn.build_clean_manifest `
  --manifest ml_workspace\coarse_crnn\manifest_full.jsonl `
  --out ml_workspace\coarse_crnn\manifest_clean.jsonl
python -m ml.scripts.coarse_crnn.train `
  --manifest ml_workspace\coarse_crnn\manifest_clean.jsonl `
  --out ml_workspace\models\coarse_crnn\coarse_crnn_clean_boundary.pt `
  --device cuda `
  --batch-size 16 `
  --max-frames 1200 `
  --boundary-head `
  --augment `
  --class-balance
```

Run one alignment:

```powershell
python -m ml.scripts.coarse_crnn.align `
  --audio path\to\sample.wav `
  --language ja `
  --model ml_workspace\models\coarse_crnn\coarse_crnn.pt `
  --out output\coarse_crnn
```

Evaluate alignment quality against gold TextGrid/timed lab rows:

```powershell
python -m ml.scripts.coarse_crnn.build_eval_splits `
  --manifest ml_workspace\coarse_crnn\manifest_full.jsonl `
  --out-dir ml_workspace\coarse_crnn\eval_splits
python -m ml.scripts.coarse_crnn.evaluate_alignment `
  --manifest ml_workspace\coarse_crnn\eval_splits\eval_all.jsonl `
  --model ml_workspace\models\coarse_crnn\coarse_crnn.pt `
  --device cuda `
  --source dataset_staged `
  --max-items 0 `
  --max-per-language 0 `
  --out ml_workspace\coarse_crnn\alignment_eval.json
```

Build and train the OTO parameter predictor:

```powershell
.\.venv\Scripts\Activate.ps1
python -m ml.scripts.coarse_crnn.build_oto_manifest `
  --dataset-staged dataset_staged `
  --out ml_workspace\coarse_crnn\oto_manifest.jsonl
python -m ml.scripts.coarse_crnn.build_oto_splits `
  --manifest ml_workspace\coarse_crnn\oto_manifest.jsonl `
  --out-dir ml_workspace\coarse_crnn\oto_splits
python -m ml.scripts.coarse_crnn.train_oto `
  --manifest ml_workspace\coarse_crnn\oto_splits\oto_train.jsonl `
  --val-manifest ml_workspace\coarse_crnn\oto_splits\oto_val.jsonl `
  --out ml_workspace\models\coarse_crnn\oto_anchor_crnn.pt `
  --device cuda `
  --batch-size 16 `
  --max-frames 1200
python -m ml.scripts.coarse_crnn.evaluate_oto `
  --manifest ml_workspace\coarse_crnn\oto_splits\oto_test.jsonl `
  --model ml_workspace\models\coarse_crnn\oto_anchor_crnn.pt `
  --device cuda `
  --max-items 0 `
  --out ml_workspace\coarse_crnn\oto_eval.json
```

Predict one `oto.ini` row:

```powershell
python -m ml.scripts.coarse_crnn.predict_oto `
  --audio path\to\sample.wav `
  --alias "ka" `
  --language japanese `
  --format-type cv `
  --model ml_workspace\models\coarse_crnn\oto_anchor_crnn.pt
```
