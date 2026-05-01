# coarse_crnn workflow

This workflow is intentionally separate from the existing `sequence` aligner.

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

Run one alignment:

```powershell
python -m ml.scripts.coarse_crnn.align `
  --audio path\to\sample.wav `
  --language ja `
  --model ml_workspace\models\coarse_crnn\coarse_crnn.pt `
  --out output\coarse_crnn
```
