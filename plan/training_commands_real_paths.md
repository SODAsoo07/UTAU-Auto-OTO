# Training Commands (Real Paths)

This document uses absolute paths from this workspace:
`C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO`

## 0) Project root
```powershell
Set-Location "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 1) Stage sources into dataset
```powershell
python -X utf8 .\ml\scripts\stage_training_sources.py `
  --config "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml\configs\training_data_roots.yaml" `
  --dataset-root "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset"
```

## 2A) Build dataset CSV without auto OTO (TextGrid required)
```powershell
.\ml\scripts\run_coupled_mel_oto_training.ps1 `
  -StageSources `
  -BuildDatasetOnly `
  -AutoOtoPolicy generate-temp
  -format cvc
```

## 2B) Optional: generate auto OTO + TextGrid first
Use this if TextGrid is missing in `C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset`.
```powershell
python -X utf8 .\ml\scripts\prepare_staged_auto_pairs.py `
  --dataset-root "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset"
```

## 3) Build rawmel patch cache (required for coupled v2 rawmel)
```powershell
python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --dataset-root "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset"
```

## 4) Train coupled (rawmel) model
```powershell
python -X utf8 .\ml\scripts\train_oto_mel_coupled_model.py `
  --lang korean `
  --format cvvc `
  --dataset "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --out-dir "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ML_models\korean\cvvc\v1_coupled" `
  --device cpu `
  --epochs 70 `
  --batch-size 192 `
  --learning-rate 0.001
```

## 5) Evaluate coupled model
```powershell
python -X utf8 .\ml\scripts\evaluate_oto_mel_coupled.py `
  --model-dir "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ML_models\korean\cvvc\v1_coupled" `
  --dataset "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --lang korean `
  --format cvvc `
  --report "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\logs\eval_korean_cvvc_coupled.json"
```

## 6) Train LightGBM (optional, for gated fallback)
```powershell
python -X utf8 .\ml\scripts\train_oto_lightgbm_model.py `
  --lang korean `
  --format cvvc `
  --dataset "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --out-dir "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ML_models\korean\cvvc\v1_lightgbm" `
  --group-column voicebank_id `
  --min-mapping-confidence 0.40
```

## 7) Enable gated runtime
```powershell
$env:UTOA_ML_GATED_ENSEMBLE_ENABLE = "1"
$env:UTOA_ML_COUPLED_ENABLE = "1"
$env:UTOA_ML_ENSEMBLE_ENABLE = "0"
$env:UTOA_ML_COUPLED_MIN_CONF = "0.55"
```
