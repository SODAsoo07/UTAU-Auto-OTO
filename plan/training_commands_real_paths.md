# 로컬 학습 명령어 (Windows, 실경로 기준)

기준 경로:
`C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO`

## 0) 공통 준비
```powershell
Set-Location "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 1) 소스 스테이징
```powershell
python -X utf8 .\ml\scripts\coupled\stage_sources.py `
  --config ".\ml\configs\training_data_roots.yaml" `
  --dataset-root ".\dataset_staged"
```

## 2) 페어 준비 (학습용 자동 OTO, ML 보정 기본 OFF)
```powershell
python -X utf8 .\ml\scripts\coupled\prepare_pairs.py `
  --dataset-root ".\dataset_staged" `
  --resume `
  --retry-failed
```

## 3) Coupled 학습 CSV 생성 (예: KR CVC)
```powershell
.\ml\scripts\run_coupled_mel_oto_training.ps1 `
  -Lang korean `
  -Format cvc `
  -BuildDatasetOnly `
  -DatasetRoot ".\dataset_staged" `
  -DatasetCsv ".\ml_workspace\datasets\korean\dataset_korean_cvc.csv" `
  -AutoOtoPolicy require
```

## 4) rawmel 캐시 생성 (v2 학습 전 필수 권장)
```powershell
python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cv.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing
```
복붙용
python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cv.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing

## 5) 단일 형식 학습 (예: KR CVC, v2 rawmel)
```powershell
python -X utf8 .\ml\scripts\train_oto_mel_coupled_model.py `
  --lang korean `
  --format cvc `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvc.csv" `
  --out-dir ".\ML_models\korean\cvc\coupled_v2_rawmel" `
  --backend coupled_nn_v2_rawmel `
  --device cuda `
  --epochs 120 `
  --batch-size 192 `
  --learning-rate 0.0007 `
  --group-column voicebank_id `
  --min-confidence 0.65
```

## 6) 일괄 학습 (언어별 여러 형식)
```powershell
python -X utf8 .\ml\scripts\coupled\train_matrix.py `
  --dataset-root ".\dataset_staged" `
  --workspace-root ".\ml_workspace" `
  --model-root ".\ML_models" `
  --languages "korean,japanese" `
  --formats "korean=cv,cvc,cvvc,vcv;japanese=cv,cvvc,vcv" `
  --auto-oto-policy require `
  --backend coupled_nn_v2_rawmel `
  --device cuda
```

## 7) fallback LightGBM 학습
```powershell
python -X utf8 .\ml\scripts\train_oto_lightgbm_model.py `
  --lang korean `
  --format cvc `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvc.csv" `
  --out-dir ".\ML_models\korean\cvc\v1_lightgbm" `
  --group-column voicebank_id `
  --min-mapping-confidence 0.40
```
