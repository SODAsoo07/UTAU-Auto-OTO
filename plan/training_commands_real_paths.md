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
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_vcv.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing
```
복붙용
python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cv.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing

python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset ".\ml_workspace\datasets\japanese\dataset_japanese_cvvc.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing

## 5) 단일 형식 학습 (예: KR CVC, v2 rawmel)
```powershell
python -X utf8 ml\scripts\coupled\train.py `
  --lang korean `
  --format cvvc `
  --dataset ml_workspace\datasets\korean\dataset_korean_cvvc.csv `
  --out-dir ML_models\korean\cvvc\v2_coupled_rawmel_baseline `
  --backend coupled_nn_v2_rawmel `
  --rawmel-cache ml_workspace\cache\mel_patches\korean\cvvc\v2 `
  --device cuda `
  --epochs 80 `
  --batch-size 192 `
  --learning-rate 0.0007 `
  --min-confidence 0.65 `
  --progress-every 50
```
복붙용
python -X utf8 ml\scripts\coupled\train.py `
  --lang korean `
  --format vcv `
  --dataset ml_workspace\datasets\korean\dataset_korean_vcv.csv `
  --out-dir ML_models\korean\vcv\v2_coupled_rawmel_baseline `
  --backend coupled_nn_v2_rawmel `
  --rawmel-cache ml_workspace\cache\mel_patches\korean\vcv\v2 `
  --device cuda `
  --epochs 80 `
  --batch-size 192 `
  --learning-rate 0.0007 `
  --min-confidence 0.655 `
  --progress-every 50

복붙용2
python -X utf8 ml\scripts\coupled\train.py `
  --lang japanese `
  --format cvvc `
  --dataset ml_workspace\datasets\japanese\dataset_japanese_cvvc.csv `
  --out-dir ML_models\japanese\cvvc\v2_coupled_rawmel_baseline `
  --backend coupled_nn_v2_rawmel `
  --rawmel-cache ml_workspace\cache\mel_patches\japanese\cvvc\v2 `
  --device cuda `
  --epochs 80 `
  --batch-size 192 `
  --learning-rate 0.0007 `
  --min-confidence 0.6 `
  --progress-every 50

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
