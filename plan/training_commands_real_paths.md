# 모델 훈련 명령어 (Windows, 실제 경로)

작업 루트:
`C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO`

## 0) 공통 환경 설정
```powershell
Set-Location "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 1) 데이터 소스 스테이징
```powershell
python -X utf8 .\ml\scripts\coupled\stage_sources.py `
  --config ".\ml\configs\training_data_roots.yaml" `
  --dataset-root ".\dataset_staged"
```

## 2) 쌍 데이터 준비 (수동 OTO 우선, ML 자동 OTO 비활성)
```powershell
python -X utf8 .\ml\scripts\coupled\prepare_pairs.py `
  --dataset-root ".\dataset_staged" `
  --resume `
  --retry-failed
```

## 3) Coupled 학습용 CSV만 생성 (예: KR CVC)
```powershell
.\ml\scripts\run_coupled_mel_oto_training.ps1 `
  -Lang korean `
  -Format cvc `
  -BuildDatasetOnly `
  -DatasetRoot ".\dataset_staged" `
  -DatasetCsv ".\ml_workspace\datasets\korean\dataset_korean_cvc.csv" `
  -AutoOtoPolicy require
```

## 4) rawmel 캐시 생성 (v2 계열 학습 준비)
```powershell
python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_vcv.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing

python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cv.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing

python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset ".\ml_workspace\datasets\japanese\dataset_japanese_cvvc.csv" `
  --dataset-root ".\dataset_staged" `
  --allow-missing
```

## 5) 단일 포맷 Coupled v2(rawmel) 학습
```powershell
python -X utf8 .\ml\scripts\coupled\train.py `
  --lang korean `
  --format cvvc `
  --dataset .\ml_workspace\datasets\korean\dataset_korean_cvvc.csv `
  --out-dir .\ML_models\korean\cvvc\v2_coupled_rawmel_baseline `
  --backend coupled_nn_v2_rawmel `
  --rawmel-cache .\ml_workspace\cache\mel_patches\korean\cvvc\v2 `
  --device cuda `
  --epochs 80 `
  --batch-size 192 `
  --learning-rate 0.0007 `
  --min-confidence 0.65 `
  --progress-every 50

python -X utf8 .\ml\scripts\coupled\train.py `
  --lang korean `
  --format vcv `
  --dataset .\ml_workspace\datasets\korean\dataset_korean_vcv.csv `
  --out-dir .\ML_models\korean\vcv\v2_coupled_rawmel_baseline `
  --backend coupled_nn_v2_rawmel `
  --rawmel-cache .\ml_workspace\cache\mel_patches\korean\vcv\v2 `
  --device cuda `
  --epochs 80 `
  --batch-size 192 `
  --learning-rate 0.0007 `
  --min-confidence 0.655 `
  --progress-every 50

python -X utf8 .\ml\scripts\coupled\train.py `
  --lang japanese `
  --format cvvc `
  --dataset .\ml_workspace\datasets\japanese\dataset_japanese_cvvc.csv `
  --out-dir .\ML_models\japanese\cvvc\v2_coupled_rawmel_baseline `
  --backend coupled_nn_v2_rawmel `
  --rawmel-cache .\ml_workspace\cache\mel_patches\japanese\cvvc\v2 `
  --device cuda `
  --epochs 80 `
  --batch-size 192 `
  --learning-rate 0.0007 `
  --min-confidence 0.60 `
  --progress-every 50
```

## 6) 다중 포맷 일괄 학습
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

## 7) Fallback LightGBM 학습
```powershell
python -X utf8 .\ml\scripts\train_oto_lightgbm_model.py `
  --lang korean `
  --format cvc `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvc.csv" `
  --out-dir ".\ML_models\korean\cvc\v1_lightgbm" `
  --group-column voicebank_id `
  --min-mapping-confidence 0.40
```

## 8) Two-stage 튜닝 실행 (broad -> narrow)
```powershell
python -X utf8 .\scripts\run_coupled_two_stage_tune.py `
  --config ".\scripts\coupled_two_stage_tune.sample.yaml"
```

trial 수 직접 지정:
```powershell
python -X utf8 .\scripts\run_coupled_two_stage_tune.py `
  --config ".\scripts\coupled_two_stage_tune.sample.yaml" `
  --stage1-max-trials 12 `
  --stage2-max-trials 8 `
  --neighbor-window 1
```

1단계만 실행:
```powershell
python -X utf8 .\scripts\run_coupled_two_stage_tune.py `
  --config ".\scripts\coupled_two_stage_tune.sample.yaml" `
  --stage1-only
```

드라이런(학습 없이 조합만 점검):
```powershell
python -X utf8 .\scripts\run_coupled_two_stage_tune.py `
  --config ".\scripts\coupled_two_stage_tune.sample.yaml" `
  --force-dry-run
```

## 9) KR 회귀 배치 + 게이트 실행
```powershell
python -X utf8 .\scripts\run_kr_regression_suite.py `
  --batch-config ".\scripts\oto_batch_cases.sample.yaml" `
  --max-validation-warnings 180 `
  --max-validation-warning-rate 0.30
```

baseline `summary.json` 대비 악화 허용폭 체크:
```powershell
python -X utf8 .\scripts\run_kr_regression_suite.py `
  --batch-config ".\scripts\oto_batch_cases.sample.yaml" `
  --baseline-summary "C:\path\to\baseline\summary.json" `
  --max-ml-fallback-rate-delta 0.06 `
  --max-blank-flag-rate-delta 0.06 `
  --max-mel-unreliable-rate-delta 0.06 `
  --max-validation-warning-rate-delta 0.03
```
