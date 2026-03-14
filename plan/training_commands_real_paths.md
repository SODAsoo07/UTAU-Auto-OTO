# Coupled Model Training Commands (Real Paths)

최종 갱신: 2026-03-15

기준 워크스페이스:
`C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO`

현재 권장 학습 백엔드:
- `coupled_nn_v2_rawmel` (1순위)
- `lightgbm` (fallback/비교군)

산출물(커플링 모델 디렉터리):
- `coupled_model.pt`
- `feature_schema.json`
- `model_meta.json`
- `eval_summary.json`

코랩 전용 전처리+훈련 노트북:
- `ml/notebooks/coupled_preprocess_train_colab_utf8.ipynb`

## 0) 프로젝트 루트 및 UTF-8
```powershell
Set-Location "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 1) (선택) 원본 보이스뱅크를 dataset으로 스테이징
```powershell
python -X utf8 .\ml\scripts\coupled\stage_sources.py `
  --config "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml\configs\training_data_roots.yaml" `
  --dataset-root "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset"
```

## 2) 전처리(prepare_pairs): TextGrid/auto oto 준비
`dataset`에 TextGrid/oto_auto_ml.ini가 불완전한 경우 실행.
```powershell
python -X utf8 .\ml\scripts\coupled\prepare_pairs.py `
  --dataset-root "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset" `
  --resume `
  --retry-failed
```

## 3) Coupled 학습 CSV 빌드
실무에서는 오케스트레이터 사용을 권장.
```powershell
.\ml\scripts\run_coupled_mel_oto_training.ps1 `
  -Lang korean `
  -Format cvvc `
  -BuildDatasetOnly `
  -DatasetRoot "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset" `
  -DatasetCsv "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  -AutoOtoPolicy generate-temp
```

## 4) rawmel patch cache 생성 (v2_rawmel 필수)
```powershell
python -X utf8 .\ml\scripts\coupled\build_mel_patch_cache.py `
  --dataset "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --dataset-root "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\dataset"
```

## 5) Coupled v2(rawmel) 학습
```powershell
python -X utf8 .\ml\scripts\train_oto_mel_coupled_model.py `
  --lang korean `
  --format cvvc `
  --dataset "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --out-dir "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ML_models\korean\cvvc\v2_coupled_rawmel" `
  --backend coupled_nn_v2_rawmel `
  --rawmel-cache "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\mel_patch_cache\korean\cvvc\v1\<spec_hash>" `
  --device cuda `
  --epochs 70 `
  --batch-size 192 `
  --learning-rate 0.001 `
  --group-column voicebank_id `
  --min-confidence 0.55
```

`--rawmel-cache`에는 `build_mel_patch_cache.py` 실행 결과로 생성된 `manifest.json`이 있는 디렉터리를 넣는다.

## 6) (선택) LightGBM fallback 학습
```powershell
python -X utf8 .\ml\scripts\train_oto_lightgbm_model.py `
  --lang korean `
  --format cvvc `
  --dataset "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --out-dir "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ML_models\korean\cvvc\v1_lightgbm" `
  --group-column voicebank_id `
  --min-mapping-confidence 0.40
```

## 7) 런타임 권장 설정(운영안)
```powershell
$env:UTOA_ML_ROUTE = "autofree_v1"
$env:UTOA_ML_COUPLED_ENABLE = "1"
$env:UTOA_ML_GATED_ENSEMBLE_ENABLE = "0"
$env:UTOA_ML_ENSEMBLE_ENABLE = "0"
$env:UTOA_ML_COUPLED_BACKEND = "v2"
$env:UTOA_ML_COUPLED_MIN_CONF = "0.55"
$env:UTOA_ML_AUTOFREE_AUX_ENABLE = "1"
```
