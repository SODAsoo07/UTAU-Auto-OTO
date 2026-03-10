# ML Training And Test Commands

이 문서는 LightGBM 경로와 `mel+oto coupled_nn_v1` 경로를 함께 다룹니다.

## 공통 준비 (PowerShell)

```powershell
$root="C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
```

## 1) LightGBM delta/selector 학습

```powershell
python "$root\ml\scripts\train_oto_ml_model.py" `
  --lang korean --format cvvc `
  --dataset "$root\ml_workspace\datasets\korean\dataset_korean_cvvc.csv" `
  --out-dir "$root\ML_models\korean\cvvc\v1" `
  --include-selector --selector-objective pointwise
```

필요 시 옵션:
- `--alias-family cv|vowel|bridge`
- `--min-mapping-confidence 0.65`
- `--exclude-nuclei-fallback`
- `--require-train-keep`
- `--use-pseudo-labels`

## 2) Coupled mel+oto 데이터셋 생성

```powershell
python "$root\ml\scripts\build_oto_mel_coupled_dataset.py" `
  --lang korean `
  --auto "$root\dataset_stage\korean\cvvc\BankA\oto_auto.ini" `
  --manual "$root\dataset_stage\korean\cvvc\BankA\oto_manual.ini" `
  --tg-dir "$root\dataset_stage\korean\cvvc\BankA\textgrid" `
  --wav-dir "$root\dataset_stage\korean\cvvc\BankA\wav" `
  --out "$root\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --voicebank-id "BankA"
```

## 3) Coupled mel+oto 모델 학습

GPU 없이 CPU 학습 강제:

```powershell
python "$root\ml\scripts\train_oto_mel_coupled_model.py" `
  --lang korean --format cvvc `
  --dataset "$root\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --out-dir "$root\ML_models\korean\cvvc\v1_coupled" `
  --device cpu `
  --epochs 70 `
  --batch-size 192 `
  --learning-rate 0.001 `
  --min-confidence 0.55
```

## 4) Coupled 모델 평가

```powershell
python "$root\ml\scripts\evaluate_oto_mel_coupled.py" `
  --model-dir "$root\ML_models\korean\cvvc\v1_coupled" `
  --dataset "$root\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --lang korean --format cvvc `
  --device cpu `
  --report "$root\logs\eval_korean_cvvc_coupled.json"
```

## 5) 번들 export/install

```powershell
python "$root\ml\scripts\export_oto_ml_bundle.py" `
  --model-dir "$root\ML_models\korean\cvvc\v1_coupled" `
  --export-root "$root\ml_workspace\exports\model_bundles" `
  --zip

python "$root\ml\scripts\install_oto_ml_bundle.py" `
  --bundle "$root\ml_workspace\exports\model_bundles\korean_cvvc_v1_coupled.zip" `
  --install-root "$root\models_installed\oto_ml"
```

참고:
- lightgbm 번들은 `model_*.txt` 5개 필수.
- coupled 번들은 `coupled_model.pt` 필수.
- 공통 필수 파일: `feature_schema.json`, `model_meta.json`, `eval_summary.json`.

## 6) 런타임 환경 변수

coupled/fallback:
- `UTOA_ML_COUPLED_ENABLE=1`
- `UTOA_ML_COUPLED_MIN_CONF=0.55`
- `UTOA_ML_COUPLED_DEVICE=auto|cpu|cuda`
- `UTOA_ML_COUPLED_STRICT_CONSTRAINT=0|1`

selector:
- `UTOA_SELECTOR_RANKING_MIN_GROUPS` (default 120)
- `UTOA_SELECTOR_MIN_MARGIN`
- `UTOA_SELECTOR_HN_LOG`

## 7) 선별 테스트

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q `
  ml/tests/test_runtime_loading.py `
  ml/tests/test_runtime_fallbacks.py `
  ml/tests/test_export_bundle.py `
  ml/tests/test_bundle_install.py `
  ml/tests/test_ja_vcv_row_v2.py `
  ml/tests/test_kr_mapping_select_v2.py `
  ml/tests/test_ml_fallback.py
```
