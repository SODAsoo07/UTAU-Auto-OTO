# ML Training And Test Commands

This document summarizes the training and test commands used for selector retraining and validation.

## Selector Training (PowerShell)

Set a root path for convenience:

```powershell
$root="C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
```

### Targeted CVVC (recommended first)

```powershell
python "$root\scripts\train_selector_bundle.py" `
  --language japanese --format cvvc `
  --delta-dataset "$root\ml_workspace\datasets\japanese\dataset_japanese_cvvc.csv" `
  --out-dir "$root\ML_models\japanese_cvvc_profile_run_new" `
  --objective auto

python "$root\scripts\train_selector_bundle.py" `
  --language korean --format cvvc `
  --delta-dataset "$root\ml_workspace\datasets\korean\dataset_korean_cvvc.csv" `
  --out-dir "$root\ML_models\korean_cvvc_profile_run_new" `
  --objective auto
```

### Full Formats (optional)

```powershell
python "$root\scripts\train_selector_bundle.py" --language japanese --format cv --delta-dataset "$root\ml_workspace\datasets\japanese\dataset_japanese_cv.csv" --out-dir "$root\ML_models\japanese_cv_profile_run_new" --objective auto
python "$root\scripts\train_selector_bundle.py" --language japanese --format vcv --delta-dataset "$root\ml_workspace\datasets\japanese\dataset_japanese_vcv.csv" --out-dir "$root\ML_models\japanese_vcv_profile_run_new" --objective auto

python "$root\scripts\train_selector_bundle.py" --language korean --format cv --delta-dataset "$root\ml_workspace\datasets\korean\dataset_korean_cv.csv" --out-dir "$root\ML_models\korean_cv_profile_run_new" --objective auto
python "$root\scripts\train_selector_bundle.py" --language korean --format cvc --delta-dataset "$root\ml_workspace\datasets\korean\dataset_korean_cvc.csv" --out-dir "$root\ML_models\korean_cvc_profile_run_new" --objective auto
python "$root\scripts\train_selector_bundle.py" --language korean --format vcv --delta-dataset "$root\ml_workspace\datasets\korean\dataset_korean_vcv.csv" --out-dir "$root\ML_models\korean_vcv_profile_run_new" --objective auto
```

### Optional Flags

Add these to the training command if needed:

- `--use-pseudo-labels`
- `--min-mapping-confidence 0.65`
- `--exclude-nuclei-fallback`
- `--require-train-keep`

## Hard-Negative Summary Report

```powershell
python "$root\scripts\report_selector_hard_negatives.py" `
  --log-path "$root\logs\selector_hard_negatives.jsonl" `
  --out-dir "$root\logs"
```

Outputs:
- `selector_hard_negative_summary.json`
- `selector_hard_negative_top_aliases.csv`

## Tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q ml/tests/test_oto_ml_selector.py ml/tests/test_oto_ml_policy.py ml/tests/test_oto_ml_refiner_clip_by_alias.py
```

## Runtime Tuning (optional)

Environment variables for selector behavior:

- `UTOA_SELECTOR_RANKING_MIN_GROUPS` (default: 120)
- `UTOA_SELECTOR_MIN_MARGIN` (overrides margin threshold)
- `UTOA_SELECTOR_HN_LOG` (override hard-negative log path)
