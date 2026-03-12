# Ensemble v1 학습 명령

## 목적
- `scripts/train_ensemble_bundle.py`로 `lightgbm + coupled_nn_v2_rawmel + OOF meta` 번들을 한 번에 학습한다.
- 출력 번들은 루트 `model_meta.json` 아래에 `lightgbm`, `coupled`, `meta` 하위 번들을 가진다.

## 전제 조건
- 입력 CSV는 delta 학습용 컬럼과 rawmel 매칭용 키 컬럼을 포함해야 한다.
- `rawmel-cache-dir`는 같은 데이터셋과 같은 patch spec로 생성된 캐시여야 한다.
- 필터 후 최소 24행이 필요하다.
- `voicebank_id` 기준 OOF를 쓰려면 최소 3개 이상 bank가 있는 편이 맞다.

## 캐시 재생성이 필요한 경우
- `mel_patch_spec`가 바뀐 경우
- `wav_norm`, `alias_norm`, `occurrence_index`, `row_index_in_wav` 규칙이 바뀐 경우
- 학습 중 rawmel cache missing key 오류가 나는 경우

그 외에는 기존 캐시를 그대로 재사용해도 된다.

## 공통 환경

```powershell
$root = "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
Set-Location $root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 권장 명령 1: 한국어 CVVC 다중 bank

현재 가장 현실적인 기본 명령이다.

```powershell
python .\scripts\train_ensemble_bundle.py `
  --language korean `
  --format cvvc `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --rawmel-cache-dir "<rawmel_cache_dir>" `
  --out-dir ".\ML_models\korean\cvvc\v1_ensemble" `
  --group-column voicebank_id `
  --min-mapping-confidence 0.40 `
  --num-folds 5 `
  --lightgbm-num-boost-round 500 `
  --lightgbm-early-stopping-rounds 50 `
  --coupled-epochs 70 `
  --coupled-batch-size 192 `
  --coupled-learning-rate 0.001 `
  --alias-type cv `
  --alias-type cv_head `
  --alias-type vc `
  --alias-type vv `
  --alias-type vcv
```

## 권장 명령 2: 한국어 CVC 다중 bank

한국어 CVC에서도 같은 구조로 돌리면 된다.

```powershell
python .\scripts\train_ensemble_bundle.py `
  --language korean `
  --format cvc `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvc_coupled.csv" `
  --rawmel-cache-dir "<rawmel_cache_dir>" `
  --out-dir ".\ML_models\korean\cvc\v1_ensemble" `
  --group-column voicebank_id `
  --min-mapping-confidence 0.40 `
  --num-folds 5 `
  --lightgbm-num-boost-round 500 `
  --lightgbm-early-stopping-rounds 50 `
  --coupled-epochs 70 `
  --coupled-batch-size 192 `
  --coupled-learning-rate 0.001 `
  --alias-type cv `
  --alias-type cv_head `
  --alias-type vc `
  --alias-type vv `
  --alias-type vcv
```

## 권장 명령 3: 단일 bank 디버그용

단일 bank만 있을 때는 `voicebank_id` OOF가 사실상 무의미하므로 `wav_norm`으로 쪼개는 편이 낫다.
이 설정은 디버그용이고, 일반화 성능 판단용으로는 약하다.

```powershell
python .\scripts\train_ensemble_bundle.py `
  --language korean `
  --format cvvc `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --rawmel-cache-dir "<rawmel_cache_dir>" `
  --out-dir ".\ML_models\korean\cvvc\v1_ensemble_debug" `
  --group-column wav_norm `
  --min-mapping-confidence 0.35 `
  --num-folds 4 `
  --lightgbm-num-boost-round 500 `
  --lightgbm-early-stopping-rounds 50 `
  --coupled-epochs 70 `
  --coupled-batch-size 128 `
  --coupled-learning-rate 0.001 `
  --alias-type cv `
  --alias-type cv_head `
  --alias-type vc `
  --alias-type vv `
  --alias-type vcv
```

## 저메모리 변형

VRAM이 부족하면 batch만 줄이고 epoch는 유지하는 편이 낫다.

```powershell
python .\scripts\train_ensemble_bundle.py `
  --language korean `
  --format cvvc `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --rawmel-cache-dir "<rawmel_cache_dir>" `
  --out-dir ".\ML_models\korean\cvvc\v1_ensemble_lowvram" `
  --group-column voicebank_id `
  --min-mapping-confidence 0.40 `
  --num-folds 5 `
  --lightgbm-num-boost-round 500 `
  --lightgbm-early-stopping-rounds 50 `
  --coupled-epochs 80 `
  --coupled-batch-size 96 `
  --coupled-learning-rate 0.001
```

## 학습 후 확인 파일
- `ML_models\korean\cvvc\v1_ensemble\model_meta.json`
- `ML_models\korean\cvvc\v1_ensemble\eval_summary.json`
- `ML_models\korean\cvvc\v1_ensemble\lightgbm\model_meta.json`
- `ML_models\korean\cvvc\v1_ensemble\coupled\model_meta.json`
- `ML_models\korean\cvvc\v1_ensemble\meta\model_meta.json`

## 실행 후 기대 구조

```text
v1_ensemble/
  feature_schema.json
  model_meta.json
  eval_summary.json
  lightgbm/
  coupled/
  meta/
```

## 해석 기준
- `lightgbm`은 구조적 fallback이다.
- `coupled`는 rawmel 기반 주 모델이다.
- `meta`는 OOF 예측 위에 얹는 stacked meta-learner다.
- 실사용에서는 `ensemble_v1` 루트 번들을 설치하거나, 기존 coupled + lightgbm 조합에서 gated ensemble만 켜도 된다.
