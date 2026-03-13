# Coupled Mel+OTO 학습 절차

## 개요
- 리팩토링 이후 mel 커플링 모델의 권장 진입점은 `ml/scripts/coupled/*.py`입니다.
- 기존 `ml/scripts/*.py`와 `ml/scripts/run_coupled_mel_oto_training.ps1`는 호환용 래퍼로 봅니다.
- 학습/평가 홀드아웃은 기본적으로 `wav_norm` 기준입니다.
  - 같은 녹음 파일 안의 VC/CV 관계가 train/valid에 섞이지 않도록 하기 위한 기본 설정입니다.

## 전처리에서 alias가 처리되는 방식
학습 전처리에서는 alias를 그대로 쓰지 않고 아래 순서로 정리합니다.

1. `prefix.map` 탐색
- `oto.ini`, `wav_dir`, `tg_dir` 기준으로 상위 폴더까지 탐색합니다.
- `prefix.map`이 있으면 우선 사용합니다.

2. 다른 `*.map` fallback 탐색
- `prefix.map`이 없으면 같은 위치의 다른 `*.map` 파일도 후보로 봅니다.
- note / prefix / suffix 규칙으로 실제 파싱되는 파일만 사용합니다.
- 일반적인 다른 용도의 `.map` 파일은 가능한 한 무시합니다.

3. prefix/suffix 제거
- note 문맥에 맞는 prefix/suffix를 alias에서 제거합니다.
- 예: `PRE_ga_SUF` -> `ga`
- 예: `ga_C4` -> `ga`

4. 음계 affix 제거
- `_A3`, `-C4`, ` F4` 같은 pitch affix를 제거합니다.

5. 주석성 접미사 제거
- alias 뒤에 붙은 한자, 한글, 메모성 문자열을 제거합니다.
- 예: `ga注釈` -> `ga`
- 예: `ga메모` -> `ga`

6. 숨소리 정규화
- `吸こ吐`, `吸き吐`, `吸き吐_A3` 같은 변형도 `br`로 정규화합니다.

이 과정을 거친 뒤 `alias_norm`, `alias_type`, `occurrence_index`가 결정됩니다.

## 다시 전처리/재학습이 필요한 경우
아래 중 하나라도 바뀌면 기존 CSV를 재사용하지 않는 편이 맞습니다.

- `FEATURE_VERSION` 변경
- `TRAIN_ROW_MATCH_VERSION` 변경
- alias 정규화 규칙 변경
- 숨소리 분류 규칙 변경
- 주석성 접미사 제거 규칙 변경
- `prefix.map` / `*.map` affix 처리 규칙 변경

실무적으로는 아래 순서가 안전합니다.
- `build_dataset` 다시 실행
- 새 CSV로 `train.py` 다시 실행
- 필요하면 `evaluate.py`와 export/install 재실행

## 사전 준비
1. 작업 디렉터리를 프로젝트 루트로 둡니다.
2. Python 의존성을 설치합니다.
3. TextGrid/MFA를 사용할 경우 관련 도구가 준비되어 있어야 합니다.

예시:

```powershell
$root = "C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
Set-Location $root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## 1. 학습 소스 스테이징
`training_data_roots.yaml` 기준으로 학습 소스를 `dataset` 아래에 모읍니다.

```powershell
python .\ml\scripts\coupled\stage_sources.py `
  --config ".\ml\configs\training_data_roots.yaml" `
  --dataset-root ".\dataset"
```

생성물:
- `dataset\_manifest\staged_sources.json`
- `dataset\_manifest\staged_sources.csv`

## 2. auto OTO / TextGrid 준비
staged dataset을 기준으로 `oto_auto_ml.ini`, lab, TextGrid 관련 산출물을 준비합니다.

```powershell
python .\ml\scripts\coupled\prepare_pairs.py `
  --dataset-root ".\dataset"
```

점검만 할 때:

```powershell
python .\ml\scripts\coupled\prepare_pairs.py `
  --dataset-root ".\dataset" `
  --dry-run
```

생성물:
- `dataset\_manifest\prepared_auto_pairs.json`

## 준비 작업 재개 / 실패 항목 재시도
`prepare_pairs.py`는 이제 `dataset\_manifest\prepared_auto_pairs.json`를 체크포인트로 사용합니다.
각 work item이 끝날 때마다 리포트가 갱신되므로, 중간에 중단돼도 다음 실행에서 이어서 진행할 수 있습니다.

중단된 지점 이후만 계속 진행:

```powershell
python .\ml\scripts\coupled\prepare_pairs.py `
  --dataset-root ".\dataset" `
  --resume
```

이전에 `align_failed:*`로 건너뛴 항목까지 다시 시도:

```powershell
python .\ml\scripts\coupled\prepare_pairs.py `
  --dataset-root ".\dataset" `
  --resume `
  --retry-failed
```

주의:
- `--resume`은 이미 `prepared` 또는 `prepared_existing`로 기록된 항목을 건너뜁니다.
- `--retry-failed`는 `skip` 상태로 기록된 항목도 다시 처리합니다.
- `-StageSources`를 다시 실행하면 `dataset` 폴더가 재구성될 수 있으므로, 준비 재개는 보통 스테이징을 다시 하지 않는 상황에서 사용하는 편이 안전합니다.

## 3. coupled 학습용 CSV 생성
단일 work 디렉터리 예시:

```powershell
python .\ml\scripts\coupled\build_dataset.py `
  --lang korean `
  --auto ".\dataset\korean\cvvc\BankA\work\oto_auto_ml.ini" `
  --manual ".\dataset\korean\cvvc\BankA\work\oto.ini" `
  --tg-dir ".\dataset\korean\cvvc\BankA\work" `
  --wav-dir ".\dataset\korean\cvvc\BankA\work" `
  --out ".\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --voicebank-id "BankA"   --format-override cvvc`

```

참고:
- 이 단계에서 alias 정규화, 숨소리 처리, 접미 메모 제거, `prefix.map`/`*.map` affix 제거가 모두 적용됩니다.
- 위 규칙이 바뀌었으면 기존 CSV 대신 새 CSV를 다시 생성해야 합니다.

## 4. 모델 학습

```powershell
python .\ml\scripts\coupled\train.py `
  --lang korean `
  --format cvvc `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --out-dir ".\ML_models\korean\cvvc\v1_coupled" `
  --device cuda `
  --epochs 70 `
  --batch-size 192 `
  --learning-rate 0.001 `
  --min-confidence 0.55 `
  --min-mapping-confidence 0.0
```

주요 옵션:
- `--alias-types cv,cv_head,vc,vv,vcv`
- `--device auto|cpu|cuda`
- `--group-column wav_norm`

## 5. 평가

```powershell
python .\ml\scripts\coupled\evaluate.py `
  --model-dir ".\ML_models\korean\cvvc\v1_coupled" `
  --dataset ".\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --lang korean `
  --format cvvc `
  --device cuda `
  --report ".\logs\eval_korean_cvvc_coupled.json"
```

주요 확인 파일:
- `ML_models\korean\cvvc\v1_coupled\model_meta.json`
- `ML_models\korean\cvvc\v1_coupled\eval_summary.json`
- `logs\eval_korean_cvvc_coupled.json`

## 6. export / install

```powershell
python .\ml\scripts\coupled\export.py `
  --model-dir ".\ML_models\korean\cvvc\v1_coupled" `
  --export-root ".\ml_workspace\exports\model_bundles" `
  --zip
```

```powershell
python .\ml\scripts\coupled\install.py `
  --bundle ".\ml_workspace\exports\model_bundles\korean_cvvc_v1.zip" `
  --install-root ".\models_installed\oto_ml"
```

## PowerShell 래퍼로 한 번에 실행

```powershell
powershell -ExecutionPolicy Bypass -File ".\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -StageSources `
  -PrepareAuto `
  -Device cuda
```

export/install까지 포함:

```powershell
powershell -ExecutionPolicy Bypass -File ".\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -StageSources `
  -PrepareAuto `
  -Device cuda `
  -ExportBundle `
  -InstallBundle
```

준비 단계만 다시 이어서 실행:

```powershell
powershell -ExecutionPolicy Bypass -File ".\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -PrepareAuto `
  -ResumePrepare
```

이전 실패 항목까지 다시 시도:

```powershell
powershell -ExecutionPolicy Bypass -File ".\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -PrepareAuto `
  -ResumePrepare `
  -RetryFailedPrepare
```

학습만
powershell -ExecutionPolicy Bypass -File ".\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -Device cuda `
  -ExportBundle `
  -InstallBundle

powershell -ExecutionPolicy Bypass -File ".\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format vcv `
  -Device cuda `
  -ExportBundle `
  -InstallBundle

## 단일 work 디렉터리 대상 실행 예시

```powershell
powershell -ExecutionPolicy Bypass -File ".\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -SingleWorkDir ".\dataset\korean\cvvc\BankA\work" `
  -SingleManualOto ".\dataset\korean\cvvc\BankA\work\oto.ini" `
  -SingleVoicebankId "BankA" `
  -Device cpu
```

## 관련 환경 변수
- `UTOA_ML_VC_CV_PAIR_WEIGHT`
- `UTOA_ML_VC_CV_MAX_GAP`

## 권장 점검 항목
- `prefix.map` 또는 다른 `*.map`이 있는 bank에서 alias가 예상대로 벗겨졌는지 확인
- 숨소리 alias가 `br`로 잘 수렴했는지 확인
- 접미 메모가 있는 alias가 같은 `alias_norm`으로 묶였는지 확인
- `wav_norm` 홀드아웃으로 train/valid 누수가 없는지 확인
