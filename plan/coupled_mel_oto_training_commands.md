# Coupled mel+oto 데이터 전처리/학습 명령어

이 문서는 `coupled_nn_v1`(mel+oto 결합) 기준으로, 데이터 준비부터 학습/평가/설치까지 실행 순서를 정리한다.

## 원클릭 스크립트

복붙용 PowerShell 스크립트:
- `ml/scripts/run_coupled_mel_oto_training.ps1`
- `ml/scripts/run_coupled_mel_oto_training_ko_expanded.ps1` (한국어 `CV/CVC/VCV` 일괄)

예시 1) 준비 리포트 기반으로 CSV 생성 + 학습 + 평가

```powershell
powershell -ExecutionPolicy Bypass -File "$root\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -DatasetRoot "$root\dataset" `
  -Device cuda
```

예시 2) 단일 작업 폴더 기반으로 CSV 생성 + 학습 + 평가

```powershell
powershell -ExecutionPolicy Bypass -File "$root\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -SingleWorkDir "$root\dataset\korean\cvvc\BankA\work" `
  -SingleManualOto "$root\dataset\korean\cvvc\BankA\work\oto.ini" `
  -SingleVoicebankId "BankA" `
  -Device cpu
```

예시 3) 스테이징 + auto 준비 + 학습 + export + install

```powershell
powershell -ExecutionPolicy Bypass -File "$root\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -StageSources `
  -PrepareAuto `
  -Device cpu `
  -ExportBundle `
  -InstallBundle
```

예시 4) 실행 없이 명령만 확인

```powershell
powershell -ExecutionPolicy Bypass -File "$root\ml\scripts\run_coupled_mel_oto_training.ps1" `
  -Lang korean `
  -Format cvvc `
  -DryRun
```

예시 5) 한국어 `CV/CVC/VCV` 확장 학습 일괄 실행

```powershell
powershell -ExecutionPolicy Bypass -File "$root\ml\scripts\run_coupled_mel_oto_training_ko_expanded.ps1" `
  -Formats "cv,cvc,vcv" `
  -DatasetRoot "$root\dataset" `
  -Device cpu `
  -ExportBundle `
  -InstallBundle
```

필요 시 `-Formats "cv,cvc,cvvc,vcv"`로 `CVVC`까지 포함 가능.
기본 동작은 데이터가 없는 포맷을 자동으로 `skip`하고 다음 포맷으로 진행한다.
모든 포맷을 반드시 성공시켜야 하면 `-RequireAllFormats`를 추가한다.

## 0) 공통 환경 설정

```powershell
$root="C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO"
Set-Location $root
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
```

권장: 가상환경에서 실행.

---

## 1) 전처리 A: 학습 소스 스테이징 (선택)

`ml/configs/training_data_roots.yaml`에 등록된 원본 경로를 `dataset` 폴더로 복사한다.

```powershell
python ml\scripts\stage_training_sources.py `
  --config "$root\ml\configs\training_data_roots.yaml" `
  --dataset-root "$root\dataset"
```

---

## 2) 전처리 B: auto OTO/TextGrid 생성 (선택)

`dataset` 내부 작업 디렉토리를 스캔해서 다음을 생성한다.
- lab / dictionary
- MFA 정렬(TextGrid)
- auto OTO: `oto_auto_ml.ini`

```powershell
python ml\scripts\prepare_staged_auto_pairs.py `
  --dataset-root "$root\dataset"
```

사전 점검(dry-run):

```powershell
python ml\scripts\prepare_staged_auto_pairs.py `
  --dataset-root "$root\dataset" `
  --dry-run
```

결과 리포트:
- `$root\dataset\_manifest\prepared_auto_pairs.json`

---

## 3) 전처리 C: coupled 학습 CSV 생성

## 3-1) 단일 보이스뱅크 예시

```powershell
python ml\scripts\build_oto_mel_coupled_dataset.py `
  --lang korean `
  --auto "$root\dataset\korean\cvvc\BankA\work\oto_auto_ml.ini" `
  --manual "$root\dataset\korean\cvvc\BankA\work\oto.ini" `
  --tg-dir "$root\dataset\korean\cvvc\BankA\work" `
  --wav-dir "$root\dataset\korean\cvvc\BankA\work" `
  --out "$root\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --voicebank-id "BankA" `
  --format-override cvvc
```

옵션:
- `--append`: 기존 CSV 뒤에 누적
- `--custom-phonemes <path>`

## 3-2) 다중 보이스뱅크 일괄 생성 예시 (리포트 기반)

아래 스니펫은 `prepared_auto_pairs.json`을 읽어 언어/포맷별 CSV로 누적 생성한다.

```powershell
$report="$root\dataset\_manifest\prepared_auto_pairs.json"
$items=(Get-Content $report -Encoding UTF8 | ConvertFrom-Json).items

$seen=@{}
foreach($it in $items){
  if($it.status -notin @("prepared","prepared_existing")){ continue }
  $lang=($it.language).ToLower()
  $fmt=($it.format_type).ToLower()
  $work=$it.work_dir
  $auto=Join-Path $work "oto_auto_ml.ini"
  $manual=$it.manual_oto
  if(!(Test-Path $auto) -or !(Test-Path $manual)){ continue }

  $outDir=Join-Path $root ("ml_workspace\datasets\" + $lang)
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $outCsv=Join-Path $outDir ("dataset_{0}_{1}_coupled.csv" -f $lang,$fmt)
  $key="$lang|$fmt"
  $append=$seen.ContainsKey($key)

  $args=@(
    "ml\scripts\build_oto_mel_coupled_dataset.py",
    "--lang",$lang,
    "--auto",$auto,
    "--manual",$manual,
    "--tg-dir",$work,
    "--wav-dir",$work,
    "--out",$outCsv,
    "--voicebank-id",(Split-Path $it.stage_root -Leaf),
    "--format-override",$fmt
  )
  if($append){ $args += "--append" }

  python @args
  if($LASTEXITCODE -ne 0){ throw "build failed: $work" }
  $seen[$key]=$true
}
```

---

## 4) coupled 모델 학습

CPU 전용 예시:

```powershell
python ml\scripts\train_oto_mel_coupled_model.py `
  --lang korean `
  --format cvvc `
  --dataset "$root\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --out-dir "$root\ML_models\korean\cvvc\v1_coupled" `
  --device cpu `
  --epochs 70 `
  --batch-size 192 `
  --learning-rate 0.001 `
  --min-confidence 0.55 `
  --group-column voicebank_id `
  --min-mapping-confidence 0.0
```

자주 쓰는 추가 옵션:
- `--alias-types cv,cv_head,vc,vv,vcv` (특정 alias_type만 학습)
- `--device auto|cpu|cuda`

---

## 5) 평가

```powershell
python ml\scripts\evaluate_oto_mel_coupled.py `
  --model-dir "$root\ML_models\korean\cvvc\v1_coupled" `
  --dataset "$root\ml_workspace\datasets\korean\dataset_korean_cvvc_coupled.csv" `
  --lang korean `
  --format cvvc `
  --device cpu `
  --report "$root\logs\eval_korean_cvvc_coupled.json"
```

---

## 6) 번들 export / install

```powershell
python ml\scripts\export_oto_ml_bundle.py `
  --model-dir "$root\ML_models\korean\cvvc\v1_coupled" `
  --export-root "$root\ml_workspace\exports\model_bundles" `
  --zip

python ml\scripts\install_oto_ml_bundle.py `
  --bundle "$root\ml_workspace\exports\model_bundles\korean_cvvc_v1_coupled.zip" `
  --install-root "$root\models_installed\oto_ml"
```

---

## 7) 런타임 환경변수 (coupled 적용)

```powershell
$env:UTOA_ML_COUPLED_ENABLE="1"
$env:UTOA_ML_COUPLED_MIN_CONF="0.55"
$env:UTOA_ML_COUPLED_DEVICE="cpu"   # auto/cpu/cuda
$env:UTOA_ML_COUPLED_STRICT_CONSTRAINT="0"  # 0/1
```

---

## 8) 빠른 체크리스트
- `oto_auto_ml.ini`, 수작업 OTO, TextGrid, WAV가 같은 작업 디렉토리에 존재하는지 확인
- 학습 CSV에서 `matched_rows`가 충분한지 확인 (너무 적으면 학습 불안정)
- 학습 후 `model_meta.json`의 `backend`가 `coupled_nn_v1`인지 확인
- 평가 리포트(`model_mae`)가 baseline 대비 개선되는지 확인
