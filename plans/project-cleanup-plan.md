# AutoOTO 프로젝트 정리 계획서

작성일: 2026-05-16  
대상 프로젝트: `SODAsoo07/Auto_OTO`  
목적: 코드 삭제보다 **역할 분리, 카테고리화, 배포 경계 고정, AI 보조 개발 안정화**를 우선하여 프로젝트 관리 난이도를 낮춘다.

---

## 0. 현재 판단 요약

AutoOTO는 현재 전체 약 10만 줄, 스크립트/모듈 약 320개 규모로 성장했다.  
하지만 전체 코드의 상당 부분이 모델 훈련, 평가, 내부 테스트, 실험 코드이고, 실제 배포 런타임에는 약 2~3만 줄만 포함되는 것으로 추정된다.

따라서 현재 문제는 다음이 아니다.

- 코드가 무조건 너무 많다
- 파일을 대량 삭제해야 한다
- 기능을 줄여야 한다

현재 핵심 문제는 다음이다.

- 런타임 코드와 학습/실험/테스트 코드가 구조상 섞여 보인다
- `scripts/`의 용도가 한눈에 보이지 않는다
- AI 보조 개발 시 잘못된 파일을 수정할 위험이 커졌다
- 배포 포함/제외 기준이 문서와 빌드 설정 양쪽에서 더 명확해야 한다
- 앞으로 자체 정렬기, CRNN, C# GUI 전환 등이 추가되면 구조 혼란이 커질 수 있다

> 결론: **코드량 감축보다 카테고리화와 경계 고정이 우선이다.**

---

## 1. 정리 목표

## 1.1 1차 목표

- [ ] 배포 런타임 코드와 개발/학습/실험 코드를 명확히 분리한다.
- [ ] `scripts/` 폴더를 목적별로 재분류한다.
- [ ] 각 주요 폴더에 `README.md`를 추가해 역할과 import 금지 규칙을 적는다.
- [ ] 배포 빌드 포함/제외 규칙을 문서화한다.
- [ ] AI/Codex가 수정해야 할 범위를 쉽게 찾을 수 있게 한다.

## 1.2 하지 않을 일

- [ ] 코드 줄 수를 억지로 줄이지 않는다.
- [ ] 실험 코드를 바로 삭제하지 않는다.
- [ ] 현재 정상 동작하는 런타임 경로를 대규모 이동하지 않는다.
- [ ] 기능 추가와 구조 정리를 동시에 크게 진행하지 않는다.
- [ ] C# 전환을 이 정리 작업과 동시에 시작하지 않는다.

---

## 2. 권장 최종 폴더 구조

현재 구조를 한 번에 크게 갈아엎기보다, 우선 `scripts/`, `ml/`, `core/`의 경계부터 정리한다.

```text
Auto_OTO/
  main.py

  ui/
    README.md
    # CustomTkinter GUI, 사용자 입력, 설정 저장/복원, 버튼 액션

  core/
    README.md
    # 배포 런타임 핵심 로직
    alignment/
    generation/
      kr/
      ja/
      no_mfa/
      en_preview/
    ml_runtime/
    validation/
    reporting/
    io/
    runtime_config/

  ml/
    README.md
    runtime/
      # 배포에 필요한 추론용 최소 코드만 위치하거나 core/ml_runtime으로 이동
    training/
      # 모델 학습
    evaluation/
      # 평가/비교
    datasets/
      # manifest, split, dataset builder
    tests/
      # ML 관련 테스트

  scripts/
    README.md
    runtime/
      # 배포 후 복구, MFA setup 등 사용자 런타임 보조
    build/
      # Nuitka, portable zip, release channel, packaging
    dev/
      # 개발자용 점검/로컬 보조
    batch/
      # batch OTO generation, regression batch
    train/
      # 학습 실행용 wrapper
    evaluate/
      # 평가 실행용 wrapper
    benchmark/
      # benchmark, leave-one-voicebank-out 등
    experiments/
      # 아직 정식 파이프라인이 아닌 실험
    deprecated/
      # 현재 진입점이 아니지만 삭제 전 보관

  tests/
    README.md
    unit/
    integration/
    regression/

  assets/
    README.md

  models/
    README.md

  docs/
    README.md
    00-문서-인덱스.md
    01-프로젝트-개요와-실행-모델.md
    03-전체-파이프라인.md
    98-문서-업데이트-내역.md
    99-사소한-업데이트-내역.md

  plans/
    README.md
    project-cleanup-plan.md
```

---

## 3. 분류 기준

## 3.1 Runtime

배포판에 포함될 수 있는 코드.

조건:

- GUI에서 직접 또는 간접 호출된다.
- `main.py` 실행에 필요하다.
- OTO 생성, 정렬, post-process, validation에 필요하다.
- 사용자 환경에서 복구 또는 설정에 필요하다.

예시:

- `ui/`
- `core/alignment_pipeline.py`
- `core/mfa_runner.py`
- `core/oto_generator.py`
- `core/ja_oto_generator.py`
- `core/no_mfa_oto_builder.py`
- `core/oto_ml_runtime.py`
- `core/model_resolver.py`
- `core/oto_validator.py`
- runtime recovery script

규칙:

- [ ] runtime 코드는 `ml/training`, `ml/evaluation`, `scripts/experiments`, `scripts/deprecated`를 import하면 안 된다.
- [ ] runtime 코드는 실패 시 사용자에게 설명 가능한 오류를 반환해야 한다.
- [ ] runtime 코드는 가능한 한 환경변수와 config 경로를 명확히 사용해야 한다.

---

## 3.2 Build

배포물을 만들 때만 필요한 코드.

예시:

- `build.py`
- `scripts/build_portable_with_models.ps1`
- release channel 생성
- portable zip 생성
- Nuitka wrapper
- build manifest 생성

규칙:

- [ ] build 코드는 runtime에서 import하지 않는다.
- [ ] build script는 포함/제외 파일 목록을 명시적으로 남긴다.
- [ ] Python 3.10 빌드 요구사항을 README에 적는다.

---

## 3.3 Training

모델 학습에 필요한 코드.

예시:

- CRNN 학습
- LightGBM 학습
- sequence residual 학습
- manifest 생성
- split 생성
- dataset builder

규칙:

- [ ] training 코드는 배포에 포함하지 않는다.
- [ ] training 코드는 GUI에서 직접 호출하지 않는다.
- [ ] training 결과물만 `models/` 또는 `assets/models/`를 통해 runtime에 전달한다.
- [ ] 학습 재현에 필요한 명령은 README에 남긴다.

---

## 3.4 Evaluation / Benchmark

모델 또는 OTO 품질을 평가하는 코드.

예시:

- `evaluate_oto.py`
- `compare_oto_eval.py`
- leave-one-voicebank-out benchmark
- batch summary 비교
- regression suite

규칙:

- [ ] evaluation 코드는 배포 제외가 기본이다.
- [ ] runtime 품질 검증과 개발 평가를 구분한다.
- [ ] 사용자용 validation은 `core/validation` 또는 runtime 경로에 둔다.
- [ ] 개발자용 benchmark는 `scripts/benchmark` 또는 `ml/evaluation`에 둔다.

---

## 3.5 Experiments

아직 정식 파이프라인이 아닌 실험.

예시:

- 새 aligner prototype
- CRNN 구조 실험
- alias role 실험
- CTC head 실험
- 영어 Preview 실험
- 임시 데이터 분석

규칙:

- [ ] experiments는 runtime에서 import 금지.
- [ ] 실험이 정식 기능이 되면 runtime 또는 training/evaluation으로 승격한다.
- [ ] 1개월 이상 쓰지 않는 실험은 deprecated 후보로 이동한다.

---

## 3.6 Deprecated

현재 메인 파이프라인이 아닌 코드.

조건:

- 문서상 현재 진입점이 아니다.
- 내부 호출이 없다.
- 과거 비교용으로만 남아 있다.
- 현재 함수 시그니처와 맞지 않는다.
- 새 구조로 대체되었다.

규칙:

- [ ] 바로 삭제하지 않고 `deprecated/README.md`에 이동 사유를 기록한다.
- [ ] 2~4주 동안 필요 없으면 삭제 후보로 올린다.
- [ ] 삭제 전 `ripgrep` 또는 import graph로 참조 여부를 확인한다.

---

## 4. 단계별 실행 계획

## Phase 1. 현황 고정

목표: 현재 프로젝트의 실제 진입점과 배포 경로를 문서화한다.

작업:

- [ ] `main.py`에서 시작되는 runtime import 경로를 확인한다.
- [ ] GUI 버튼별 호출 core 함수를 목록화한다.
- [ ] 배포 빌드에 포함되는 파일 목록을 확인한다.
- [ ] 배포 제외되는 학습/평가/테스트 파일 목록을 확인한다.
- [ ] 현재 `scripts/` 전체 목록을 추출한다.
- [ ] 현재 `core/`, `ml/`, `ui/`, `scripts/`별 파일 수와 줄 수를 기록한다.

산출물:

- [[Runtime-진입점-목록]]
- [[배포-포함-제외-목록]]
- [[Scripts-현황-인벤토리]]

권장 명령:

```bash
python -m compileall .
python scripts/audit_imports.py
rg "from ml|import ml|scripts/" core ui main.py
```

---

## Phase 2. scripts 카테고리화

목표: 스크립트 수가 많아도 목적이 바로 보이게 만든다.

작업:

- [ ] `scripts/runtime/` 생성
- [ ] `scripts/build/` 생성
- [ ] `scripts/dev/` 생성
- [ ] `scripts/batch/` 생성
- [ ] `scripts/train/` 생성
- [ ] `scripts/evaluate/` 생성
- [ ] `scripts/benchmark/` 생성
- [ ] `scripts/experiments/` 생성
- [ ] `scripts/deprecated/` 생성
- [ ] 기존 script를 하나씩 이동한다.
- [ ] 이동 후 import/path 깨짐을 수정한다.
- [ ] 각 폴더에 `README.md`를 추가한다.

주의:

- 경로를 하드코딩한 스크립트가 있을 수 있으므로 이동 후 바로 테스트한다.
- PowerShell, bat 파일은 상대경로 기준이 깨지기 쉽다.
- build script는 이동 전후로 portable build smoke test가 필요하다.

산출물:

- [[Scripts-분류표]]
- [[Scripts-README-템플릿]]

---

## Phase 3. ML 코드 경계 정리

목표: 학습/평가 코드와 런타임 추론 코드를 분리한다.

작업:

- [ ] runtime inference에 필요한 최소 ML 파일 목록 작성
- [ ] 학습 전용 파일 목록 작성
- [ ] 평가 전용 파일 목록 작성
- [ ] dataset builder 목록 작성
- [ ] CRNN runtime과 CRNN training의 경계 확인
- [ ] LightGBM runtime과 training 경계 확인
- [ ] GUI가 training/evaluation 파일을 직접 import하지 않는지 확인
- [ ] 배포 빌드에서 training/evaluation 제외 규칙을 고정한다.

권장 구조:

```text
ml/
  runtime/
  training/
  evaluation/
  datasets/
  tests/
```

또는 runtime 추론 코드는 `core/ml_runtime/`으로 옮기고, `ml/`은 개발 전용으로 유지한다.

추천 판단:

- 배포 추론에 필요한 코드는 `core/ml_runtime/` 쪽이 더 명확하다.
- 학습/평가/데이터셋 코드는 `ml/`에 남기는 것이 좋다.

산출물:

- [[ML-런타임-경계]]
- [[학습-평가-배포-제외-목록]]

---

## Phase 4. core 내부 패키지 정리

목표: core가 “모든 것이 들어 있는 폴더”가 아니라 기능별 runtime library처럼 보이게 한다.

권장 구조:

```text
core/
  alignment/
    mfa/
    sequence/
    coarse_crnn/
  generation/
    kr/
    ja/
    no_mfa/
    en_preview/
    common/
  ml_runtime/
  validation/
  reporting/
  runtime_config/
  io/
```

우선순위:

1. 한국어 생성기는 이미 분리 중이므로 현재 구조를 안정화한다.
2. 일본어 생성기를 한국어와 비슷한 구조로 2차 분리한다.
3. alignment 관련 파일을 묶는다.
4. validation/reporting/io를 묶는다.
5. import compatibility layer를 둬서 기존 호출을 한 번에 깨지 않게 한다.

주의:

- 대규모 이동은 한 번에 하지 않는다.
- 파일 이동보다 먼저 public entry function을 고정한다.
- 예: `generate_oto()`, `generate_ja_oto()`, `run_alignment_with_fallback()` 같은 진입점은 유지한다.

산출물:

- [[Core-패키지-구조]]
- [[Runtime-Public-API]]
- [[Import-Compatibility-Plan]]

---

## Phase 5. Deprecated 후보 격리

목표: 삭제가 아니라 “현재 경로가 아님”을 명확히 한다.

작업:

- [ ] 문서에서 삭제/제거되었다고 언급된 과거 경로 확인
- [ ] 현재 import되지 않는 visual compare, legacy eval, old pipeline 후보 확인
- [ ] `deprecated_manifest.md` 작성
- [ ] deprecated 이동
- [ ] 테스트 통과 확인
- [ ] 2~4주 후 삭제 여부 결정

`deprecated_manifest.md` 예시:

```md
# Deprecated Manifest

## scripts/old_eval/foo.py

- 이동일: 2026-05-16
- 사유: 현재 batch/evaluation 경로에서 사용하지 않음
- 대체 경로: scripts/evaluate/bar.py
- 삭제 예정: 2026-06-15 이후
- 복구 조건: 과거 결과 재현 필요 시
```

산출물:

- [[Deprecated-Manifest]]
- [[삭제-후보-검토]]

---

## Phase 6. 문서와 AI 작업 규칙 갱신

목표: 사람이 봐도, AI가 봐도 수정 범위를 헷갈리지 않게 한다.

작업:

- [ ] [[00-문서-인덱스]] 갱신
- [ ] [[01-프로젝트-개요와-실행-모델]]에 새 폴더 구조 반영
- [ ] [[03-전체-파이프라인]]에 runtime/train/eval 경계 반영
- [ ] [[13-테스트-검증-수정-가이드]]에 수정 범위 규칙 추가
- [ ] `AGENTS.md` 또는 `CODEX.md` 작성
- [ ] 각 폴더 README 작성

AI 작업 규칙 예시:

```md
# AI 작업 규칙

- runtime 수정 요청이면 `core/`, `ui/`, `scripts/runtime/`, `scripts/build/`만 우선 확인한다.
- training 요청이면 `ml/training/`, `ml/datasets/`, `scripts/train/`을 확인한다.
- evaluation 요청이면 `ml/evaluation/`, `scripts/evaluate/`, `scripts/benchmark/`를 확인한다.
- `deprecated/` 코드는 사용자의 명시 요청 없이는 수정하지 않는다.
- runtime 코드에서 training/evaluation/experiments를 import하지 않는다.
```

산출물:

- [[AI-작업-규칙]]
- [[폴더별-README-목록]]
- [[문서-업데이트-체크리스트]]

---

## 5. 우선순위별 작업 목록

## 5.1 최우선

- [ ] 배포 포함/제외 파일 목록 문서화
- [ ] `scripts/` 카테고리화
- [ ] `ml` 학습/평가/런타임 경계 확정
- [ ] runtime에서 training/evaluation import 금지 규칙 추가
- [ ] `AGENTS.md` 또는 `CODEX.md` 작성

## 5.2 중요

- [ ] core public entrypoint 정리
- [ ] 한국어 생성기 1차 분리 구조 안정화
- [ ] 일본어 생성기 2차 분리 계획 수립
- [ ] deprecated 후보 격리
- [ ] CI 또는 로컬 smoke test 명령 고정

## 5.3 나중에 해도 됨

- [ ] 폴더 이름 전체 리네이밍
- [ ] C# GUI 전환
- [ ] 자체 정렬기 대규모 통합
- [ ] 영어 확장
- [ ] 실험 코드 삭제

---

## 6. 권장 README 템플릿

## 6.1 runtime 폴더 README

```md
# Runtime Folder

이 폴더는 배포 런타임에서 사용되는 코드만 포함한다.

## 포함 기준

- GUI 또는 batch runtime에서 직접 호출됨
- OTO 생성, 정렬, 검증, post-process에 필요함
- 사용자 배포 패키지에 포함될 수 있음

## 금지

- training 코드 import 금지
- evaluation/benchmark 코드 import 금지
- experiments/deprecated 코드 import 금지

## 주요 진입점

- ...
```

## 6.2 training 폴더 README

```md
# Training Folder

이 폴더는 모델 학습용 코드다. 배포 패키지에는 포함하지 않는다.

## 포함

- dataset builder
- manifest/split 생성
- model training
- training config

## 출력

- trained model
- evaluation json
- logs

## Runtime과의 관계

학습 결과물만 runtime에 전달한다.
runtime이 이 폴더를 import하면 안 된다.
```

## 6.3 deprecated 폴더 README

```md
# Deprecated Folder

현재 메인 파이프라인에서 사용하지 않는 코드의 임시 보관소다.

## 규칙

- 새 기능은 이 폴더에 추가하지 않는다.
- 사용자의 명시 요청 없이 수정하지 않는다.
- 복구 필요성이 없으면 삭제 후보로 이동한다.
```

---

## 7. 검증 체크리스트

정리 작업 후 반드시 확인할 것.

## 7.1 기본 실행

- [ ] GUI 실행
- [ ] config 로딩
- [ ] 모델 폴더 탐색
- [ ] MFA 경로 감지
- [ ] 로그 폴더 생성
- [ ] Windows frozen build 기준 경로 확인

## 7.2 한국어 경로

- [ ] Lab 생성
- [ ] Dictionary 생성
- [ ] MFA alignment
- [ ] Sequence fallback
- [ ] 한국어 CV OTO 생성
- [ ] 한국어 CVC/COC OTO 생성
- [ ] 한국어 CVVC OTO 생성
- [ ] post-file pipeline
- [ ] OTO-ML 적용
- [ ] validation

## 7.3 일본어 경로

- [ ] Lab 생성
- [ ] Dictionary 생성
- [ ] 일본어 CV OTO 생성
- [ ] 일본어 CVVC OTO 생성
- [ ] 일본어 VCV OTO 생성
- [ ] post-file pipeline
- [ ] validation

## 7.4 No-MFA / 특수 경로

- [ ] No-MFA template remap
- [ ] C+V 경로
- [ ] Preview 영어 경로
- [ ] TextGrid 없는 경우 preflight 오류 처리

## 7.5 ML 경로

- [ ] bundled model 탐색
- [ ] LightGBM 추론
- [ ] Coupled/Ensemble/E2E fallback
- [ ] 모델 없을 때 fallback report
- [ ] training 파일이 배포에 포함되지 않는지 확인

## 7.6 빌드

- [ ] Nuitka build
- [ ] portable output 생성
- [ ] stable channel
- [ ] preview channel
- [ ] MFA runtime 복구
- [ ] packaged script 경로 확인

---

## 8. 위험 관리

## 8.1 가장 큰 위험

- 파일 이동 후 상대경로 깨짐
- build script가 오래된 경로를 참조
- GUI에서 실험 기능 import
- runtime이 training/evaluation을 암묵적으로 import
- 테스트는 통과하지만 packaged build에서 실패
- 문서와 실제 경로 불일치

## 8.2 대응

- [ ] 이동은 한 번에 한 폴더씩 진행
- [ ] 이동 후 즉시 smoke test
- [ ] public entrypoint compatibility 유지
- [ ] import alias 또는 thin wrapper 임시 유지
- [ ] `deprecated_manifest.md`로 삭제 전 추적
- [ ] 문서 업데이트를 코드 변경 PR/커밋의 필수 조건으로 둔다

---

## 9. 추천 커밋 단위

## Commit 1

목표: 문서와 규칙 추가

- `project-cleanup-plan.md`
- `AGENTS.md`
- `scripts/README.md`
- `ml/README.md`
- `core/README.md`

## Commit 2

목표: scripts 1차 분류

- `scripts/build`
- `scripts/runtime`
- `scripts/dev`
- `scripts/batch`

## Commit 3

목표: training/evaluation/benchmark 분류

- `scripts/train`
- `scripts/evaluate`
- `scripts/benchmark`
- `ml/training`
- `ml/evaluation`

## Commit 4

목표: deprecated 격리

- `scripts/deprecated`
- `deprecated_manifest.md`

## Commit 5

목표: build 포함/제외 규칙 고정

- build script 수정
- portable package 검증
- 문서 반영

## Commit 6

목표: core 구조 1차 안정화

- public entrypoint 문서화
- import compatibility 확인
- regression test

---

## 10. 최종 완료 기준

이 계획의 완료 기준은 코드 줄 수가 줄어드는 것이 아니다.

완료 기준:

- [ ] 새 개발자가 봐도 각 폴더의 역할을 이해할 수 있다.
- [ ] AI에게 작업을 맡길 때 수정 범위를 명확히 지정할 수 있다.
- [ ] 배포에 포함되는 코드와 제외되는 코드가 명확하다.
- [ ] runtime 코드가 training/evaluation/experiments를 import하지 않는다.
- [ ] scripts가 목적별로 분류되어 있다.
- [ ] deprecated 후보가 격리되어 있다.
- [ ] GUI 실행, 한국어/일본어 OTO 생성, ML 추론, validation, build가 정상 동작한다.
- [ ] 문서 인덱스와 실제 경로가 일치한다.

---

## 11. 관련 문서 링크

- [[00-문서-인덱스]]
- [[01-프로젝트-개요와-실행-모델]]
- [[03-전체-파이프라인]]
- [[13-테스트-검증-수정-가이드]]
- [[14-Coarse-CRNN-OTO-모델]]
- [[15-추후-업데이트-수정-사항]]
- [[98-문서-업데이트-내역]]
- [[99-사소한-업데이트-내역]]

---

## 12. 한 줄 원칙

> **AutoOTO는 줄여야 하는 프로젝트가 아니라, 경계를 고정해야 하는 프로젝트다.**  
> 코드 삭제보다 먼저 `runtime / build / training / evaluation / experiment / deprecated`를 분리한다.
