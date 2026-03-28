# Script Cleanup + Debug/Test Refactor Plan (2026-03-28)

## 1) 목적

- `scripts/`에 있는 실행 스크립트 중 **실제로 쓰이지 않는 후보**를 추려 삭제 리스크를 낮춘다.
- 디버깅/테스트를 반복할 때 진입점이 분산되어 생기는 시간 낭비를 줄인다.
- 설치/복구/빌드/학습 스크립트의 책임을 분리하고, 실패 원인 추적을 빠르게 만든다.

## 2) 조사 범위/방법

- 기준 브랜치: `main`
- 조사 대상: `scripts/*.py`, `scripts/*.ps1`, `scripts/*.bat`
- 조사 방식:
  - 정적 참조 검색(`git grep`)으로 호출 지점 확인
  - CI(workflows), 빌드 엔트리(`build.py`), 설치 스크립트(`setup_mfa.bat`, `setup_ctc.bat`) 교차 확인
  - 문서 참조 여부 확인(README + `ml/*.md`)
- 한계:
  - 사용자의 로컬 수동 실행 이력은 정적 분석만으로 100% 판단 불가
  - 따라서 삭제는 `즉시 삭제`와 `보류/관찰`로 분리

## 3) 사용 현황 분류

## 3-1) 삭제 금지(운영 핵심)

- `scripts/runtime_recovery.ps1`
  - `setup_mfa.bat`, `build.py`에서 직접 참조
- `scripts/startup_diagnose.ps1`, `scripts/startup_diagnose.bat`
  - `build.py`/설치 패키징 경로에서 사용
- `scripts/build_installer.ps1`, `scripts/build_portable_with_models.ps1`
  - GitHub Actions 워크플로우에서 직접 사용
- `scripts/run_oto_generation_batch.py`
  - 문서/회귀 배치 시나리오에서 사용 중
- `scripts/train_cvn_backend.py`
  - 코드 내 자동 참조는 없지만 최근 실사용 학습 엔트리로 확인됨(삭제 대상 아님)

## 3-2) 즉시 삭제 후보 (A등급: 고신뢰)

1. `scripts/setup_ctc_runtime.py`
   - 코드/CI/문서 내 참조 없음
   - 현재 CTC 설치 표준은 `setup_ctc.bat` 기반 `.env_ctc` 분리 방식
   - 기능 중복 + 운영 경로 불일치(현재 파이프라인에서 비표준)

## 3-3) 보류 삭제 후보 (B등급: 중간 신뢰, 1~2주 관찰 후 삭제)

1. `scripts/build_bundle_profiles.ps1`
   - 외부 참조 없음, 수동 배포용 통합 스크립트
   - 대체 경로: `build.py` + `build_installer.ps1` + `build_portable_with_models.ps1`
2. `scripts/build_release.bat`
   - 외부 참조 없음, 로컬 편의 래퍼 성격
3. `scripts/retrain_all_formats_staged.ps1`
4. `scripts/retrain_from_dataset_staged.ps1`
   - 상호 및 내부 연계는 있으나 상위 호출자 없음
   - ML 파이프라인 재정리 시 통합 가능
5. `scripts/promote_best_models.py`
   - 상위 호출자 없음
6. `scripts/run_mel_env_autotune.py`
   - 상위 호출자 없음
7. `scripts/sandbox_smoke_check.ps1`
8. `scripts/windows_mfa_alignment_smoke.ps1`
   - QA 도구 성격, 자동 호출 지점 없음

## 3-4) 관찰 대상 (C등급: 유지 권장)

- `scripts/portable_smoke_check.ps1`
  - `build_release.bat`에서 참조되므로 B등급 스크립트 정리 후 재판단
- `scripts/run_coupled_experiment_matrix.py`, `scripts/run_coupled_two_stage_tune.py`, `scripts/run_kr_regression_suite.py`
  - ML 문서/튜닝 흐름 연결이 있어 당장 삭제 비권장

## 4) 삭제 실행 정책 (실사용 안전장치)

- 원칙: **즉시 물리 삭제 금지**, 먼저 `deprecated` 표식 + 경고 출력 + 대체 경로 안내
- 2단계 절차:
  1. 스크립트 상단에 deprecation 경고 추가(대체 명령 명시)
  2. 1~2주 관찰 후 호출 이력(문서/CI/팀 사용) 재검토 뒤 삭제
- 예외:
  - `setup_ctc_runtime.py`는 운영 충돌 가능성이 있어 우선 제거 가능

## 5) 디버깅/테스트 효율화를 위한 리팩토링 계획

## Phase 0: 인벤토리 고정 (반나절)

1. `scripts_manifest.json` 추가
   - 각 스크립트의 `owner`, `purpose`, `input`, `output`, `status(active/deprecated)` 명시
2. CI에서 manifest와 실제 파일 불일치 체크

성과 기준:
- 새 스크립트 추가 시 목적/책임이 누락되지 않음

## Phase 1: 엔트리포인트 정리 (1일)

1. 공용 실행 진입점 추가 (`python -m tools ...` 또는 `scripts/run.ps1`)
2. 기존 산발적 실행 명령을 서브커맨드로 수렴
   - `tools setup mfa`
   - `tools setup ctc`
   - `tools batch oto`
   - `tools train cvn`

성과 기준:
- 신규 인원이 "어떤 스크립트를 돌려야 하는지" 1분 내 판단 가능

## Phase 2: 로그/진단 표준화 (1일)

1. 모든 스크립트 공통 옵션 통일:
   - `--dry-run`
   - `--verbose`
   - `--log-file`
   - `--json-report`
2. 실패 코드 표준화:
   - `E_INPUT`, `E_ENV`, `E_RUNTIME`, `E_DEP`, `E_INTERNAL`
3. 출력 경로 고정:
   - `artifacts/logs/<tool>/<timestamp>.log|json`

성과 기준:
- 재현 실패 케이스에서 로그만으로 원인 계층(입력/환경/내부) 분리 가능

## Phase 3: 스크립트 스모크 테스트 도입 (1~1.5일)

1. `tests/scripts/test_smoke_cli.py` 추가
   - 핵심 스크립트 `--help`/`--dry-run` 최소 검증
2. 설치 스크립트는 실제 설치 대신 사전검증 경로만 테스트
3. GitHub Actions에 `script-smoke` job 추가

성과 기준:
- 커밋마다 "실행 자체가 깨진 스크립트"를 사전 탐지

## Phase 4: 삭제/이관 실행 (0.5~1일)

1. A등급 후보 삭제
2. B등급 후보는 `scripts/legacy/`로 이관 후 경고
3. 1~2주 후 호출 증거 없으면 최종 삭제

성과 기준:
- `scripts/` 최상위에는 운영상 필요한 스크립트만 남김

## 6) 제안하는 1차 실행 우선순위

1. `setup_ctc_runtime.py` deprecate 또는 삭제
2. `scripts_manifest.json` 도입
3. 핵심 5개 스크립트(`runtime_recovery`, `startup_diagnose`, `build_installer`, `build_portable_with_models`, `run_oto_generation_batch`)에 공통 로그 옵션 표준화
4. 스모크 테스트 CI 추가

## 7) 리스크/대안

- 리스크: "실제로는 수동 사용 중인 스크립트"를 삭제할 가능성
- 대안:
  - 삭제 전 `legacy` 이관 + 경고 기간 운영
  - 문서와 실행 진입점을 동시에 갱신해 우회 경로 제공

---

이 계획은 "코드 품질 개선"보다 "실사용 안정성 + 디버깅 속도"를 우선으로 설계했다.

## 8) 실행 현황 (2026-03-28 반영)

- 완료:
  - `scripts/setup_ctc_runtime.py` 삭제
  - 아래 레거시 스크립트 9개 삭제
    - `scripts/build_bundle_profiles.ps1`
    - `scripts/build_release.bat`
    - `scripts/portable_smoke_check.ps1`
    - `scripts/promote_best_models.py`
    - `scripts/retrain_all_formats_staged.ps1`
    - `scripts/retrain_from_dataset_staged.ps1`
    - `scripts/run_mel_env_autotune.py`
    - `scripts/sandbox_smoke_check.ps1`
    - `scripts/windows_mfa_alignment_smoke.ps1`
  - `scripts/scripts_manifest.json` 추가 및 상태 기준 정비
  - `scripts/audit_script_usage.py` 추가 (삭제 안전성 자동 점검)
  - `ml/tests/test_script_audit_usage.py` 추가 (manifest drift/삭제 안전성 테스트)

- 검증:
  - 감사 결과: `scripts=14`, `manifest_entries=14`, `safe_delete_candidates=0`
  - 테스트 결과: `ml/tests/test_script_audit_usage.py` 3 passed
