# Scripts

`scripts/`는 **실사용 실행 경로**(런타임/빌드/학습/평가/배치) 스크립트를 목적별로 분리한 폴더다.
테스트/스모크/벤치 보조 스크립트는 `tests/scripts/`로 분리한다.

## 구조 원칙

- 루트 `scripts/*.py|*.ps1`는 **호환성 래퍼(entrypoint)** 로 유지한다.
- 실제 구현은 카테고리 하위 폴더에 둔다.
- 기존 CI/문서 명령(`scripts/build_installer.ps1` 등)은 그대로 동작해야 한다.
- `scripts/`에는 실작동에 필요한 엔트리포인트만 둔다.
- 테스트 전용 스크립트는 `tests/scripts/<purpose>/` 하위로 이동한다.

## 카테고리

- `scripts/runtime/`: 사용자 런타임 복구/진단 관련(현재 핵심 스크립트는 배포 계약상 루트 유지)
- `scripts/build/`: 빌드/패키징 구현 스크립트
- `scripts/dev/`: 개발자 점검/감사용 스크립트
- `scripts/batch/`: 배치 OTO 생성 구현
- `scripts/train/`: 학습 실행 구현
- `scripts/evaluate/`: 평가/튜닝 실행 구현
- `scripts/benchmark/`: 회귀/벤치마크 실행 구현
- `scripts/experiments/`: 정식 파이프라인이 아닌 실험용 영역
- `scripts/deprecated/`: 비활성/보관 영역

## 테스트 스크립트 위치

- `tests/scripts/alignment/`: 정렬 샘플 구성, 시각 비교 등 점검 도구
- `tests/scripts/benchmark/`: 실험/벤치 전용 비교 도구
- `tests/scripts/evaluate/`: 수동 평가/내보내기 보조 도구
- `tests/scripts/smoke/`: 배포/설치 스모크 점검 도구
- `tests/scripts/training/`: 실험성 학습 전처리/베이스라인 점검 도구

## 보조 폴더

- `scripts/config/`: 실행 설정 YAML 샘플
- `scripts/docs/`: 스크립트 사용 문서
- `scripts/logs/`: 배치 실행 로그(산출물)

## 호환성 래퍼(루트 유지)

- `scripts/build_installer.ps1` -> `scripts/build/build_installer.ps1`
- `scripts/build_portable_with_models.ps1` -> `scripts/build/build_portable_with_models.ps1`
- `scripts/run_oto_generation_batch.py` -> `scripts/batch/run_oto_generation_batch.py`
- `scripts/train_cvn_backend.py` -> `scripts/train/train_cvn_backend.py`
- `scripts/audit_script_usage.py` -> `scripts/dev/audit_script_usage.py`
- `scripts/run_test_tools.py` -> `scripts/dev/run_test_tools.py`
- `scripts/analyze_legacy_wrappers.py` -> `scripts/dev/analyze_legacy_wrappers.py`
- `scripts/dry_run_candidate_delete.py` -> `scripts/dev/dry_run_candidate_delete.py`

운영 기준:

- CI/배포 계약 경로: `protected`
- 새 실행 경로를 이미 가진 루트 호환 래퍼: `legacy`
- 실제 구현 파일: `active`

## 테스트 도구 통합 실행

- 목록: `python scripts/run_test_tools.py list`
- 실행: `python scripts/run_test_tools.py run <tool-key> -- <tool-args>`
- 경로 확인: `python scripts/run_test_tools.py path <tool-key>`

## 레거시 래퍼 사용 감사

- 실행: `python scripts/analyze_legacy_wrappers.py`
- JSON 출력: `python scripts/analyze_legacy_wrappers.py --json`
- 기본 리포트 파일: `scripts/logs/legacy_wrapper_usage_report.json`

## 삭제 드라이런

- 실행: `python scripts/dry_run_candidate_delete.py`
- JSON 출력: `python scripts/dry_run_candidate_delete.py --json`
- 기본 리포트 파일: `scripts/logs/candidate_delete_dry_run.json`

## core 래퍼 lifecycle 감사

- 실행: `python scripts/audit_core_wrapper_lifecycle.py`
- JSON 출력: `python scripts/audit_core_wrapper_lifecycle.py --json`
- 기본 리포트 파일: `scripts/logs/core_wrapper_lifecycle_report.json`
- CI 강제 체크: `python scripts/audit_core_wrapper_lifecycle.py --enforce-clean`

## 금지 규칙

- runtime 코드(`core/`, `ui/`)에서 `scripts/experiments`, `scripts/deprecated`를 import하지 않는다.
- 카테고리 이동 시 `scripts/scripts_manifest.json`을 반드시 갱신한다.
