# MFA 병행 이중 파이프라인 + 공유 코어 설계안

## 1) 목표

- 기존 MFA 기반 파이프라인은 그대로 유지하고 품질 개선을 계속 진행한다.
- MFA 미사용(no-MFA) 파이프라인을 별도로 구축해 독립적으로 실험/운영한다.
- 두 파이프라인은 "기본 OTO 계산/검증/가드"만 공유하고, 앞단(경계/정렬)은 분리한다.

## 2) 설계 원칙

- 안정성 우선: 기존 사용자 경로(MFA)는 기본값을 유지한다.
- 실험 분리: no-MFA 실패가 MFA 경로에 영향을 주지 않게 코드 경계를 분리한다.
- 공통화 최소화: 공유는 수학식/검증기/후처리처럼 변동이 적은 코어에 한정한다.
- 관측 가능성: 두 경로의 리포트 필드를 통일해 A/B 비교가 가능해야 한다.

## 3) 아키텍처 개요

## 3.1 파이프라인 A: MFA 유지 경로 (기본)

1. `wav + lab/dict` 준비
2. MFA 정렬(TextGrid)
3. OTO 초안 생성
4. ML 후처리(legacy/autofree/two_stage 확장)
5. 공통 가드/검증/출력

기존 진입점 유지:
- `core/alignment_pipeline.py`
- `core/post_file_pipeline.py`
- `core/oto_ml_refiner.py`

## 3.2 파이프라인 B: no-MFA 경로 (신규)

1. audio-only 경계 탐지 (onset/nucleus/tail + blank/voicing conf)
2. 파일명/alias 기반 단조 정렬(monotonic align)
3. OTO 초안 생성(또는 기본값 seed)
4. OTO 매핑 + 파라미터 모델 추론
5. 공통 가드/검증/출력

핵심: 정렬 소스만 TextGrid -> boundary model로 대체하고, 뒤쪽 계산/검증은 공유한다.

## 4) 모듈 경계

## 4.1 공유 코어 (공통 사용)

- OTO 파라미터 수학식/정규화
- validator (`validate_oto_params` 계열)
- anchor lock lite / destination guard / wav-duration hard guard
- selector abstain, confidence gate
- 리포트 코드 및 필드 스키마

대상 파일(재사용):
- `core/oto_ml_refiner.py` (후반부 finalize/guard 로직)
- `core/oto_generator.py`, `core/ja_oto_generator.py` (검증 함수)
- `core/post_file_pipeline.py`
- `core/pipeline_status.py`

## 4.2 MFA 전용 모듈

- `core/mfa_runner.py`
- `core/alignment_pipeline.py`
- TextGrid ingest 관련 모듈

## 4.3 no-MFA 전용 모듈 (신규)

- `core/no_mfa_pipeline.py` (오케스트레이션)
- `core/oto_ml/boundary/*` (boundary model 계약/로더/추론)
- `core/no_mfa_monotonic_align.py` (파일명 순서 + alias 제약 정렬)
- `core/no_mfa_feature_bridge.py` (boundary 출력 -> shared feature row 변환)

## 5) 라우팅/설정 전략

- 기본값: 기존과 동일(MFA 경로)
- 신규 설정:
  - `UTOA_ALIGN_ENGINE=mfa|none`
  - `UTOA_NO_MFA_ENABLE=0|1`
  - `UTOA_NO_MFA_BOUNDARY_DIR=<path>`
  - `UTOA_NO_MFA_FAIL_OPEN=1` (실패 시 기존 계산식 유지)
- ML route와 독립:
  - `UTOA_ML_ROUTE`는 후처리 모델 선택용으로 계속 사용
  - `UTOA_ALIGN_ENGINE`은 정렬 소스 선택용

## 6) 데이터/모델 운영 분리

- 모델 디렉터리 분리:
  - MFA 경로 모델: 기존 위치 유지
  - no-MFA 경계 모델: `models_installed/oto_ml_boundary/...`
  - no-MFA OTO 모델: `models_installed/oto_ml_nomfa/...`
- 학습 데이터 분리:
  - MFA supervised 데이터와 no-MFA weak label 데이터를 섞지 않는다.
- 공통 feature schema는 버전 필드로 분기:
  - 예: `feature_version=v13_mfa`, `feature_version=v13_nomfa`

## 7) 런타임 동작 계약

## 7.1 공통 입력/출력 계약

- 입력: `oto_path`, `wav_dir`, `language`, `format_type`, row context
- 출력: 최종 `offset, cons, cutoff, pre, ovl`
- 최종 출력 전에는 항상 공통 validator + guard를 거친다.

## 7.2 no-MFA 추가 계약

- boundary stage가 row마다 `boundary_confidence`를 반환해야 함
- confidence 임계치 미달 row는 "적용 중지(abstain)" 후 기본값 유지
- 파일 단위 실패 시 fail-open 정책에 따라 no-op 처리

## 8) 품질/안전 게이트

- 하드 게이트:
  - `boundary_confidence < threshold` -> no-MFA 보정 비적용
  - 파라미터 기하 제약 위반 -> 공통 guard로 clamp
- 소프트 게이트:
  - format/alias family별 적용 범위 제한
  - 초기에는 `korean cv/cvc` 우선 적용
- 회귀 방지:
  - `MFA on` 경로 출력 diff 0 보장 테스트

## 9) PR 순서 (권장)

1. PR-01: 파이프라인 스위치 골격
- `UTOA_ALIGN_ENGINE` 도입, 기본 `mfa`
- no-MFA 경로는 아직 no-op

2. PR-02: no-MFA 오케스트레이터 추가
- `core/no_mfa_pipeline.py` 생성
- 실패 시 기존 계산 유지

3. PR-03: boundary 계약/로더 추가
- `core/oto_ml/boundary/schema.py`, `inference.py` 추가
- shadow 모드(예측만, 결과 미적용)

4. PR-04: monotonic 정렬 모듈 추가
- 파일명 순서 + alias family 제약
- 정렬 품질 지표 로그 추가

5. PR-05: shared feature bridge 연결
- boundary 출력을 기존 OTO 모델 입력 feature로 주입
- 여전히 gated 적용

6. PR-06: no-MFA 적용 범위 제한 오픈
- 특정 포맷/alias에서만 실제 반영
- fallback reason/coverage 리포트 강화

7. PR-07: 테스트/문서/운영 가이드
- 경로별 회귀 테스트
- 실험 플래그, 롤백 절차 문서화

## 10) 테스트 계획

- 단위 테스트:
  - no-MFA 경로 활성/비활성 분기
  - boundary confidence gate
  - monotonic 정렬 계약(순서 역전 금지)
- 회귀 테스트:
  - `UTOA_ALIGN_ENGINE=mfa`에서 결과 불변
  - 모델 없음/로드 실패 시 fail-open 동작
- 통합 테스트:
  - 한국어 CV/CVC, 일본어 CVVC 샘플셋에서 성공률/에러율 비교

## 11) 운영 권장안

- 단계 1: 내부 실험만 no-MFA on, 사용자 기본값은 MFA 유지
- 단계 2: 포맷 제한 공개 베타(no-MFA opt-in)
- 단계 3: 품질 기준 충족 포맷만 no-MFA default 후보 검토

## 12) 결론

- "MFA 유지 + no-MFA 별도 구축 + 공유 코어"는 현재 코드베이스에서 가장 현실적이고 안전한 전략이다.
- 앞단만 분기하고 뒷단 계산/검증을 공유하면, 개발 속도와 안정성을 동시에 확보할 수 있다.
- 구현은 반드시 fail-open과 단계별 게이트를 전제로 진행해야 한다.

