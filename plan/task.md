# Mapping Core v2 Task Status

기준일: 2026-03-10

이 문서는 예전 계획 초안이 아니라 현재 코드 기준 상태를 반영한다.
`[x]`는 코드 반영 완료, `[/]`는 일부 완료 또는 후속 작업 남음, `[ ]`는 미착수다.

## Core Rewrite

- [x] 파일 단위 전역 매핑 planner 도입
- [x] alignment ingest 계층 분리
- [x] candidate selection / candidate filtering 계층 분리
- [x] runtime confidence / low-trust policy 공통화
- [x] row-level abstain 정책 공통화
- [x] diagnostics / trace / unset collector 공통화
- [x] anchor graph 도입 및 VC/VV 브리지에 연결
- [x] JA/KR anchor lock adapter 공통화

## Row Execution Split

- [x] Japanese `VCV` row executor 분리
- [x] Japanese `CV_HEAD` row executor 분리
- [x] Japanese general row executor 분리
- [x] Korean `VCV` row executor 분리
- [x] Korean `CV_HEAD` row executor 분리
- [x] Korean general row executor 분리
- [x] row finalize / row output helper 공통화

## Mapping Selection Split

- [x] Korean `VCV` 선택 로직 helper 분리
- [x] Korean `CV_HEAD` forced index helper 분리
- [x] Japanese `CV/CV_HEAD` forced index helper 분리
- [x] Japanese `VCV` 선택 로직 helper 분리
- [x] Japanese exact/vowel/forward-shift scoring 유틸 분리
- [x] Korean `CV` 음절 선택 scoring 유틸 분리
- [x] Korean 일반 `CV/VV` 선택 게이트/모음 보정 helper 분리
- [/] generator 내부의 언어별 세부 scoring 함수 완전 분리
설명: 일본어 핵심 scoring 유틸과 한국어 `CV/VCV` 핵심 선택 보조는 v2 모듈로 이동했다. 일부 보조 scoring 유틸은 아직 generator 안에 남아 있다.

## Timing / Guard / Postprocess

- [x] Japanese timing helper 분리
- [x] Korean timing helper 분리
- [x] `_stabilize_params_to_phone_activity` 블렌딩 가드 반영
- [x] `_apply_soft_mel_offset_cutoff_guard` onset 보조 탐지 반영
- [x] 파일 일관성 후처리 강화
- [x] alias-type aware ML delta clipping 반영

## Selector Improvements

- [x] selector objective auto (ranking with group threshold)
- [x] selector quality weights by format/alias family
- [x] selector score margin abstain + hard-negative logging
- [x] selector risk features (silence/jump)
- [x] selector retraining script
- [x] hard-negative summary report script

## Diagnostics / Verification

- [x] v2 공통 모듈 대상 단위 테스트 추가
- [x] JA/KR 매핑 선택 회귀 테스트 추가
- [x] row/finalize/output 계층 테스트 추가
- [x] 타깃 pytest 회귀 통과
현재 기준:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q ml/tests/test_sinsy_label_ingest.py ml/tests/test_oto_runtime_policy.py ml/tests/test_oto_row_abstain.py ...` -> `52 passed`
- 전체 `ml/tests`는 일부 레거시/외부 의존 import 오류 때문에 수집 단계에서 중단됨

## Phase A Policy Work

- [/] low-margin runtime policy / row abstain 1차 반영
설명: `mapping_margin`을 runtime policy에 연결했고, low-margin `CV/CV_HEAD` 행은 skip할 수 있게 했다.
- [/] optional sinsy syllable label ingest / planner anchor 반영
설명: `USE_SINSY_LABELS`, `SINSY_LABEL_PATH`, 환경 변수 override, `core/sinsy_label_ingest.py`를 추가했고 JA/KR planner가 opt-in으로 sinsy 라벨을 우선 사용할 수 있다.
- [x] JA/KR plain-glide/youon reject 강화
- [x] VV pair scoring 도입
- [x] low-trust 시 forced index / free search 추가 축소
- [x] sinsy vs MFA 충돌 경고/진단 추가

## Legacy / Cleanup

- [x] 사용 중단된 분석 스크립트 일부를 `scripts/legacy_mapping_v1/`로 이동
- [x] 수동 점검용 레거시 스크립트를 `scripts/legacy_misc_tools/`로 이동
- [x] SOFA를 핵심 정렬 경로에서 제외하는 방향으로 설계 정리
- [/] generator 내부 잔여 wrapper / scoring 유틸 정리
설명: 동작상 불필요한 얇은 wrapper는 대부분 줄였지만, 설명용/호환용 로컬 함수가 일부 남아 있다.

## Remaining High-Value Work

- [ ] generator 밖으로 남은 scoring 유틸 완전 이관
- [ ] 배치 평가 재실행 후 수치 문서화
- [ ] 실제 샘플 oto 생성 및 청감 검증 기록화
- [ ] `mapping_core_v2_design` 문서와 구현 결과의 최종 동기화
