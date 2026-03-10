# Mapping Core v2 Current Status

기준일: 2026-03-10

## 1. 개발 의도

이번 재작성의 목적은 `파라미터 숫자 미세조정`이 아니라 `잘못된 음절/모라 선택 자체를 줄이는 것`이다.
기존 구조는 매핑, 타이밍 계산, 후처리 가드, ML 보정, fallback이 한 흐름에 섞여 있어서 다음 문제가 반복됐다.

- CV/CV_HEAD/VCV가 잘못된 음절에 붙는다.
- 반복 음절에서 한 칸씩 밀리며 연쇄 오매핑이 난다.
- 무음/저활성 구간에도 alias가 생성된다.
- ML이 연결감을 개선해도 잘못 매핑된 대상을 되살릴 수는 없다.

v2의 의도는 다음과 같다.

- 매핑을 먼저 확정하고 timing은 그 다음에 계산한다.
- 파일 전체를 한 번에 보는 전역 planner를 우선 사용한다.
- low-trust 상태에서는 자유도를 줄이고 보수적으로 생성한다.
- ML은 `VC/VV/V-CV` 같은 연결 타이밍 자연화에 집중시킨다.

## 2. 현재 구현된 구조

현재 코드는 완전한 신규 엔진으로 독립한 상태는 아니지만, 핵심 계층은 이미 v2 모듈로 분리되어 있다.

### 공통 계층

- `alignment_ingest`
  - [core/alignment_ingest.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/alignment_ingest.py)
- `candidate selection / filtering`
  - [core/oto_candidate_selection.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_candidate_selection.py)
  - [core/oto_mapping_candidates.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_mapping_candidates.py)
- `mapping planner / confidence / policy`
  - [core/oto_mapping_plan.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_mapping_plan.py)
  - [core/oto_mapping_confidence.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_mapping_confidence.py)
  - [core/oto_mapping_policy.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_mapping_policy.py)
  - [core/oto_runtime_policy.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_runtime_policy.py)
- `optional sinsy syllable label ingest`
  - [core/sinsy_label_ingest.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/sinsy_label_ingest.py)
- `anchor graph`
  - [core/oto_anchor_graph.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_anchor_graph.py)
- `row policy / abstain / diagnostics`
  - [core/oto_row_policy.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_row_policy.py)
  - [core/oto_row_abstain.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_row_abstain.py)
  - [core/oto_diagnostics.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_diagnostics.py)
  - [core/oto_diagnostics_adapter_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_diagnostics_adapter_v2.py)
- `row output / finalize`
  - [core/oto_row_output_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_row_output_v2.py)
  - [core/oto_row_finalize_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_row_finalize_v2.py)
- `anchor lock adapter`
  - [core/anchor_lock_adapter_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/anchor_lock_adapter_v2.py)

### 일본어 계층

- mapping helper
  - [core/ja_mapping_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_mapping_v2.py)
  - [core/ja_mapping_select_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_mapping_select_v2.py)
  - [core/ja_mapping_scoring_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_mapping_scoring_v2.py)
- timing helper
  - [core/ja_timing_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_timing_v2.py)
- row runtime / finalize
  - [core/ja_row_runtime_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_row_runtime_v2.py)
  - [core/ja_row_finalize_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_row_finalize_v2.py)
  - [core/ja_vcv_row_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_vcv_row_v2.py)
  - [core/ja_cv_head_row_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_cv_head_row_v2.py)
  - [core/ja_general_row_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_general_row_v2.py)
- anchor lock
  - [core/ja_anchor_lock_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_anchor_lock_v2.py)

### 한국어 계층

- mapping helper
  - [core/kr_mapping_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_mapping_v2.py)
  - [core/kr_mapping_select_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_mapping_select_v2.py)
  - [core/kr_mapping_scoring_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_mapping_scoring_v2.py)
  - [core/kr_candidate_selection_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_candidate_selection_v2.py)
- timing helper
  - [core/kr_timing_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_timing_v2.py)
  - [core/kr_oto_vc.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_oto_vc.py)
- row runtime / finalize
  - [core/kr_row_runtime_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_row_runtime_v2.py)
  - [core/kr_row_finalize_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_row_finalize_v2.py)
  - [core/kr_vcv_row_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_vcv_row_v2.py)
  - [core/kr_cv_head_row_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_cv_head_row_v2.py)
  - [core/kr_general_row_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_general_row_v2.py)
- anchor lock / feedback
  - [core/kr_anchor_lock_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_anchor_lock_v2.py)
  - [core/kr_runtime_feedback_v2.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_runtime_feedback_v2.py)

## 3. 현재 generator의 역할

현재 [core/ja_oto_generator.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_oto_generator.py) 와 [core/oto_generator.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_generator.py) 는 예전보다 역할이 줄었다.

주요 역할은 다음으로 정리됐다.

- 파일별 context / ingest 준비
- 후보 source 선택
- planner 실행 및 runtime policy 적용
- alias family 분기
- 선택된 인덱스를 각 row executor에 전달
- 최종 export와 postprocess 연결

즉, generator는 예전의 “모든 로직을 직접 수행하는 파일”에서 “오케스트레이션 중심 파일”로 이동 중이다.

## 4. 실제로 개선된 점

현재 구조에서 기대하는 실효 개선 포인트는 다음과 같다.

- `CV/CV_HEAD/VCV`의 잘못된 음절 선택 감소
- 반복 음절에서의 누적 드리프트 감소
- 무음/저활성 구간 매핑 감소
- low-trust 정렬 결과에서 과도한 자유 탐색 억제
- low-margin row를 보수적으로 skip해서 잘못된 `CV/CV_HEAD` 생성 억제
- 사용자가 제공한 sinsy 음절 라벨을 opt-in anchor source로 사용 가능
- plain/glide/youon 오매핑을 줄이기 위한 forward jump/모음 보정 가드 강화
- `VV`를 vowel pair 기준으로 재평가해 마지막 모음만 보고 매핑하는 문제를 완화
- `VC/VV/V-CV`는 기존 ML/후처리 장점을 유지

특히 일본어 `CVVC`와 한국어 `CVVC/VCV`에서 문제였던
`plain vowel -> glide/youon 오매핑`, `wrong occurrence`, `silence placement`
방지 방향으로 구조가 바뀌었다.

## 5. 아직 남은 것

핵심 재작성은 사실상 끝났지만, 완전히 마감된 상태는 아니다.

남은 고가치 작업은 다음과 같다.

- generator 내부에 남은 언어별 scoring 유틸 완전 이관
- 한국어 `VCV` 주변의 잔여 보조 scoring / drift repair 유틸 정리
- batch evaluation 재실행 후 v2 전/후 수치 문서화
- 실제 oto 생성 샘플 청감 검증 정리
- 필요 시 `mapping_core_v2_design` 문서를 현재 구현 기준으로 재서술

즉, 현재 단계는 `핵심 구조 이관 완료, 검증/마감 정리 단계`로 보는 것이 맞다.

## 6. 검증 현황

현재 타깃 테스트는 아래 기준으로 통과했다.

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q ml/tests/test_sinsy_label_ingest.py ml/tests/test_oto_runtime_policy.py ml/tests/test_oto_row_abstain.py ...`
- 결과: `52 passed`

이번 v2 작업 중 추가된 테스트 범위는 다음을 포함한다.

- planner / anchor graph
- runtime policy / row policy / abstain
- optional sinsy label ingest / guided anchor plan
- diagnostics / output / finalize
- JA/KR mapping selection
- JA/KR row executor
- JA/KR timing helper

전체 `ml/tests`는 현재도 일부 레거시/외부 의존 테스트의 import 오류 때문에 수집 단계에서 중단된다.

## 7. 레거시 처리 방침

더 이상 핵심 흐름에서 쓰지 않는 일부 분석 스크립트는 다음 폴더로 이동했다.

- [scripts/legacy_mapping_v1/README.md](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/scripts/legacy_mapping_v1/README.md)
- [scripts/legacy_misc_tools/README.md](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/scripts/legacy_misc_tools/README.md)

이 폴더는 참고/회귀용 보관 영역이며, 새 기능 개발의 기본 경로는 아니다.

## 8. 운영 판단

현재 코드베이스는 “기존 시스템을 유지한 채 새 매핑 코어를 병렬로 심는 단계”를 넘어,
실질적으로는 JA/KR 핵심 형식의 매핑 엔진이 v2 구조를 사용하도록 상당 부분 전환된 상태다.

따라서 다음 판단이 유효하다.

- 새 기능은 가능하면 v2 모듈에 추가한다.
- generator 본문에는 orchestration 외의 새 복잡 로직을 되도록 넣지 않는다.
- ML은 timing refiner 역할을 유지하고, mapping 결정권은 planner/policy 쪽에 둔다.
- low-trust 상황에서는 더 공격적으로 찾기보다 더 보수적으로 생성한다.
