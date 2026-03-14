# 경계탐지 + OTO지정 2단 모델 호환 설계안

## 1) 목표

- 목적: OTO 품질 문제를 `경계/매핑` 문제와 `파라미터 배치` 문제로 분리해 정확도와 디버깅 가능성을 동시에 높인다.
- 핵심 아이디어:
  - 1단: 경계 탐지 특화 모델 (`공백/음성`, `온셋`, `핵(vowel nucleus)`, `tail`).
  - 2단: 1단 출력 + 기존 문맥 feature로 OTO 파라미터(`offset, cons, cutoff, pre, ovl`) 산출.
- 전제: 기존 파이프라인/환경변수/모델 번들 구조를 유지하면서 점진적으로 도입한다.

## 2) 현재 코드와의 호환 기준

- ML 후처리 진입점은 이미 고정되어 있음:
  - `core/post_file_pipeline.py::run_ml_post_stage()`
  - `core/oto_ml_refiner.py::apply_oto_ml_to_oto_file()`
- 현재 라우트:
  - `legacy` (lightgbm/ensemble/coupled 계열)
  - `autofree_v1`
- 런타임 안전장치:
  - confidence gate, selector abstain, anchor lock lite, wav duration hard guard, fallback 모델.
- 이미 경계 보조 신호가 존재:
  - `core/oto_ml/features/schema.py`의 `AUX_TARGET_NAMES`
  - `core/oto_ml/coupled/training.py`의 boundary auxiliary loss
  - `core/oto_ml/autofree/schema.py`의 `onset_ms/nucleus_start_ms/nucleus_end_ms/tail_ms/source_confidence`

즉, 완전 신규 체계가 아니라 기존 체계를 확장하는 방식이 가장 안전하다.

## 3) 제안 아키텍처 (호환형 2단)

## 3.1 Stage A: Boundary Model (경계 탐지)

- 역할:
  - row 단위로 경계 anchor와 경계 신뢰도 예측
  - `speech/silence`, `voiced/unvoiced/breath` 확률 제공
- 출력 계약 (`BoundaryResultV1`, 신규):
  - `boundary_onset_ms`
  - `boundary_nucleus_start_ms`
  - `boundary_nucleus_end_ms`
  - `boundary_tail_ms`
  - `boundary_blank_conf`
  - `boundary_voiced_conf`
  - `boundary_unvoiced_conf`
  - `boundary_breath_conf`
  - `boundary_confidence`
  - `boundary_source` (`textgrid`, `audio_only`, `hybrid`)

## 3.2 Stage B: OTO Parameter Model (기존 보정 확장)

- 역할:
  - 기존 feature + `BoundaryResultV1`를 받아 최종 OTO 파라미터 산출
- 출력:
  - 기존과 동일 (`delta_*` 또는 `target_*_ms`)
- 적용 규칙:
  - `boundary_confidence`가 낮으면 Stage A 값을 강제 적용하지 않고 기존 feature/기본 계산 사용
  - 기존 fallback 체계(legacy, selector-only, base 유지)를 그대로 사용

## 3.3 범위 밖(초기 거절/보류)

- `자모 완전 분리`를 음향만으로 단독 해결하는 목표는 초기 범위에서 제외한다.
  - 이유: 데이터 요구량과 언어/뱅크별 편차가 크고 운영 리스크가 큼.
  - 대안: 자모 후보는 기존 alias/파일명/사전 기반 제약을 유지하고, 음향 모델은 경계 위치와 blank/voicing 판단에 집중.

## 4) 코드 변경 설계 (파일 단위)

## 4.1 신규 모듈

- `core/oto_ml/boundary/schema.py`
  - Boundary feature/target schema 정의
- `core/oto_ml/boundary/data.py`
  - 학습/추론용 row 생성 (`TextGrid 우선`, 없으면 `audio_only`)
- `core/oto_ml/boundary/model.py`
  - LightGBM 기반 baseline (추후 coupled head 재사용 가능)
- `core/oto_ml/boundary/inference.py`
  - 번들 로드/예측 API
- `core/oto_ml_boundary.py`
  - 기존 스타일과 맞춘 re-export shim

## 4.2 기존 파일 확장

- `core/oto_ml_refiner.py`
  - `ml_route`에 `two_stage_v1` 추가
  - row 루프에서 Stage A 결과를 `feat`에 주입 후 Stage B 호출
  - boundary gate (`boundary_confidence` 임계치) 적용
- `core/post_file_pipeline.py`
  - 기존 `ml_route` 인자 전달 구조 유지 (호출부 변경 최소)
- `core/oto_ml_runtime.py`
  - 기존 OTO backend 로더는 유지
  - Stage A는 별도 loader를 refiner에서 호출 (기존 런타임 계약 비침범)
- `core/oto_ml/features/schema.py`
  - Stage B 신규 feature를 위한 `feature_version` 상향(`v13`) 계획
  - 기존 `v12` 모델과 동시 운영 가능하도록 canonicalization 기본값 유지

## 4.3 학습/CLI 스크립트 추가

- `ml/scripts/build_oto_boundary_dataset.py`
- `ml/scripts/train_oto_boundary_model.py`
- `ml/scripts/evaluate_oto_boundary.py`
- `ml/scripts/run_two_stage_training.ps1` (선택)

명명 규칙은 기존 `build_oto_*`, `train_oto_*`, `evaluate_oto_*` 패턴을 따른다.

## 5) 데이터/피처 설계

## 5.1 Stage A 입력 (최소)

- 기존 mel/window 계열
- alias_type, format_type, row 위치 정보
- TextGrid 기반 phone interval (존재 시)
- filename token 기반 약한 제약

## 5.2 Stage A 라벨

- 우선순위:
  - 1순위: manual OTO + TextGrid 정합 기반 라벨
  - 2순위: TextGrid 직접 라벨
  - 3순위: audio-only 약라벨 (낮은 sample_weight)

## 5.3 Stage B 입력 확장

- 기존 `FEATURE_NAMES` + `BoundaryResultV1` 파생값
  - 예: `boundary_onset_to_pre_ms`, `boundary_tail_to_cutoff_ms`, `boundary_confidence`
- 기존 feature가 없던 모델을 깨지 않도록 누락값 기본 0 처리 유지

## 6) 런타임 플로우 (two_stage_v1)

1. 기존 방식으로 `extract_feature_rows()` 수행
2. Stage A boundary predict
3. `boundary_confidence` 평가
4. 신뢰 높음:
   - boundary 기반 보정 feature 생성 후 Stage B 실행
5. 신뢰 낮음:
   - 기존 feature 기반 Stage B 또는 기존 fallback
6. 기존 finalize guard(validator/anchor lock/wav duration) 그대로 적용

이 구조는 현재 `apply_oto_ml_to_oto_file()`의 fallback 체계와 충돌하지 않는다.

## 7) 환경변수/설정 제안

- `UTOA_ML_ROUTE=two_stage_v1`
- `UTOA_ML_BOUNDARY_ENABLE=1`
- `UTOA_ML_BOUNDARY_MIN_CONF=0.62`
- `UTOA_ML_BOUNDARY_DIR=<model_dir>`
- `UTOA_ML_TWO_STAGE_FAIL_OPEN=1`
  - 1: Stage A 실패 시 legacy로 자동 폴백
  - 0: Stage A 실패를 ML 실패로 처리

기본값은 fail-open으로 둬서 운영 안전성을 우선한다.

## 8) 단계별 도입 계획

## Phase 0: Shadow (동작 불변)

- Stage A 결과를 계산만 하고 실제 파라미터 적용에는 미반영
- 로그/리포트에 boundary 품질 지표만 수집

완료 기준:
- 처리 실패율 증가 없음
- runtime latency 증가 허용 범위 이내

## Phase 1: Gated 적용 (제한 포맷)

- 한국어 `cvvc/cvc`의 `cv/cv_head`에만 적용
- `boundary_confidence` 기준 미달 시 자동 비적용

완료 기준:
- 오매핑/타이밍 붕괴 케이스 감소
- fallback 비율 과도 상승 없음

## Phase 2: 범위 확장

- `vc/vv/vcv`, 일본어 `cvvc/vcv`로 확대
- alias family별 임계치 튜닝

## 9) 평가 지표

## 9.1 Stage A 단독

- onset MAE, nucleus_start MAE, tail MAE
- blank/speech 분류 F1
- confidence calibration (reliability curve)

## 9.2 End-to-End

- alias_type별 OTO 파라미터 MAE
- cutoff overreach 비율
- low-confidence fallback 비율
- 실제 생성 결과에서 rhythm drift/bridge disconnect 보고 건수

## 10) 테스트 설계

- `ml/tests/test_boundary_pipeline.py`
  - boundary dataset build/load/predict contract
- `ml/tests/test_two_stage_route.py`
  - `ml_route=two_stage_v1` fallback 동작 검증
- 기존 테스트 보강:
  - `ml/tests/test_runtime_loading.py`
  - `ml/tests/test_autofree_pipeline.py`

필수 회귀 포인트:
- `legacy`, `autofree_v1` 경로 동작 불변
- 모델 미설치/번들 불일치 시 기존 코드와 동일한 안전 폴백

## 11) 구현 우선순위 (현실적 순서)

1. Phase 0 shadow 구현 (behavior 변화 없음)
2. Stage A dataset/train/eval CLI 추가
3. `two_stage_v1` 라우트 추가 + gated 적용
4. feature schema v13 확장 및 신규 모델 학습
5. A/B 리포트 기반 임계치 튜닝

## 12) 결론

- 제안한 2단 분리는 현재 코드와 충돌하지 않고 도입 가능하다.
- 단, 성공의 핵심은 `완전 분리`가 아니라 `기존 안전장치 위에 경계 모델을 점진적으로 얹는 것`이다.
- 초기에는 `자모 음향 단독 분리`를 목표로 삼지 않고, 경계/blank/voicing 안정화에 집중해야 실사용 품질 개선을 빠르게 얻을 수 있다.

