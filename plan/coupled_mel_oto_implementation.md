# Coupled mel+oto 모델 구현 정리 (현재 코드 기준)

## 1. 목표와 위치
- 백엔드 이름: `coupled_nn_v1`
- 핵심 구현 파일:
  - `core/oto_ml_coupled.py` (데이터셋 빌드/학습/추론/평가)
  - `core/oto_ml_features.py` (mel+oto 결합 feature 추출)
  - `core/oto_ml_refiner.py` (런타임 적용, fallback, 제약 정렬)
  - `core/oto_ml_runtime.py` (번들 로딩 라우팅)
  - `core/oto_generator.py` (`_mel_envelope`에서 프레임 클래스 생성)

모델은 OTO 5개 파라미터 델타(`delta_offset/delta_cons/delta_cutoff/delta_pre/delta_ovl`)를 동시에 예측하고, 별도 confidence를 함께 출력한다.

---

## 2. 데이터 구성 방식 (mel + row feature 결합)

### 2.1 학습 row 생성
- 진입점: `build_training_rows(...)` (`core/oto_ml_features.py`)
- 입력:
  - 자동 OTO (`auto_oto`)
  - 수작업 OTO (`manual_oto`)
  - TextGrid, WAV
- 매칭:
  - `(wav_norm, alias_norm)` 그룹으로 우선 매칭
  - 반복 alias는 `occurrence_index` 우선
  - 실패 시 시간 근접 매칭
- 타깃 생성:
  - 수작업 OTO 기준 절대/상대 해석 후
  - `delta_* = manual - base`로 5타깃 생성

### 2.2 mel 프레임 클래스 (신호 계층)
- 생성 위치: `core/oto_generator.py::_mel_envelope`
- 프레임별 class:
  - `cls_voiced_formant` (F2/F3 + 유성 단서)
  - `cls_silence_sparse` (저에너지/저스펙트럼)
  - `cls_unvoiced_diffuse` (고역/확산형 무성 성분)
  - `cls_breath_like` (저에너지+고역 비중)
- 이 class가 `core/oto_ml_features.py::_compute_segment_stats`에서 집계되어 row feature로 들어간다.

### 2.3 coupled 전용 feature
- 비율/경계:
  - `mel_voiced_formant_ratio`
  - `mel_silence_sparse_ratio`
  - `mel_unvoiced_diffuse_ratio`
  - `mel_breath_like_ratio`
  - `blank_span_confidence`
  - `mel_offset_candidate_ms`
  - `mel_cutoff_candidate_ms`
- patch feature(모델 patch branch 입력):
  - `onset_patch_energy_mean`
  - `onset_patch_voiced_ratio`
  - `onset_patch_unvoiced_ratio`
  - `tail_patch_energy_mean`
  - `tail_patch_silence_ratio`
  - `blank_span_confidence`

---

## 3. 모델 구조
- 정의 위치: `core/oto_ml_coupled.py::_build_model`
- 입력 2가지:
  - 일반 row feature 벡터 (`FEATURE_NAMES`)
  - patch 벡터 (`PATCH_FEATURES`)
- 네트워크:
  - `feature_net`: Linear(in,160) -> ReLU -> Linear(160,160) -> ReLU
  - `patch_net`: Linear(patch,64) -> ReLU -> Linear(64,64) -> ReLU
  - `joint`: concat -> Linear(224,160) -> ReLU -> Linear(160,80) -> ReLU
  - `delta_head`: Linear(80,5)
  - `conf_head`: Linear(80,1) + Sigmoid
- 범주형 feature는 문자열 해시(`_stable_hash_to_unit`)로 0~1 스칼라로 변환 후 입력한다.

---

## 4. 학습 로직

### 4.1 split
- `GroupShuffleSplit(test_size=0.2)` 우선 (`group_column=voicebank_id`)
- group split 불가 시 순차 80/20 분할

### 4.2 손실
- 기본 회귀:
  - 가중 SmoothL1(Huber), 타깃 가중치 `[1.00, 0.90, 0.95, 1.00, 0.65]`
  - row별 `sample_weight` 반영
- 제약 페널티:
  - `ovl <= pre`
  - `cons >= pre + 10`
  - `|cutoff| >= cons + 10`
  - `offset >= 0`
- 경계 정렬 손실:
  - `(offset, cutoff_abs)` vs `(mel_offset_candidate_ms, mel_cutoff_candidate_ms)` SmoothL1
- confidence 손실:
  - target: `exp(-mean_abs_error/80)`
  - BCE 사용
- 최종:
  - `total = base + 0.25*penalty + 0.12*align + 0.05*conf`

### 4.3 학습 제어
- `epochs` 기본 70
- `patience` 10 (validation total loss 기준 early stop)
- device:
  - `auto`: CUDA 가능 시 CUDA, 아니면 CPU
  - `cpu/cuda` 명시 지원

---

## 5. 산출물(번들)
- `coupled_model.pt`
- `model_meta.json`
  - `backend: coupled_nn_v1`
  - `feature_version`, `mel_patch_spec`
  - `min_confidence`
  - `fallback_order: [coupled_nn_v1, lightgbm, base]`
- `feature_schema.json`
- `eval_summary.json`

---

## 6. 런타임 적용 및 fallback

### 6.1 경로 선택
- `core/oto_ml_refiner.py`에서 route별 모델 탐색:
  - coupled 우선, 실패 시 lightgbm fallback
- 한국어 `VCV`에서 받침 성격 `VC(stop/nasal/liquid)` alias는 `cvc` route를 우선 사용하도록 확장됨.

### 6.2 coupled confidence gate
- env: `UTOA_ML_COUPLED_MIN_CONF` (기본 0.55)
- coupled confidence가 임계치 미만이면 `CoupledLowConfidenceError` 발생
- lightgbm fallback 번들이 있으면 즉시 하향 추론

### 6.3 제약 재정렬(공동 solver)
- `_solve_joint_constraints(...)`에서 최종 파라미터 일관성 재보정
- strict 모드:
  - env `UTOA_ML_COUPLED_STRICT_CONSTRAINT=1`
  - margin이 일반(10/10)보다 강화(16/14)

### 6.4 리포트 필드
- 런타임 리포트에 다음이 기록됨:
  - `route`
  - `model_confidence`
  - `fallback_reason`
  - `blank_confidence`
  - `constraint_adjust_count`

---

## 7. CLI 엔트리포인트
- 데이터셋 생성:
  - `ml/scripts/build_oto_mel_coupled_dataset.py`
- 학습:
  - `ml/scripts/train_oto_mel_coupled_model.py`
- 평가:
  - `ml/scripts/evaluate_oto_mel_coupled.py`

모든 스크립트는 `core/runtime_encoding.py::bootstrap_utf8_runtime()`을 통해 UTF-8 런타임을 먼저 고정한다.

---

## 8. CPU-only 운영 가능성 (현재 구현)
- 가능.
- `--device cpu` 또는 `UTOA_ML_COUPLED_DEVICE=cpu`로 고정 가능.
- CUDA가 없으면 `auto`도 CPU로 자동 하향.
- coupled 로드/추론 실패 시 런타임은 lightgbm -> base 순으로 하향하도록 설계되어 기능 중단을 피한다.
