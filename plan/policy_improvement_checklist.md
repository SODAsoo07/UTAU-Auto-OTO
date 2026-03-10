# OTO Quality Improvement Checklist

기준일: 2026-03-10

이 문서는 현재 남은 핵심 품질 과제를 구현 단위로 나눈 체크리스트와 구현안이다.
우선순위는 다음 세 가지 문제를 기준으로 잡는다.

1. 음절 오매핑 발생 빈도 추가 감소
2. 한국어 CVVC의 불안정한 파라미터 지정 개선
3. `VC` 등 연결 음소 자연스러움 강화

---

## 전체 전략

핵심 원칙은 다음과 같다.

- 매핑 문제와 타이밍 문제를 분리한다.
- `CV/CV_HEAD/VCV`는 전역 planner와 하드 제약으로 안정화한다.
- `VC/VV/V-CV`는 매핑이 맞은 뒤 자연스러움 보정에 집중한다.
- low-trust 상태에서는 더 넓게 찾지 말고 더 보수적으로 생성한다.
- sinsy 라벨은 선택형 보조 입력으로만 사용한다.
  없으면 기존 파이프라인 그대로 동작하고, 있을 때만 음절 anchor 우선순위를 높인다.

---

## Phase A. Mapping Hardening

목표:

- 잘못된 음절 선택을 더 줄인다.
- `plain vowel <-> glide/youon` 혼동을 더 강하게 차단한다.
- low-trust 파일에서 억지 선택을 줄인다.

### A-1. KR/JA plain-glide/youon reject 강화

- [ ] 일본어 `CV/CV_HEAD/VCV`에서 plain mora와 youon/inserted mora 불일치 패널티 강화
- [ ] 한국어 `CV/CV_HEAD/VCV`에서 plain vowel과 glide vowel 불일치 패널티 강화
- [ ] `exact match`가 아니면 `+1 forward jump` 허용 조건 축소
- [ ] 반복 음절 구간에서 forward fix는 `exact + vowel match + small delta` 동시 충족 시에만 허용

후보 파일:

- `core/ja_mapping_scoring_v2.py`
- `core/ja_mapping_select_v2.py`
- `core/kr_mapping_scoring_v2.py`
- `core/kr_mapping_select_v2.py`

구현안:

- 일본어는 `should_allow_ja_soft_forward_shift`, `find_ja_cv_vowel_match_index`, `prefer_vcv_candidate_index` 쪽에서:
  - target이 plain이면 youon/inserted 후보를 거의 reject
  - expected가 이미 target vowel을 만족하면 forward fix를 더 강하게 차단
- 한국어는 `resolve_cv_syllable_index`, `select_kr_vcv_index`, `select_kr_general_cv_index` 쪽에서:
  - `go -> gyo`, `do -> dyo`, `o -> yo`, `reo -> ryeo` 케이스에 강한 penalty
  - expected가 same vowel이면 onset-only improvement로는 전진 허용 금지
- 공통으로:
  - `forward jump`는 `best_gain`, `vowel exact`, `distance <= 1`을 모두 만족할 때만 허용
  - 반복 음절 구간은 `exact token match`가 없으면 순서 고정 우선

### A-2. VV를 vowel pair 기준으로 재평가

- [ ] 한국어 `VV`에 `pair scoring` 추가
- [ ] 일본어 `VV`도 필요 시 `pair scoring` 추가
- [ ] `last vowel only` 성격의 느슨한 fallback 사용 축소

후보 파일:

- `core/kr_mapping_select_v2.py`
- `core/ja_general_row_v2.py`
- 필요 시 새 모듈: `core/kr_vv_mapping_v2.py`, `core/ja_vv_mapping_v2.py`

구현안:

- `VV` alias를 `prev_vowel + next_vowel` pair로 normalize
- candidate 평가 시:
  - next vowel 일치만으로는 충분치 않음
  - prev vowel과 next vowel이 모두 맞을 때만 high confidence
- pair confidence가 낮으면:
  - `VV`를 보수 생성하거나 abstain
  - 필요 시 adjacent anchors만 남기고 fallback

### A-3. Low-margin abstain 강화

- [x] planner score margin 또는 local confidence margin 계산 추가
- [x] margin이 낮은 `CV/CV_HEAD`는 `row abstain` 빈도 증가
- [ ] low-trust 파일에서는 `forced index`와 `free search` 허용폭 축소

후보 파일:

- `core/oto_mapping_confidence.py`
- `core/oto_runtime_policy.py`
- `core/oto_row_abstain.py`

구현안:

- `top1 - top2` margin 또는 `expected vs selected` margin을 계산
- 기준:
  - high margin: 기존 동작 유지
  - mid margin: jump 축소, exact/vowel fix만 허용
  - low margin: `CV/CV_HEAD` abstain 또는 order lock
- file-level trust와 row-level margin을 함께 사용

### A-4. 사용자 sinsy 라벨 활용

- [x] sinsy 음절 라벨 ingest 추가
- [x] sinsy 라벨이 있으면 planner input anchor로 우선 사용
- [/] MFA phone tier는 내부 자음/모음 경계 추정용 보조로만 사용
- [ ] sinsy와 MFA가 크게 충돌하면 low-trust 경고 기록

정책:

- 필수 입력이 아니라 선택 기능으로 유지
- 사용자가 직접 제공한 경우에만 활성화
- 라벨이 없으면 현재 `MFA + planner + audio/phone 보조` 경로 유지
- 라벨이 있어도 `VC/VV` 내부 위치는 기존 audio/phone 추정 로직을 계속 사용

후보 파일:

- 새 모듈 `core/sinsy_label_ingest.py`
- `core/alignment_ingest.py`
- `core/oto_mapping_plan.py`
- `core/oto_runtime_policy.py`

구현안:

- sinsy 라벨 형식은 최소한 `start`, `end`, `label`을 가진 음절 구간으로 읽음
- planner는 `candidate_syllables` 대신 `sinsy syllable anchors`를 우선 사용
- `CV/CV_HEAD/VCV`는 sinsy anchor 기반으로 assignment
- `VC/VV`는 sinsy 음절 경계 + phones/audio로 내부 위치 추정
- 현재 구현 상태:
  - `core/sinsy_label_ingest.py` 추가
  - JA/KR generator에 `USE_SINSY_LABELS`, `SINSY_LABEL_PATH`, `UTOA_USE_SINSY_LABELS`, `UTOA_SINSY_LABEL_PATH` 연결
  - row-level abstain은 planner/runtime margin과 연결되어 low-margin `CV/CV_HEAD`를 건너뛸 수 있음
- UI/CLI에서는 `use_sinsy_labels` 같은 opt-in 플래그로만 켜지게 함

### A-5. 평가 지표 확장

- [ ] `wrong_syllable_rate` 추가
- [ ] `plain_glide_confusion_count` 추가
- [ ] `silence_placement_count` 추가
- [ ] `VV pair mismatch count` 추가

후보 파일:

- 배치 평가 스크립트
- 평가 리포트 생성 스크립트
- 필요 시 `scripts/run_oto_generation_batch.py` 주변 요약 로직

검증 기준:

- `wrong_syllable_rate` 감소
- `plain_glide_confusion_count` 감소
- selector top1 비퇴행

---

## Phase B. KR CVVC Timing Split

목표:

- 한국어 CVVC의 파라미터가 alias type별로 더 일관되게 계산되도록 한다.
- `CV`, `-CV`, `VC`, `VV`를 하나의 공통 감각으로 처리하지 않도록 분리한다.

### B-1. Alias type별 timing policy 분리

- [ ] `CV` timing policy 분리
- [ ] `CV_HEAD` timing policy 분리
- [ ] `VC` timing policy 분리
- [ ] `VV` timing policy 분리

권장 파일:

- 새 모듈 `core/kr_cv_timing_policy_v2.py`
- 새 모듈 `core/kr_cv_head_timing_policy_v2.py`
- 새 모듈 `core/kr_vc_timing_policy_v2.py`
- 새 모듈 `core/kr_vv_timing_policy_v2.py`

구현안:

- 각 모듈은 공통 출력만 맞춘다.
  - 반환: `offset, consonant, cutoff, pre, ovl`
- 계산 기준은 type별로 분리한다.
  - `CV`: onset usable region + vowel entry 기준
  - `CV_HEAD`: phrase-initial onset 기준, overlap 의미 축소
  - `VC`: prev vowel tail -> consonant center -> next vowel leakage 방지
  - `VV`: vowel handoff 유지, 다음 vowel body 과포함 방지

### B-2. 자음군별 preset policy 추가

- [ ] `plosive`
- [ ] `affricate`
- [ ] `fricative`
- [ ] `nasal`
- [ ] `liquid`
- [ ] `glide`

정책:

- `pre`, `ovl`, `cutoff`, `cons` 목표 비율을 자음군별로 분리
- 파열음은 짧고 단단하게
- 비음/유음은 더 넓고 부드럽게

구현안:

- `alias_text` 또는 onset class를 기준으로 class table을 만든다.
- table에는 최소한 아래 값을 둔다.
  - `pre_ratio`
  - `ovl_ratio`
  - `cons_extension`
  - `cutoff_margin`
- table은 `VC`, `CV`, `CV_HEAD`에서 다르게 적용 가능하게 한다.

### B-3. 종성 유무 분리

- [ ] 종성 있는 CV와 없는 CV를 다른 policy로 계산
- [ ] 종성 포함 syllable에서 `cons/cutoff` 과연장 방지
- [ ] 종성 없는 syllable에서 과도한 stiff timing 완화

구현안:

- `_split_kr_syllable_parts` 결과의 coda 존재를 timing policy 입력으로 사용
- coda가 있으면:
  - `cons`를 지나치게 길게 늘리지 않음
  - `cutoff`는 다음 vowel leakage 방지 쪽으로 당김
- coda가 없으면:
  - `pre`와 `ovl`을 조금 더 부드럽게 허용

### B-4. 파라미터 순서 계산식 단계 강제

- [ ] 계산식 단계에서 `ovl < pre <= cons < |cutoff|` 강제
- [ ] postprocess 이전에 alias type별 기본 geometry 정합 확인
- [ ] smoothing은 geometry가 맞는 row에만 적용

후보 파일:

- `core/kr_timing_v2.py`
- `core/kr_general_row_v2.py`
- `core/kr_cv_head_row_v2.py`
- `core/kr_oto_vc.py`
- `core/kr_oto_file_consistency.py`

구현안:

- type별 timing policy 내부에서 geometry clamp를 먼저 적용
- postprocess는 geometry correction이 아니라 미세보정 역할만 하게 함
- smoothing은:
  - same alias class
  - same consonant group
  - geometry-valid row
  에만 적용

검증 기준:

- 한국어 CVVC의 `off/pre/cut/ovl` 분산 축소
- `VC`가 다음 모음을 먹는 사례 감소
- `OpenUtau` 렌더 오류성 row 미생성 유지

---

## Phase C. Bridge Naturalness

목표:

- `VC`, `VV`, `V-CV` 연결이 더 자연스럽게 들리도록 한다.
- 자연스러움 강화 과정에서도 다음 음소 누출은 계속 억제한다.

### C-1. VC naturalness score 도입

- [ ] `prev vowel tail`
- [ ] `consonant center`
- [ ] `next vowel leakage`

세 축으로 브리지 상태를 점수화하고, 점수에 따라 shaping 강도를 조절한다.

권장 파일:

- 새 모듈 `core/kr_bridge_naturalness_v2.py`
- 새 모듈 `core/ja_bridge_naturalness_v2.py`

구현안:

- `prev vowel tail score`
  - 이전 모음 감쇠 구간을 얼마나 자연스럽게 포함하는가
- `consonant center score`
  - 자음 핵심 구간이 `pre/ovl` 기준으로 얼마나 안정적인가
- `next vowel leakage score`
  - cutoff가 다음 모음 body를 침범하는가
- 최종 score에 따라:
  - high: 자연스러운 overlap 허용
  - mid: 기본 shaping만 적용
  - low: cutoff를 보수적으로 당기고 overlap 축소

### C-2. alias class별 `ovl/pre` 비율 정책

- [ ] 파열음 bridge
- [ ] 마찰음 bridge
- [ ] 비음 bridge
- [ ] 유음 bridge
- [ ] glide bridge

정책:

- `overlap`은 `preutterance`보다 앞에 두되, 자음군마다 목표 비율을 다르게 설정
- bridge naturalness보다 `next vowel leakage 방지`를 항상 우선

구현안:

- class별 목표 구간 예시:
  - plosive: overlap 짧게, pre 명확히
  - fricative: overlap 조금 넓게
  - nasal/liquid: overlap 더 부드럽게
  - glide: 과도한 겹침 방지
- 목표 비율에서 크게 벗어나면 refiner가 다시 당기거나 줄이게 함

### C-3. cutoff guard 우선순위 고정

- [ ] 자연스러움 shaping 후에도 cutoff guard 재적용
- [ ] `VC`는 다음 모음 body 시작 전 종료 강제
- [ ] `VV`는 pair handoff 유지 범위 안에서만 연장 허용

후보 파일:

- `core/kr_oto_vc.py`
- `core/ja_timing_v2.py`
- `core/oto_generator.py`
- `core/ja_oto_generator.py`

구현안:

- naturalness shaping -> guard 재적용 -> file consistency 순으로 고정
- `VC`는 next vowel onset 또는 next vowel body 시작점을 soft/hard bound로 둠
- `VV`는 pair handoff 유지 길이를 넘으면 cutoff를 당김

검증 기준:

- 청감상 연결 자연스러움 개선
- `VC` 누출 사례 감소
- bridge timing MAE 비퇴행

---

## Phase D. Verification and Close-out

목표:

- 구조 변경이 실제 품질 향상으로 이어졌는지 수치와 청감으로 확인한다.

### D-1. Batch evaluation 재실행

- [ ] KR CVVC
- [ ] KR VCV
- [ ] JA CVVC
- [ ] JA VCV

### D-2. 새 지표 반영

- [ ] wrong syllable rate
- [ ] silence placement count
- [ ] plain/glide confusion count
- [ ] VV pair mismatch count

### D-3. Listening checks

- [ ] 반복 음절
- [ ] plain vs glide contrast
- [ ] 저음량/저활성 파일
- [ ] dense VC/VV chain
- [ ] 한국어 CVVC 실사용 곡 재생 확인

### D-4. 문서 동기화

- [ ] `plan/task.md` 갱신
- [ ] `plan/mapping_core_v2_status.md` 갱신
- [ ] 필요한 경우 운영 메모 문서 추가

---

## Recommended Execution Order

1. Phase A
2. Phase B
3. Phase C
4. Phase D

이 순서가 맞는 이유:

- 매핑이 틀리면 timing 개선의 의미가 작다.
- 한국어 CVVC는 현재 사용상 불만이 가장 큰 파라미터 영역이다.
- bridge naturalness는 geometry가 맞은 뒤 강화해야 효과가 크다.

---

## Practical Stop Condition

아래 조건을 만족하면 1차 마감으로 본다.

- 오매핑 체감 빈도가 추가로 감소
- 한국어 CVVC의 파라미터 뒤죽박죽 사례가 눈에 띄게 축소
- `VC/VV` 연결 청감이 개선
- 전체 회귀 테스트 비퇴행
- 배치 평가에서 핵심 지표 비퇴행 또는 개선
