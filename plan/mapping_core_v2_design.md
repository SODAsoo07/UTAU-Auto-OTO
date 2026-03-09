# Mapping Core v2 Design

## Summary

현재 품질 병목은 `offset/pre/cutoff` 계산식 자체보다, `어느 음절/모라에 alias를 매핑할지`를 결정하는 구조에 있다.

따라서 v2의 목표는 다음과 같다.

1. 매핑 확정과 타이밍 계산을 분리한다.
2. `row-by-row local fix` 구조를 `file-level global planning` 구조로 바꾼다.
3. 무음/저활성 구간 매핑을 구조적으로 차단한다.
4. 기존 ML의 장점인 `VC/VV/V-CV` 자연스러움 보정은 유지한다.
5. 기존 UI, 평가, export, postprocess 자산은 최대한 재사용한다.

이 문서는 "전체 전면 재작성"이 아니라, `mapping core 중심 반재작성`을 위한 설계 초안이다.

---

## Why Rewrite This Part

현재 구조의 핵심 문제:

- `매핑`, `타이밍 계산`, `보정 가드`, `ML delta`, `fallback`이 한 흐름에 섞여 있다.
- CV/CV_HEAD/VCV가 파일 전체 맥락이 아니라 개별 행 기준으로 보정된다.
- 오매핑이 한 줄에서 발생하면 뒤쪽 줄까지 누적 드리프트가 전파된다.
- 무음/저활성 구간을 사후 보정으로만 막고 있어, 구조적으로는 여전히 후보에 남아 있다.
- low-confidence 상황에서 "생성하지 않음"보다 "어딘가에라도 배치" 쪽으로 기울어 있다.

이 문제는 기존 구조 위에 규칙을 더 붙여도 줄일 수는 있지만, 장기적으로는 품질 상한이 낮다.

---

## Scope

### Rewrite

- file-level mapping planner
- mapping confidence policy
- candidate filtering / activity gating
- format-specific anchor resolution
- abstain / skip policy

### Reuse

- UI
- batch eval / logging / trace
- `oto.ini` read/write
- wav-duration sanitize / validate
- file consistency postprocess
- ML timing refiner
- alias classification / normalization utilities
- format export and OpenUtau compatibility code

### Remove Or De-prioritize

- SOFA alignment path
- local-only remap cascade as the primary strategy
- mapping and timing mixed in one decision stage

---

## Core Principles

1. Mapping first, timing second.
2. Global sequence optimization first, local correction only as fallback.
3. Hard constraints first, ML second.
4. If confidence is low, reduce freedom instead of increasing search.
5. `CV/CV_HEAD/VCV` are mapping-sensitive.
6. `VC/VV` are timing-sensitive after anchors are correct.

---

## v2 Architecture

### 1. `alignment_ingest`

역할:

- MFA TextGrid 읽기
- phones / words / optional energy-derived activity 정보 정리
- file-level trust score 계산

출력:

- `phones`
- `words`
- `alignment_trust`
- `timeline_meta`

### 2. `candidate_builder`

역할:

- phone/word 기반으로 syllable candidate 생성
- silence / low-activity / no-vowel candidate 제거
- candidate별 token, onset, vowel, coda, activity metrics 계산

출력:

- `candidate_syllables`
- `candidate_features`

### 3. `mapping_planner`

역할:

- filename/alias 기반 expected token sequence 생성
- expected token과 candidate syllable의 점수 행렬 생성
- monotonic DP/Viterbi로 전역 경로 선택

출력:

- `planned_cv_indices`
- `planned_anchor_map`
- `plan_score`
- `plan_margin`

### 4. `confidence_policy`

역할:

- alignment trust
- planner score / margin
- activity coverage
- format constraints

를 종합해 `high / mid / low / abstain` 결정

출력:

- `mapping_tier`
- `fallback_mode`
- `allow_generation`

### 5. `anchor_graph`

역할:

- 확정된 `CV/CV_HEAD/VCV` anchor를 기반으로
- `VC`, `VV`, `V-CV` 등 연결 관계를 그래프로 구성

출력:

- `anchor_by_idx`
- `bridge_edges`

### 6. `timing_engine`

역할:

- 형식별 기본 파라미터 계산
- anchor graph 기준으로 `offset/pre/ovl/cutoff` 산출

출력:

- raw oto rows

### 7. `timing_refiner_ml`

역할:

- 이미 확정된 anchor 위에서만
- `VC`, `VV`, `V-CV` 연결감 자연화
- delta clipping / post-guard 적용

주의:

- 이 단계는 인덱스를 변경하지 않는다.

### 8. `postprocess`

역할:

- validate
- wav duration clamp
- file consistency
- export

---

## Internal Score Design

planner score는 아래 항목의 가중 합으로 구성한다.

### Positive Terms

- exact token match
- vowel match
- onset match
- coda structure match
- glide/youon class match
- nearby position prior
- local continuity prior
- activity bonus
- filename-order prior

### Negative Terms

- silence candidate penalty
- no-vowel penalty for CV/CV_HEAD/VCV
- plain vs glide mismatch penalty
- plain vs youon mismatch penalty
- coda mismatch penalty
- distance penalty
- over-jump penalty
- low-trust free-search penalty

### Hard Reject Rules

다음은 점수 이전에 후보 탈락:

- CV/CV_HEAD인데 모음이 없음
- candidate가 silence/sp/pau/spn 중심
- active duration이 최소치 미만
- target plain인데 candidate가 glide/youon이고 exact match도 아님
- target에 coda가 없는데 candidate에 coda만 남아 있는 경우

---

## Why Global DP / Viterbi

행 단위 탐색은 "지금 줄에 가장 좋아 보이는 후보"를 고른다.
하지만 실제 필요한 것은 "파일 전체에서 가장 일관된 경로"다.

DP/Viterbi를 쓰는 이유:

- monotonic order 보장
- repeated syllable drift 감소
- local optimum보다 global optimum 우선
- one-step correction 누적 실패 감소
- confidence margin 계산이 쉬움

v2에서는 local remap을 기본 전략이 아니라 fallback 전략으로만 사용한다.

---

## Low-Trust Strategy

TextGrid 신뢰도가 낮을수록 자유도를 줄여야 한다.

### High Trust

- MFA phones/words 적극 사용
- planner full score 사용
- local correction 일부 허용
- ML timing full strength

### Mid Trust

- filename order prior 강화
- jump 제한 강화
- activity filter 강화
- local correction window 축소
- ML timing은 유지하되 conservative clipping

### Low Trust

- filename order hard lock
- active-only candidate
- no global free search
- weak local repair only
- low-confidence row abstain 허용
- ML timing limited mode

### Very Low Trust

- conservative fallback or skip
- invalid rows are not generated

---

## Format Rules

### CVVC

가장 먼저 v2로 이관할 형식.

- `CV/CV_HEAD`: planner가 확정
- `VC/VV`: 확정된 양옆 anchor 기준 계산
- planner confidence가 낮으면 skip 또는 conservative fallback

### VCV

- `VCV` / `-CV` 모두 file-level plan 사용
- local remap은 1-step exact recovery 수준으로 제한

### CVC

- CV anchor는 planner 기반
- coda bridge는 adjacent anchor 기반

### CV

- planner 단순 버전 적용 가능
- 후순위 이관

---

## Language-Specific Notes

### Japanese

중점:

- youon / inserted mora mismatch penalty
- repeated mora occurrence stability
- VCV drift suppression
- silence candidate rejection

### Korean

중점:

- glide vowel mismatch penalty
- coda/no-coda distinction
- `go -> gyo`, `do -> dyo`, `reo -> ryeo` 차단
- low-volume nuclei instability 대응

---

## Migration Plan

### Phase A

- new planner core 추가
- current generator에서 optional path로 병렬 연결
- Japanese CVVC, Korean CVVC부터 적용

### Phase B

- Japanese VCV
- Korean VCV
- Korean CVC

### Phase C

- common planner interface 정리
- old local remap path 축소
- low-confidence abstain 정식 도입

### Phase D

- ML reranker 추가 검토
- planner score + ML rank ensemble

---

## Code Organization Proposal

신규 또는 분리 대상:

- `core/oto_mapping_plan.py`
- `core/oto_mapping_confidence.py`
- `core/oto_mapping_candidates.py`
- `core/oto_mapping_policy.py`
- `core/ja_mapping_v2.py`
- `core/kr_mapping_v2.py`

기존 generator는 최종 orchestration 역할로 축소:

- ingest
- planner call
- timing engine call
- ML refiner call
- export

---

## Verification Plan

### Unit Tests

- monotonic planning path correctness
- silence candidate rejection
- glide/youon mismatch blocking
- repeated token stability
- low-trust fallback behavior
- abstain behavior

### Regression Tests

- Japanese CVVC
- Korean CVVC
- Korean VCV

확인 지표:

- CV/CV_HEAD mis-mapping rate
- silence-placement count
- repeated-token drift count
- selector top1 non-regression
- bridge timing MAE non-regression

### Listening Checks

- repeated syllables
- glide vs plain vowel contrast
- low-volume recordings
- dense VC/VV chains

---

## Success Criteria

v2 도입 성공 기준:

1. CV/CV_HEAD 오매핑이 현저히 감소
2. 무음 공백 매핑이 구조적으로 거의 제거
3. VC/VV 연결 자연스러움은 유지 또는 개선
4. low-trust 파일에서 "틀리게 생성"보다 "보수 생성/스킵" 비율 증가
5. batch eval과 실제 청감이 동시에 개선

---

## Recommendation

권장 전략은 다음과 같다.

- 기존 시스템 전체를 버리지 않는다.
- `mapping core`만 v2로 병렬 재설계한다.
- Japanese CVVC, Korean CVVC부터 A/B 비교한다.
- timing ML과 postprocess 자산은 그대로 재사용한다.

이 접근이 품질 향상 대비 위험이 가장 낮다.
