# OTO 정확도 개선 2차 계획서

> **기반 문서**: `텍스트그리드_약화_실행_계획.md` (적용 완료), `멜_커플링_파라미터_보정_동작.md`, `약한_엔드투엔드_단계별_구현_계획서.md`, UTAU OTO Principles skill  
> **방향성**: TextGrid 정렬 의존도를 계속 낮추고, mel/F0/voicing 신호로 **자음-모음 경계**, **무음 구간**, **alias 타이밍** 정확도를 직접 높인다.

---

## 0. 현재 상태 요약

| 항목 | 상태 |
|---|---|
| alignment_weight / anchor_lock lite | 적용 완료 (1차 계획) |
| mel class/blank 신호 → ML feature | v11 feature로 포함 |
| coupled_nn_v1 + lightgbm fallback | 운영 중 |
| v2_rawmel (raw mel patch encoder) | 데이터 캐시 단계 구현됨, 학습/운영은 미완 |
| f0_voicing 계산 | `_estimate_f0_voicing_strength` → `f0_voicing_mean`, `f0_voicing_near_pre`로 feature화 |
| 무음 감지 | `db_silence_ratio`, `blank_span_confidence`, `mel_silence_sparse_ratio`로 feature화 |
| 자모음 구분 | `onset_class`, `voicing_class`(voiced/voiceless), 정적 음소 테이블 기반 분류 |

### 잔여 문제

1. **자모음 경계 판정이 정적 테이블에 의존** — mel/F0 실측 데이터가 있어도 히스토그램 기반 boundary 추정이 없다.
2. **무음 구간 offset/cutoff 보정이 feature 수준에서 멈춤** — heuristic base OTO가 이미 무음 안으로 진입한 뒤 ML delta로 보정하면 범위가 부족하다.
3. **F0 voicing이 pre/cons 결정에 직접 반영되지 않음** — feature로만 존재하고, base OTO 계산 시점에서 voicing 신호를 anchor로 사용하지 않는다.
4. **VC/VV 에일리어스 cutoff이 다음 모음을 침범** — tail patch silence 비율이 높아도 heuristic cutoff이 고정 규칙에 묶여 있다.
5. **앵커 프로필 파라미터가 정적** — 녹음 환경(볼륨/속도/발음 습관) 차이를 반영하지 못한다.

---

## 1. mel 기반 자모음 경계 직접 추정 (Voiced Onset / Vowel Nucleus Detector)

### 목적
- TextGrid 없이도 자음→모음 전환점(voiced onset)과 모음핵 시작/끝을 mel + F0에서 직접 추정한다.
- 이 추정값을 heuristic base OTO의 `pre`, `cons`, `ovl` 산출 시 **anchor 후보**로 제공한다.

### 핵심 변경

| 파일 | 내용 |
|---|---|
| `core/oto_generator.py` — `_mel_envelope` | F0 voicing + mel class 신호에서 `mel_voiced_onset_ms`, `mel_vowel_nucleus_start_ms`, `mel_vowel_nucleus_end_ms`를 추출하는 로직 추가 |
| `core/oto_generator.py` — CV/VCV 행 계산 | `mel_voiced_onset_ms`가 존재하면 base `pre` 계산 시 TextGrid 경계 대신 이 값을 우선 사용. alignment_weight < 0.65이면 mel 우선 비율을 높임 |
| `core/ja_oto_generator.py` — VCV/CVVC 행 계산 | 동일 로직을 일본어 측에도 적용 |
| `core/oto_ml_features.py` | `mel_voiced_onset_ms`, `mel_vowel_nucleus_start_ms`, `mel_vowel_nucleus_end_ms`를 피처 스키마에 추가 (FEATURE_VERSION 업) |

### 추정 알고리즘 (개요)

```
1. f0_voicing 배열에서 연속 voiced 구간 (f0_voicing >= 0.5) 추출
2. cls_voiced_formant + energy 배열에서 formant-stable 구간 추출
3. 두 신호의 교집합 중 현재 alias 의 pre 근방 (±80ms) 에서 가장 이른 onset을 mel_voiced_onset_ms로 설정
4. formant-stable 구간의 에너지 피크 ±20ms 내 에너지 안정 구간을 모음핵으로 설정
```

### 적용 조건
- mel context가 존재하고 `f0_voicing` 배열이 비어있지 않을 때만 동작.
- alignment_weight >= 0.75이고 TextGrid trust_tier == "high"이면 기존 TextGrid 경계를 유지하여 품질 회귀를 방지.

---

## 2. mel 기반 무음 구간 offset/cutoff 1차 보정 (Pre-ML Safety Clamp)

### 목적
- base OTO가 무음 구간 안에 offset/cutoff을 잡는 경우를 ML 이전 단계에서 미리 잡아낸다.
- ML delta가 보정할 수 없는 대폭 이탈을 방지한다.

### 핵심 변경

| 파일 | 내용 |
|---|---|
| `core/oto_generator.py` — `_apply_mel_safety_clamp` (신규) | `mel_offset_candidate_ms`, `mel_cutoff_candidate_ms` + `blank_span_confidence`를 사용해 base OTO의 offset/cutoff을 pre-ML 단계에서 soft-clamp |
| `core/ja_oto_generator.py` | KR과 동일 로직 적용 |
| `core/kr_oto_file_finalize.py` / `core/ja_oto_file_finalize.py` | 파일 단위 finalize 시 clamp 적용 여부 로깅 |

### 동작 규칙

```
IF blank_span_confidence >= 0.72 AND alias_type in {cv, cv_head, vcv}:
  IF base_offset < mel_offset_candidate_ms - 12:
    base_offset = mel_offset_candidate_ms - 12
  IF base_cutoff_abs > mel_cutoff_candidate_ms + 8:
    base_cutoff_abs = mel_cutoff_candidate_ms + 8

IF alias_type in {vc, vv}:
  tail_silence_ratio 기반으로 cutoff을 당기되,
  최소 cons + 20ms 이상 유지하여 너무 짧아지는 것을 방지
```

### 가드레일
- alignment_weight >= 0.78이고 mel_offset_candidate와 TextGrid 경계가 30ms 이내이면 clamp 건너뜀 (이미 정확).
- clamp 적용 시 `mel_safety_clamped` 플래그를 row에 기록 → ML feature + 로그에 반영.

---

## 3. F0 voicing을 anchor lock 시점에 반영

### 목적
- 현재 anchor lock은 profile의 정적 window로 `pre`를 조정하지만, **voicing이 실제로 시작되는 위치**를 참조하지 않는다.
- F0 voicing 신호를 anchor lock에 직접 주입하여, pre가 무음/무성 구간 안에 빠지는 것을 방지한다.

### 핵심 변경

| 파일 | 내용 |
|---|---|
| `core/timing_anchor_runtime.py` — `apply_anchor_lock` | `AnchorTimingContext`에 `voiced_onset_ms: Optional[float]` 필드 추가. `pre` 조정 시 voiced_onset_ms가 존재하면 pre가 이 값보다 앞으로 가지 않도록 floor 적용 |
| `core/anchor_lock_adapter_v2.py` | `voiced_onset_ms` 파라미터 전달 경로 추가 |
| `core/timing_anchor_profiles.py` — `AnchorTimingProfile` | `voiced_onset_weight: float = 0.4` 필드 추가 — voiced onset floor의 강도 조절 |

### 동작 요약
```
IF voiced_onset_ms is not None AND pre_candidate < voiced_onset_ms:
  pre_result = lerp(pre_candidate, voiced_onset_ms, voiced_onset_weight)
```

---

## 4. VC/VV cutoff 다음 모음 침범 방지 강화

### 목적
- VC/VV 에일리어스에서 cutoff이 다음 CV의 모음핵까지 침범하면, 우타우 렌더러에서 크로스페이드가 탁해지거나 다음 모음이 이중으로 들린다.
- 현재 `cut_to_next_onset_allow_ms` / `cut_to_next_vowel_allow_ms`가 존재하지만, mel 기반 정밀 경계가 반영되지 않음.

### 핵심 변경

| 파일 | 내용 |
|---|---|
| `core/kr_oto_bridge.py`, `core/ja_oto_bridge.py` | `mel_cutoff_candidate_ms`와 `next_mel_voiced_onset_ms`를 조합하여 VC/VV cutoff 상한을 재설정. 다음 onset의 mel voiced onset 이전 위치로 cutoff을 제한 |
| `core/timing_anchor_profiles.py` | VC/VV 프로필에 `max_cut_to_next_voiced_onset_ms` 필드 추가 (기본 -6.0, 즉 다음 voiced onset 6ms 전에 끊음) |

### 동작 규칙
```
next_voiced_onset = 다음 CV/VCV row의 mel_voiced_onset_ms
IF next_voiced_onset is not None:
  max_cut = next_voiced_onset + max_cut_to_next_voiced_onset_ms
  IF base_cutoff_abs > max_cut:
    base_cutoff_abs = max_cut
```

---

## 5. 파일-적응형 앵커 프로필 스케일링

### 목적
- 녹음 볼륨이 작거나, 발음 속도가 빠르거나, 배경 노이즈가 높은 파일에서는 정적 앵커 프로필 파라미터가 맞지 않을 수 있다.
- 파일 단위로 측정한 mel 통계를 기반으로 프로필 파라미터를 런타임에서 미세 조정한다.

### 핵심 변경

| 파일 | 내용 |
|---|---|
| `core/timing_anchor_runtime.py` — `adapt_profile_to_file` (신규) | 파일 단위 mel 통계 (평균 에너지, voiced 비율, 평균 음절 길이)로 프로필의 window/gap 파라미터를 ±20% 범위에서 스케일링 |
| `core/kr_oto_file_finalize.py`, `core/ja_oto_file_finalize.py` | 파일 시작 시 `adapt_profile_to_file`을 호출하여 해당 파일의 모든 row에 적용할 adapted profile 생성 |

### 스케일링 규칙 (개요)
```
energy_ratio = file_mean_energy / global_reference_energy
speed_ratio  = global_reference_syllable_dur / file_mean_syllable_dur

pre_window_scale  = clamp(speed_ratio, 0.80, 1.20)
cons_gap_scale    = clamp(speed_ratio, 0.82, 1.18)
cut_gap_scale     = clamp(energy_ratio * 0.3 + speed_ratio * 0.7, 0.80, 1.20)
```

---

## 6. ML feature 확장 및 모델 재학습 가이드

### 6.1 신규 피처 추가 (FEATURE_VERSION 업)

| 피처 | 유형 | 설명 |
|---|---|---|
| `mel_voiced_onset_ms` | numeric | 1항에서 산출한 voiced onset 절대 시간 |
| `mel_vowel_nucleus_start_ms` | numeric | 모음핵 시작 절대 시간 |
| `mel_vowel_nucleus_end_ms` | numeric | 모음핵 끝 절대 시간 |
| `mel_voiced_onset_to_pre_ms` | numeric | voiced onset과 base pre 사이 거리 — 양수이면 pre가 onset보다 뒤 |
| `mel_safety_clamped` | binary | 2항의 pre-ML safety clamp 적용 여부 |
| `file_mean_energy` | numeric | 파일 단위 평균 에너지 |
| `file_voiced_ratio` | numeric | 파일 단위 voiced 프레임 비율 |
| `file_mean_syllable_dur_ms` | numeric | 파일 단위 평균 음절 길이 (ms) |

### 6.2 학습 시 반영 사항
- `FEATURE_VERSION`과 `TRAIN_ROW_MATCH_VERSION`을 모두 올린다 → 캐시 무효화.
- `coupled_nn_v1` 재학습 시 신규 피처를 scalar input에 추가.
- `coupled_nn_v2_rawmel` 구현이 완료되면 onset/tail raw mel patch 안에 자동으로 voiced onset 정보가 포함되므로 별도 scalar 추가 불필요.
- aux 타깃 (`aux_vowel_start_rel`)이 mel 기반 모음핵과 겹치므로, 두 신호의 상관을 loss에서 soft penalty로 추가 장려 가능.

---

## 7. 구현 우선순위

| 순위 | 항목 | 영향 범위 | 난이도 |
|---|---|---|---|
| **P0** | 2. Pre-ML Safety Clamp | KR+JA base OTO 전체 | ★★☆ |
| **P1** | 1. Voiced Onset / Vowel Nucleus Detector | base OTO pre/cons + ML feature | ★★★ |
| **P2** | 4. VC/VV cutoff 다음 모음 침범 방지 | KR+JA bridge 행 | ★★☆ |
| **P3** | 3. F0 voicing → anchor lock 반영 | anchor lock 적용 행 | ★★☆ |
| **P4** | 5. 파일-적응형 앵커 프로필 스케일링 | anchor lock 적용 파일 | ★★☆ |
| **P5** | 6. ML feature 확장 + 재학습 | ML 모델 전체 | ★★★ |

> P0~P2는 **base OTO 품질 자체를 올려서** ML delta가 보정할 수 있는 범위를 넓히는 데 집중한다.  
> P3~P5는 **anchor lock과 ML 모델의 입력 품질을 개선**하여 보정 정밀도를 높인다.

---

## 8. 검증 계획

### 8.1 자동 테스트
- 기존 `test_auto.py` 확장: 각 항목(safety clamp, voiced onset, VC cutoff 제한)에 대해 더미 WAV + TextGrid로 파라미터 범위 검증.
- oto_validator의 blank_span/offset/cutoff 경고 카운트가 개선 전 대비 감소하는지 비교.
- `pytest tests/` 로 실행.

### 8.2 샘플 뱅크 기반 비교 (수동)
1. KR CVVC 샘플 뱅크 1~2개, JA VCV 샘플 뱅크 1~2개로 auto-oto 실행.
2. 개선 전/후 `oto.ini`를 diff하여:
   - offset이 무음 안으로 진입한 행 수 비교.
   - VC/VV cutoff이 다음 모음까지 침범한 행 수 비교.
   - pre가 voiced onset에서 40ms 이상 벗어난 행 수 비교.
3. 대표 5~10행을 우타우/OpenUtau에서 실제 재생하여 청각 품질 확인.

### 8.3 ML 재학습 후 평가
- holdout MAE 비교 (`v1` baseline vs 신규 피처 포함 재학습).
- confidence 분포 변화 확인.
- coupled 결과에서 `mel_safety_clamped=1`인 행의 delta 분포가 합리적인지 확인.

---

## 9. 주의사항

- 모든 mel 기반 보정은 **mel context가 없는 환경에서 graceful fallback** 해야 한다 (f0_voicing 배열 없음 → 기존 로직 유지).
- `FEATURE_VERSION` 업 전에 기존 캐시로 학습된 모델이 운영 중이면, 신규 빌드 전까지 feature 추가 피처가 0.0으로 채워져도 추론이 깨지지 않도록 방어 코드를 둔다.
- KR과 JA의 음소 구조가 다르므로 voiced onset 알고리즘의 하이퍼파라미터(window, threshold)는 언어별로 분리 가능한 구조로 작성한다.
- 1차 계획서의 `alignment_weight` 공식은 그대로 유지하되, 신규 mel 신호가 alignment_weight와 모순되면 mel 신호를 우선하는 방향으로 정책을 설정한다.
