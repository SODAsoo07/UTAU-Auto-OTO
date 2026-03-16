# 정렬산출물 없는 OTO 생성 구현안

## 1. 문서 목적

이 문서는 `TextGrid`, MFA 정렬 결과, 기타 외부 정렬 산출물이 없어도 `oto.ini`를 생성할 수 있는 실사용 경로를 설계한다.

핵심 목표는 다음과 같다.

- 현재의 `No-MFA (Experimental)` 경로를 실사용 가능한 수준으로 끌어올린다.
- "순수 음향만으로 모든 alias를 맞춘다"가 아니라, `파일명/리클리스트/토큰맵 + 음향 경계 추정`을 결합한 하이브리드 경로를 만든다.
- 기존 MFA 경로를 버리지 않고, fail-open 가능한 병행 구조로 유지한다.

이 문서의 결론은 명확하다.

- 가능하다.
- 그러나 권장 구현은 `순수 음향 only`가 아니라 `토큰 유도형 no-alignment`이다.
- 첫 적용 대상은 `CV`, `CVC`, 단순 `CVVC`이고, `VCV`와 복잡한 bridge는 후순위가 맞다.

## 2. 현재 코드 상태 요약

현재 코드베이스에는 이미 no-alignment 방향의 발판이 있다.

- UI에 `No-MFA (Experimental)` 선택지가 있다.
- aligner 이름을 `none`으로 정규화하는 분기가 있다.
- `autofree_v1` 스키마와 런타임이 존재한다.
- `audio_only` 모드도 존재한다.

하지만 현재 `audio_only`는 실사용 메인 경로로 쓰기에는 구조가 단순하다.

현재 한계:

1. `audio_only` row 생성은 실제 음향 분할기가 아니라, 파일을 토큰 개수만큼 균등 분할하는 수준이다.
2. alias 종류 판정은 음향에서 직접 뽑는 것이 아니라, alias 문자열이나 파일명 토큰에 강하게 의존한다.
3. 기존 강한 OTO ML feature는 `TextGrid` 기반 phone/vowel/syllable 경계를 많이 사용한다.
4. 따라서 지금 상태 그대로는 "정렬 산출물 없이 고품질 생성"보다 "실험적 fallback"에 가깝다.

즉, 지금 필요한 것은 `autofree`를 폐기하는 것이 아니라, `audio_only`의 입력 품질을 실질적으로 높이는 것이다.

## 3. 목표와 비목표

### 3.1 목표

- MFA/TextGrid 없이도 `oto.ini`를 생성한다.
- `CV`, `CVC`, `CVVC`에서 수동 보정량을 줄인다.
- 음향 경계 추정 실패 시 자동으로 보수적 fallback을 적용한다.
- 기존 MFA 경로와 결과 비교가 가능하도록 동일한 guard/validator를 재사용한다.

### 3.2 비목표

- 초기에 "파일명도 토큰도 없이" 음향만으로 alias 문자열 전체를 복원하는 것은 목표로 두지 않는다.
- 초기에 `VCV`, 복잡한 `VV`, 한국어 glide 혼합, 특수 alias 전부를 고정밀로 해결하는 것은 목표로 두지 않는다.
- MFA 경로를 제거하지 않는다.

## 4. 권장 전략

권장 전략은 아래 한 줄로 요약된다.

`토큰 순서는 파일명/리클리스트/외부 맵에서 받고, 경계 위치는 음향에서 잡고, alias 매핑은 포맷 규칙과 단조 정렬로 결정한다.`

즉, 아래 세 가지를 분리한다.

1. **무엇을 녹음했는가**
- `wav` 파일명
- reclist 순서
- 외부 token map
- 기존 alias family 규칙

2. **어디서 소리가 시작하고 끝나는가**
- 무음/발성
- 유성/무성
- 모음 핵 시작/종료
- tail

3. **그 경계를 어떤 alias에 배정할 것인가**
- `CV`, `CV_HEAD`, `VC`, `VV`, `VCV`, `BR`
- 언어별/포맷별 monotonic mapping

이 분리가 중요한 이유는, 현재 문제의 본질이 "음향에서 전부 해석 못 한다"가 아니라 "정렬 없이도 충분한 제약을 줄 수 있느냐"이기 때문이다.

## 5. 제안 아키텍처

## 5.1 Stage A: 경계 추정기

입력:

- `wav`
- 파일 단위 토큰 수 또는 alias 수
- 언어/포맷 정보

출력:

- `speech_start_ms`
- `voiced_onset_ms`
- `vowel_nucleus_start_ms`
- `vowel_nucleus_end_ms`
- `tail_ms`
- `blank_confidence`
- `voiced_confidence`
- `unvoiced_confidence`
- `boundary_confidence`

구현 원칙:

- 처음에는 대규모 신경망보다, 기존 mel/energy 기반 feature를 재사용하는 경량 모델 또는 규칙 혼합 방식이 낫다.
- 이미 코드에 있는 `mel_window_energy_mean`, `mel_window_silence_ratio`, `mel_window_voiced_ratio`, blank 관련 신뢰도 로직을 적극 활용한다.
- Stage A는 "정확한 음소 인식"이 아니라 "실용적인 경계 후보 생성"만 책임진다.

## 5.2 Stage B: 토큰 공급기

토큰 공급 우선순위:

1. 외부 token map
2. 사용자 제공 reclist/메타
3. 파일명 토큰 분해
4. 기존 `oto` seed의 alias 순서

여기서 중요한 점은, no-alignment 경로에서 가장 값비싼 오류가 "경계가 조금 틀린 것"보다 "아예 다른 alias로 매핑된 것"이라는 것이다.

따라서 토큰 순서를 외부 정보로 강하게 잡는 편이 맞다.

## 5.3 Stage C: 단조(monotonic) alias 매핑

토큰과 경계 후보를 받아 alias row를 만든다.

예시:

- `CV`, `CV_HEAD`: 주 onset과 모음 핵 기준
- `VC`: 앞모음 tail과 다음 onset 사이를 좁게 배정
- `VV`: 두 모음 핵 사이 구간을 배정
- `BR`: blank confidence가 충분히 높을 때만 생성

핵심 원칙:

- 시간축은 항상 앞으로만 간다.
- 앞 row가 뒤 row를 침범하지 않게 한다.
- `VC/VV`는 탐욕적으로 길게 잡지 않는다.
- 신뢰도 낮은 row는 생성하더라도 conservative하게 clamp한다.

## 5.4 Stage D: provisional OTO seed 생성

이 단계에서는 완성형 OTO를 한 번에 만들지 않고, 먼저 seed를 만든다.

seed에서 정하는 것:

- `offset`
- `pre`
- `cutoff_abs`
- `cons`
- `ovl`

초기 값은 규칙 기반으로 보수적으로 잡는다.

예시:

- `CV`: `pre`는 voiced onset 또는 nucleus 시작 근방
- `VC`: `cutoff`는 다음 모음이 새기기 전에 끊는다
- `VV`: `offset`은 첫 번째 모음 tail 쪽, `cutoff`는 두 번째 모음 핵 전까지

## 5.5 Stage E: ML 보정

보정 방식은 두 갈래 중 하나를 택한다.

1. 기존 legacy/coupled feature 일부를 no-alignment용으로 축약 재구성
2. `autofree_v1`를 실질적 no-alignment backend로 확장

권장안은 2번이다.

이유:

- 이미 `autofree_v1`의 스키마와 런타임이 있다.
- absolute target 기반이라 TextGrid 없는 경로와 더 잘 맞는다.
- 현재 부족한 것은 backend의 존재가 아니라 feature 품질이다.

즉, `autofree`를 새로 만드는 것이 아니라 `autofree 입력을 강화`하는 쪽이 맞다.

## 5.6 Stage F: guard, abstain, fallback

이 단계는 필수다.

없으면 no-alignment 경로는 쉽게 "그럴듯하지만 망가진 oto"를 만든다.

필수 보호 장치:

- `validate_oto_params`
- wav duration hard guard
- tail overreach clamp
- bridge 과신 방지
- low confidence row abstain
- 파일 단위 fallback

fallback 우선순위:

1. no-alignment full apply
2. no-alignment conservative apply
3. seed only
4. 기존 row 유지
5. MFA 경로 사용 가능 시 재시도

## 6. 구체 구현 항목

## 6.1 기존 파일 수정

### `core/oto_ml/autofree/data.py`

수정 목적:

- 현재의 균등 분할 `audio_only`를 실제 음향 경계 기반 분할로 교체

핵심 변경:

- `start = idx * duration / count` 방식 제거 또는 fallback 전용으로 격하
- 경계 추정 결과를 사용해 `onset_ms`, `nucleus_start_ms`, `nucleus_end_ms`, `tail_ms` 채움
- `source_mode`를 `audio_only` 하나로 두더라도 내부 `source_detail`을 추가해 구분
  - `uniform_split`
  - `boundary_guided`
  - `boundary_guided_conservative`

### `core/oto_ml/autofree/schema.py`

추가 권장 feature:

- `boundary_confidence`
- `voiced_onset_ms`
- `tail_margin_ms`
- `nucleus_span_ms`
- `token_position_ratio`
- `neighbor_boundary_gap_ms`
- `boundary_source_detail`

주의:

- 스키마 버전은 올려야 한다.
- 구버전 호환을 위해 canonicalize 단계에서 기본값을 채워야 한다.

### `core/oto_ml_autofree_runtime.py`

수정 목적:

- boundary confidence를 이용해 apply 강도를 조절

추가 내용:

- row 단위 confidence gate
- alias family별 max delta 제한
- `VC/VV`에 대한 stricter clamp
- 파일 단위 fallback reason 기록 강화

### `core/oto_ml_features.py`

수정 목적:

- no-alignment 경로에서도 재사용 가능한 mel/blank/voicing feature를 shared helper로 분리

권장:

- 현재 TextGrid 의존 feature와 no-alignment 공용 feature를 분리
- 공용 feature는 `boundary` 또는 `shared_audio_features` 쪽으로 빼는 것이 낫다

## 6.2 신규 파일 추가 권장

### `core/no_mfa_boundary.py`

역할:

- 파일 단위 음향 경계 추정 API

예상 함수:

- `predict_file_boundaries(...)`
- `build_boundary_rows_for_tokens(...)`

### `core/no_mfa_mapping.py`

역할:

- token sequence + boundary candidates -> alias row 매핑

예상 함수:

- `map_cv_rows(...)`
- `map_cvvc_rows(...)`
- `map_vcv_rows(...)`

### `core/no_mfa_seed.py`

역할:

- 매핑 결과를 provisional OTO seed로 변환

### `core/no_mfa_pipeline.py`

역할:

- 전체 no-alignment 실행 orchestration

## 6.3 CLI/학습 스크립트

추가 권장:

- `ml/scripts/autofree/build_boundary_guided_dataset.py`
- `ml/scripts/autofree/train_boundary_guided_model.py`
- `ml/scripts/autofree/evaluate_boundary_guided_model.py`

처음부터 분리 학습 CLI를 크게 벌리기보다, `autofree` 기존 CLI 옆에 붙이는 편이 유지보수상 낫다.

## 7. 추천 개발 순서

## Phase 1: CV/CVC 우선

범위:

- 한국어 `CV`, `CVC`
- 일본어 `CV`

목표:

- 무음/발성 구간
- onset
- 모음 핵
- tail

이 단계에서는 `VC/VV`를 공격적으로 다루지 않는다.

성공 기준:

- seed만으로도 수동 보정량이 줄어든다.
- offset/pre가 파일 앞 공백으로 무너지지 않는다.
- validator 에러가 증가하지 않는다.

## Phase 2: 단순 CVVC

범위:

- `VC`
- 단순 `VV`

추가 과제:

- bridge 길이 과대 추정 방지
- 다음 모음 leakage 차단

## Phase 3: VCV 및 복합 bridge

범위:

- 일본어 `VCV`
- glide 혼합
- 반복 모음 위치 혼동 감소

이 단계부터는 alias family별 세분화 모델 또는 규칙 분기가 필요할 가능성이 높다.

## 8. 학습 데이터 전략

초기 학습 데이터는 기존 수동 OTO + TextGrid가 있는 자산을 역이용하는 것이 맞다.

즉:

1. 기존 TextGrid를 사용해 경계 정답을 만든다.
2. 런타임에서는 그 모델을 TextGrid 없이 사용한다.

이 접근은 현실적이다.

이유:

- 운영 시 TextGrid가 없으면 되는 것이지, 학습 때까지 TextGrid를 금지할 이유는 없다.
- 이미 보유한 고품질 자산을 최대한 활용할 수 있다.

권장 sample weight:

- MFA/TextGrid 기반 고신뢰 학습 row: 높게
- audio-only pseudo row: 낮게
- bridge 계열 pseudo row: 더 낮게

## 9. 평가 계획

정렬 산출물 없는 경로는 듣기 전용 평가만으로 가면 위험하다.

최소 지표:

- onset MAE
- nucleus start/end MAE
- tail MAE
- target별 OTO MAE
  - offset
  - cons
  - cutoff
  - pre
  - ovl
- validator warning/error 수
- alias family별 실패율
- low-confidence fallback 비율

청감 평가 항목:

- 리듬 드리프트
- 자음 어택 손실
- bridge 끊김
- 다음 모음 누수
- 맨 앞 공백 쏠림

## 10. 예상 성능

이 수치는 현재 코드 기준 추정치다.

전제:

- 토큰 순서는 파일명/리클리스트/외부 맵으로 준다.
- 음향 경계 추정은 no-alignment로 한다.
- MFA/TextGrid는 런타임에서 사용하지 않는다.

예상 체감 성능:

- `CV`, `CVC`: 기존 MFA 경로 대비 75% ~ 90%
- 단순 `CVVC`: 65% ~ 80%
- 복잡한 `CVVC/VCV`: 50% ~ 70%

실무 해석:

- CV/CVC는 충분히 투자 가치가 있다.
- CVVC는 bridge에 대한 conservative policy가 필수다.
- VCV까지 한 번에 잡으려 하면 일정이 길어지고 실패 위험이 높다.

## 11. 채택 판단

채택 권장.

다만 조건이 있다.

1. 첫 버전은 `CV/CVC` 중심으로 시작한다.
2. `순수 음향 only`를 고집하지 않는다.
3. `token-guided + boundary-guided` 구조를 사용한다.
4. low-confidence fallback을 강하게 둔다.
5. 기존 MFA 경로는 유지한다.

## 12. 최종 제안

실행 우선순위는 아래가 맞다.

1. `autofree audio_only`의 균등 분할을 `boundary-guided` 방식으로 교체
2. no-alignment 전용 boundary 추정 모듈 추가
3. `CV/CVC` 우선 적용
4. confidence gate + conservative bridge 정책 도입
5. 이후 `CVVC`, 마지막으로 `VCV` 확대

한 줄로 정리하면,

`정렬 결과 없이 oto를 생성하는 것은 충분히 가능하며, 가장 현실적인 구현은 순수 음향 해석기가 아니라 토큰 유도형 음향 경계 추정 파이프라인이다.`

## 13. 실행 체크리스트

이 체크리스트는 실제 구현 순서에 맞춰 작성한다.

## 13.1 착수 전 체크

- 현재 `No-MFA` 경로가 실제로 어디서 분기되는지 다시 고정한다.
- `autofree audio_only`가 균등 분할이라는 점을 문서와 로그에 명시한다.
- 기준 샘플 보이스뱅크를 최소 4종 확보한다.
- 평가용 포맷 범위를 확정한다.
- 1차 목표를 `CV/CVC`로 제한한다.
- 기존 MFA 경로 회귀 기준을 정한다.

권장 샘플 세트:

- 한국어 `CV`
- 한국어 `CVC`
- 한국어 `CVVC`
- 일본어 `CV`

## 13.2 공용 기반 작업 체크

- `No-MFA` 실행 경로에서 사용할 설정 키를 확정한다.
- boundary 결과 구조체 또는 dict contract를 확정한다.
- 파일 단위 confidence와 row 단위 confidence를 분리한다.
- fallback reason 코드를 표준화한다.
- 로그에 `source_mode`, `boundary_confidence`, `fallback_reason`를 남기게 한다.
- validator, wav-duration guard, overreach clamp를 재사용 가능한 공용 함수로 정리한다.

## 13.3 Stage A 경계 추정기 체크

- 무음/발성 구분 기준값을 고정한다.
- voiced onset 후보 산출 로직을 만든다.
- nucleus start/end 후보 산출 로직을 만든다.
- tail 후보 산출 로직을 만든다.
- blank/voiced/unvoiced confidence를 계산한다.
- 저신뢰 파일에 대해 conservative 결과를 강제하는 분기를 넣는다.
- 결과를 파일 단위와 토큰 단위 둘 다 참조할 수 있게 만든다.

완료 조건:

- 최소한 `speech_start_ms`, `voiced_onset_ms`, `vowel_nucleus_start_ms`, `tail_ms`가 채워진다.
- 무성 시작 음절과 저음량 샘플에서 맨 앞 공백 쏠림이 줄어든다.
- 실패 시 예외 대신 fallback 가능한 기본값을 반환한다.

## 13.4 token-guided 매핑 체크

- 외부 token map 우선순위를 구현한다.
- 파일명 토큰 분해 fallback을 유지한다.
- 기존 alias 순서를 seed 정보로 사용할지 결정한다.
- 포맷별 token count와 row count 불일치 정책을 정한다.
- monotonic mapping 규칙을 `CV/CVC`부터 구현한다.
- `BR` 생성 조건을 blank confidence 중심으로 재정의한다.

완료 조건:

- token 수와 경계 수가 맞지 않아도 런타임이 죽지 않는다.
- `CV/CVC`에서 row 순서가 뒤집히지 않는다.
- 잘못된 token source일 때 confidence가 명시적으로 낮아진다.

## 13.5 provisional seed 생성 체크

- `CV`용 seed 규칙을 확정한다.
- `CVC`용 tail/coda 규칙을 확정한다.
- `VC`, `VV`는 1차 버전에서 보수적으로 비활성 또는 제한 적용한다.
- `offset`, `pre`, `cutoff_abs`, `cons`, `ovl` 산출 순서를 고정한다.
- parameter 관계 검증을 모든 row에 적용한다.

완료 조건:

- validator error 없이 seed OTO가 생성된다.
- `offset < pre < cutoff_abs` 계열 관계가 무너지지 않는다.
- tail overreach로 다음 음이 유입되는 경우가 감소한다.

## 13.6 autofree 확장 체크

- `autofree` feature schema를 올린다.
- boundary 관련 feature를 추가한다.
- 구버전 번들 로딩 시 graceful fallback 되게 한다.
- `audio_only` 내부 detail을 `uniform_split`과 `boundary_guided`로 구분한다.
- confidence gate에 boundary confidence를 반영한다.
- alias family별 delta clamp를 강화한다.

완료 조건:

- 새 스키마에서 학습/추론/번들 로드가 모두 동작한다.
- no-alignment row에 대해 예전보다 과도한 이동량이 줄어든다.
- `VC/VV`가 저신뢰일 때 자동으로 보수화된다.

## 13.7 학습/평가 체크

- boundary-guided dataset build 스크립트를 만든다.
- 학습용 CSV에 boundary target과 confidence 관련 컬럼을 넣는다.
- holdout split 기준을 voicebank 단위로 유지한다.
- MFA 기반 정답과 no-alignment 입력을 섞어 학습한다.
- 평가 리포트에 target별 MAE와 fallback 비율을 넣는다.
- alias family별 세부 지표를 저장한다.

완료 조건:

- 최소 1회 end-to-end 학습이 재현 가능하다.
- 평가 리포트만 보고도 regress 여부를 판단할 수 있다.
- `CV/CVC` 기준으로 baseline 대비 개선 방향이 숫자로 보인다.

## 13.8 UI/운영 체크

- UI에 현재 no-MFA 품질 수준을 과장 없이 표시한다.
- 로그에 uniform split 사용 여부를 노출한다.
- low-confidence apply 여부를 사용자 로그에서 확인 가능하게 한다.
- 실패 시 MFA 재시도 가능 여부를 명시한다.
- 설정 파일에 no-MFA 관련 키를 저장한다.

완료 조건:

- 사용자가 "왜 품질이 낮았는지" 로그만 보고 대략 이해할 수 있다.
- no-MFA 사용 여부가 세션 간 일관되게 유지된다.

## 13.9 1차 릴리스 체크

- 한국어 `CV/CVC` 최소 샘플에서 수동 보정량이 줄었는지 확인한다.
- validator error 증가가 없는지 확인한다.
- 기존 MFA 경로 회귀가 없는지 확인한다.
- no-MFA 실패 시 fallback이 안전하게 작동하는지 확인한다.
- 문서에 지원 범위와 미지원 범위를 명확히 적는다.

1차 릴리스에서 반드시 문서화할 항목:

- 지원 포맷
- 미지원 또는 품질 낮은 포맷
- 권장 입력 조건
- 실패 시 권장 대응

## 14. PR 단위 세부 리스트

PR은 작게 끊어야 한다. 한 PR에서 경계 추정, 매핑, 학습, UI를 동시에 건드리면 리뷰와 회귀 추적이 어려워진다.

## PR-01: no-MFA 실행 경로 정리

목표:

- 기존 `No-MFA` 분기와 설정 키를 명확히 정리한다.

대상 파일:

- `ui/layout_mixin.py`
- `core/pipeline_status.py`
- `core/post_file_pipeline.py`
- 필요 시 `config.json` 관련 로드/저장 경로

작업 항목:

- no-MFA 관련 설정명 정리
- aligner=`none` 분기 점검
- no-MFA 사용 시 로그 메시지 정리
- uniform split은 아직 임시 구현임을 로그에 표시

완료 조건:

- no-MFA 경로가 어디서 활성화되는지 코드상 분명하다.
- 사용자가 no-MFA 활성화 사실을 UI와 로그에서 확인할 수 있다.

테스트:

- aligner=`MFA`
- aligner=`No-MFA`
- config 저장/재실행

## PR-02: boundary 결과 contract 추가

목표:

- 경계 추정 결과 구조를 공용 contract로 고정한다.

대상 파일:

- 신규 `core/no_mfa_boundary.py`
- 필요 시 `core/oto_ml/autofree/schema.py`

작업 항목:

- boundary result field 정의
- 기본값/실패값 정의
- confidence field 정의
- source detail 정의

완료 조건:

- 이후 PR이 동일한 결과 구조를 참조할 수 있다.
- 실패한 경우에도 contract가 깨지지 않는다.

테스트:

- 정상 wav
- 짧은 wav
- 무음 wav
- 읽기 실패 wav

## PR-03: heuristic boundary baseline 구현

목표:

- 균등 분할을 대체할 최소 실용 경계 추정기를 만든다.

대상 파일:

- 신규 `core/no_mfa_boundary.py`
- 필요 시 `core/silence_profile_runtime.py`
- 필요 시 `core/oto_ml_features.py` 공용 함수 분리

작업 항목:

- energy/silence 기반 speech start 추정
- voiced onset 추정
- nucleus 구간 추정
- tail 추정
- confidence 계산

완료 조건:

- uniform split보다 명백히 나은 onset 후보를 만든다.
- 극단 케이스에서 죽지 않는다.

테스트:

- 무성 시작 자음
- 저음량 샘플
- 짧은 CV
- 긴 CVC

## PR-04: `autofree audio_only`를 boundary-guided로 교체

목표:

- 현재 `audio_only` row 생성의 핵심 한계를 제거한다.

대상 파일:

- `core/oto_ml/autofree/data.py`
- `core/oto_ml/autofree/schema.py`

작업 항목:

- 균등 분할 로직을 fallback으로 격하
- boundary-guided row 생성 추가
- `source_detail` 또는 동등 필드 추가
- 새 feature 채움

완료 조건:

- no-MFA row가 더 이상 기본적으로 균등 분할되지 않는다.
- fallback인 경우만 uniform split이 사용된다.

테스트:

- token map 있음
- token map 없음
- token count와 row count 불일치

## PR-05: seed 생성 및 guard 강화

목표:

- 경계 후보를 실제 OTO seed로 안전하게 바꾼다.

대상 파일:

- 신규 `core/no_mfa_seed.py`
- `core/oto_ml_autofree_runtime.py`
- 필요 시 `core/oto_generator.py`
- 필요 시 `core/ja_oto_generator.py`

작업 항목:

- `CV/CVC` seed 규칙 추가
- parameter relation guard 추가
- cutoff overreach clamp 추가
- low confidence conservative apply 추가

완료 조건:

- 생성된 seed가 validator 기준에서 크게 무너지지 않는다.
- 맨 앞 공백 쏠림과 다음 모음 누수가 줄어든다.

테스트:

- CV
- CVC
- 무성 시작
- 저신뢰 파일

## PR-06: token-guided monotonic mapping 추가

목표:

- 경계와 토큰을 실제 alias row로 연결한다.

대상 파일:

- 신규 `core/no_mfa_mapping.py`
- `core/oto_ml/autofree/data.py`

작업 항목:

- token priority 구현
- `CV/CVC` monotonic mapping 구현
- `BR` 생성 규칙 구현
- mismatch 정책 구현

완료 조건:

- `CV/CVC` row 순서가 안정적이다.
- token mismatch가 나도 런타임이 죽지 않는다.

테스트:

- 정상 token sequence
- 누락 token
- 중복 token
- 과다 token

## PR-07: autofree 스키마/학습 경로 확장

목표:

- no-alignment 강화 feature로 학습 가능한 상태를 만든다.

대상 파일:

- `core/oto_ml/autofree/schema.py`
- `core/oto_ml/autofree/model.py`
- `ml/scripts/autofree/build_dataset.py`
- 신규 평가/학습 스크립트 필요 시 추가

작업 항목:

- feature schema version 상향
- 학습 CSV 컬럼 확장
- meta/eval summary에 no-alignment 관련 지표 추가
- 구버전 호환 처리

완료 조건:

- 새 schema로 학습/추론이 모두 동작한다.
- eval summary에서 regress를 파악할 수 있다.

테스트:

- dataset build
- train
- bundle load
- inference

## PR-08: 런타임 fallback 및 진단 강화

목표:

- no-MFA 실패 시 안전하게 물러나고 원인을 남긴다.

대상 파일:

- `core/oto_ml_autofree_runtime.py`
- `core/post_file_pipeline.py`
- 필요 시 `core/log_events.py`

작업 항목:

- row 단위 abstain reason
- 파일 단위 fallback reason
- confidence 로그
- route별 attempted/fallback report 정리

완료 조건:

- 실패 원인이 로그에 남는다.
- fallback이 조용히 품질을 망치지 않는다.

테스트:

- boundary 실패
- wav 읽기 실패
- token 부족
- low confidence 강제

## PR-09: UI/문서/운영 메시지 정리

목표:

- 사용자가 기능 상태를 오해하지 않게 한다.

대상 파일:

- `ui/layout_mixin.py`
- 필요 시 `README.md`
- 필요 시 추가 계획 문서

작업 항목:

- 지원 범위 문구 수정
- no-MFA 경고/권장 사용 범위 표기
- 로그 문구 정리

완료 조건:

- 기능 상태가 "실험적"인지 "권장"인지 혼동되지 않는다.
- 지원 포맷과 미지원 포맷이 명시된다.

테스트:

- UI 표시
- 로그 표시
- 설정 저장 후 재실행

## PR-10: `CVVC` 보수 확장

목표:

- bridge 계열을 제한적으로 지원한다.

대상 파일:

- `core/no_mfa_mapping.py`
- `core/no_mfa_seed.py`
- `core/oto_ml_autofree_runtime.py`

작업 항목:

- `VC` 보수 규칙 추가
- `VV` 보수 규칙 추가
- bridge low-confidence clamp 추가
- 다음 모음 leakage 억제

완료 조건:

- `CVVC`에서 bridge가 과도하게 길어지지 않는다.
- validator warning이 급증하지 않는다.

테스트:

- 한국어 `CVVC`
- 일본어 `CVVC`
- bridge-heavy 샘플

## PR-11: `VCV` 실험 확장

목표:

- VCV를 별도 실험 단계로 올린다.

대상 파일:

- `core/no_mfa_mapping.py`
- `core/no_mfa_seed.py`
- 필요 시 `core/ja_*` 계열 보조 로직

작업 항목:

- `-CV`와 medial `VCV` 분리 규칙
- repeated mora 위치 혼동 완화
- glide/contracted kana 예외 처리

완료 조건:

- `VCV`가 최소한 "완전히 틀린 매핑"보다 "조금 덜 정확한 매핑" 수준으로 들어온다.
- 미지원 케이스는 과감히 abstain 또는 fallback한다.

테스트:

- head alias
- 반복 모라
- 축약음
- voiced consonant 연속

## PR-12: 회귀 평가 자동화

목표:

- 앞으로 no-MFA 품질을 반복 측정할 수 있게 한다.

대상 파일:

- `scripts/run_kr_regression_suite.py`
- 신규 no-MFA 평가 스크립트
- 필요 시 batch 실행 yaml

작업 항목:

- no-MFA 전용 regression 케이스 추가
- target별 MAE 저장
- validator warning/error 수집
- 청감 검토용 로그/출력 정리

완료 조건:

- PR마다 최소 회귀 검증이 가능하다.
- 숫자와 로그만으로 대략적 품질 비교가 가능하다.

테스트:

- CV
- CVC
- CVVC
- 실패 fallback 케이스

## 15. 추천 병합 순서

병합 순서는 아래가 가장 안전하다.

1. PR-01
2. PR-02
3. PR-03
4. PR-04
5. PR-05
6. PR-06
7. PR-07
8. PR-08
9. PR-09
10. PR-10
11. PR-11
12. PR-12

실제로 1차 사용자 가치가 생기는 시점은 `PR-05` 또는 `PR-06` 이후다.

즉, 1차 목표는 아래 수준으로 잡는 것이 맞다.

- `CV/CVC`에서 no-MFA로도 일단 들어볼 만한 `oto.ini`가 생성된다.
- 실패 시 기존 행 유지 또는 보수 fallback이 된다.
- MFA 경로는 그대로 살아 있다.
