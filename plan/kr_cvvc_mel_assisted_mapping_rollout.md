# 한국어 CVVC 초기 매핑 품질 개선 상세 구현안 (Mel-Assisted v1)

작성일: 2026-03-11
대상: 한국어 CVVC 우선

## 1. 목표
- CVVC에서 "초기 매핑" 자체의 오정렬을 줄인다.
- 기존 ML/커플링 보정은 유지하되, 보정 전에 매핑 품질을 끌어올린다.
- 저신뢰 구간에서 mel 신호(유성/무성/무음/숨소리)를 직접 매핑 선택에 반영한다.

## 2. 적용 범위 (1차)
- 파일 포맷: `cvvc`
- 매핑 타입: `cv`, `cv_head`, `vcv`
- 동작 조건: 저신뢰 행 또는 blank high 구간에만 국소적으로 적용

## 3. 단계별 진행 순서

### 단계 1. CVVC 저신뢰 구간 mel-guided 국소 재선택 (완료)
상태: 완료

핵심 변경:
1. `select_kr_general_cv_index`/`select_kr_vcv_index` 내부에 mel-guided 재선택 단계 추가
2. 기존 순서:
   - forced/planned index
   - base resolver
   - vowel fix
   - order clamp
   - blank guard
3. 추가 순서:
   - 위 절차 이후, `cvvc` + 저신뢰 조건에서만 mel 점수 재랭킹 수행
4. 점수 구성:
   - text score(`cv_match`)
   - mel bonus(`voiced_formant`, `unvoiced_diffuse`)
   - mel penalty(`blank_conf`, `silence_sparse`, `breath_like`)
   - jump penalty(기대 인덱스와 거리)
5. 보수적 적용:
   - 개선 이득이 충분할 때만 index 교체
   - 극단 blank 후보는 교체 금지

추가 데이터 연결:
- 음절 단위 annotation에 아래 mel class confidence 저장
  - `mel_voiced_formant_conf`
  - `mel_silence_sparse_conf`
  - `mel_unvoiced_diffuse_conf`
  - `mel_breath_like_conf`

적용 파일:
- `core/kr_mapping_select_v2.py`
- `core/oto_generator.py`
- `ml/tests/test_kr_mapping_select_v2.py`

검증:
- `python -m unittest ml.tests.test_kr_mapping_select_v2 -v`

---

### 단계 2. CVVC reclist 스타일별 토큰 규칙 분리 (진행 중)
상태: 1차 완료 (스타일 분기 + 말차식 구분자 정규화)

목표:
- 말차식 CVVC 등 스타일별 파일명 토큰화/슬롯 규칙을 분리해 초기 후보 자체를 안정화

구현 포인트:
1. 스타일 감지 키(패턴 기반)
2. 스타일별 split 규칙/예외 규칙
3. 기존 tokenizer 대비 A/B 로그

1차 반영 내용:
1. 파일명 스타일 감지 추가(`matcha_cvvc`/`apostrophe_cvvc`/`generic`)
2. 말차식 구분자(`_'`, `'_`, `-'` 변형) 정규화 후 토큰 분리
3. typographic apostrophe(’/`) 입력 허용
4. 기존 coda tail marker 병합(`lR`, `ngH`) 회귀 유지

검증 지표:
- tokenization 실패율
- occurrence map 누락율

---

### 단계 3. 파일 단위 모노토닉 DP 매핑 (진행 중)
상태: 1차 완료 (mel score 포함 DP 플랜 적용)

목표:
- 행 단위 greedy를 보완해 파일 전체 경로 최적화

구현 포인트:
1. 상태: row i -> syllable j
2. 전이 제약: 역행 금지, 점프 페널티
3. 점수: text + mel + blank penalty + type compatibility
4. 저신뢰 행만 DP 재평가(전체 비용 억제)

1차 반영 내용:
1. `build_kr_cv_anchor_plan`에 mel score 가중 반영(옵션)
2. `cvvc` + 저신뢰 징후(텍스트그리드 낮음/blank 높음/phone quality 낮음)일 때 mel plan 활성화
3. 기본 DP 플랜은 동일, mel 신호가 있을 때만 점수 보정

검증 지표:
- 음절 오매핑률
- CVVC 행별 index jump 빈도

---

### 단계 4. 신뢰도 라우팅 강화 (다음)
상태: 1차 완료

목표:
- 저신뢰 파일에서 공격적 보정 억제

구현 포인트:
1. 파일 신뢰도/행 신뢰도 임계치 분리
2. `cvvc` 전용 정책(강한 abstain, 제한적 remap)
3. runtime report에 매핑 단계 근거 필드 추가

1차 반영 내용:
1. `resolve_runtime_mapping_policy`에 파일/행 신뢰도 임계치 분리(`file_conf_floor`, `row_conf_floor`, `row_margin_floor`) 적용
2. `cvvc` strict 모드에서 row floor 상향(+conf,+margin) 및 low_conf reason 추적
3. row abstain 로직에 `row_confidence`, `blank_confidence` 추가
4. `cvvc` + blank_conf_mean 높을 때 row blank floor 적용 (`UTOA_KR_CVVC_ROW_BLANK_FLOOR`, 기본 0.68)
5. runtime report에 `mapping` 요약 필드 추가(신뢰도/plan/mel 관련 근거 기록)

---

### 단계 5. 배치 평가/릴리즈 (다음)
상태: 예정

필수 리포트:
- format별(`cvvc`,`cvc`,`vcv`,`cv`) 오매핑률
- `offset/cutoff` MAE
- blank 탐지 재현율
- OpenUtau 렌더 오류(`cutoff before offset`) 0건 여부

## 4. 운영 원칙
- 고신뢰 구간은 기존 결정 유지(과보정 방지)
- 저신뢰 구간만 mel-assisted 재선택 적용
- 모든 변경은 row-level 안전 제약 + wav-duration safety를 최종 통과해야 함

## 5. 지금 바로 실행 가능한 검증 명령
```powershell
python -m unittest ml.tests.test_kr_mapping_select_v2 -v
python -m unittest ml.tests.test_kr_oto_file_finalize -v
```

## 6. 다음 작업 시작점
- 단계 2(CVVC 스타일별 tokenizer 분리)부터 진행
- 대상 우선순위: 말차식 CVVC reclist 패턴
