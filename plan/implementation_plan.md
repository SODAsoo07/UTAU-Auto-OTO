# OTO 품질 향상 구현 계획

UTAU OTO 자동 생성 품질(정확도, 연결감) 향상을 위해 6개 Phase로 나누어 구현합니다.
파라미터 순서 꼬임 방지를 최우선으로, 파일 단위 일관성(8-1)을 구조적 중심으로 진행합니다.

## Proposed Changes

---

### Phase 1: 파라미터 순서 강제 (최우선)

`팁 (파라미터 설명).txt`의 규칙에 따라 파라미터들이 절대 순서가 꼬이지 않도록 강제합니다.

UTAU 규칙: `ovl < pre ≤ consonant < |cutoff|`, 그리고 `offset ≥ 0`

#### [MODIFY] [oto_generator.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_generator.py)
- `validate_oto_params` 함수를 강화:
  - alias_type 파라미터 추가 (선택적, 기존 호출 호환 유지)
  - `ovl < pre` 강제 (현재 있음, 유지)
  - `pre ≤ consonant` 강제에 alias_type별 최소 간격 적용:
    - CV/CV_HEAD: `cons ≥ pre + 20`
    - VC: `cons ≥ pre + 10`
    - VV: `cons ≥ pre + 16`
    - 기본: `cons ≥ pre + 14`
  - `consonant < |cutoff|` 강제에 최소 간격 적용:
    - VC (파열음 받침): `|cutoff| ≥ cons + 8`
    - 기본: `|cutoff| ≥ cons + 12`
  - `ovl ≥ 0`, `pre ≥ 0`, `offset ≥ 0` (현재 있음)
  - 추가: 전체 구간이 논리적으로 유효한지 최종 검증

#### [MODIFY] [kr_oto_bridge.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_oto_bridge.py)
- `_validate_oto_params`를 메인 `validate_oto_params`를 import해서 사용하도록 변경 (코드 중복 제거)

---

### Phase 2: 파일 단위 일관성 후처리 (8-1, 구조적 핵심)

같은 WAV 파일 내 인접 에일리어스들의 파라미터 관계를 최적화합니다.

#### [NEW] [kr_oto_file_consistency.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_oto_file_consistency.py)
새 모듈에 3가지 핵심 기능 구현:

1. **`enforce_adjacent_continuity`**: 인접 에일리어스 간 cutoff↔offset 연속성
   - 이전 VC의 cutoff 끝이 다음 CV의 offset 시작과 합리적으로 연결되는지 확인
   - 과도한 겹침(>50ms)이나 간격(>80ms) 감지 시 보정
   - CV→VC, VC→CV 전환점에서 타이밍 보간

2. **`smooth_abrupt_changes`**: 파라미터 급변 감지 및 스무딩
   - 인접한 같은 타입 에일리어스들의 pre/cons/cutoff 간 변화량 계산
   - 급격한 변화(>30% 차이)를 감지해 블렌딩으로 완화
   - 단, 음소 성질 변화(파열음→공명음 등)에 의한 변화는 보존

3. **`apply_file_level_validation`**: 파일 전체 일관성 검증
   - 위 보정 후 `validate_oto_params` 재적용으로 순서 강제
   - 파일 전체 통계 수집 (변경된 라인 수, 보정 유형별 카운트)

#### [MODIFY] [kr_oto_file_finalize.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_oto_file_finalize.py)
- `run_kr_post_file_pipeline`에 일관성 단계 추가 (bridge coherence 다음에 실행)
- `KrPostFilePipelineContext`에 옵셔널 파라미터 추가

---

### Phase 3: CV 자음 분기 세분화

#### [MODIFY] [kr_oto_cv.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/kr_oto_cv.py)
- `_compute_kr_cv_timing`에서 기존 4분류를 6분류로 확장:
  - 기존: 경음(`is_tense`) / 공명음(`is_sonorant`) / 파열음(`is_plosive`) / 기타
  - 추가: **마찰음**(ㅅ,ㅆ,ㅎ,f) / **격음**(ㅋ,ㅌ,ㅍ,ㅊ)
  - 마찰음: pre를 넓게(onset 소음이 길다), overlap 약간 넓게
  - 격음: pre를 넓게(기식이 길다), overlap 약간 넓게

#### [MODIFY] [oto_generator.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_generator.py)
- `adaptive_overlap`에서 경음/격음 세분화:
  - 경음(ㄲ,ㄸ,ㅃ,ㅆ,ㅉ): `ratio -= 0.20` (VOT 짧음)
  - 격음(ㅋ,ㅌ,ㅍ,ㅊ): `ratio += 0.02` (기식 길음)

---

### Phase 4: 후처리 가드 개선

#### [MODIFY] [oto_generator.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_generator.py)
- `_stabilize_params_to_phone_activity`: alias_type에 따라 snap 방향 차등화
  - CV/CV_HEAD: 뒤쪽(자음 끝) snap 우선
  - VC/VV: 앞쪽(모음 끝) snap 우선
  - snap 거리가 30ms 초과 시 블렌딩으로 전환

---

### Phase 5: ML 리파이너 개선

#### [MODIFY] [oto_ml_features.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_ml_features.py)
- `KR_DELTA_CLIP_LIMITS`를 alias_type별로 분리:
  ```python
  KR_DELTA_CLIP_LIMITS_BY_TYPE = {
    "cv": {"delta_offset": [-160, 160], "delta_pre": [-140, 140], ...},
    "vc": {"delta_offset": [-200, 200], "delta_pre": [-160, 160], ...},
    "vv": {"delta_offset": [-220, 220], "delta_pre": [-120, 120], ...},
    # cv_head, vcv 등
  }
  ```
- `get_delta_clip_limits`에 alias_type 파라미터 추가 (기존 호출 호환 유지)

#### [MODIFY] [oto_ml_refiner.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_ml_refiner.py)
- `_clip_delta`에서 alias_type별 클리핑 범위 사용

---

### Phase 6: 음절 매핑 정확도

#### [MODIFY] [oto_generator.py](file:///c:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_generator.py)
- `_resolve_kr_mapping_conf_threshold`에서 phone tier 품질에 따른 동적 조정:
  - phone 품질 점수 < 0.4 → 임계값을 0.05 낮춤
  - phone 품질 점수 > 0.8 → 임계값을 0.03 올림

---

## Verification Plan

### Automated Tests

1. **기존 테스트 통과 확인**:
   ```bash
   cd c:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO
   python -m pytest ml\tests\ -x -q
   ```

2. **새 유닛 테스트** `ml/tests/test_oto_file_consistency.py`:
   - `validate_oto_params` 강화 테스트: 순서 꼬인 입력 → 올바른 출력 검증
   - `enforce_adjacent_continuity` 테스트: 인접 에일리어스 겹침/간격 보정 검증
   - `smooth_abrupt_changes` 테스트: 급변 감지 및 스무딩 검증
   - `adaptive_overlap` 세분화 테스트: 경음/격음/마찰음별 다른 ratio 확인
   - delta clip per alias type 테스트
   ```bash
   python -m pytest ml\tests\test_oto_file_consistency.py -v
   ```

3. **기존 관련 테스트 실행**:
   ```bash
   python -m pytest ml\tests\test_post_file_pipeline.py ml\tests\test_kr_oto_file_finalize.py -v
   ```
