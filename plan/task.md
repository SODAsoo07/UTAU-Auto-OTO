# OTO 품질 향상: 파라미터 강제 순서, 파일 일관성, 타이밍 세분화

## Phase 1: 파라미터 순서 강제 로직 (최우선)
- [x] `validate_oto_params` 강화 — alias type별 최소 간격, 순서 제약 강제
- [x] `kr_oto_bridge._validate_oto_params` 동기화

## Phase 2: 파일 단위 일관성 후처리 (8-1, 구조적 핵심)
- [x] 새 모듈 `core/kr_oto_file_consistency.py` 생성
  - [x] 인접 에일리어스 간 cutoff↔offset 연속성 보정
  - [x] 같은 WAV 내 파라미터 급변 감지 및 스무딩
  - [x] 파일 전체 일관성 검증 + 순서 강제 재적용
- [x] `run_kr_post_file_pipeline`에 일관성 단계 통합

## Phase 3: CV 자음 분기 세분화 (영역 3)
- [x] `kr_oto_cv._compute_kr_cv_timing` — 마찰음(ㅅ,ㅆ,ㅎ) 전용 분기 추가
- [x] `adaptive_overlap` — 경음/격음 세분화
- [x] 이중모음 glide 구간 길이 직접 반영

## Phase 4: 후처리 가드 개선 (영역 4)
- [x] `_stabilize_params_to_phone_activity` — alias type별 snap 방향 차등
- [x] `_apply_soft_mel_offset_cutoff_guard` — onset detection 기반 강화

## Phase 5: ML 리파이너 개선 (영역 5)
- [x] `KR_DELTA_CLIP_LIMITS` alias type별 분리

## Phase 6: 음절 매핑 정확도 (영역 2)
- [x] 동적 매핑 신뢰도 임계값
- [x] 매핑 점프 순서 일관성 검증

## 검증
- [/] 기존 테스트 통과 확인
- [x] 새 유닛 테스트 작성 (validate, file consistency)
