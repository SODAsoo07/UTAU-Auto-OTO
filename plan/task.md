# OTO 정확도 개선 2차 구현

## P0: Pre-ML Safety Clamp (무음 구간 offset/cutoff 1차 보정)
- [x] `core/mel_safety_clamp.py` 신규 모듈 구현
- [/] KR generator 파일 단위 finalize에서 clamp 호출
- [/] JA generator 파일 단위 finalize에서 clamp 호출
- [/] clamp 플래그 로깅

## P1: Voiced Onset / Vowel Nucleus Detector
- [ ] `_mel_envelope`에서 voiced onset + 모음핵 추출 로직 추가
- [ ] KR base OTO pre/cons 계산 시 mel voiced onset 참조
- [ ] JA base OTO pre/cons 계산 시 mel voiced onset 참조
- [ ] ML feature schema에 신규 피처 추가

## P2: VC/VV cutoff 다음 모음 침범 방지
- [ ] `kr_oto_bridge.py`에서 cutoff 상한 재설정
- [ ] `ja_oto_bridge.py`에서 cutoff 상한 재설정

## P3: F0 voicing → anchor lock 반영
- [ ] `AnchorTimingContext`에 `voiced_onset_ms` 추가
- [ ] `apply_anchor_lock`에서 voiced_onset_ms floor 적용
- [ ] adapter에 전달 경로 추가

## P4: 파일-적응형 앵커 프로필 스케일링
- [ ] `adapt_profile_to_file` 함수 구현
- [ ] KR/JA finalize에서 호출

## P5: ML feature 확장 + 재학습 가이드
- [ ] FEATURE_VERSION 업 + 신규 피처 추가
- [ ] 방어 코드 (0.0 fallback)
