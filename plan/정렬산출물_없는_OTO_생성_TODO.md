# 정렬산출물 없는 OTO 생성 TODO

> 기반 문서: `plan/정렬산출물_없는_OTO_생성_구현안.md`
> 1차 목표: `CV`, `CVC`에서 `No-MFA`로도 실사용 가능한 `oto.ini` 생성
> 비목표: 1차에서 `VCV` 완성, 순수 음향 only alias 복원

## 작업 원칙

- [ ] 1차 범위를 `CV/CVC`로 고정한다.
- [ ] 기존 MFA 경로 회귀를 만들지 않는다.
- [ ] `uniform_split`은 fallback으로만 남기고 메인 경로에서 내린다.
- [ ] low-confidence 구간은 공격적으로 보정하지 않는다.
- [ ] 실패 시 seed only 또는 기존 row 유지로 안전하게 빠진다.

## P0. 착수 전 고정값 정리

- [ ] no-MFA 설정 키 이름을 확정한다.
- [ ] no-MFA 로그 메시지 형식을 확정한다.
- [ ] 평가용 샘플 보이스뱅크 4종을 고정한다.
- [ ] 1차 지원 포맷을 문서와 코드 주석에 명시한다.
- [ ] 기존 `audio_only`가 균등 분할이라는 사실을 로그에 드러낸다.

완료 기준:

- 구현 중 범위가 흔들리지 않는다.
- no-MFA 경로의 현재 한계가 사용자 로그에서 보인다.

## P1. 실행 경로 정리

대상:

- `ui/layout_mixin.py`
- `core/pipeline_status.py`
- `core/post_file_pipeline.py`

TODO:

- [ ] aligner=`none` 분기 위치를 다시 확인하고 정리한다.
- [ ] no-MFA 활성화 시 route/log/report 필드를 정리한다.
- [ ] no-MFA 사용 시 경고 문구를 현재 상태에 맞게 조정한다.
- [ ] config 저장/복원 시 no-MFA 상태가 일관되게 유지되게 한다.

완료 기준:

- no-MFA 경로가 어디서 켜지고 꺼지는지 명확하다.
- 실행 로그만 봐도 MFA/no-MFA 구분이 된다.

## P2. boundary contract 추가

대상:

- 신규 `core/no_mfa_boundary.py`

TODO:

- [ ] boundary result 필드 목록을 고정한다.
- [ ] 기본값/실패값 contract를 만든다.
- [ ] `boundary_confidence` 계산 결과를 포함한다.
- [ ] `source_detail` 또는 동등 필드를 포함한다.

추천 필드:

- [ ] `speech_start_ms`
- [ ] `voiced_onset_ms`
- [ ] `vowel_nucleus_start_ms`
- [ ] `vowel_nucleus_end_ms`
- [ ] `tail_ms`
- [ ] `blank_confidence`
- [ ] `voiced_confidence`
- [ ] `unvoiced_confidence`
- [ ] `boundary_confidence`
- [ ] `source_detail`

완료 기준:

- 이후 단계가 이 구조만 믿고 붙을 수 있다.
- 실패한 오디오도 예외 대신 정상 구조를 반환한다.

## P3. heuristic boundary baseline 구현

대상:

- 신규 `core/no_mfa_boundary.py`
- 필요 시 `core/silence_profile_runtime.py`
- 필요 시 공용 audio helper

TODO:

- [ ] speech start 추정 로직 구현
- [ ] voiced onset 추정 로직 구현
- [ ] vowel nucleus start/end 추정 로직 구현
- [ ] tail 추정 로직 구현
- [ ] blank/voiced/unvoiced confidence 계산 구현
- [ ] 짧은 파일/무음 파일 fallback 처리 구현

테스트 케이스:

- [ ] 짧은 CV
- [ ] 긴 CVC
- [ ] 무성 시작 자음
- [ ] 저음량 샘플
- [ ] 거의 무음에 가까운 샘플

완료 기준:

- 균등 분할보다 onset 후보가 명확히 낫다.
- boundary 계산 실패가 전체 파이프라인 실패로 번지지 않는다.

## P4. `autofree audio_only`를 boundary-guided로 교체

대상:

- `core/oto_ml/autofree/data.py`
- `core/oto_ml/autofree/schema.py`

TODO:

- [ ] `audio_only` 기본 로직을 boundary-guided로 바꾼다.
- [ ] `uniform_split`은 fallback 전용으로 격하한다.
- [ ] `source_detail=boundary_guided|uniform_split` 구분을 넣는다.
- [ ] boundary 관련 feature 컬럼을 스키마에 추가한다.
- [ ] 구버전 row canonicalize fallback을 넣는다.

추천 신규 feature:

- [ ] `boundary_confidence`
- [ ] `voiced_onset_ms`
- [ ] `nucleus_span_ms`
- [ ] `tail_margin_ms`
- [ ] `neighbor_boundary_gap_ms`
- [ ] `boundary_source_detail`

완료 기준:

- no-MFA row 생성의 기본값이 더 이상 균등 분할이 아니다.
- fallback 상황만 명시적으로 uniform split을 사용한다.

## P5. token-guided 매핑 추가

대상:

- 신규 `core/no_mfa_mapping.py`
- `core/oto_ml/autofree/data.py`

TODO:

- [ ] token 우선순위 규칙 구현
- [ ] 외부 token map 우선 적용
- [ ] 파일명 token fallback 유지
- [ ] `CV/CVC` monotonic mapping 구현
- [ ] token 수 불일치 시 정책 구현
- [ ] `BR` 생성 규칙 구현

정책 메모:

- [ ] row 순서는 절대 뒤집지 않는다.
- [ ] token mismatch는 예외가 아니라 confidence 하락으로 처리한다.
- [ ] `VC/VV`는 1차에서 적극 생성하지 않는다.

완료 기준:

- `CV/CVC`에서 row 순서가 안정적이다.
- token mismatch가 나도 파이프라인이 죽지 않는다.

## P6. provisional seed 생성

대상:

- 신규 `core/no_mfa_seed.py`
- 필요 시 `core/oto_ml_autofree_runtime.py`

TODO:

- [ ] `CV` seed 규칙 구현
- [ ] `CVC` seed 규칙 구현
- [ ] `offset/pre/cutoff_abs/cons/ovl` 산출 순서 고정
- [ ] seed 단계 validator 적용
- [ ] 과도한 tail overreach clamp 추가
- [ ] 저신뢰 row conservative seed 추가

완료 기준:

- validator error 없이 seed OTO가 생성된다.
- 맨 앞 공백 쏠림이 줄어든다.
- 다음 음 누수가 줄어든다.

## P7. runtime guard와 fallback 강화

대상:

- `core/oto_ml_autofree_runtime.py`
- `core/post_file_pipeline.py`

TODO:

- [ ] row 단위 abstain reason 추가
- [ ] 파일 단위 fallback reason 추가
- [ ] boundary confidence 기반 apply 강도 조절
- [ ] alias family별 clamp 강화
- [ ] no-MFA 실패 시 seed only 또는 기존 row 유지 분기 추가

완료 기준:

- 실패 원인이 로그에 남는다.
- 위험한 row는 조용히 망가지지 않고 보수 fallback 된다.

## P8. 학습 경로 확장

대상:

- `core/oto_ml/autofree/model.py`
- `ml/scripts/autofree/build_dataset.py`
- 필요 시 신규 train/eval 스크립트

TODO:

- [ ] 새 schema version 정의
- [ ] boundary-guided dataset build 경로 추가
- [ ] holdout voicebank split 유지
- [ ] eval summary에 target별 MAE 저장
- [ ] no-MFA 관련 지표 저장

학습 데이터 원칙:

- [ ] TextGrid 기반 고신뢰 row는 높은 weight
- [ ] audio-only pseudo row는 낮은 weight
- [ ] bridge 계열 pseudo row는 더 낮은 weight

완료 기준:

- 새 schema로 dataset build/train/eval이 한 번 끝까지 돈다.
- 리포트만 보고 regress 여부를 알 수 있다.

## P9. 테스트 추가

대상:

- `ml/tests/`
- 필요 시 `tests/`

TODO:

- [ ] boundary contract 테스트
- [ ] boundary-guided row 생성 테스트
- [ ] token mismatch fallback 테스트
- [ ] seed validator 테스트
- [ ] no-MFA runtime fallback 테스트
- [ ] 기존 MFA 경로 회귀 테스트

완료 기준:

- 최소 핵심 경로가 자동 테스트로 보호된다.

## P10. UI/문서 정리

대상:

- `ui/layout_mixin.py`
- `README.md`
- 관련 계획 문서

TODO:

- [ ] no-MFA 지원 범위 문구 정리
- [ ] 미지원 또는 품질 낮은 포맷 문구 정리
- [ ] uniform split fallback 사용 시 로그/문구 노출
- [ ] 권장 입력 조건 문서화

완료 기준:

- 사용자가 현재 지원 범위를 오해하지 않는다.

## P11. 1차 릴리스 게이트

- [ ] 한국어 `CV` 샘플 검증
- [ ] 한국어 `CVC` 샘플 검증
- [ ] 일본어 `CV` 샘플 검증
- [ ] validator error 증가 없음 확인
- [ ] 기존 MFA 경로 회귀 없음 확인
- [ ] no-MFA 실패 시 fallback 정상 동작 확인

릴리스 조건:

- [ ] `CV/CVC`에서 수동 보정량이 기존 no-MFA 대비 체감상 줄어든다.
- [ ] "아예 틀린 alias"보다 "조금 덜 정확한 경계" 수준으로 수렴한다.
- [ ] 위험한 경우에는 과감히 덜 적용하고 안전하게 빠진다.

## 지금 바로 시작할 순서

- [ ] P0 완료
- [ ] P1 완료
- [ ] P2 contract 초안 추가
- [ ] P3 heuristic boundary baseline 초안 구현
- [ ] P4에서 `autofree audio_only` 메인 경로 교체

## 보류 항목

- [ ] `VC/VV` 적극 지원
- [ ] `CVVC` 본격 확장
- [ ] `VCV` 실전 품질 개선
- [ ] pure audio only alias 복원

## 메모

- 구현 시작 시 첫 작업 브랜치는 `codex/no-mfa-boundary-baseline` 또는 동등한 이름으로 시작하는 편이 관리가 쉽다.
- 한 PR에서 경계 추정, 학습 스키마, UI를 동시에 건드리지 않는다.
- 1차 목표는 "완벽한 no-MFA"가 아니라 "안전한 no-MFA"다.
