# 한국어 mel+oto 통합 보정기(하이브리드) 구현안 v1

## 1) 범위/원칙
- 1차 적용 언어: 한국어.
- 런타임 전략: `ML 메인 + 실패 시 fallback`.
- 보정 순서: `coupled_nn_v1 -> lightgbm -> base`.
- `torch` 사용은 선택적: 미설치/모델 누락/추론 실패 시 즉시 하향.
- 일본어는 안정화 핫픽스(`JaPostprocessContext.file_format`)만 포함.

## 2) 사용자 이슈 선반영
- `JaPostprocessContext.file_format` 누락 방어:
  - `post_ctx.file_format` 직접 접근을 `getattr(post_ctx, "file_format", "")`로 변경.
  - 컨텍스트 생성 시 `file_format` 누락이면 1회 경고 후 기본값으로 진행.
- UTF-8 런타임 정규화:
  - 공통 부트스트랩(`core/runtime_encoding.py`)으로 통합.
  - `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, stdio reconfigure, Windows codepage 65001 설정 시도.
  - 배치/학습/평가 스크립트 진입점에서 동일 함수 사용.

## 3) 스펙트럼 신호 계층
- mel 프레임 클래스:
  - `voiced_formant`: 기음/F2/F3 강한 구간
  - `silence_sparse`: 주파수 성분 희박
  - `unvoiced_diffuse`: 성분은 있으나 흐림(치찰/파열 후보)
  - `breath_like`: 저에너지/고역 비중
- row feature 확장(v7):
  - 클래스 비율, `blank_span_confidence`, `mel_offset_candidate_ms`, `mel_cutoff_candidate_ms`,
  - onset/tail patch 요약(`onset_patch_*`, `tail_patch_*`) 추가.

## 4) 매핑 결합(한국어)
- CV/CV_HEAD/VCV 점수에 blank penalty 반영.
- 저신뢰/blank 고구간에서 jump 허용 폭 축소 및 모노토닉 제약 강화.
- blank guard로 공백 구간 forward 매핑을 되돌려 오매핑 감소.

## 5) 결합 보정 모델
- 데이터셋: 기존 row feature + 스펙트럼 feature + mel patch 결합.
- 타깃: `delta_offset/delta_cons/delta_cutoff/delta_pre/delta_ovl` 5개 동시 예측.
- 학습:
  - 멀티헤드 NN(`coupled_nn_v1`)
  - 가중 Huber + 제약 페널티 + 경계 정렬 손실.
- 제약 solver(후처리):
  - `offset>=0`, `ovl<=pre`, `cons>=pre+margin`, `|cutoff|>=cons+margin`.

## 6) 런타임/리포트
- 리포트 확장:
  - `route`, `model_confidence`, `fallback_reason`, `blank_confidence`, `constraint_adjust_count`.
- coupled 저신뢰(`UTOA_ML_COUPLED_MIN_CONF`) 또는 coupled 오류 시 lightgbm 즉시 하향.
- coupled 로드 실패 시에도 lightgbm load fallback 경로를 우선 사용.

## 7) CLI/API 변경
- 신규 CLI:
  - `ml/scripts/build_oto_mel_coupled_dataset.py`
  - `ml/scripts/train_oto_mel_coupled_model.py`
  - `ml/scripts/evaluate_oto_mel_coupled.py`
- 번들 메타:
  - `backend: coupled_nn_v1`
  - `feature_version`, `mel_patch_spec`, `min_confidence`, `fallback_order` 포함.
- 런타임 옵션:
  - `UTOA_ML_COUPLED_ENABLE`
  - `UTOA_ML_COUPLED_MIN_CONF`
  - `UTOA_ML_COUPLED_DEVICE`
  - `UTOA_ML_COUPLED_STRICT_CONSTRAINT`

## 8) CPU 실행 정책
- 추론:
  - `UTOA_ML_COUPLED_DEVICE=cpu`로 강제 가능.
  - GPU 없어도 동작(속도 저하는 허용).
- 학습:
  - CPU 학습 가능(학습 시간 증가).
  - 사용자 환경에서 torch 미지원 시 lightgbm 경로를 유지.

## 9) OTO 기본 규칙 반영 근거
- 참고 문서:
  - `oto 설정의 기본.txt`
  - `팁 (파라미터 설명).txt`
- 반영 원칙:
  - 선행/오버랩/고정자음/컷오프 관계를 위배하지 않도록 제약 기반 후처리 적용.
  - 공백/호흡/무성 구간을 offset/cutoff 후보에 반영해 컷 위치 안정화.

## 10) 검증 체크리스트
- 단위:
  - file_format 누락 방어
  - blank confidence/blank guard
  - coupled confidence 기반 fallback
  - constraint solver
- 통합:
  - torch 미설치/모델 누락/추론 예외에서 즉시 하향
  - 리포트 필드 확장값 기록 확인
- 회귀:
  - 음수 offset, 파라미터 순서 위반, 행 누락 재발 없음
