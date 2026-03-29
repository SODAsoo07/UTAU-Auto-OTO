# OTO ML / 개발현황

`Auto_OTO`의 OTO 자동 생성 후 보정용 ML 자산과 학습 보조 문서를 둡니다.  
기본 문서 인코딩은 `UTF-8`입니다.

## 현재 구성
- `schemas/`: feature schema, model meta schema
- `configs/`: 학습/수집 관련 설정
- `scripts/`: 데이터 수집, 학습, 평가, 번들 export/install CLI
- `tests/`: ML 관련 테스트

## 현재 런타임 정책
- OTO 자동 생성 보정 런타임은 `LightGBM` 중심입니다.
- 대용량 학습 산출물은 기본적으로 `../ml_workspace` 아래를 사용합니다.
- 기본 분리 정책: `CV`는 공용 유지, `korean/cvc`, `korean/cvvc`, `japanese/cvvc`만 `cv`/`bridge` family 분리를 권장합니다.

## 개발 현황 (2026-03-28)
- 완료: 한국어 `C+V`는 Preview 채널에서만 노출/실행되도록 제한.
- 완료: 일본어 `CVVC` 매핑/보정 로직 개선 백포트(오매핑 억제, 과오버랩 완화 방향).
- 완료: 음절 탐지/경계 보정 관련 코어 로직 백포트(한국어/일본어 공통 보정 경로).
- 완료: 접미사 파싱 강화 백포트.
  - `_` + 영문/숫자 접미사 인식 강화
  - 한자 포함 접미사 처리 강화
  - 호흡성 한자 토큰은 접미사 제거 예외 처리
- 완료: CTC/MFA 생성 파이프라인 관련 코어 변경 백포트.

## 백포트 범위 메모
- 반영 완료(주요):  
  `core/kr_mapping_select_v2.py`, `core/ja_mapping_select_v2.py`,  
  `core/ja_generator_setup.py`, `core/ja_oto_mapping.py`, `core/oto_generator.py`,  
  `core/mfa_runner.py`, `core/ctc_runner.py`, `core/oto_ml_refiner.py` 외 다수 코어 파일.
- 의도적으로 유지(로컬 변경 충돌 방지):  
  `core/oto_ml_runtime.py`, `core/oto_ml_ensemble.py`는 현재 저장소 로컬 버전 우선.

## 지원 형식
- 한국어: `CV`, `CVC`, `CVVC`, `C+V(Preview)`, `general`
- 일본어: `CV`, `VCV`, `CVVC`, `general`

## 현재 표준 흐름
1. `ml/scripts/stage_training_sources.py`
2. `ml/scripts/prepare_staged_auto_pairs.py`
3. `ml/scripts/build_oto_mel_coupled_dataset.py` 또는 `ml/scripts/build_oto_autofree_dataset.py`
4. `ml/scripts/train_oto_lightgbm_model.py` 또는 `ml/scripts/train_oto_mel_coupled_model.py`
5. `ml/scripts/evaluate_oto_mel_coupled.py`
6. `ml/scripts/export_oto_ml_bundle.py`
7. `ml/scripts/install_oto_ml_bundle.py`

## 참고 문서
- [모델 학습 명령어 가이드](C:\Users\oyh57\Documents\GitHub\UTAU-Auto-OTO\ml\모델_학습_명령어_가이드.md)
- [오매핑 데이터 수집 가이드](C:\Users\oyh57\Documents\GitHub\UTAU-Auto-OTO\ml\오매핑_데이터_수집_가이드.md)
- [셀렉터 운용 정책/AB 테스트](C:\Users\oyh57\Documents\GitHub\UTAU-Auto-OTO\ml\셀렉터_운용_정책과_AB_테스트_절차.md)

## 권장 분리 정책
- `CV`: 한국어/일본어 모두 공용 모델 유지
- `korean/cvc`: `cv`, `bridge` 분리 권장
- `korean/cvvc`: `cv`, `bridge` 분리 권장
- `japanese/cvvc`: `cv`, `bridge` 분리 권장
- `vcv`: 현재 공용 유지

family 분리 기본 정책은 `core/oto_ml_policy.py`의 `recommended_alias_family_splits()`에도 반영되어 있습니다.
