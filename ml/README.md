# OTO ML

`Auto_OTO`의 OTO 자동 생성 후 보정용 ML 자산과 학습 보조 문서를 둡니다.

## 현재 구성
- `schemas/`: feature schema, model meta schema
- `configs/`: 학습/수집 관련 설정
- `scripts/`: 데이터 수집, 학습, 평가, 번들 export/install CLI
- `tests/`: ML 관련 테스트

## 현재 런타임 정책
- OTO 자동 생성 보정 런타임은 `LightGBM`만 사용합니다.
- 과거의 `PyTorch` OTO 보정 파이프라인은 제거되었습니다.
- 대용량 학습 산출물은 기본적으로 `../ml_workspace` 아래를 사용합니다.

## 지원 형식
- 한국어: `CV`, `CVC`, `CVVC`, `general`
- 일본어: `CV`, `VCV`, `CVVC`, `general`

## 현재 표준 흐름
1. `stage_training_sources.py`
2. `prepare_staged_auto_pairs.py`
3. `collect_oto_ml_training_data.py`
4. `train_oto_ml_model.py`
5. `evaluate_oto_ml_model.py`
6. `export_oto_ml_bundle.py`
7. `install_oto_ml_bundle.py`

## 참고
- 재학습 절차: `ml/학습_재빌드_절차.md`
- 런타임 번들 위치: `assets/models/oto_ml`, `models_installed/oto_ml`

## 추가 문서
- 오매핑 데이터 수집 가이드: `C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml\오매핑_데이터_수집_가이드.md`
- 모델 학습 명령어 가이드: `C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\ml\모델_학습_명령어_가이드.md`

`?? ???? ????.

## 권장 분리 정책
- `CV`: 한국어/일본어 모두 공용 모델 유지
- `korean/cvc`: `cv`, `bridge` 분리 권장
- `korean/cvvc`: `cv`, `bridge` 분리 권장
- `japanese/cvvc`: `cv`, `bridge` 분리 권장
- `vcv`: 현재 공용 유지

- family 분리 기본 정책은 `core/oto_ml_policy.py`의 `recommended_alias_family_splits()`에도 반영되어 있습니다.
