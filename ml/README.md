# OTO ML 작업 폴더

`Auto_OTO`의 OTO 수치 보정 AI 관련 코드와 설정을 관리하는 폴더다.

구성:
- `schemas/`: feature schema, model meta schema
- `configs/`: 학습 기본 설정, 학습 데이터 루트 목록
- `scripts/`: 데이터셋 생성, 재귀 수집, 학습, 평가 CLI
- `tests/`: feature/dataset/runtime 회귀 테스트

현재 원칙:
- 1차 모델은 `LightGBM` 기반 delta 회귀를 사용한다.
- 런타임 구조는 나중에 `PyTorch` backend를 추가할 수 있게 분리되어 있다.
- 실제 대용량 데이터셋, 실험 로그, 임시 산출물은 저장소 바깥 `../ml_workspace`에 둔다.

일본어 형식 규칙:
- `VC` 또는 `VV` 연결이 있으면 `CVVC`
- `VCV` alias가 있으면 `VCV`
- 그 외는 `CV`
- 기존 `CVC` 표기는 일본어에서는 내부적으로 `CVVC` 또는 `CV`로 흡수한다.

에일리어스 매칭 규칙:
- 학습 데이터셋 생성 시 alias 끝의 음계 접미사(`_C4`, `_D4`, `-A4`, ` F4` 등)는 제거하고 비교한다.
- 음계 접미사 외에도, alias 코어의 `alias_type`을 유지한 채 뒤에 붙은 비음운 suffix(음색/스타일 문자, 한자, 한글, 가나, 영문 꼬리 등)는 반복적으로 제거한다.
- 단, `a か` 같은 실제 `VCV`/`VC` 구조를 깨뜨리는 제거는 허용하지 않는다.
- 매칭 키는 `wav_norm + alias_norm + occurrence_index`를 사용한다.

권장 작업 순서:
1. `build_oto_ml_dataset.py`로 자동 OTO / 수동 OTO / TextGrid / WAV를 묶어 CSV 생성
2. 여러 루트를 한꺼번에 훑으려면 `collect_oto_ml_training_data.py`로 후보 manifest와 형식별 CSV를 자동 생성
3. `train_oto_ml_model.py`로 언어/형식별 모델 학습
4. `evaluate_oto_ml_model.py`로 baseline 대비 보정 성능 비교
5. 충분히 검증된 모델만 `assets/models/oto_ml/...`에 배치
