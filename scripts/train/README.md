# Train Scripts

모델 학습 실행 진입 스크립트 구현 경로.

이 폴더 코드는 배포 런타임에서 직접 import하지 않는다.

실험성 전처리/베이스라인 점검 스크립트는 `tests/scripts/training/`으로 분리했다.

현재 `scripts/train/`에는 실사용 학습 엔트리포인트(`train_cvn_backend.py`)만 유지한다.
