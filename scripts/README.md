# scripts 폴더 구조

`scripts/` 루트에는 실행 진입점(보호 대상) 위주만 남기고, 수동/실험 스크립트는 카테고리 하위 폴더로 분리했다.

## 루트 (진입점/보호 대상)

- `run_oto_generation_batch.py`
- `run_kr_regression_suite.py`
- `run_coupled_experiment_matrix.py`
- `run_coupled_two_stage_tune.py`
- `train_cvn_backend.py`
- `build_installer.ps1`
- `build_portable_with_models.ps1`
- `runtime_recovery.ps1`
- `startup_diagnose.ps1`
- `startup_diagnose.bat`
- `audit_script_usage.py`
- `scripts_manifest.json`

## 하위 카테고리

- `scripts/ml/data/`
  - 데이터 분류/프로파일 생성 보조
- `scripts/ml/sequence/`
  - sequence residual 데이터셋/학습/벤치마크
- `scripts/ml/coupled/`
  - coupled ONNX export / ensemble 학습 보조
- `scripts/alignment/tools/`
  - 정렬 검증/변환/전처리 보조
- `scripts/config/`
  - 배치/튜닝 YAML 샘플 및 실행 설정
- `scripts/docs/`
  - 스크립트 사용법 문서
- `scripts/testing/`
  - 샌드박스/스모크 테스트 스크립트
