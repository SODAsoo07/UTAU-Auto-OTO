# Mel Coupling ML 리팩토링 마이그레이션 가이드

## 개요

`core/oto_ml_coupled.py`와 `core/oto_ml_features.py`를 기능 단위 모듈로 분리하고,
`ml/scripts/` 내 스크립트를 `ml/scripts/coupled/`로 재배치하였습니다.

## Import 호환성

**기존 import 경로는 100% 유지됩니다.**

```python
# 기존 코드 — 변경 없이 동작
from core.oto_ml_coupled import train_coupled_bundle, load_coupled_bundle
from core.oto_ml_features import FEATURE_NAMES, extract_feature_rows
```

새로운 경로도 사용 가능합니다:

```python
# 새 모듈 경로 (선택사항)
from core.oto_ml.coupled.model import _build_model, COUPLED_BACKEND
from core.oto_ml.coupled.training import train_coupled_bundle
from core.oto_ml.coupled.inference import load_coupled_bundle, predict_coupled_deltas
from core.oto_ml.features.schema import FEATURE_NAMES, TARGET_NAMES, get_feature_schema
from core.oto_ml.features.caches import _feature_cache_path
from core.oto_ml.pairing.vc_cv_pairing import _build_vc_cv_pair_map
```

## 모듈 매핑

### core/oto_ml_coupled.py → core/oto_ml/coupled/

| 기존 위치 | 새 위치 |
|---|---|
| `_build_model`, `COUPLED_BACKEND`, `PATCH_FEATURES` | `coupled/model.py` |
| `train_coupled_bundle`, `build_and_save_coupled_dataset`, `evaluate_coupled_bundle` | `coupled/training.py` |
| `load_coupled_bundle`, `predict_coupled_deltas` | `coupled/inference.py` |
| `_build_vc_cv_pair_map`, `_batch_pair_positions` | `pairing/vc_cv_pairing.py` |

### core/oto_ml_features.py → core/oto_ml/features/

| 기존 위치 | 새 위치 |
|---|---|
| `FEATURE_NAMES`, `TARGET_NAMES`, `get_feature_schema`, `canonicalize_feature_row`, `dataset_fieldnames`, `write_dataset_csv`, delta clip constants | `features/schema.py` |
| `_feature_cache_path`, `_load_feature_cache`, `_save_feature_cache` | `features/caches.py` |
| `extract_feature_rows`, `build_training_rows`, 기타 피처 추출 로직 | `oto_ml_features.py` (기존 위치 유지) |

## 스크립트 매핑

### 활성 스크립트 재배치

| 기존 경로 | 새 경로 |
|---|---|
| `build_oto_mel_coupled_dataset.py` | `coupled/build_dataset.py` |
| `train_oto_mel_coupled_model.py` | `coupled/train.py` |
| `evaluate_oto_mel_coupled.py` | `coupled/evaluate.py` |
| `export_oto_ml_bundle.py` | `coupled/export.py` |
| `install_oto_ml_bundle.py` | `coupled/install.py` |
| `stage_training_sources.py` | `coupled/stage_sources.py` |
| `prepare_staged_auto_pairs.py` | `coupled/prepare_pairs.py` |

기존 경로의 스크립트는 redirect wrapper로 유지되므로, PowerShell 스크립트 호환에 문제 없습니다.

### 삭제된 레거시 스크립트

| 삭제 파일 | 대체 |
|---|---|
| `build_oto_ml_dataset.py` | `coupled/build_dataset.py` |
| `collect_oto_ml_training_data.py` | `coupled/stage_sources.py` + `coupled/prepare_pairs.py` |
| `train_oto_ml_model.py` | `coupled/train.py` |
| `evaluate_oto_ml_model.py` | `coupled/evaluate.py` |
| `build_oto_ml_selector_dataset.py` | _(사용하지 않음)_ |
| `train_oto_ml_selector_model.py` | _(사용하지 않음)_ |
| `evaluate_oto_ml_selector_model.py` | _(사용하지 않음)_ |
| `evaluate_oto_ml_batch.py` | _(사용하지 않음)_ |
| `train_recommended_ml_models.ps1` | _(사용하지 않음)_ |

## 테스트 개선

- `ml/tests/conftest.py` 추가: `sys.path` 부트스트랩을 한 곳에서 관리
- `pytest.ini` 추가: `mel_coupled`, `slow` 마커 정의
- mel coupling 전용 테스트 실행: `python -m pytest ml/tests/ -m mel_coupled -v`
