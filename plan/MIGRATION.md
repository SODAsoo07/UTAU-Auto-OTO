# Mel Coupling ML 마이그레이션 안내

## 개요
리팩토링 이후 mel coupling 관련 코드와 스크립트는 기능별 모듈로 분리되었습니다.
기존 엔트리 포인트 일부는 호환용 래퍼로 남아 있지만, 앞으로는 새 모듈 경로를 기준으로 작업하는 것이 맞습니다.

## 핵심 변경점
- `core/oto_ml_coupled.py`, `core/oto_ml_features.py`에 있던 책임이 `core/oto_ml/` 아래 세부 모듈로 분리되었습니다.
- `ml/scripts/`에 흩어져 있던 coupled 학습 스크립트는 `ml/scripts/coupled/` 아래로 정리되었습니다.
- 학습 홀드아웃 기본 기준은 `wav_norm`입니다.
- mel class / blank patch 입력, aux head, VC↔CV pairing loss가 현재 기본 coupled 학습 경로에 포함됩니다.

## import 경로 변경
기존 import 예시:

```python
from core.oto_ml_coupled import train_coupled_bundle, load_coupled_bundle
from core.oto_ml_features import FEATURE_NAMES, extract_feature_rows
```

권장 import 예시:

```python
from core.oto_ml.coupled.model import _build_model, COUPLED_BACKEND, PATCH_FEATURES
from core.oto_ml.coupled.training import (
    build_and_save_coupled_dataset,
    evaluate_coupled_bundle,
    train_coupled_bundle,
)
from core.oto_ml.coupled.inference import (
    load_coupled_bundle,
    predict_coupled_deltas,
)
from core.oto_ml.features.schema import (
    FEATURE_NAMES,
    TARGET_NAMES,
    AUX_TARGET_NAMES,
    get_feature_schema,
    canonicalize_feature_row,
    dataset_fieldnames,
    write_dataset_csv,
)
from core.oto_ml.features.caches import (
    _feature_cache_path,
    _load_feature_cache,
    _save_feature_cache,
    _training_row_cache_path,
)
from core.oto_ml.pairing.vc_cv_pairing import (
    _build_vc_cv_pair_map,
    _batch_pair_positions,
)
```

## 파일 이동 매핑
### `core/oto_ml_coupled.py`

| 기존 심볼 | 새 위치 |
|---|---|
| `_build_model`, `COUPLED_BACKEND`, `PATCH_FEATURES` | `core/oto_ml/coupled/model.py` |
| `train_coupled_bundle`, `build_and_save_coupled_dataset`, `evaluate_coupled_bundle` | `core/oto_ml/coupled/training.py` |
| `load_coupled_bundle`, `predict_coupled_deltas` | `core/oto_ml/coupled/inference.py` |
| `_build_vc_cv_pair_map`, `_batch_pair_positions` | `core/oto_ml/pairing/vc_cv_pairing.py` |

### `core/oto_ml_features.py`

| 기존 심볼 | 새 위치 |
|---|---|
| `FEATURE_NAMES`, `TARGET_NAMES`, `AUX_TARGET_NAMES`, `get_feature_schema`, `canonicalize_feature_row`, `dataset_fieldnames`, `write_dataset_csv`, delta clip constants | `core/oto_ml/features/schema.py` |
| `_feature_cache_path`, `_load_feature_cache`, `_save_feature_cache`, `_training_row_cache_path` | `core/oto_ml/features/caches.py` |
| `extract_feature_rows`, `build_training_rows` 등 feature row 생성 로직 | `core/oto_ml/features/feature_extraction.py` |

## 스크립트 이동 매핑
### coupled 전용 스크립트

| 기존 스크립트 | 새 스크립트 |
|---|---|
| `build_oto_mel_coupled_dataset.py` | `ml/scripts/coupled/build_dataset.py` |
| `train_oto_mel_coupled_model.py` | `ml/scripts/coupled/train.py` |
| `evaluate_oto_mel_coupled.py` | `ml/scripts/coupled/evaluate.py` |
| `export_oto_ml_bundle.py` | `ml/scripts/coupled/export.py` |
| `install_oto_ml_bundle.py` | `ml/scripts/coupled/install.py` |
| `stage_training_sources.py` | `ml/scripts/coupled/stage_sources.py` |
| `prepare_staged_auto_pairs.py` | `ml/scripts/coupled/prepare_pairs.py` |

참고:
- 기존 스크립트 중 일부는 redirect wrapper로 남아 있을 수 있습니다.
- 새 작업은 가능하면 `ml/scripts/coupled/` 경로 기준으로 진행하는 편이 안전합니다.
- PowerShell 래퍼는 `ml/scripts/run_coupled_mel_oto_training.ps1`를 사용합니다.

### 레거시 스크립트 처리 방향
| 레거시 스크립트 | 처리 방향 |
|---|---|
| `build_oto_ml_dataset.py` | `ml/scripts/coupled/build_dataset.py`로 통합 |
| `collect_oto_ml_training_data.py` | `stage_sources.py` + `prepare_pairs.py`로 분리 대체 |
| `train_oto_ml_model.py` | `ml/scripts/coupled/train.py`로 대체 |
| `evaluate_oto_ml_model.py` | `ml/scripts/coupled/evaluate.py`로 대체 |
| `build_oto_ml_selector_dataset.py` | 미사용 후보 |
| `train_oto_ml_selector_model.py` | 미사용 후보 |
| `evaluate_oto_ml_selector_model.py` | 미사용 후보 |
| `evaluate_oto_ml_batch.py` | 미사용 후보 |
| `train_recommended_ml_models.ps1` | 미사용 후보 |

## 현재 coupled 파이프라인 동작 포인트
- 학습 홀드아웃은 `wav_norm` 기준입니다.
- mel class / blank 신호가 patch 입력에 직접 들어갑니다.
- aux head는 `aux_vowel_start_rel`, `aux_vowel_end_rel`, `aux_next_onset_rel`을 예측합니다.
- VC↔CV 관계 학습을 위해 pairing loss가 포함됩니다.

## alias 전처리 변경 사항
현재 전처리는 아래 순서로 alias를 정리합니다.

1. `prefix.map` 탐색
- `oto.ini`, `wav_dir`, `tg_dir` 기준으로 상위 폴더까지 탐색합니다.
- `prefix.map`이 있으면 우선 사용합니다.

2. 다른 `*.map` fallback 탐색
- `prefix.map`이 없으면 affix 규칙으로 파싱 가능한 다른 `*.map` 파일도 후보로 봅니다.
- 일반적인 다른 용도의 `.map` 파일은 가능한 한 무시합니다.

3. map 기반 prefix/suffix 제거
- 예: `PRE_ga_SUF` -> `ga`
- 예: `ga_C4` -> `ga`

4. pitch affix 제거
- `_A3`, `-C4`, ` F4` 같은 음계 affix를 제거합니다.

5. 주석성 접미사 제거
- 예: `ga注釈` -> `ga`
- 예: `ga메모` -> `ga`

6. 숨소리 정규화
- `吸こ吐`, `吸き吐`, `吸き吐_A3` 같은 변형도 `br`로 정규화합니다.

이 과정을 거친 뒤 `alias_norm`, `alias_type`, `occurrence_index`가 결정됩니다.

## 테스트 경로
- `ml/tests/conftest.py`에서 테스트용 `sys.path`가 정리되어 있습니다.
- mel coupling 관련 검증은 아래 명령으로 실행합니다.

```powershell
python -m pytest ml/tests -m mel_coupled -v
```

- `pytest.ini`에는 `mel_coupled`, `slow` marker가 정의되어 있습니다.

## 마이그레이션 시 주의사항
- `FEATURE_VERSION` 또는 `TRAIN_ROW_MATCH_VERSION`이 바뀌면 dataset CSV와 cache를 다시 생성해야 합니다.
- alias 정규화 규칙, 숨소리 규칙, 접미 메모 제거 규칙, `prefix.map` / `*.map` affix 처리 규칙이 바뀐 경우도 마찬가지로 `build_dataset`부터 다시 수행하는 편이 맞습니다.
- 새 코드 추가 시에는 먼저 `core/oto_ml/`와 `ml/scripts/coupled/` 안에서 대응 모듈을 찾고, 래퍼나 레거시 스크립트에 직접 기능을 넣지 않는 쪽이 좋습니다.
- 레거시 래퍼는 호환성 때문에 남아 있을 수 있으므로, 새 기능은 래퍼가 아니라 실제 구현 모듈에 넣어야 합니다.
