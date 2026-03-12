# Ensemble v1 추천 설정값

## 목적
- 한국어 CVVC/CVC에서 `CV`가 너무 이르게 들어오거나 공백으로 점프하는 문제를 줄이는 방향으로 학습과 운영 값을 정리한다.
- 기준은 현재 코드에 들어간 `ensemble_v1`, `gated ensemble`, `coupled_nn_v2_rawmel` 경로다.

## 1. 다중 bank 학습 권장값

12개 안팎 bank를 묶어 학습할 때의 시작점이다.

| 항목 | 권장값 | 이유 |
|---|---:|---|
| `group-column` | `voicebank_id` | bank 누수를 줄이고 leave-bank-out 성격을 확보하기 쉽다 |
| `num-folds` | `5` | OOF 안정성과 시간 비용의 균형이 가장 무난하다 |
| `min-mapping-confidence` | `0.40` | `0.30`은 오매핑 샘플이 너무 많이 섞일 수 있다 |
| `lightgbm-num-boost-round` | `500` | 현재 코드 기본값과 일치하고 과하게 크지 않다 |
| `lightgbm-early-stopping-rounds` | `50` | OOF 학습에서 과적합을 적당히 끊는다 |
| `coupled-epochs` | `70` | 현재 기본값, rawmel v2에서 첫 기준점으로 적당하다 |
| `coupled-batch-size` | `192` | 12GB 이상 VRAM에서 권장 |
| `coupled-learning-rate` | `0.001` | 현재 학습 기본값 |

## 2. 단일 bank 디버그 권장값

단일 bank만 가지고 빠르게 경향을 볼 때의 값이다.

| 항목 | 권장값 | 이유 |
|---|---:|---|
| `group-column` | `wav_norm` | `voicebank_id`는 그룹 수가 1이라 OOF 분리가 안 된다 |
| `num-folds` | `4` | 너무 많은 fold보다 안정적이다 |
| `min-mapping-confidence` | `0.35` | 데이터가 적을 때는 너무 높게 자르지 않는 편이 낫다 |
| `coupled-batch-size` | `128` | 소규모 실험에서 메모리 부담을 줄인다 |

## 3. alias 학습 범위 권장값

한국어 CVVC/CVC에서 ensemble 학습에는 아래 alias 타입을 먼저 넣는 편이 낫다.

- `cv`
- `cv_head`
- `vc`
- `vv`
- `vcv`

이유:
- `cv`, `cv_head`는 가장 치명적인 오매핑이 발생하는 구간이다.
- `vc`, `vv`, `vcv`를 같이 넣어야 연결부를 meta가 배울 수 있다.

## 4. 운영 런타임 권장값

환경변수는 아래 조합이 안전하다.

```powershell
$env:UTOA_ML_ENSEMBLE_ENABLE = "1"
$env:UTOA_ML_GATED_ENSEMBLE_ENABLE = "1"
$env:UTOA_ML_COUPLED_BACKEND = "v2"
```

의미:
- `UTOA_ML_ENSEMBLE_ENABLE=1`: `ensemble_v1` 루트 번들이 있으면 우선 사용한다.
- `UTOA_ML_GATED_ENSEMBLE_ENABLE=1`: explicit ensemble가 없어도 `coupled + lightgbm` 조합에서 confidence gate를 탄다.
- `UTOA_ML_COUPLED_BACKEND=v2`: rawmel 인코더 경로를 우선한다.

## 5. confidence 권장값

현재 코드의 기본 동작은 아래와 같다.

- 전역 기본: `UTOA_ML_COUPLED_MIN_CONF` 기본값 `0.55`
- 한국어 `cv`: 내부 floor `0.66`
- 한국어 `cv_head`: 내부 floor `0.68`
- 한국어 `cv/cv_head` + `cvc/cvvc`: 내부적으로 `+0.02`
- 한국어 `vc`: 내부 floor `0.56`
- 한국어 `vv`, `vcv`: 내부 floor `0.58`

즉, 아무 것도 설정하지 않아도 한국어 CVC/CVVC에서는 대략 아래 수준으로 동작한다.

| alias_type | 실질 기본값 |
|---|---:|
| `cv` in `cvc/cvvc` | `0.68` |
| `cv_head` in `cvc/cvvc` | `0.70` |
| `vc` | `0.56` |
| `vv` | `0.58` |
| `vcv` | `0.58` |

권장:
- 우선은 alias별 confidence 환경변수를 비워 두고 이 기본값을 그대로 쓰는 편이 낫다.
- 초기부터 더 낮게 내리면 공백 점프와 음절 오매핑이 다시 늘 가능성이 높다.

## 6. explicit ensemble_v1 번들 사용 시 주의

현재 `scripts/train_ensemble_bundle.py`가 쓰는 루트 `model_meta.json`에는 아래 값이 들어간다.

- `gating.min_coupled_confidence = 0.0`

즉, explicit `ensemble_v1` 루트 번들만 단독 사용하면 내부 gate가 다소 느슨할 수 있다.

운영 권장:
- 첫 배포 전에는 루트 `model_meta.json`의 `gating.min_coupled_confidence`를 `0.58` 전후로 수동 조정해서 비교한다.
- 특히 한국어 CVVC/CVC의 `cv`, `cv_head` 조기 진입 문제가 남아 있으면 `0.60`까지 올려보는 편이 낫다.

## 7. mapping confidence 권장 해석

`min-mapping-confidence`는 낮을수록 데이터 양은 늘지만 노이즈가 늘어난다.

| 값 | 권장 용도 |
|---|---|
| `0.30` | 데이터가 매우 적을 때의 탐색용 |
| `0.35` | 단일 bank 디버그 |
| `0.40` | 다중 bank 기본값 |
| `0.45` | 공백 점프와 음절 오매핑 억제를 더 우선할 때 |

현재 문제 맥락에서는 `0.30`보다 `0.40`에서 시작하는 편이 더 안전하다.

## 8. 배치 크기 권장값

| VRAM 대략치 | 권장 batch |
|---|---:|
| `8GB` | `96` |
| `10GB` | `128` |
| `12GB+` | `192` |

batch를 줄일 때는 lr를 먼저 내리기보다 epoch를 조금 늘리는 편이 보통 안정적이다.

## 9. 우선 실험 순서

실험 우선순위는 이 순서가 맞다.

1. `min-mapping-confidence 0.40`로 ensemble 재학습
2. explicit `ensemble_v1` 루트 `model_meta.json`의 `gating.min_coupled_confidence`를 `0.58`로 올려 A/B
3. 필요하면 `0.60`까지 올려 CV 공백 점프가 줄어드는지 확인
4. 그 다음에만 `min-mapping-confidence 0.45`를 시도

## 10. 추천 운용 프로필

### 프로필 A: 기본 배포
- `group-column=voicebank_id`
- `min-mapping-confidence=0.40`
- `num-folds=5`
- `epochs=70`
- `batch=192`
- `lr=0.001`
- runtime gate on
- alias별 confidence env는 비워 두기

### 프로필 B: 보수적 배포
- `group-column=voicebank_id`
- `min-mapping-confidence=0.45`
- `num-folds=5`
- `epochs=70`
- `batch=192`
- `lr=0.001`
- root `gating.min_coupled_confidence=0.60`

### 프로필 C: 단일 bank 디버그
- `group-column=wav_norm`
- `min-mapping-confidence=0.35`
- `num-folds=4`
- `epochs=70`
- `batch=128`
- `lr=0.001`
