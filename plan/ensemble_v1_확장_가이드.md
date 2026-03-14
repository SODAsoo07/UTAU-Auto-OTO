# ensemble v1 확장 가이드 (향후)

이 문서는 **데이터가 충분할 때** 확장 가능한 ensemble v1 학습/운영안을 정리합니다.
현재는 **voicebank가 1개인 데이터셋**이므로 권장하지 않습니다.

## 전제 조건
- 최소 3개 이상 voicebank가 필요 (OOF 분할 안정성).
- rawmel patch cache가 준비되어야 함.
- LightGBM + coupled + meta까지 학습/저장할 수 있는 환경.

## 핵심 구조
- LightGBM: 구조적 fallback.
- coupled(rawmel): 주 모델.
- meta: OOF 예측 위에 얹는 stacked learner.
- 번들 구조:
  - `v1_ensemble/`
    - `lightgbm/`
    - `coupled/`
    - `meta/`

## 학습 흐름
1. rawmel patch cache 생성.
2. `scripts/train_ensemble_bundle.py`로 OOF + meta 학습.
3. 생성된 번들을 runtime 모델 경로에 설치.

## 런타임 동작
- `UTOA_ML_ENSEMBLE_ENABLE=1`이면 ensemble v1이 우선 로드됨.
- ensemble이 없으면 coupled/lightgbm 게이팅으로 폴백.

## 비용 요약
- OOF 및 meta 학습으로 **학습 시간/디스크/관리 비용 증가**.
- 단일 voicebank에서는 성능 이득이 제한적.
