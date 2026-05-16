# Core OTO-ML Runtime

`core/oto_ml/`은 OTO-ML 런타임 추론/선택/보정 로직을 둔다.

## 책임 범위

- 모델 번들 로딩/설치/검증
- feature 구성, selector/refiner, 신뢰도/정책 적용
- coupled/autofree/e2e 런타임 결합 보조

## 배치 규칙

- 학습 스크립트/데이터셋 빌드는 `ml/scripts/*`에 둔다.
- `core/oto_ml/`은 런타임에서 직접 호출되는 코드만 유지한다.
- 루트 `core/oto_ml_*.py` 신규 실구현 추가 금지(호환 래퍼만 유지).

## 금지

- 학습 루프/평가 루프를 직접 포함하지 않는다.
- 실험성 코드(`scripts/experiments`)에 직접 의존하지 않는다.

## 검증 체크

- 번들 로딩 실패 시 원인 코드/메시지를 명확히 남긴다.
- 배포 경로(`assets/models`, `models`) 해석이 바뀌는 변경은 build smoke와 함께 검증한다.
