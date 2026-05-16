# ML Workspace

`ml/`은 학습/평가/데이터셋/실험 중심 개발 영역이다.

## 원칙

- 학습/평가 코드는 배포 런타임에서 직접 import하지 않는다.
- 배포에 필요한 것은 모델 결과물과 최소 런타임 연결 코드만 사용한다.
- 실행 스크립트는 `scripts/train`, `scripts/evaluate`, `scripts/benchmark` 경로를 우선 사용한다.
