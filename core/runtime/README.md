# Core Runtime Helpers

`core/runtime/`은 실행 시점 공통 환경/경로/진단 유틸리티를 둔다.

## 책임 범위

- 런타임 인코딩/경로/설정 로딩
- 모델 리졸버/상태 코드/로그 이벤트
- preflight 공통 검증

## 배치 규칙

- 생성 알고리즘(OTO row 조립, mapping 정책)은 `core/generation/*`로 둔다.
- 학습/평가 전용 유틸은 `ml/`로 둔다.
- 루트 `core/*.py` 신규 실구현 금지(호환 래퍼만 허용).

## 금지

- `ml/training`, `ml/evaluation` 직접 의존 금지
- GUI 상태/위젯 객체 의존 금지
- 실험성 경로(`scripts/experiments`, `scripts/deprecated`) 의존 금지

## 검증 체크

- 신규 runtime 모듈은 side-effect 최소화(모듈 import만으로 환경 변경 금지)
- `python -m compileall core`와 주요 호출 import smoke 확인
