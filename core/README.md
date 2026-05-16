# Core Runtime

`core/`는 AutoOTO 배포 런타임 핵심 로직이다.

## 포함

- OTO 생성/정렬/후처리/검증
- GUI가 직접 호출하는 비즈니스 로직
- 런타임 ML 추론 연결

## 현재 폴더 구조(정리 2차 기준)

- `core/alignment/`: 정렬/라벨/sequence aligner 관련 구현
- `core/generation/ja/`: 일본어 생성 구현
- `core/generation/kr/`: 한국어 생성 구현
- `core/generation/*.py`: 공통 생성 런타임 보조
- `core/coarse_crnn/`: boundary scorer/decoder 계열
- `core/oto_ml/`: OTO-ML 서브패키지

## 호환성 규칙

- 기존 루트 모듈(`core/ja_*.py`, `core/kr_*.py`, 일부 alignment 모듈)은 import 호환 래퍼를 유지한다.
- 신규 구현은 하위 폴더(`alignment`, `generation/ja`, `generation/kr`)에만 추가한다.

## 금지

- `ml/training`, `ml/evaluation`, `scripts/experiments`, `scripts/deprecated` import 금지
- 사용자 설명 없는 silent failure 금지
