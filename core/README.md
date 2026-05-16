# Core Runtime

`core/`는 AutoOTO 배포 런타임 핵심 로직이다.

## 포함

- OTO 생성/정렬/후처리/검증
- GUI가 직접 호출하는 비즈니스 로직
- 런타임 ML 추론 연결

## 현재 폴더 구조(정리 5차 기준)

- `core/alignment/`: 정렬/라벨/sequence aligner 관련 구현
- `core/generation/ja/`: 일본어 생성 구현
- `core/generation/kr/`: 한국어 생성 구현
- `core/generation/common/`: alias/oto/mapping/e2e 공통 생성 런타임 구현
- `core/generation/*.py`: generation 하위 구현을 가리키는 루트 호환 래퍼
- `core/coarse_crnn/`: boundary scorer/decoder 계열
- `core/oto_ml/`: OTO-ML 서브패키지 구현
- `core/cvn/`: CVN 분류/특징/학습 타입 구현
- `core/runtime/`: 런타임 설정/경로/진단/상태/로깅 구현
- `core/timing/`: timing anchor/멜 보정/무음 프로파일 구현
- `core/common/`: 공통 유틸/정책/변환 보조 구현

## 호환성 규칙

- 기존 루트 모듈(`core/ja_*.py`, `core/kr_*.py`, `core/oto_*.py`, `core/mapping_*.py`, alignment/cvn/runtime/timing/oto_ml 래퍼)은 import 호환 래퍼를 유지한다.
- 신규 구현은 하위 카테고리 폴더에만 추가한다.

## 루트 래퍼 lifecycle(초안)

- `active`: 외부 진입점이 실제로 사용하는 래퍼. 즉시 삭제 금지.
- `legacy`: 구현 경로가 이미 카테고리 폴더로 이관되었고, 과거 호환을 위해 임시 유지하는 래퍼.
- `candidate_delete`: 외부 참조가 없고 문서/manifest 참조만 남은 래퍼. 다음 정리 사이클에서 삭제 대상.
- 삭제 원칙: 래퍼 삭제 전 반드시 호출 경로를 구현 경로(`core/<category>/...`)로 교체한 뒤, compile/import smoke를 통과해야 한다.

## 금지

- `ml/training`, `ml/evaluation`, `scripts/experiments`, `scripts/deprecated` import 금지
- 사용자 설명 없는 silent failure 금지
