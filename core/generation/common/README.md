# Core Generation Common

`core/generation/common/`은 언어 공통 OTO 생성 런타임 구현을 둔다.

## 책임 범위

- alias/anchor/mapping 점수 및 선택
- OTO row/파일 후처리 및 정책
- no-MFA fallback 생성 경로
- 런타임 진단/정규화 보조

## 배치 규칙

- `ja_*`, `kr_*` 언어 특화 로직은 각각 `core/generation/ja`, `core/generation/kr`에 둔다.
- 언어 불문 재사용 로직만 이 폴더에 둔다.
- 루트 `core/*.py`에는 신규 실구현을 추가하지 않는다. 필요한 경우 루트는 호환 래퍼만 유지한다.

## 금지

- `ml/training`, `ml/evaluation`, `scripts/experiments`, `scripts/deprecated` import 금지
- UI 계층(`ui/*`) 의존 금지

## 검증 체크

- 신규 파일 추가 후 최소 `python -m compileall core` 통과
- 호출 경로가 기존 루트 래퍼를 참조 중이면 카테고리 경로로 치환 후 반영
