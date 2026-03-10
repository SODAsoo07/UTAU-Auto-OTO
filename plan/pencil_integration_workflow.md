# Pencil MCP 통합 워크플로우 (초안)

## 목표
- `.pen` 기반 UI 설계 결과를 `CustomTkinter` 구현에 안정적으로 반영한다.
- "완전 자동 코드 생성" 대신, 실사용 가능한 반자동 동기화(토큰/스펙 중심)를 사용한다.

## 전제
- 이 프로젝트의 실제 실행 UI는 Python 코드다.
- 핵심 진입점: `main.py`, `ui/layout_mixin.py`.
- 디자인 소스는 `.pen` 파일로 별도 관리한다.

## 폴더/파일 제안
- `design/`: `.pen` 원본 저장 위치
- `plan/pencil_integration_workflow.md`: 운영 절차 문서
- `ui/theme_tokens.py`: 디자인 토큰 단일 소스

## 일일 작업 절차
1. 디자인 편집기 확인
- Codex에서 `pencil/get_editor_state` 호출.
- "열린 파일 없음"이면 `pencil/open_document`로 새 문서 또는 기존 `.pen` 열기.

2. 화면 설계
- `.pen`에서 화면 구조/간격/색상 조정.
- 컴포넌트 이름은 Python UI 영역과 매핑 가능한 이름으로 지정.
- 예: `path_frame`, `status_group`, `advanced_toggle`.

3. 토큰 확정
- 색상/상태색(기본, 경고, 성공, 위험)을 토큰으로 정리.
- 확정된 값만 `ui/theme_tokens.py`에 반영.

4. 구현 반영
- `ui/layout_mixin.py` 등에서 하드코딩 값을 제거하고 토큰 참조로 교체.
- 신규 UI 코드 작성 시 하드코딩 금지, 토큰 우선 사용.

5. 검증
- 정적 검증: `python -m py_compile main.py ui/*.py`
- 실행 검증: `python main.py`
- 시각 검증: 주요 상태(한국어/일본어 전환, 고급 옵션 토글, 버튼 hover) 확인.

## 변경 정책
- 토큰 변경은 PR/커밋에서 단독으로 관리(의도 추적 용이).
- 레이아웃 구조 변경과 토큰 변경은 분리 커밋 권장.
- 릴리즈 직전에는 토큰 동결(Hotfix 제외).

## 비권장/비현실 구간
- `.pen` -> `CustomTkinter` 완전 자동 변환
- `CustomTkinter` -> `.pen` 역변환

위 두 항목은 구조 차이로 유지보수 비용이 매우 높아, 현재 프로젝트에서는 채택하지 않는다.

## 다음 확장안
- `ui/theme_tokens.py`의 일부를 JSON로 추출해 디자이너-개발자 간 리뷰에 사용.
- 상태별 스냅샷 체크리스트(언어/정렬 옵션/오류 상태) 문서화.
