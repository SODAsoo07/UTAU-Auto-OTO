# Legacy Misc Tools

이 폴더는 현재 런타임, 배치 파이프라인, 테스트 기본 경로에서 쓰이지 않는
수동 점검용 스크립트를 보관한다.

## Moved Files

- `test_profile_timing_presets.py`
  - 프로파일 프리셋의 수동 smoke test 성격 스크립트
  - 현재 pytest 스위트나 앱 런타임에서 직접 사용하지 않음

## Rule

- 새 기능 개발은 이 폴더의 스크립트에 의존하지 않는다.
- 필요 시 참고용으로만 사용하고, 재활성화하려면 별도 테스트/문서와 함께 복귀시킨다.
