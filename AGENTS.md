# AutoOTO Agent Rules

## 목적

프로젝트 정리 원칙에 따라 런타임/빌드/학습/평가/실험 경계를 유지한다.

## 우선 탐색 범위

- runtime 수정: `ui/`, `core/`, `scripts/runtime/`, `scripts/build/`
: `core` 내부에서는 `core/alignment/`, `core/generation/ja/`, `core/generation/kr/` 구현 경로를 우선 본다.
- training 수정: `ml/`, `scripts/train/`
- evaluation 수정: `ml/`, `scripts/evaluate/`, `scripts/benchmark/`

## 금지

- runtime 코드에서 training/evaluation/experiments/deprecated import 금지
- `scripts/deprecated/`는 사용자 명시 요청 없이는 수정 금지

## 변경 시 필수

- 경로 이동 시 `scripts/scripts_manifest.json` 갱신
- 문서(`AutoOTO-Docs`) 변경 내역 반영
