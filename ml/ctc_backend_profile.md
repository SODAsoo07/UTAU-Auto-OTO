# CTC 백엔드 모델 운영 정책 (저부하 + 고신뢰)

## 범위
- 본 문서는 `ctc-forced-aligner` 백엔드 통합 브랜치에서 사용할 모델/기본값 정책을 정의한다.
- 현재 메인 파이프라인은 유지한다.
  - `MFA (accurate/fast 프로필)` -> `OTO 생성`

## 통합 방향
- MFA를 1차(기본) 경로로 유지한다.
- CTC 백엔드는 다음 두 방식으로 추가한다.
  - 수동 정렬 엔진 선택 옵션
  - MFA 실패 시 자동 폴백
- 백엔드 구현은 아래 커밋 기준 별도 브랜치에서 진행한다.
  - `ca3d39496302e32b894e1fc2ed30859a4722f83d`

## 모델 기본값

### 한국어
- Base(저부하): `Kkonjeong/wav2vec2-base-korean`
  - 약 360MB
- Upgrade(고신뢰 재정렬): `kresnik/wav2vec2-large-xlsr-korean`
  - 약 1.2GB

### 일본어
- Base(저부하): `reazon-research/japanese-wav2vec2-base-rs35kh`
  - 약 369MB
- Upgrade(고신뢰 재정렬): `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`
  - 약 1.2GB

### 공통 비상 폴백
- `MahmoudAshraf/mms-300m-1130-forced-aligner`
  - KR/JP 기본 품질 경로에는 사용하지 않고, 비상용으로만 사용한다.

## 런타임 프로필
- `balanced` (기본값)
  - Base 모델 먼저 실행
  - 저신뢰 파일만 Upgrade 모델로 재실행
- `low_load`
  - Base 모델만 실행
- `high_accuracy`
  - 모든 파일을 Upgrade 모델로 실행

## 저신뢰 재실행 트리거 (초기값)
- 아래 조건 중 하나라도 만족하면 해당 파일을 재실행한다.
  - 정렬 생성 실패
  - 세그먼트 결과가 비어 있음
  - 세그먼트 시간 순서가 단조 증가하지 않음
  - coverage 비율 `< 0.97` (정렬된 길이 / 오디오 길이)
  - 정규화 신뢰도 `< 0.58` (실데이터로 추후 보정)

## 안전장치 제안
- `balanced` 모드에서 재실행 파일 수 상한:
  - `max(2, ceil(total_files * 0.30))`
- Upgrade 재실행 후에도 저신뢰인 경우:
  - MFA 결과가 있으면 MFA 결과 우선 사용
  - MFA 결과가 없으면 CTC 결과 중 최선값을 채택하고 로그에 저신뢰 상태를 기록

## 라이선스 메모
- `MahmoudAshraf97/ctc-forced-aligner` 라이선스는 `CC-BY-NC 4.0`이다.
- 본 정책은 비상업적 사용을 전제로 한다.
