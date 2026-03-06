# WhisperX 백엔드 통합 계획 (MFA 유지)

## 범위
- 본 문서는 기존 CTC 백엔드 계획을 대체한다.
- 메인 파이프라인은 유지한다.
  - `MFA (accurate/fast 프로필)` -> `OTO 생성`
- WhisperX는 보조 정렬 엔진으로 추가한다.
  - 수동 선택 실행
  - MFA 실패/저신뢰 시 자동 폴백

## 통합 원칙
- MFA를 1차 경로로 유지한다.
- WhisperX는 결과 품질 개선과 저사양 대체 경로 제공이 목적이다.
- 화자 분리(diarization)는 기본 비활성화한다.
  - 의존성/라이선스/속도 리스크를 줄이기 위함

## 구현 단계
1. `core/whisperx_runner.py` 추가
   - 모델 로딩, 정렬 실행, TextGrid 변환, 오류/로그 래핑
2. 파이프라인 연결
   - 정렬 엔진 선택지에 `WhisperX` 추가
   - MFA 실패/저신뢰 시 WhisperX 재시도 분기 추가
3. 신뢰도/선택 정책 통합
   - 기존 임계치 기반 신뢰도 계산을 공유
   - MFA/WhisperX 동시 결과 시 점수 우선 + 안정성 규칙 적용
4. 운영 안정화
   - 모델 캐시 경로/용량 정책 추가
   - 정렬 중간 산출물(JSON/세그먼트) 자동 정리 옵션 추가

## 언어별 기본 모델 (상업 이용 가능 라이선스 우선)

### 한국어
- `low_load`: `Kkonjeong/wav2vec2-base-korean` (`apache-2.0`)
- `balanced`/`high_accuracy`: `kresnik/wav2vec2-large-xlsr-korean` (`apache-2.0`)

### 일본어
- `low_load`: `reazon-research/japanese-wav2vec2-base-rs35kh` (`apache-2.0`)
- `balanced`/`high_accuracy`: `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` (`apache-2.0`)

## 기본 실행 프로필
- `balanced` (기본값)
  - MFA 실행 -> 실패/저신뢰 파일만 WhisperX 재정렬
- `low_load`
  - MFA fast 우선, WhisperX는 저부하 모델만 사용
- `high_accuracy`
  - MFA accurate + WhisperX 고신뢰 모델 재검증

## 저신뢰 재실행 트리거 (초기값)
- 정렬 생성 실패
- 세그먼트 비어 있음
- 세그먼트 시간 순서 비단조
- coverage `< 0.97`
- 정규화 신뢰도 `< 0.58`

## 라이선스 메모
- `WhisperX`는 `BSD-2-Clause` 라이선스다.
- `openai/whisper`는 `MIT` 라이선스다.
- 위 KR/JP 정렬 모델은 모두 `Apache-2.0` 표기다.
- 2차 산출물 상업 이용은 가능하되, 모델/코드 재배포 시 각 라이선스 고지 의무를 준수한다.
- 보이스뱅크 원저작권/음성 제공자 약관은 별도로 확인한다.
