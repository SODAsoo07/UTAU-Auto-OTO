# OTO 배치 청음 테스트 사용법

## 1) 설정 파일 준비
1. `scripts/oto_batch_cases.sample.yaml`을 복사해서 예: `scripts/oto_batch_cases.yaml`로 만든다.
2. 각 `cases`에 보이스뱅크 경로와 형식을 넣고 `enabled: true`로 바꾼다.
3. 기본적으로 결과는 각 보이스뱅크에 `oto.{run_tag}.ini` 형태로 생성된다.

## 2) 실행
```powershell
python scripts/run_oto_generation_batch.py --config scripts/oto_batch_cases.yaml
```

## 3) 옵션
1. 기존 `oto.ini`를 바로 교체:
```powershell
python scripts/run_oto_generation_batch.py --config scripts/oto_batch_cases.yaml --replace
```

2. 자동 검증 생략:
```powershell
python scripts/run_oto_generation_batch.py --config scripts/oto_batch_cases.yaml --skip-validation
```

3. 오류 시 즉시 중단:
```powershell
python scripts/run_oto_generation_batch.py --config scripts/oto_batch_cases.yaml --stop-on-error
```

## 4) 결과 확인
1. 실행 요약:
`logs/oto_batch/<run_tag>/summary.json`
2. 케이스별 상세 로그:
`logs/oto_batch/<run_tag>/*.log`
3. 검증 리포트:
생성된 `oto.*.ini` 옆의 `*.validation.txt`
