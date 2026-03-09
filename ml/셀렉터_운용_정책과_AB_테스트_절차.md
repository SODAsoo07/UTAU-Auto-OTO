# 셀렉터 운용 정책과 A/B 테스트 절차

## 1. 현재 기본 정책

기준 파일:
- `C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\core\oto_ml_policy.py`

현재 기본 정책은 아래와 같다.

### Delta
- `delta` 보정은 전 형식에서 기본 `ON`

### Selector 기본 `ON`
- `korean / cv`
- `korean / cvc`

### Selector 기본 `OFF`
- `korean / cvvc`
- `korean / vcv`
- `japanese / cv`
- `japanese / cvvc`
- `japanese / vcv`

설명:
- `korean / cv`, `korean / cvc`는 재학습 후 selector 이득이 비교적 안정적으로 확인된 형식이다.
- 나머지 형식은 아직 실사용 A/B 검증이 더 필요하므로 기본 `OFF`로 둔다.
- 특히 `japanese / vcv`는 이전 평가에서 selector가 오히려 악화되는 경향이 있었으므로 계속 `OFF`가 맞다.

## 2. 환경변수로 selector 강제 제어

런타임에서는 아래 환경변수로 selector를 강제로 켜거나 끌 수 있다.

### 강제 끄기
```powershell
$env:UTOA_DISABLE_OTO_SELECTOR = "1"
Remove-Item Env:UTOA_FORCE_OTO_SELECTOR -ErrorAction SilentlyContinue
```

### 강제 켜기
```powershell
$env:UTOA_FORCE_OTO_SELECTOR = "1"
Remove-Item Env:UTOA_DISABLE_OTO_SELECTOR -ErrorAction SilentlyContinue
```

### 기본 정책으로 복귀
```powershell
Remove-Item Env:UTOA_FORCE_OTO_SELECTOR -ErrorAction SilentlyContinue
Remove-Item Env:UTOA_DISABLE_OTO_SELECTOR -ErrorAction SilentlyContinue
```

주의:
- 이 환경변수는 현재 PowerShell 세션에만 적용된다.
- 새 창을 열면 다시 기본 정책으로 돌아간다.

## 3. A/B 테스트 대상 우선순위

기본 정책 `OFF` 상태에서 우선 확인할 형식은 아래 순서가 좋다.

1. `korean / cvvc`
2. `japanese / cvvc`
3. `korean / vcv`
4. `japanese / cv`

이유:
- `cvvc` 계열은 실제 사용자 체감 문제인 `CV` 오매핑과 연결 불일치가 많다.
- `japanese / cvvc`는 selector 수치가 좋더라도 실제로는 mora 오매핑이 남을 수 있어 반드시 미학습 보이스뱅크로 확인해야 한다.

## 4. A/B 테스트 준비 원칙

테스트는 아래 조건을 반드시 맞춘다.

- 같은 보이스뱅크
- 같은 `wav`
- 같은 `TextGrid`
- 같은 배치 설정
- 출력 `oto.ini` 파일명만 다르게 저장

즉 바뀌는 것은 `selector ON/OFF` 하나만 남겨야 한다.

## 5. 추천 테스트 절차

예시는 `japanese / cvvc` 또는 `korean / cvvc`에 그대로 적용하면 된다.

### A안: delta only
```powershell
$env:UTOA_DISABLE_OTO_SELECTOR = "1"
Remove-Item Env:UTOA_FORCE_OTO_SELECTOR -ErrorAction SilentlyContinue

python -X utf8 C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\scripts\run_oto_generation_batch.py `
  --config C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\scripts\oto_batch_cases.single_test.yaml `
  --run-tag ab_delta_only
```

### B안: delta + selector
```powershell
$env:UTOA_FORCE_OTO_SELECTOR = "1"
Remove-Item Env:UTOA_DISABLE_OTO_SELECTOR -ErrorAction SilentlyContinue

python -X utf8 C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\scripts\run_oto_generation_batch.py `
  --config C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\scripts\oto_batch_cases.single_test.yaml `
  --run-tag ab_selector_on
```

### 테스트 후 기본 정책 복귀
```powershell
Remove-Item Env:UTOA_FORCE_OTO_SELECTOR -ErrorAction SilentlyContinue
Remove-Item Env:UTOA_DISABLE_OTO_SELECTOR -ErrorAction SilentlyContinue
```

## 6. 비교할 결과 파일

배치 결과는 아래를 같이 본다.

- `summary.json`
- 개별 케이스 로그
- 생성된 `oto.ini`
- validation 결과 파일

예:
- `C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\logs\oto_batch\ab_delta_only\summary.json`
- `C:\Users\oyh57\SODAsoo1\Devs\UTAU_Auto_OTO_v3\Auto_OTO\logs\oto_batch\ab_selector_on\summary.json`

## 7. 실제 판정 기준

수치만 보지 말고 아래 순서로 본다.

### 1순위: 음절 오매핑
- `CV`
- `-CV`
- 일본어 `CVVC`의 mora 선택
- 한국어 `CVC/CVVC`의 `CV`가 다른 음절을 가리키는지

이 항목이 하나라도 늘어나면 selector는 `OFF` 유지가 맞다.

### 2순위: 연결감
- `VC -> 다음 CV`
- `VV -> 다음 CV`
- 한국어 종성 마감
- 일본어 브리지 연결

청감상 “삐걱거림”이 줄어드는지 본다.

### 3순위: validation 경고
- 경고 수가 줄었는지
- 특히 `late offset`, `active region`, `overlap` 관련 경고가 늘었는지 확인

### 4순위: 로그의 selector 사용량
- selector가 켜졌는데도 거의 모든 row가 `base`만 고르면 실익이 작다.
- 반대로 selector가 자주 개입하는데 실제 청감이 나빠지면 `OFF`가 맞다.

## 8. 형식별 판정 기준

### Korean CVC
- `CV`의 음절 오매핑이 늘지 않아야 함
- 종성 이후 `VC` 마감이 더 안정적이어야 함

### Korean CVVC
- `CV` head가 다른 음절로 밀리지 않아야 함
- `VC -> CV` 연결이 덜 어색해야 함

### Japanese CVVC
- `CV`, `-CV`가 다른 mora를 가리키면 바로 탈락
- 요음, 장음, `ん`, `っ` 포함 파일을 우선 확인

### Japanese CV
- 단음 자체는 비교적 단순하므로 과보정 여부를 중점 확인

### VCV
- 브리지 연결은 좋아질 수 있지만, alias 선택이 흔들리면 기본 `OFF` 유지

## 9. 운영 결론 반영 규칙

아래 조건을 만족하면 기본 `ON` 후보로 올릴 수 있다.

- 미학습 보이스뱅크 2개 이상에서 테스트
- `CV/-CV` 오매핑 증가 없음
- 연결감이 동일하거나 개선
- validation 경고가 악화되지 않음

아래 중 하나라도 있으면 기본 `OFF` 유지가 맞다.

- 음절 오매핑 증가
- mora 선택 오류 증가
- `VC/CV` 연결이 더 어색해짐
- 특정 보이스뱅크에서만 좋아지고 다른 곳에서 무너짐

## 10. 권장 운용

현재는 아래처럼 쓰는 것이 가장 안전하다.

- 기본 정책 그대로 사용
- `korean / cvvc`, `japanese / cvvc`만 별도 A/B 테스트
- 테스트 통과 전까지는 selector를 강제로 켜지 않음
