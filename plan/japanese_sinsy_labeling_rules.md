# Japanese Sinsy Labeling Rules

기준일: 2026-03-10

이 문서는 일본어 `sinsy` 음절/모라 라벨을 현재 OTO 생성기에서 사용할 때의 권장 라벨링 규칙을 정리한다.
여기서 말하는 `sinsy` 라벨은 필수 입력이 아니라 선택형 보조 anchor source다.

적용 대상:

- 일본어 `CVVC`
- 일본어 `CV`
- 일본어 `VCV`의 `CV/CV_HEAD/VCV` anchor 보조

---

## 1. 목적

일본어 `sinsy` 라벨의 목적은 모라 경계를 사람이 직접 지정해서 다음 문제를 줄이는 것이다.

- 반복 모라에서 한 칸 밀리는 오매핑
- `do -> dyo`, `ko -> kyo` 같은 plain/youon 오매핑
- 무음 구간 또는 저활성 구간으로의 잘못된 매핑
- `VV`가 마지막 모음만 보고 잘못된 위치로 가는 문제

현재 구현에서는 `sinsy` 라벨을 `CV/CV_HEAD/VCV`의 모라 anchor로 사용하고,
세부 `VC/VV` 위치는 계속 phone/audio 기반 로직으로 보조 추정한다.

---

## 2. 기본 형식

각 줄은 아래 형식을 따른다.

```txt
start end label
```

예:

```txt
0 500000 ka
500000 1000000 kyo
1000000 1500000 N
1500000 2000000 a
```

조건:

- 각 줄은 최소 3개 컬럼이어야 한다.
- `start < end` 여야 한다.
- 마지막 컬럼 `label`은 공백 없는 단일 토큰이 권장된다.
- 주석 줄은 `#`로 시작하면 무시된다.

현재 구현은 다음 시간 단위를 자동 추정한다.

- 초
- 밀리초
- HTS/Sinsy 계열 큰 정수 시간값

---

## 3. 핵심 원칙

일본어에서 가장 중요한 원칙은 다음 두 가지다.

1. `label`은 현재 reclist/alias와 같은 내부 토큰이어야 한다.
2. 라벨 단위는 가능한 한 `모라` 기준을 유지해야 한다.

즉 다음이 더 중요하다.

- 실제 가나 문자
- 표준 로마자 표기
- 사람이 읽기 좋은 설명형 이름

보다 더 중요한 것은:

- 현재 alias 토큰과 1:1 대응
- plain mora와 youon mora 구분
- `ん`, `っ`, 장음 계열 구분
- `CVVC`에서 실제 anchor로 쓰일 모라 단위 보존

---

## 4. 허용 토큰 원칙

## 4-1. 기본 CV / V 토큰

예:

- `a i u e o`
- `ka ki ku ke ko`
- `sa shi su se so`
- `ta chi tsu te to`
- `ra ri ru re ro`

이 토큰들은 각각 실제 녹음 모라 anchor를 뜻한다.

예:

- `ka` -> `か`
- `shi` -> `し`
- `tsu` -> `つ`
- `o` -> 독립 모음 `お`

## 4-2. youon / 축약 모라 토큰

예:

- `kya kyu kyo`
- `gya gyu gyo`
- `sha shu sho`
- `cha chu cho`
- `rya ryu ryo`

이 토큰들은 `ki + ya`, `shi + yo`처럼 둘로 나누지 않는다.
현재 매핑 관점에서는 독립 모라로 봐야 한다.

좋은 예:

- `kyo`
- `sha`
- `ryu`

나쁜 예:

- `ki yo`
- `sh i yo`
- `ri yu`

plain mora와 youon mora는 별개 anchor다.

즉 아래는 서로 다른 토큰이다.

- `ko` / `kyo`
- `shi` / `sha`
- `ri` / `ryu`

## 4-3. `ん`, `っ`, 장음 계열

프로젝트마다 토큰명은 다를 수 있지만, 중요한 것은 독립 단위로 유지하는 것이다.

예:

- `N` 또는 프로젝트에서 쓰는 비음 토큰
- `cl` / `q` / `xtsu` 계열의 促音 토큰
- 장음은 프로젝트 정책에 따라
  - 실제 모음 반복으로 유지하거나
  - 별도 토큰으로 유지

중요:

- `ん`은 주변 모라에 흡수해서 쓰지 않는다.
- `っ`도 다음 자음 일부로 적당히 뭉개지지 않게 독립 단위로 유지하는 편이 안전하다.
- 장음은 reclist/alias 설계와 같은 방식으로 적는다.

## 4-4. `CVVC`에서의 `VV`

`VV` 매핑 품질 때문에, 독립 모음과 모음 연쇄는 마지막 모음만 보고 적으면 안 된다.

예:

- `a i`가 있다면 각 모라 anchor는 `a`, `i`로 분리
- `o u`, `e i` 같은 연쇄도 실제 모라 순서를 그대로 유지

즉:

- `ai` 같은 합성 표기보다
- `a`, `i` 두 모라 anchor가 더 안전하다

단, 이 문서는 anchor 라벨 기준이므로 실제 사용 토큰은 reclist 설계를 우선한다.

---

## 5. 좋은 예 / 나쁜 예

좋은 예:

```txt
0 420000 ka
420000 860000 kyo
860000 1180000 N
1180000 1500000 a
1500000 1840000 shi
1840000 2200000 cl
2200000 2600000 to
```

나쁜 예:

```txt
0 420000 か
420000 860000 ki yo
860000 1180000 ん sound
1180000 1500000 xx^xx-kyo+a=...
1500000 1840000 shiyo
```

이유:

- 한글/가나 label은 현재 구현이 직접 정규화하지 않는다.
- 공백 포함 label은 안전하지 않다.
- 설명형 suffix는 planner score 비교에 불리하다.
- full-context HTS label은 현재 구현 대상이 아니다.
- `kyo`를 `ki yo`로 쪼개면 모라 수가 틀어진다.

---

## 6. 권장 운영 규칙

실무적으로는 아래 규칙을 권장한다.

1. 라벨 토큰 집합을 현재 reclist 토큰 집합과 동일하게 유지한다.
2. plain mora와 youon mora를 절대 합치지 않는다.
3. `ん`, `っ`, 장음 계열은 프로젝트 정책대로 독립 단위 유지한다.
4. `CVVC`에서 독립 모음 모라는 실제 순서대로 유지한다.
5. 라벨이 없으면 기존 `MFA + planner + audio/phone 보조` 경로를 그대로 사용한다.
6. 라벨이 있어도 `VC/VV` 내부 위치는 기존 audio/phone 추정 로직을 계속 사용한다.

---

## 7. validator 기준

일본어 `sinsy` 라벨 validator를 만든다면 최소한 아래를 검사해야 한다.

필수 검사:

- 각 줄이 `start end label` 형식인지
- `start < end` 인지
- `label`에 공백이 없는지
- `label`이 허용 토큰 사전에 존재하는지

권장 검사:

- plain/youon 혼동 가능성이 높은 오탈자 탐지
  - 예: `ko` 대신 `kyo`
  - 예: `sho` 대신 `sho`가 아닌 비표준 분해형
- `ki yo`, `shi ya` 같은 분해형 youon 탐지
- full-context label 형태 포함 여부 확인
- 독립 모음 연쇄에서 순서 누락 여부 확인

경고 예시:

- unknown token
- possible plain/youon confusion
- split youon token detected
- unsupported full-context style label
- suspicious vowel-chain labeling

---

## 8. 구현과의 관계

현재 구현 기준:

- `USE_SINSY_LABELS`
- `SINSY_LABEL_PATH`
- `UTOA_USE_SINSY_LABELS`
- `UTOA_SINSY_LABEL_PATH`

를 통해 opt-in으로 활성화할 수 있다.

관련 코드:

- [core/sinsy_label_ingest.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/sinsy_label_ingest.py)
- [core/alignment_ingest.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/alignment_ingest.py)
- [core/ja_oto_generator.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/ja_oto_generator.py)

현재 동작은 다음과 같다.

- 라벨 파일이 있으면 planner가 `sinsy_labels` source를 우선 시도한다.
- 라벨이 없거나 매핑 실패 시 기존 planner로 fallback 한다.
- low-margin 행은 `row abstain`으로 건너뛸 수 있다.

---

## 9. 결론

일본어 `sinsy` 라벨의 label 값은
`현재 reclist/alias에서 실제로 사용하는 로마자 토큰과 동일하게 적는 것`
이 가장 중요하다.

다시 말해:

- `ko`는 `ko`
- `kyo`는 `kyo`
- `shi`는 `shi`
- `sha`는 `sha`
- `N`은 `N`
- `cl/q/xtsu` 계열은 프로젝트 토큰 그대로

이 원칙을 지키면 일본어 `CVVC/CV/VCV`의 모라 anchor 안정성이 크게 올라간다.
