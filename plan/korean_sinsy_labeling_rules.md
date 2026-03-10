# Korean Sinsy Labeling Rules

기준일: 2026-03-10

이 문서는 한국어 `sinsy` 음절 라벨을 현재 OTO 생성기에서 사용할 때의 권장 라벨링 규칙을 정리한다.
여기서 말하는 `sinsy` 라벨은 필수 입력이 아니라 선택형 보조 anchor source다.

적용 대상:

- 한국어 `CVVC`
- 한국어 `CVC`
- 필요 시 한국어 `VCV`의 `CV/CV_HEAD/VCV` anchor 보조

---

## 1. 목적

`sinsy` 라벨의 목적은 음절 경계를 사람이 직접 지정해서 다음 문제를 줄이는 것이다.

- 반복 음절에서 한 칸 밀리는 오매핑
- `go -> gyo`, `reo -> ryeo` 같은 plain/glide 오매핑
- 무음 구간 또는 저활성 구간으로의 잘못된 매핑
- 한국어 `CVVC/CVC`에서 음절 순서가 꼬이는 문제

현재 구현에서는 `sinsy` 라벨을 내부 음절 anchor로 사용하고,
세부 `VC/VV` 위치는 계속 phone/audio 기반 로직으로 보조 추정한다.

---

## 2. 기본 형식

각 줄은 아래 형식을 따른다.

```txt
start end label
```

예:

```txt
0 500000 ga
500000 1000000 gi
1000000 1500000 gya
1500000 2000000 gan
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

가장 중요한 원칙은 다음 하나다.

`label`은 사람이 읽기 좋은 이름이 아니라 현재 reclist/alias와 같은 내부 토큰이어야 한다.

즉 다음이 우선이다.

- 실제 한글 철자
- 표준 로마자 표기
- 일반 한국어 발음 표기

보다 더 중요한 것은:

- 현재 reclist에 존재하는 토큰과 1:1 대응
- plain/glide 구분 보존
- `r/l` 구분 보존
- 종성 포함 음절 토큰 보존

---

## 4. 허용 토큰 원칙

## 4-1. 기본 CV 토큰

예:

- `ga gi gu ge go geu geo`
- `na ni nu ne no neun neon`
- `da di du de do deu deo`

이 토큰들은 각각 실제 녹음 음절 anchor를 뜻한다.

예:

- `ga` -> `가`
- `ge` -> `개/게` 계열로 reclist 내부에서 사용하는 front vowel token
- `geu` -> `그`
- `geo` -> `거`

주의:

- 이 프로젝트의 `e`는 표준 국어학 표기 구분보다 reclist 내부 토큰 기준이 우선이다.
- 따라서 `e`는 “reclist에서 그렇게 부르는 front vowel class”로 취급한다.

## 4-2. glide / 복합모음 토큰

예:

- `gya gye gyeo gyo gyu`
- `gwa gwe gwi gweo`
- `ya ye yeo yo yu eui`
- `wa we wi weo`

이 토큰들은 기본 CV의 변형이 아니라 독립 anchor다.

즉 아래는 서로 다른 토큰이다.

- `go` / `gyo`
- `o` / `yo`
- `reo` / `ryeo`
- `ga` / `gwa`

현재 생성기 정책에서도 plain/glide 혼동은 별도 오매핑으로 본다.

## 4-3. 종성 포함 토큰

예:

- `gan gim gong`
- `gam geom`
- `gang ging geong`
- `ral reol reul`

이 토큰들은 `ga + n`처럼 분해해서 쓰지 않는다.
라벨 안에서는 하나의 독립 토큰으로 유지하는 것이 원칙이다.

좋은 예:

- `gan`
- `gam`
- `gang`

나쁜 예:

- `ga n`
- `ga-ng`
- `ga+n`

## 4-4. `ㄹ`의 `r/l` 구분

이 프로젝트의 reclist는 `ㄹ`을 두 계열로 나눈다.

- `r`: 초성/탄설 쪽
- `l`: 종성/설측 쪽

예:

- `rara ...`는 `나라면`의 `라` 쪽 발음
- `lala ...`는 `올라`의 `라` 쪽 발음

따라서 아래는 서로 다른 토큰이다.

- `ra` / `la`
- `ryeo` / `lyeo`
- `reui` / `leui`

이 둘을 한 label class로 합치면 안 된다.

## 4-5. 특수 토큰

다음과 같은 특수 항목도 alias 이름 그대로 유지한다.

- `eunR`
- `eumR`
- `eungR`
- `eulR`
- `aLong`
- `iLong`
- `uLong`
- `aH`
- `iH`

이 토큰들은 일반 CV/CVC 규칙으로 다시 쓰지 않는다.

---

## 5. 좋은 예 / 나쁜 예

좋은 예:

```txt
0 420000 go
420000 860000 gyo
860000 1300000 gan
1300000 1740000 gang
1740000 2160000 reo
2160000 2600000 ryeo
```

나쁜 예:

```txt
0 420000 고
420000 860000 교
860000 1300000 ga n
1300000 1740000 ㄱㅏㅇ
1740000 2160000 reo plain
2160000 2600000 xx^xx-ryeo+ga=...
```

이유:

- 한글 label은 현재 구현이 직접 정규화하지 않는다.
- 공백 포함 label은 안전하지 않다.
- 설명형 suffix는 planner score 비교에 불리하다.
- full-context HTS label은 현재 구현 대상이 아니다.

---

## 6. 권장 운영 규칙

실무적으로는 아래 규칙을 권장한다.

1. 라벨 토큰 집합을 현재 reclist 토큰 집합과 동일하게 유지한다.
2. plain vowel과 glide vowel은 절대 합치지 않는다.
3. `r/l`은 절대 합치지 않는다.
4. 종성 포함 음절은 독립 토큰으로 유지한다.
5. 특수 토큰은 임의 변형하지 않는다.
6. 라벨이 없으면 기존 `MFA + planner + audio/phone` 경로를 그대로 사용한다.
7. 라벨이 있어도 `VC/VV` 내부 위치는 기존 audio/phone 추정 로직을 계속 사용한다.

---

## 7. validator 기준

한국어 `sinsy` 라벨 validator를 만든다면 최소한 아래를 검사해야 한다.

필수 검사:

- 각 줄이 `start end label` 형식인지
- `start < end` 인지
- `label`에 공백이 없는지
- `label`이 허용 토큰 사전에 존재하는지

권장 검사:

- plain/glide 혼동 가능성이 높은 오탈자 탐지
  - 예: `go` 대신 `gyo`
  - 예: `reo` 대신 `ryeo`
- `r/l` 혼동 탐지
- 종성 포함 토큰이 분해형으로 적히지 않았는지 확인
- full-context label 형태 포함 여부 확인

경고 예시:

- unknown token
- possible plain/glide confusion
- possible r/l confusion
- split coda token detected
- unsupported full-context style label

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
- [core/oto_generator.py](C:/Users/oyh57/SODAsoo1/Devs/UTAU_Auto_OTO_v3/Auto_OTO/core/oto_generator.py)

현재 동작은 다음과 같다.

- 라벨 파일이 있으면 planner가 `sinsy_labels` source를 우선 시도한다.
- 라벨이 없거나 매핑 실패 시 기존 planner로 fallback 한다.
- low-margin 행은 `row abstain`으로 건너뛸 수 있다.

---

## 9. 결론

한국어 `sinsy` 라벨의 label 값은
`현재 reclist/alias에서 실제로 사용하는 로마자 토큰과 동일하게 적는 것`
이 가장 중요하다.

다시 말해:

- `go`는 `go`
- `gyo`는 `gyo`
- `gan`은 `gan`
- `ra`와 `la`는 따로
- `reo`와 `ryeo`는 따로

이 원칙만 지켜도 음절 anchor 안정성은 크게 올라간다.
