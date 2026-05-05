# UTAU Auto OTO

한국어 · 일본어 UTAU 음원을 위한 **자동 OTO 생성 / 음원 준비 워크플로우** 도구입니다.  
Windows 독립 실행형 GUI 프로그램 (CustomTkinter 기반, Python 3.10+)

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **자동 OTO 생성** | MFA 음성 정렬 또는 No-MFA 방식으로 OTO 파라미터를 자동 계산합니다 |
| **No-MFA 자동 설정** | MFA 설치 없이 베이스 OTO 재매핑 또는 CRNN 신경망 예측으로 OTO를 생성합니다 |
| **한국어 / 일본어 지원** | VCV, CV, CVVC, CVC, 연속음 등 다양한 음원 형식을 지원합니다 |
| **실시간 로그** | 진행 상황과 오류를 앱 내 로그 창에서 실시간으로 확인합니다 |
| **프로파일 관리** | 언어별 · 음원 유형별 설정 프로파일을 저장하고 불러옵니다 |

---

## 자동 OTO 설정 모드

### MFA 정렬 방식 (기본)

Montreal Forced Aligner(MFA)를 사용하여 WAV 파일과 발음 레이블을 정렬한 뒤 OTO를 생성합니다.
초기 실행 시 `setup_mfa.bat`으로 MFA 환경을 설치해야 합니다.

### No-MFA 자동 설정 모드

MFA 환경 없이 사용 가능한 두 가지 서브 모드가 있습니다.

#### 베이스 OTO 재매핑 + 보정

기존 OTO(베이스 OTO)를 기반으로 파라미터를 재매핑하고 보정 공식을 적용합니다.
MFA 없이 빠르게 OTO를 조정하는 데 적합합니다.

#### CRNN(실험적)

WAV 파일과 소스 OTO를 입력으로 받아 CRNN(합성곱-순환 신경망) 모델로 OTO 파라미터를 예측합니다.
예측 일부가 실패하더라도 성공한 라인은 파일에 저장되며, 실패 내역은 로그로 출력됩니다.

> **참고:** CRNN 모드는 현재 실험적 기능입니다. 예측 품질은 모델과 음원 특성에 따라 달라질 수 있습니다.

---

## 빌드 / 개발 환경

### 요구 사항

- Python 3.10+
- 의존성 설치: `pip install -r requirements.txt`
- ML 기능 사용 시: `pip install -r requirements-ml.txt`

### 빌드 명령어

```bat
REM Nuitka (기본, 권장)
python build.py

REM PyInstaller 백엔드 사용
python build.py --backend pyinstaller

REM 릴리즈 채널 지정 (stable / preview)
python build.py --channel stable
python build.py --channel preview

REM 도움말
python build.py --help
```

빌드 결과물은 `dist_nuitka\UTAU_Auto_OTO\` (Nuitka) 또는 `dist\UTAU_Auto_OTO\` (PyInstaller)에 생성됩니다.

### 릴리즈 패키징

```bat
python build.py --package
```

패키지 명령은 다음을 수행합니다:
1. 앱 빌드
2. 런타임 데이터(프로파일, 모델, 설정 파일 등) 복사
3. `setup_mfa.bat`, `requirements.txt`, 스크립트 파일 포함
4. 개발 전용 파일(`.cache`, `.claude`, `__pycache__` 등) 제거

---

## MFA 환경 설정 (`setup_mfa.bat`)

MFA 정렬 방식을 사용하려면 최초 1회 MFA 환경을 설치해야 합니다.

```bat
setup_mfa.bat
```

### 기존 환경 재사용

이전 배포본으로 설치한 micromamba 가상환경(`.env`)이 감지되면,
재다운로드 없이 기존 환경을 재사용할지 묻는 메시지가 표시됩니다.

```
[INFO] 이전 설치에서 생성된 가상환경이 감지되었습니다:
       C:\Users\...\UTAU_Auto_OTO_v3\.env
이 환경을 재사용하시겠습니까? (재다운로드 불필요) [Y/N]:
```

`Y`를 선택하면 기존 환경의 Python과 MFA 실행 파일을 검증한 뒤 재사용합니다.
검증 실패 시 자동으로 새 환경을 설치합니다.

탐색 경로:
- `%LOCALAPPDATA%\UTAU_Auto_OTO_v3\.env`
- `%LOCALAPPDATA%\UTAU_Auto_OTO\.env`
- `%PUBLIC%\UTAU_Auto_OTO_v3\.env`
- `%PUBLIC%\UTAU_Auto_OTO\.env`

---

## 최근 변경사항

### v3.2.1

- **CRNN(실험적) 모드 항상 표시**: 개발자 설정 활성화 없이도 CRNN 모드를 선택할 수 있습니다
- **CRNN 저장 안정성 개선**: 일부 예측 실패 시에도 생성된 OTO 라인을 파일에 저장하고 실패 내역만 로그로 출력합니다
- **micromamba 환경 재사용**: 이전 배포본의 가상환경을 감지하여 재사용 여부를 선택할 수 있습니다
- **빌드 파일 목록 정리**: 존재하지 않는 릴리즈 파일 항목 제거, ml/configs 런타임 필요 파일만 선택적 포함
- **CRNN 에일리어스 롤 분류 시스템**: VC · VV 에일리어스 역할 기반 손실 가중치 적용 (학습 개선)
- **CVVC 품질 개선**: VC/VV 롤 Prior 조정, 타겟 윈도우 비활성화

---

## 기본 인코딩

앱의 기본 텍스트/파일 인코딩은 **UTF-8**입니다.

---

## License Summary

This repository is not an open-source project in the OSI sense.

- The source code is published under the source-available freeware terms in
  [LICENSE](./LICENSE).
- The official bundled models are licensed separately under
  [MODEL_LICENSE.md](./MODEL_LICENSE.md).
- Output created with the official models, such as generated oto settings,
  prepared voicebanks, and songs created with those voicebanks, may be used
  commercially under the terms described in `MODEL_LICENSE.md`.

### What You Can Do

- Use the software for free.
- Read and modify the source code.
- Share original or modified copies for free with the license notices kept intact.
- Use the generated outputs in commercial creative work.

### What You Cannot Do

- Sell the software itself or modified versions of the software.
- Charge for access to the software.
- Sell, host, or otherwise commercially exploit the official bundled models.

### Important Notes

- Third-party dependencies remain under their own licenses.
- The official model package and the source code are licensed separately.
- This repository may contain model files and metadata that are covered by
  `MODEL_LICENSE.md`, not by `LICENSE`.
- If you use your own models or your own data, you remain responsible for the
  rights to those materials.

---

## Draft Status

The files `LICENSE` and `MODEL_LICENSE.md` are project drafts intended to define
the distribution policy of this repository. Review and adjust them before public
release if you want tighter wording for contributor, data, or jurisdiction-specific issues.
