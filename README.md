# UTAU Auto OTO

한국어 · 일본어 UTAU 음원을 위한 **자동 OTO 생성** 도구입니다.  
Windows 독립 실행형 GUI 프로그램 (CustomTkinter 기반, Python 3.10+)

---
**이 툴의 소스코드는 AI를 이용해 생성했으며**, 인간이 검수 후 일부 수정을 거쳤습니다.</br>
**The source code of this tool was created using AI**, and it was modified by humans after inspection.

</br>이 프로그램은 '있는 그대로' 제공되며, 제작자는 이 프로그램을 사용하여 입은 손해에 대해 어떤 것도 책임지지 않습니다.</br>
This program is provided 'as it is' and the producer is not responsible for any damages incurred using this program.

---

### 지원 형식:
한국어 Korean
- CV (단음/연단음)
- CVC
- CVVC
- VCV (한연음)

일본어 Japanese
- CV (단음/연단음)
- CVVC
- VCV (연속음)


(모든 형식/언어에서 생성된 oto는 정확함을 보증할 수 없습니다. 음성인식의 한계로, 음절 단위로 에일리어스와 실제 발음이 불일치하는 현상이 자주 발생합니다.)

---

**이 프로그램은 64비트 윈도우 환경에서만 실행 가능하며, 해당 툴 실행에는 Microsoft Visual C++ 2015-2022 (x64) 런타임이 필요합니다. 꼭 설치해주세요. 미설치 시 프로그램이 아예 실행되지 않을 수 있습니다.
 마이크로소프트 공식 다운로드 링크: https://aka.ms/vs/17/release/vc_redist.x64.exe**

**64bit Windows environment and Microsoft Visual C++ 2015-2022 (x64) runtime is required to run the tool. Make sure to install it. If not, the program may not run at all.
Microsoft Official Download Link: https://aka.ms/vs/17/release/vc_redist.x64.exe**

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

## 기본 인코딩

앱의 기본 텍스트/파일 인코딩은 **UTF-8**입니다.

---

---

## Draft Status

The files `LICENSE` and `MODEL_LICENSE.md` are project drafts intended to define
the distribution policy of this repository. Review and adjust them before public
release if you want tighter wording for contributor, data, or jurisdiction-specific issues.
