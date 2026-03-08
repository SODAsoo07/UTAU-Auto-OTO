# 제3자 고지문

이 문서는 Auto OTO에서 사용하는 제3자 소프트웨어를 요약한 고지문입니다.

이 문서는 편의를 위한 요약본이며, 각 구성요소의 원본 라이선스를 대체하지 않습니다.
이 문서의 요약과 상위 프로젝트의 원본 라이선스가 충돌하는 경우, 원본
라이선스가 우선합니다.

## 적용 범위

이 고지문은 다음 항목을 다룹니다.

- 소스코드 또는 선택적 ML 기능에서 사용하는 Python 패키지
- 배포용 바이너리를 생성할 때 사용하는 빌드 도구
- Windows 배포본에 포함될 수 있는 제3자 런타임 구성요소

이 문서는 각 구성요소의 원래 라이선스가 부여하는 범위를 넘는 권리를 부여하지
않습니다.

## 프로젝트 라이선스 분리

Auto OTO는 다음 항목을 서로 분리하여 관리합니다.

- 프로젝트 소스코드 및 프로젝트 자산
- 공식 모델 파일
- 제3자 의존성

제3자 구성요소는 각자의 라이선스를 그대로 유지하며, Auto OTO의 자체 라이선스로
다시 라이선스되지 않습니다.

## 런타임 및 소스 의존성

### customtkinter

- 배포처: Tom Schimansky / CustomTkinter
- 웹사이트: https://customtkinter.tomschimansky.com
- 라이선스: MIT License
- 사용 목적: GUI 프레임워크

### TextGrid

- 배포처: Kyle Gorman, Max Bane, Morgan Sonderegger
- 패키지명: `TextGrid`
- 라이선스: MIT License
- 사용 목적: TextGrid 파싱 및 처리

### NumPy

- 배포처: NumPy Developers
- 웹사이트: https://numpy.org
- 라이선스: BSD 3-Clause License
- 사용 목적: 검증 및 ML 관련 코드에서 사용하는 수치 처리
- 참고: NumPy의 바이너리 배포본에는 별도 고지 의무가 있는 추가 런타임
  구성요소가 포함될 수 있습니다

### LightGBM

- 배포처: Microsoft / LightGBM contributors
- 웹사이트: https://github.com/microsoft/LightGBM
- 라이선스: MIT License
- 사용 목적: 선택적 OTO ML 추론 및 학습 백엔드

### pandas

- 배포처: pandas contributors
- 웹사이트: https://pandas.pydata.org
- 라이선스: BSD 3-Clause License
- 사용 목적: 선택적 ML 학습 및 평가 워크플로우

### scikit-learn

- 배포처: scikit-learn developers
- 웹사이트: https://scikit-learn.org
- 라이선스: BSD 3-Clause License
- 사용 목적: 선택적 ML 학습 및 평가 워크플로우
- 참고: 일부 바이너리 배포본에는 별도 조건이 있는 추가 런타임 파일이 포함될
  수 있습니다

### PyTorch

- 배포처: PyTorch contributors
- 웹사이트: https://pytorch.org
- 라이선스: BSD 3-Clause License
- 사용 목적: 선택적 환경 점검 및 고급 ML 워크플로우
- 패키징 참고: 현재 경량 빌드 설정에서는 기본 배포본에서 `torch`,
  `torchaudio`, `torchvision`을 제외합니다

## 빌드 도구

### PyInstaller

- 배포처: PyInstaller Development Team
- 웹사이트: https://pyinstaller.org
- 라이선스: GPL v2 이상, 단 PyInstaller bootloader exception 포함
- 사용 목적: Windows 실행 파일 패키징용 빌드 도구
- 참고: 이 예외 조항 덕분에 비오픈소스 배포나 상업/비상업 배포에도 일반적으로
  사용할 수 있지만, PyInstaller 자체는 여전히 자체 라이선스를 따릅니다

## 번들될 수 있는 런타임 구성요소

### FFmpeg

- 배포처: FFmpeg 프로젝트 및 함께 포함된 코덱/라이브러리 기여자
- 웹사이트: https://ffmpeg.org
- 현재 빌드 스크립트가 참조하는 Windows 바이너리 배포처:
  https://www.gyan.dev/ffmpeg/builds/
- 사용 목적: Windows 배포본의 오디오/미디어 처리
- 중요: FFmpeg 바이너리를 Auto OTO와 함께 재배포하는 경우, 해당 FFmpeg 빌드에
  적용되는 라이선스 고지, 저작권 고지, 소스 참조 정보 등을 함께 제공해야 합니다
- 중요: 정확한 의무 범위는 실제로 포함한 FFmpeg 빌드의 조건에 따라 달라집니다

## 현재 빌드 설정 메모

이 문서 작성 시점의 프로젝트 빌드 스크립트는 다음과 같습니다.

- 빌드 시 `PyInstaller`를 설치함
- `requirements.txt`와 `requirements-ml.txt`의 의존성을 설치함
- 기본 앱 번들에서 `torch`, `torchaudio`, `torchvision`, `ml`을 제외함
- Windows 배포본에 FFmpeg 바이너리를 복사함

배포 담당자는 빌드 파이프라인이나 의존성 목록이 바뀔 때마다 이 문서를 다시
검토해야 합니다.

## 재배포 권장 사항

Auto OTO의 바이너리 배포본 또는 패키징된 소스 배포본을 재배포할 때는 다음을
함께 포함하는 것을 권장합니다.

- 이 고지문 파일
- 실제로 포함된 각 제3자 구성요소의 원본 라이선스 전문
- FFmpeg를 번들하는 경우 FFmpeg 관련 별도 고지문
- 바이너리 wheel 또는 함께 포함된 런타임 라이브러리가 요구하는 추가 고지문

## 무보증

제3자 소프트웨어는 각자의 라이선스 조건에 따라 제공됩니다. 보증 부인과 기타
조건은 각 상위 프로젝트의 라이선스를 확인하십시오.
