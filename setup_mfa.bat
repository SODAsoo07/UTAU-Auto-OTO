@echo off
@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title UTAU Auto OTO - MFA 설치

set "INSTALL_ML=0"
set "DELETE_OLD_AFTER_INSTALL=0"
set "AUTO_ML=0"
set "NON_INTERACTIVE=0"
set "DIRECT_SETUP=0"
set "NO_RECOVERY_SHIM=0"
set "FORCE_MENU=0"
set "REQUESTED_MODE="
set "WRAPPER_MODE="
set "SETUP_MFA_SELF=%~f0"
set "SETUP_MFA_DIR=%~dp0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="--with-ml" set "INSTALL_ML=1"
if /i "%~1"=="--install-ml" set "INSTALL_ML=1"
if /i "%~1"=="--non-interactive" set "NON_INTERACTIVE=1"
if /i "%~1"=="--yes" set "NON_INTERACTIVE=1"
if /i "%~1"=="--menu" set "FORCE_MENU=1"
if /i "%~1"=="--install" set "REQUESTED_MODE=install"
if /i "%~1"=="--recovery" set "REQUESTED_MODE=recovery"
if /i "%~1"=="--recover" set "REQUESTED_MODE=recovery"
if /i "%~1"=="--mode=install" set "REQUESTED_MODE=install"
if /i "%~1"=="--mode=recovery" set "REQUESTED_MODE=recovery"
if /i "%~1"=="--mode=menu" set "FORCE_MENU=1"
if /i "%~1"=="--direct-setup" set "DIRECT_SETUP=1"
if /i "%~1"=="--no-recovery-shim" set "NO_RECOVERY_SHIM=1"
shift
goto :parse_args

:show_help
echo 사용법: setup_mfa.bat [--install ^| --recovery ^| --menu] [--with-ml] [--non-interactive]
echo 스크립트 폴더^(.env^)에 Micromamba 기반 로컬 MFA 환경을 설치합니다.
echo 옵션:
echo   --install                 설치 모드 강제 실행
echo   --recovery / --recover    복구 모드 강제 실행
echo   --menu                    시작 메뉴 강제 표시
echo   --with-ml / --install-ml  ML 의존성^ (pandas/sklearn/lightgbm/pytorch^) 설치
echo   --non-interactive         비대화형 실행^ (자동 분기: env 있으면 복구, 없으면 설치^)
echo   --direct-setup            내부 사용^: 기존 설치 엔진 직접 실행
echo   --no-recovery-shim        복구 래퍼 비활성화^ (설치 모드 강제^)
exit /b 0

:args_done

for %%I in ("%SETUP_MFA_SELF%") do set "APP_DIR=%%~dpI"
if defined APP_DIR set "APP_DIR=%APP_DIR:~0,-1%"
if not defined APP_DIR set "APP_DIR=%CD%"
if /i "%APP_DIR%"=="%WINDIR%\System32" (
    echo [WARN] setup_mfa.bat이 System32 경로에서 실행되었습니다.
    echo        MFA 작업 경로를 LOCALAPPDATA로 전환합니다.
    set "APP_DIR=%LOCALAPPDATA%\UTAU_Auto_OTO_v3"
)

if "%DIRECT_SETUP%"=="1" goto :direct_setup_start

if "%NO_RECOVERY_SHIM%"=="1" set "REQUESTED_MODE=install"
call :resolve_runtime_recovery_script

if "%FORCE_MENU%"=="1" set "WRAPPER_MODE=menu"
if not defined WRAPPER_MODE if defined REQUESTED_MODE set "WRAPPER_MODE=%REQUESTED_MODE%"
if not defined WRAPPER_MODE if "%INSTALL_ML%"=="1" set "WRAPPER_MODE=install"
if not defined WRAPPER_MODE if "%NON_INTERACTIVE%"=="1" (
    if defined RUNTIME_RECOVERY_SCRIPT if exist "%APP_DIR%\.env" (
        set "WRAPPER_MODE=recovery"
    ) else (
        set "WRAPPER_MODE=install"
    )
)
if not defined WRAPPER_MODE set "WRAPPER_MODE=menu"

if /i "%WRAPPER_MODE%"=="menu" (
    call :select_start_mode
    if /i "!WRAPPER_MODE!"=="exit" exit /b 2
)

:wrapper_dispatch
if /i "%WRAPPER_MODE%"=="install" (
    call :preflight_install_tools
    if errorlevel 1 exit /b 1
    set "WRAPPER_ARGS=--direct-setup --no-recovery-shim"
    if "%INSTALL_ML%"=="1" set "WRAPPER_ARGS=!WRAPPER_ARGS! --with-ml"
    if "%NON_INTERACTIVE%"=="1" set "WRAPPER_ARGS=!WRAPPER_ARGS! --non-interactive"
    echo [INFO] 설치 모드 실행: !WRAPPER_ARGS!
    call "%SETUP_MFA_SELF%" !WRAPPER_ARGS!
    exit /b !ERRORLEVEL!
)

if /i "%WRAPPER_MODE%"=="recovery" (
    if not defined RUNTIME_RECOVERY_SCRIPT (
        echo [WARN] runtime_recovery.ps1를 찾지 못했습니다.
        if "%NON_INTERACTIVE%"=="1" (
            echo [FAILED] 비대화형 복구를 시작할 수 없습니다. setup_mfa.bat --install 로 설치를 실행해 주세요.
            exit /b 1
        )
        echo [INFO] 설치 모드로 전환합니다.
        set "WRAPPER_MODE=install"
        goto :wrapper_dispatch
    )
    echo [INFO] 복구 모드 실행: !RUNTIME_RECOVERY_SCRIPT!
    if "%NON_INTERACTIVE%"=="1" (
        if "%INSTALL_ML%"=="1" (
            powershell -NoProfile -ExecutionPolicy Bypass -File "!RUNTIME_RECOVERY_SCRIPT!" -SetupScriptPath "%SETUP_MFA_SELF%" -RuntimeRoot "%APP_DIR%" -Language korean -NonInteractive -WithMl
        ) else (
            powershell -NoProfile -ExecutionPolicy Bypass -File "!RUNTIME_RECOVERY_SCRIPT!" -SetupScriptPath "%SETUP_MFA_SELF%" -RuntimeRoot "%APP_DIR%" -Language korean -NonInteractive
        )
    ) else (
        if "%INSTALL_ML%"=="1" (
            powershell -NoProfile -ExecutionPolicy Bypass -File "!RUNTIME_RECOVERY_SCRIPT!" -SetupScriptPath "%SETUP_MFA_SELF%" -RuntimeRoot "%APP_DIR%" -Language korean -WithMl
        ) else (
            powershell -NoProfile -ExecutionPolicy Bypass -File "!RUNTIME_RECOVERY_SCRIPT!" -SetupScriptPath "%SETUP_MFA_SELF%" -RuntimeRoot "%APP_DIR%" -Language korean
        )
    )
    set "RECOVERY_RC=!ERRORLEVEL!"
    if not "!RECOVERY_RC!"=="0" (
        echo [FAILED] runtime_recovery.ps1 실행 실패 ^(code=!RECOVERY_RC!^)
    )
    exit /b !RECOVERY_RC!
)

echo [WARN] 알 수 없는 실행 모드입니다: %WRAPPER_MODE%
set "WRAPPER_MODE=install"
goto :wrapper_dispatch

:resolve_runtime_recovery_script
set "RUNTIME_RECOVERY_SCRIPT="
if exist "%APP_DIR%\runtime_recovery.ps1" (
    set "RUNTIME_RECOVERY_SCRIPT=%APP_DIR%\runtime_recovery.ps1"
)
if not defined RUNTIME_RECOVERY_SCRIPT if exist "%APP_DIR%\scripts\runtime_recovery.ps1" (
    set "RUNTIME_RECOVERY_SCRIPT=%APP_DIR%\scripts\runtime_recovery.ps1"
)
goto :eof

:select_start_mode
echo.
echo ====================================================
echo   UTAU Auto OTO - 실행 모드 선택
echo ====================================================
echo   [1] 기본 설치 ^(권장^)
echo   [2] 설치 + ML 의존성
echo   [3] 복구 메뉴 실행 ^(문제 진단/선택 복구^)
echo   [4] 자동 복구 ^(비대화형^)
echo   [0] 종료
choice /C 12340 /N /M "선택 ^(1/2/3/4/0^): "
if errorlevel 5 (
    set "WRAPPER_MODE=exit"
    goto :eof
)
if errorlevel 4 (
    set "WRAPPER_MODE=recovery"
    set "NON_INTERACTIVE=1"
    goto :eof
)
if errorlevel 3 (
    set "WRAPPER_MODE=recovery"
    set "NON_INTERACTIVE=0"
    goto :eof
)
if errorlevel 2 (
    set "WRAPPER_MODE=install"
    set "INSTALL_ML=1"
    goto :eof
)
if errorlevel 1 (
    set "WRAPPER_MODE=install"
    goto :eof
)
set "WRAPPER_MODE=install"
goto :eof

:preflight_install_tools
where powershell >nul 2>nul
if errorlevel 1 (
    echo [FAILED] powershell 명령을 찾지 못했습니다.
    exit /b 1
)
where certutil >nul 2>nul
if errorlevel 1 (
    echo [FAILED] certutil 명령을 찾지 못했습니다.
    exit /b 1
)
where tar >nul 2>nul
if errorlevel 1 (
    echo [FAILED] tar 명령을 찾지 못했습니다.
    exit /b 1
)
goto :eof

:direct_setup_start

echo ====================================================
echo   UTAU Auto OTO - MFA 경량 환경 설치
echo   이 스크립트는 최초 1회만 실행하면 됩니다.
echo ====================================================
echo.
echo [안내] 설치에는 인터넷 연결이 필요합니다.
echo        PC 환경에 따라 10~20분 정도 소요될 수 있습니다.
echo        설치 중에는 창을 닫지 말아 주세요.
echo        완료 후 MFA 정렬 기능을 바로 사용할 수 있습니다.
echo.

if not exist "%APP_DIR%" mkdir "%APP_DIR%" >nul 2>nul
pushd "%APP_DIR%" >nul 2>nul
set "OLD_PUBLIC_ROOT=%PUBLIC%"
if not defined OLD_PUBLIC_ROOT set "OLD_PUBLIC_ROOT=C:\Users\Public"
set "OLD_ENV_DIR=%OLD_PUBLIC_ROOT%\UTAU_Auto_OTO_v3\.env"
set "OLD_MICROMAMBA_ROOT=%OLD_PUBLIC_ROOT%\UTAU_Auto_OTO_v3\micromamba"
set "ENV_DIR=%APP_DIR%\.env"
set "MICROMAMBA_ROOT=%APP_DIR%\micromamba"
set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\Library\bin\micromamba.exe"
set "MICROMAMBA_ARCHIVE=%TEMP%\utau_auto_oto_micromamba-win-64-latest.tar.bz2"
set "MICROMAMBA_LATEST_URL=https://micro.mamba.pm/api/micromamba/win-64/latest"
set "MICROMAMBA_MD5_EXPECTED="
set "MICROMAMBA_MD5_ACTUAL="
set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
set "MFA_PYTHON_VERSION=3.10"

echo [INFO] MFA 환경 경로: %ENV_DIR%
echo [INFO] MFA Micromamba 경로: %MICROMAMBA_ROOT%

if exist "%APP_DIR%\ML_models" set "AUTO_ML=1"
if exist "%APP_DIR%\models_installed\oto_ml" set "AUTO_ML=1"
if "%AUTO_ML%"=="1" if "%INSTALL_ML%"=="0" (
    echo [INFO] ML 번들 자산이 감지되어 ML 의존성 설치를 활성화합니다.
    set "INSTALL_ML=1"
)
if exist "%OLD_ENV_DIR%" (
    if /i not "%OLD_ENV_DIR%"=="%ENV_DIR%" (
        call :handle_old_env
    )
)

if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.bat" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"

if exist "%MFA_EXE%" (
    call :get_env_python_version "%ENV_DIR%" MFA_ENV_PYTHON
    if defined MFA_ENV_PYTHON (
        echo [INFO] 기존 MFA Python 버전: !MFA_ENV_PYTHON!
        call :python_requires_rebuild "!MFA_ENV_PYTHON!" MFA_ENV_REBUILD
        if "!MFA_ENV_REBUILD!"=="1" (
            echo [WARN] Python !MFA_ENV_PYTHON! 버전은 Windows MFA 의존성 흐름과 호환되지 않습니다.
            echo        Python %MFA_PYTHON_VERSION% 기준으로 환경을 재구성합니다...
            call :remove_env_dir
            if errorlevel 1 exit /b 1
        )
    )
)

if exist "%MFA_EXE%" goto :existing_env_ready
goto :install_micromamba

:existing_env_ready
echo [OK] MFA가 이미 설치되어 있습니다.
echo      경로: %MFA_EXE%
echo.
call :bootstrap_python_tools
if errorlevel 1 exit /b 1
call :install_textgrid
if errorlevel 1 exit /b 1
call :verify_textgrid
if errorlevel 1 exit /b 1
call :install_audio_deps
if errorlevel 1 exit /b 1
call :install_korean_support
if errorlevel 1 exit /b 1
call :install_japanese_support
if errorlevel 1 exit /b 1
if "%INSTALL_ML%"=="1" (
    call :install_ml_requirements
    if errorlevel 1 exit /b 1
    call :verify_ml_runtime
    if errorlevel 1 exit /b 1
)
echo.
echo 한국어 음향 모델 확인 중...
call :download_acoustic_model korean_mfa
if errorlevel 1 exit /b 1
echo.
echo 일본어 음향 모델 확인 중...
call :download_acoustic_model japanese_mfa
if errorlevel 1 exit /b 1
call :cleanup_env_caches
if errorlevel 1 exit /b 1
call :cleanup_old_env_if_requested
echo.
echo 완료되었습니다. 이제 UTAU_Auto_OTO.exe를 실행할 수 있습니다.
pause
exit /b 0

:install_micromamba
echo [INFO] 설치 비용을 줄이기 위해 Micromamba 부트스트랩을 사용합니다.
echo.
echo [1/5] Micromamba 다운로드 중... ^(약 15MB^)
call :resolve_micromamba_expected_md5
if errorlevel 1 (
    echo [WARN] 신뢰 가능한 Micromamba 해시 메타데이터를 가져오지 못했습니다.
    if "%NON_INTERACTIVE%"=="1" (
        echo [FAILED] 비대화형 모드에서는 해시 검증 메타데이터가 필요합니다.
        exit /b 1
    )
    choice /C YN /N /M "신뢰 해시 메타데이터 없이 계속할까요? [Y/N]: "
    if errorlevel 2 (
        echo [FAILED] 사용자에 의해 중단되었습니다.
        exit /b 1
    )
)
if not exist "%MICROMAMBA_ARCHIVE%" (
    powershell -NoProfile -Command "& {$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MICROMAMBA_LATEST_URL%' -OutFile '%MICROMAMBA_ARCHIVE%'}"
    if errorlevel 1 (
        echo [FAILED] Micromamba 다운로드에 실패했습니다.
        pause
        exit /b 1
    )
) else (
    echo    이미 다운로드되어 있어 건너뜁니다.
)
call :verify_micromamba_archive
if errorlevel 1 (
    echo [FAILED] Micromamba 해시 검증에 실패했습니다.
    pause
    exit /b 1
)
echo [OK] 다운로드가 완료되었습니다.
echo.

echo [2/5] Micromamba 압축 해제 중...
if not exist "%MICROMAMBA_EXE%" (
    if exist "%MICROMAMBA_ROOT%" (
        echo [INFO] 기존 Micromamba 루트를 제거합니다...
        rmdir /s /q "%MICROMAMBA_ROOT%" >nul 2>nul
    )
    mkdir "%MICROMAMBA_ROOT%" >nul 2>nul
    tar -xjf "%MICROMAMBA_ARCHIVE%" -C "%MICROMAMBA_ROOT%"
    if errorlevel 1 (
        echo [FAILED] Micromamba 압축 해제에 실패했습니다.
        pause
        exit /b 1
    )
    if not exist "%MICROMAMBA_EXE%" if exist "%MICROMAMBA_ROOT%\bin\micromamba.exe" (
        set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\bin\micromamba.exe"
    )
    if not exist "%MICROMAMBA_EXE%" if exist "%MICROMAMBA_ROOT%\micromamba.exe" (
        set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\micromamba.exe"
    )
)
if not exist "%MICROMAMBA_EXE%" (
    echo [FAILED] 압축 해제 후 Micromamba 실행 파일을 찾지 못했습니다.
    pause
    exit /b 1
)
echo [OK] Micromamba 준비 완료.
echo.

echo [3/5] Montreal Forced Aligner 설치 중... ^(약 3~10분^)
if exist "%ENV_DIR%" if not exist "%MFA_EXE%" call :remove_env_dir
set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
"%MICROMAMBA_EXE%" create -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge python=%MFA_PYTHON_VERSION% montreal-forced-aligner colorama
if errorlevel 1 (
    echo [FAILED] MFA 설치에 실패했습니다.
    pause
    exit /b 1
)
call :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
call :bootstrap_python_tools
if errorlevel 1 exit /b 1
call :install_textgrid
if errorlevel 1 exit /b 1
call :verify_textgrid
if errorlevel 1 exit /b 1

echo [4/5] 한국어/일본어 토크나이저 의존성 설치 중...
call :install_korean_support
if errorlevel 1 exit /b 1
call :install_japanese_support
if errorlevel 1 exit /b 1
call :install_audio_deps
if errorlevel 1 exit /b 1
if "%INSTALL_ML%"=="1" (
    call :install_ml_requirements
    if errorlevel 1 exit /b 1
    call :verify_ml_runtime
    if errorlevel 1 exit /b 1
)

echo 설치 캐시 정리 중...
if exist "%MICROMAMBA_ARCHIVE%" del "%MICROMAMBA_ARCHIVE%" >nul 2>nul

echo [OK] MFA 설치가 완료되었습니다.
echo.
echo [최종] 한국어/일본어 음향 모델 다운로드 중... ^(약 1~2분^)
call :download_acoustic_model korean_mfa
if errorlevel 1 exit /b 1
call :download_acoustic_model japanese_mfa
if errorlevel 1 exit /b 1
call :cleanup_env_caches
if errorlevel 1 exit /b 1
call :cleanup_old_env_if_requested
echo.
echo ====================================================
echo   설치가 완료되었습니다.
echo   다음 단계:
echo   1) UTAU_Auto_OTO.exe 실행 ^(릴리즈 빌드^)
echo   2) 또는 run.bat 실행 ^(소스 체크아웃^)
echo   3) 이후 "3. Voice Alignment"를 눌러 진행
echo ====================================================
echo.
pause
exit /b 0

:bootstrap_python_tools
echo MFA Python 패키지 도구 점검 중...
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 런타임을 찾을 수 없습니다.
    pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import pip, pkg_resources, wheel" >nul 2>nul
if not errorlevel 1 (
    echo [OK] pip/setuptools/wheel이 준비되어 있습니다.
    goto :eof
)
echo [INFO] pip/setuptools/wheel 복구 중...
"%ENV_DIR%\python.exe" -m ensurepip --upgrade
if errorlevel 1 (
    echo [WARN] ensurepip이 완전히 끝나지 않았습니다. pip 복구를 계속 시도합니다...
)
if exist "%ENV_DIR%\Scripts\pip.exe" (
    "%ENV_DIR%\Scripts\pip.exe" install --upgrade "setuptools<81" wheel
) else (
    "%ENV_DIR%\python.exe" -m pip install --upgrade "setuptools<81" wheel
)
if errorlevel 1 (
    echo [FAILED] pip/setuptools/wheel 복구에 실패했습니다.
    pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import pip, pkg_resources, wheel" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] 복구 후에도 Python 패키지 도구를 사용할 수 없습니다.
    pause
    exit /b 1
)
echo [OK] pip/setuptools/wheel 복구 완료.
goto :eof

:install_japanese_support
echo 일본어 토크나이저 의존성 점검 중...
set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
call :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
if exist "%MICROMAMBA_EXE%" (
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge spacy sudachipy sudachidict-core
    if errorlevel 1 (
        echo [FAILED] 일본어 토크나이저 의존성 설치에 실패했습니다.
        pause
        exit /b 1
    )
    goto :eof
)
echo [WARN] Micromamba 명령을 찾지 못해 일본어 토크나이저 의존성 설치를 건너뜁니다.
goto :eof

:install_audio_deps
if exist "%ENV_DIR%\Library\bin\libsndfile.dll" goto :eof
echo 오디오 런타임 의존성 점검 중 ^(libsndfile^)...
if exist "%MICROMAMBA_EXE%" (
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge libsndfile pysoundfile
    if errorlevel 1 (
        echo [FAILED] 오디오 의존성 설치에 실패했습니다.
        pause
        exit /b 1
    )
    goto :eof
)
if exist "%ENV_DIR%\Scripts\conda.exe" (
    "%ENV_DIR%\Scripts\conda.exe" install -y --solver classic -p "%ENV_DIR%" -c conda-forge libsndfile pysoundfile
    if errorlevel 1 (
        echo [FAILED] 오디오 의존성 설치에 실패했습니다.
        pause
        exit /b 1
    )
    goto :eof
)
echo [WARN] micromamba/conda를 찾지 못해 오디오 의존성 설치를 건너뜁니다.
goto :eof

:install_textgrid
echo textgrid 모듈 점검 중...
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 런타임을 찾을 수 없습니다.
    pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import textgrid" >nul 2>nul
if not errorlevel 1 goto :eof
echo [INFO] textgrid 모듈 설치 중...
if exist "%ENV_DIR%\Scripts\pip.exe" (
    "%ENV_DIR%\Scripts\pip.exe" install --upgrade "textgrid>=1.5"
) else (
    "%ENV_DIR%\python.exe" -m pip install --upgrade "textgrid>=1.5"
)
if errorlevel 1 (
    echo [FAILED] textgrid 설치에 실패했습니다.
    pause
    exit /b 1
)
goto :eof

:verify_textgrid
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 런타임을 찾을 수 없습니다.
    pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import textgrid" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] textgrid import 검사에 실패했습니다.
    echo        환경 복구를 위해 setup_mfa.bat를 다시 실행해 주세요.
    pause
    exit /b 1
)
goto :eof

:install_ml_requirements
echo 선택 ML 의존성 설치 중...
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 런타임을 찾을 수 없습니다.
    pause
    exit /b 1
)
if not exist "%APP_DIR%\requirements.txt" (
    echo [FAILED] %APP_DIR%에서 requirements.txt를 찾을 수 없습니다.
    pause
    exit /b 1
)
if not exist "%APP_DIR%\requirements-ml.txt" (
    echo [FAILED] %APP_DIR%에서 requirements-ml.txt를 찾을 수 없습니다.
    pause
    exit /b 1
)
if exist "%MICROMAMBA_EXE%" (
    echo [INFO] micromamba로 ML 런타임 패키지 설치 중...
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge pandas lightgbm onnxruntime
    if errorlevel 1 (
        echo [WARN] micromamba ML 설치에 실패했습니다. pip로 폴백합니다.
    ) else (
        echo [OK] micromamba ML 패키지 설치 완료.
    )
)
echo [INFO] requirements.txt 설치 중
if exist "%ENV_DIR%\Scripts\pip.exe" (
    "%ENV_DIR%\Scripts\pip.exe" install --upgrade -r "%APP_DIR%\requirements.txt"
) else (
    "%ENV_DIR%\python.exe" -m pip install --upgrade -r "%APP_DIR%\requirements.txt"
)
if errorlevel 1 (
    echo [FAILED] requirements.txt 설치에 실패했습니다.
    pause
    exit /b 1
)
if not exist "%MICROMAMBA_EXE%" (
    echo [INFO] pip로 requirements-ml.txt 설치 중 ^(micromamba를 찾지 못함^)
    if exist "%ENV_DIR%\Scripts\pip.exe" (
        "%ENV_DIR%\Scripts\pip.exe" install --upgrade -r "%APP_DIR%\requirements-ml.txt"
    ) else (
        "%ENV_DIR%\python.exe" -m pip install --upgrade -r "%APP_DIR%\requirements-ml.txt"
    )
    if errorlevel 1 (
        echo [FAILED] ML 의존성 설치에 실패했습니다.
        echo        lightgbm 빌드를 위해 Microsoft Visual C++ Build Tools가 필요할 수 있습니다.
        pause
        exit /b 1
    )
)
echo [OK] ML 런타임 의존성 설치 완료.
goto :eof

:verify_ml_runtime
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 런타임을 찾을 수 없습니다.
    pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import pandas, lightgbm, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] ML 런타임 import 검사에 실패했습니다. pandas/lightgbm/onnxruntime가 필요합니다.
    if exist "%MICROMAMBA_EXE%" (
        echo        시도: "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge pandas lightgbm onnxruntime
    ) else (
        echo        setup_mfa.bat --with-ml 을 다시 실행해 주세요.
    )
    pause
    exit /b 1
)
goto :eof

:cleanup_env_caches
echo 최종 설치 용량 축소를 위해 패키지 캐시 정리 중...
if exist "%MICROMAMBA_EXE%" (
    "%MICROMAMBA_EXE%" clean -a -y -r "%MICROMAMBA_ROOT%" >nul 2>nul
)
if exist "%ENV_DIR%\python.exe" (
    "%ENV_DIR%\python.exe" -m pip cache purge >nul 2>nul
)
echo [OK] 캐시 정리 완료.
goto :eof

:install_korean_support
echo 한국어 토크나이저 의존성 점검 중...
call :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 런타임을 찾을 수 없습니다.
    pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import eunjeon, jamo" >nul 2>nul
if not errorlevel 1 goto :patch_korean_support
echo [INFO] 한국어 토크나이저 의존성 ^(eunjeon, jamo^) 설치 중...
if exist "%ENV_DIR%\Scripts\pip.exe" (
    "%ENV_DIR%\Scripts\pip.exe" install --upgrade eunjeon jamo
) else (
    "%ENV_DIR%\python.exe" -m pip install --upgrade eunjeon jamo
)
if errorlevel 1 (
    echo [FAILED] 한국어 토크나이저 의존성 설치에 실패했습니다.
    echo        eunjeon 빌드를 위해 Microsoft Visual C++ Build Tools가 필요할 수 있습니다.
    pause
    exit /b 1
)
:patch_korean_support
set "PYTHONPATH=%APP_DIR%"
"%ENV_DIR%\python.exe" -c "from core.mfa_runner import patch_mfa_korean_support; patch_mfa_korean_support(r'%MFA_EXE%')" >nul 2>nul
if errorlevel 1 (
    echo [WARN] 한국어 MFA 패치 단계에 실패했습니다. 정렬은 동작할 수 있으나 한국어 토크나이저 지원이 불완전할 수 있습니다.
)
goto :eof

:resolve_micromamba_expected_md5
set "MICROMAMBA_MD5_EXPECTED="
set "MICROMAMBA_MD5_FILE=%TEMP%\utau_auto_oto_micromamba_expected_md5.txt"
if exist "%MICROMAMBA_MD5_FILE%" del "%MICROMAMBA_MD5_FILE%" >nul 2>nul
powershell -NoProfile -Command ^
 "$ErrorActionPreference='Stop';" ^
 " [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;" ^
 " $latest='%MICROMAMBA_LATEST_URL%';" ^
 " $resp=$null;" ^
 " $redirect=$null;" ^
 " try { $resp=Invoke-WebRequest -Uri $latest -MaximumRedirection 0 -UseBasicParsing -TimeoutSec 30 } catch { if ($_.Exception.Response) { $resp=$_.Exception.Response } };" ^
 " if ($resp -and $resp.Headers) { $redirect=$resp.Headers['Location'] };" ^
 " if (-not $redirect) { throw 'redirect location not found' };" ^
 " if ($redirect.StartsWith('//')) { $redirect='https:' + $redirect };" ^
 " $leaf=Split-Path $redirect -Leaf;" ^
 " $target='win-64/' + $leaf;" ^
 " $files=Invoke-RestMethod -Uri 'https://api.anaconda.org/package/conda-forge/micromamba/files' -TimeoutSec 60;" ^
 " $hit=$files | Where-Object { $_.basename -eq $target } | Select-Object -First 1;" ^
 " if (-not $hit) { throw ('metadata not found: ' + $target) };" ^
 " if (-not $hit.md5) { throw 'metadata md5 is empty' };" ^
 " Set-Content -LiteralPath '%MICROMAMBA_MD5_FILE%' -Value ($hit.md5.ToLowerInvariant()) -Encoding ASCII;"
if errorlevel 1 (
    if exist "%MICROMAMBA_MD5_FILE%" del "%MICROMAMBA_MD5_FILE%" >nul 2>nul
    exit /b 1
)
if not exist "%MICROMAMBA_MD5_FILE%" exit /b 1
set /p MICROMAMBA_MD5_EXPECTED=<"%MICROMAMBA_MD5_FILE%"
if exist "%MICROMAMBA_MD5_FILE%" del "%MICROMAMBA_MD5_FILE%" >nul 2>nul
if not defined MICROMAMBA_MD5_EXPECTED exit /b 1
echo [INFO] 신뢰 Micromamba MD5 메타데이터: %MICROMAMBA_MD5_EXPECTED%
exit /b 0

:verify_micromamba_archive
if not exist "%MICROMAMBA_ARCHIVE%" (
    echo [FAILED] Micromamba 아카이브 파일이 없습니다: %MICROMAMBA_ARCHIVE%
    exit /b 1
)
set "MICROMAMBA_MD5_ACTUAL="
for /f "tokens=1" %%h in ('certutil -hashfile "%MICROMAMBA_ARCHIVE%" MD5 ^| findstr /R /I "^[0-9A-F][0-9A-F]"') do (
    if not defined MICROMAMBA_MD5_ACTUAL set "MICROMAMBA_MD5_ACTUAL=%%h"
)
if not defined MICROMAMBA_MD5_ACTUAL (
    echo [FAILED] Micromamba MD5 해시를 계산하지 못했습니다.
    exit /b 1
)
echo [INFO] Micromamba 아카이브 MD5: %MICROMAMBA_MD5_ACTUAL%
if defined MICROMAMBA_MD5_EXPECTED (
    if /I not "%MICROMAMBA_MD5_ACTUAL%"=="%MICROMAMBA_MD5_EXPECTED%" (
        echo [FAILED] Micromamba 해시 불일치가 감지되었습니다.
        echo         기대값: %MICROMAMBA_MD5_EXPECTED%
        echo         실제값: %MICROMAMBA_MD5_ACTUAL%
        del "%MICROMAMBA_ARCHIVE%" >nul 2>nul
        exit /b 1
    )
    echo [OK] Micromamba 해시 검증 통과.
) else (
    echo [WARN] 신뢰 Micromamba 해시 메타데이터를 사용할 수 없습니다.
    if "%NON_INTERACTIVE%"=="1" exit /b 1
    choice /C YN /N /M "검증된 Micromamba 해시 없이 계속할까요? [Y/N]: "
    if errorlevel 2 exit /b 1
)
exit /b 0

:handle_old_env
echo.
echo [WARN] 레거시 MFA 환경이 감지되었습니다:
echo        %OLD_ENV_DIR%
echo.
echo 레거시 환경 처리 방법을 선택하세요:
echo   [M] 마이그레이션 ^(로컬 환경 재구성 후 기존 환경 삭제^)
echo   [D] 기존 환경을 지금 바로 삭제
echo   [K] 기존 환경 유지 ^(삭제하지 않음^)
choice /C MDK /N /M "선택 ^(M/D/K^): "
if errorlevel 3 goto :keep_old_env
if errorlevel 2 goto :delete_old_env_now
if errorlevel 1 goto :migrate_old_env
goto :eof

:migrate_old_env
set "DELETE_OLD_AFTER_INSTALL=1"
echo [INFO] 로컬 설치가 성공하면 레거시 환경을 삭제합니다.
goto :eof

:delete_old_env_now
call :remove_dir "%OLD_ENV_DIR%"
call :remove_dir "%OLD_MICROMAMBA_ROOT%"
goto :eof

:keep_old_env
echo [INFO] 레거시 환경을 유지한 채 로컬 설치를 계속합니다.
goto :eof

:cleanup_old_env_if_requested
if not "%DELETE_OLD_AFTER_INSTALL%"=="1" goto :eof
if /i "%OLD_ENV_DIR%"=="%ENV_DIR%" goto :eof
call :remove_dir "%OLD_ENV_DIR%"
call :remove_dir "%OLD_MICROMAMBA_ROOT%"
echo [OK] 레거시 MFA 환경을 제거했습니다.
goto :eof

:remove_dir
set "TARGET_DIR=%~1"
if "%TARGET_DIR%"=="" goto :eof
if not exist "%TARGET_DIR%" goto :eof
echo [INFO] 제거 중: %TARGET_DIR%
rmdir /s /q "%TARGET_DIR%" >nul 2>nul
if exist "%TARGET_DIR%" (
    echo [WARN] 제거 실패: %TARGET_DIR%
)
goto :eof

:ensure_mfa_entrypoint
if exist "%MFA_EXE%" goto :eof
if exist "%ENV_DIR%\Scripts\mfa.bat" (
    set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"
    goto :eof
)
if exist "%ENV_DIR%\Scripts\mfa.cmd" (
    set "MFA_EXE=%ENV_DIR%\Scripts\mfa.cmd"
    goto :eof
)
if exist "%MICROMAMBA_EXE%" (
    if not exist "%ENV_DIR%\Scripts" mkdir "%ENV_DIR%\Scripts" >nul 2>nul
    >"%ENV_DIR%\Scripts\mfa.bat" echo @echo off
    >>"%ENV_DIR%\Scripts\mfa.bat" echo set "CONDA_PREFIX=%ENV_DIR%"
    >>"%ENV_DIR%\Scripts\mfa.bat" echo set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%%PATH%%"
    >>"%ENV_DIR%\Scripts\mfa.bat" echo "%ENV_DIR%\python.exe" -m montreal_forced_aligner.command_line.mfa %%*
    set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"
    goto :eof
)
echo [FAILED] 환경 생성 후 MFA 실행 파일을 찾지 못했습니다.
pause
exit /b 1

:download_acoustic_model
set "MODEL_NAME=%~1"
if not defined MODEL_NAME (
    echo [FAILED] 음향 모델 이름이 비어 있습니다.
    pause
    exit /b 1
)
if exist "%ENV_DIR%\python.exe" (
    set "CONDA_PREFIX=%ENV_DIR%"
    set "MFA_ROOT_DIR=%APP_DIR%\.mfa_root_ascii"
    set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%PATH%"
    "%ENV_DIR%\python.exe" -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME% --ignore_cache
    exit /b %errorlevel%
)
if exist "%MICROMAMBA_EXE%" (
    set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
    set "MFA_ROOT_DIR=%APP_DIR%\.mfa_root_ascii"
    "%MICROMAMBA_EXE%" run -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" python -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME% --ignore_cache
    exit /b %errorlevel%
)
echo [FAILED] 모델 다운로드에 사용할 실행 가능한 MFA 환경을 찾지 못했습니다.
pause
exit /b 1

:remove_env_dir
if not exist "%ENV_DIR%" goto :eof
echo [INFO] 기존 환경 제거 중: %ENV_DIR%
rmdir /s /q "%ENV_DIR%" >nul 2>nul
if exist "%ENV_DIR%" (
    echo [FAILED] 기존 MFA 환경을 제거하지 못했습니다.
    pause
    exit /b 1
)
goto :eof

:get_env_python_version
set "%~2="
if exist "%~1\python.exe" (
    for /f "usebackq delims=" %%i in (`"%~1\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2^>nul`) do set "%~2=%%i"
    goto :eof
)
if exist "%~1\Scripts\python.exe" (
    for /f "usebackq delims=" %%i in (`"%~1\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2^>nul`) do set "%~2=%%i"
)
goto :eof

:python_requires_rebuild
set "%~2=0"
set "PY_CHECK=%~1"
set "PY_MAJOR="
set "PY_MINOR="
for /f "tokens=1,2 delims=." %%a in ("%PY_CHECK%") do (
    set /a PY_MAJOR=%%a
    set /a PY_MINOR=%%b
)
if not defined PY_MAJOR goto :eof
if %PY_MAJOR% GTR 3 (
    set "%~2=1"
    goto :eof
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% GEQ 13 set "%~2=1"
goto :eof
