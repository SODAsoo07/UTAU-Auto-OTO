@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title UTAU Auto OTO - MFA Setup

if /i "%~1"=="--help" (
    echo Usage: setup_mfa.bat
    echo Installs a shared MFA environment in C:\Users\Public\UTAU_Auto_OTO_v3\.env using Micromamba.
    exit /b 0
)

echo ====================================================
echo   UTAU Auto OTO - MFA Lightweight Environment Setup
echo   This script only needs to run once.
echo ====================================================
echo.
echo [안내] 처음 설치 또는 복구에는 시간이 오래 걸릴 수 있습니다.
echo        네트워크 속도와 PC 성능에 따라 10-20분, 경우에 따라 그 이상 걸릴 수 있습니다.
echo        설치 중에는 창을 닫지 말고 기다려 주세요.
echo        설치 후에는 현재 언어용 MFA 모델 다운로드가 이어질 수 있습니다.
echo.

set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
set "PUBLIC_ROOT=%PUBLIC%"
if not defined PUBLIC_ROOT set "PUBLIC_ROOT=C:\Users\Public"
set "ENV_DIR=%PUBLIC_ROOT%\UTAU_Auto_OTO_v3\.env"
set "MICROMAMBA_ROOT=%PUBLIC_ROOT%\UTAU_Auto_OTO_v3\micromamba"
set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\Library\bin\micromamba.exe"
set "MICROMAMBA_ARCHIVE=%APP_DIR%\micromamba-win-64-latest.tar.bz2"
set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
set "MFA_PYTHON_VERSION=3.10"

if not exist "%PUBLIC_ROOT%\UTAU_Auto_OTO_v3" mkdir "%PUBLIC_ROOT%\UTAU_Auto_OTO_v3" >nul 2>nul
echo [INFO] MFA environment path: %ENV_DIR%
echo [INFO] MFA Micromamba path: %MICROMAMBA_ROOT%

if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.bat" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"

if exist "%MFA_EXE%" (
    call :get_env_python_version "%ENV_DIR%" MFA_ENV_PYTHON
    if defined MFA_ENV_PYTHON (
        echo [INFO] Existing MFA Python version: !MFA_ENV_PYTHON!
        call :python_requires_rebuild "!MFA_ENV_PYTHON!" MFA_ENV_REBUILD
        if "!MFA_ENV_REBUILD!"=="1" (
            echo [WARN] Python !MFA_ENV_PYTHON! is not compatible with the Windows MFA dependency flow.
            echo        Rebuilding the environment with Python %MFA_PYTHON_VERSION%...
            call :remove_env_dir
            if errorlevel 1 exit /b 1
        )
    )
)

if exist "%MFA_EXE%" goto :existing_env_ready
goto :install_micromamba

:existing_env_ready
echo [OK] MFA is already installed.
echo      Path: %MFA_EXE%
echo.
call :bootstrap_python_tools
if errorlevel 1 exit /b 1
call :install_japanese_support
if errorlevel 1 exit /b 1
echo.
echo Checking Korean acoustic model...
call :download_acoustic_model korean_mfa
if errorlevel 1 exit /b 1
echo.
echo Done. You can now launch UTAU_Auto_OTO.exe.
pause
exit /b 0

:install_micromamba
echo [INFO] Using Micromamba bootstrap to reduce installation cost.
echo.
echo [1/4] Downloading Micromamba... ^(about 15MB^)
if not exist "%MICROMAMBA_ARCHIVE%" (
    powershell -NoProfile -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' -OutFile '%MICROMAMBA_ARCHIVE%'}"
    if errorlevel 1 (
        echo [FAILED] Micromamba download failed.
        pause
        exit /b 1
    )
) else (
    echo    Already downloaded. Skipping.
)
echo [OK] Download complete.
echo.

echo [2/4] Extracting Micromamba...
if not exist "%MICROMAMBA_EXE%" (
    if exist "%MICROMAMBA_ROOT%" (
        echo [INFO] Removing previous Micromamba root...
        rmdir /s /q "%MICROMAMBA_ROOT%" >nul 2>nul
    )
    mkdir "%MICROMAMBA_ROOT%" >nul 2>nul
    tar -xjf "%MICROMAMBA_ARCHIVE%" -C "%MICROMAMBA_ROOT%"
    if errorlevel 1 (
        echo [FAILED] Micromamba extraction failed.
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
    echo [FAILED] Micromamba executable was not found after extraction.
    pause
    exit /b 1
)
echo [OK] Micromamba ready.
echo.

echo [3/4] Installing Montreal Forced Aligner... ^(3-10 min^)
if exist "%ENV_DIR%" if not exist "%MFA_EXE%" call :remove_env_dir
set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
"%MICROMAMBA_EXE%" create -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge python=%MFA_PYTHON_VERSION% montreal-forced-aligner colorama
if errorlevel 1 (
    echo [FAILED] MFA install failed.
    pause
    exit /b 1
)
call :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
call :bootstrap_python_tools
if errorlevel 1 exit /b 1

echo [4/4] Installing Japanese tokenizer dependencies...
call :install_japanese_support
if errorlevel 1 exit /b 1

echo Cleaning up installer cache...
if exist "%MICROMAMBA_ARCHIVE%" del "%MICROMAMBA_ARCHIVE%" >nul 2>nul

echo [OK] MFA installed successfully.
echo.
echo [Final] Downloading Korean acoustic model... ^(1-2 min^)
call :download_acoustic_model korean_mfa
if errorlevel 1 exit /b 1
echo.
echo ====================================================
echo   Setup complete.
echo   Next step:
echo   1) Launch UTAU_Auto_OTO.exe ^(release build^)
echo   2) Or run run.bat ^(source checkout^)
echo   3) Then click "3. Voice Alignment" to continue
echo ====================================================
echo.
pause
exit /b 0

:bootstrap_python_tools
echo Checking MFA Python package tools...
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python runtime was not found.
    pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import pip, pkg_resources, wheel" >nul 2>nul
if not errorlevel 1 (
    echo [OK] pip/setuptools/wheel are ready.
    goto :eof
)
echo [INFO] Repairing pip/setuptools/wheel...
"%ENV_DIR%\python.exe" -m ensurepip --upgrade
if errorlevel 1 (
    echo [WARN] ensurepip did not complete cleanly. Trying pip repair anyway...
)
if exist "%ENV_DIR%\Scripts\pip.exe" (
    "%ENV_DIR%\Scripts\pip.exe" install --upgrade "setuptools<81" wheel
) else (
    "%ENV_DIR%\python.exe" -m pip install --upgrade "setuptools<81" wheel
)
if errorlevel 1 (
    echo [FAILED] pip/setuptools/wheel repair failed.
    pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import pip, pkg_resources, wheel" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] Python package tools are still unavailable after repair.
    pause
    exit /b 1
)
echo [OK] pip/setuptools/wheel repair complete.
goto :eof

:install_japanese_support
echo Checking Japanese tokenizer dependencies...
set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
call :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
if exist "%MICROMAMBA_EXE%" (
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge spacy sudachipy sudachidict-core
    if errorlevel 1 (
        echo [FAILED] Japanese tokenizer dependency install failed.
        pause
        exit /b 1
    )
    goto :eof
)
echo [WARN] Micromamba command was not found. Skipping Japanese tokenizer dependency install.
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
echo [FAILED] MFA executable was not found after environment creation.
pause
exit /b 1

:download_acoustic_model
set "MODEL_NAME=%~1"
if not defined MODEL_NAME (
    echo [FAILED] Acoustic model name is missing.
    pause
    exit /b 1
)
if exist "%ENV_DIR%\python.exe" (
    set "CONDA_PREFIX=%ENV_DIR%"
    set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%PATH%"
    "%ENV_DIR%\python.exe" -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME% --ignore_cache
    exit /b %errorlevel%
)
if exist "%MICROMAMBA_EXE%" (
    set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
    "%MICROMAMBA_EXE%" run -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" python -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME% --ignore_cache
    exit /b %errorlevel%
)
echo [FAILED] No runnable MFA environment was found for model download.
pause
exit /b 1

:remove_env_dir
if not exist "%ENV_DIR%" goto :eof
echo [INFO] Removing existing env: %ENV_DIR%
rmdir /s /q "%ENV_DIR%" >nul 2>nul
if exist "%ENV_DIR%" (
    echo [FAILED] Could not remove the old MFA environment.
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
