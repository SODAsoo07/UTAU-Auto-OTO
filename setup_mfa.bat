@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title UTAU Auto OTO - MFA Setup

if /i "%~1"=="--help" (
    echo Usage: setup_mfa.bat
    echo Installs MFA portable environment at app folder .env ^(or public fallback when needed^).
    exit /b 0
)

echo ====================================================
echo   UTAU Auto OTO - MFA Portable Environment Setup
echo   This script only needs to run ONCE.
echo ====================================================
echo.

set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
set "PUBLIC_ROOT=%PUBLIC%"
if not defined PUBLIC_ROOT set "PUBLIC_ROOT=C:\Users\Public"
set "ENV_DIR=%APP_DIR%\.env"
set "INSTALLER=%APP_DIR%\Miniconda3-latest-Windows-x86_64.exe"
set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"

REM === Non-ASCII app path fallback ===
set "APP_DIR_NONASCII=0"
for /f %%i in ('powershell -NoProfile -Command "$p=$env:APP_DIR; if($p -match '[^\x00-\x7F]'){'1'} else {'0'}"') do set "APP_DIR_NONASCII=%%i"
if "!APP_DIR_NONASCII!"=="1" (
    set "ENV_DIR=%PUBLIC_ROOT%\UTAU_Auto_OTO_v3\.env"
    set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
    echo [WARN] APP path contains non-ASCII characters.
    echo        Using fallback ENV path: %ENV_DIR%
    if not exist "%PUBLIC_ROOT%\UTAU_Auto_OTO_v3" mkdir "%PUBLIC_ROOT%\UTAU_Auto_OTO_v3" >nul 2>nul
)

REM === Check for existing Conda ===
set "SYSTEM_CONDA="
where conda >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where conda') do (
        set "SYSTEM_CONDA=%%i"
        goto :found_conda
    )
)
:found_conda

REM === Already installed? ===
if exist "%MFA_EXE%" (
    echo [OK] MFA is already installed!
    echo      Path: %MFA_EXE%
    echo.
    echo Checking dependencies...
    if exist "%ENV_DIR%\Scripts\conda.exe" (
        "%ENV_DIR%\Scripts\conda.exe" run -p "%ENV_DIR%" pip install --no-cache-dir eunjeon jamo textgrid
        if errorlevel 1 (
            echo [FAILED] Extra Python dependency install failed.
            echo [HINT] If the log mentions "Microsoft Visual C++ 14.0 or greater is required",
            echo        install C++ Build Tools from:
            echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
            pause
            exit /b 1
        )
        "%ENV_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels spacy sudachipy sudachidict-core
        if errorlevel 1 (
            echo [FAILED] Japanese tokenizer dependency install failed.
            pause
            exit /b 1
        )
    ) else if defined SYSTEM_CONDA (
        "%SYSTEM_CONDA%" run -p "%ENV_DIR%" pip install --no-cache-dir eunjeon jamo textgrid
        if errorlevel 1 (
            echo [FAILED] Extra Python dependency install failed.
            echo [HINT] If the log mentions "Microsoft Visual C++ 14.0 or greater is required",
            echo        install C++ Build Tools from:
            echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
            pause
            exit /b 1
        )
        "%SYSTEM_CONDA%" install -y -p "%ENV_DIR%" -c conda-forge --override-channels spacy sudachipy sudachidict-core
        if errorlevel 1 (
            echo [FAILED] Japanese tokenizer dependency install failed.
            pause
            exit /b 1
        )
    )
    echo.
    echo Checking Korean model...
    "%MFA_EXE%" model download acoustic korean_mfa --ignore_cache
    echo.
    echo Done! You can now run main.py.
    pause
    exit /b 0
)

if defined SYSTEM_CONDA (
    echo [INFO] Existing Conda found: %SYSTEM_CONDA%
    echo        Skipping Miniconda download.
    echo.
    echo [1/4] Creating local environment and installing MFA... ^(5-10 min^)
    "%SYSTEM_CONDA%" create -y -p "%ENV_DIR%" -c conda-forge --override-channels montreal-forced-aligner colorama
    if errorlevel 1 (
        echo [FAILED] MFA install failed.
        pause
        exit /b 1
    )
    
    echo [2/4] Installing extra Python dependencies...
    "%SYSTEM_CONDA%" run -p "%ENV_DIR%" pip install --no-cache-dir eunjeon jamo textgrid
    if errorlevel 1 (
        echo [FAILED] Extra Python dependency install failed.
        echo [HINT] If the log mentions "Microsoft Visual C++ 14.0 or greater is required",
        echo        install C++ Build Tools from:
        echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
        pause
        exit /b 1
    )

    echo [3/4] Installing Japanese tokenizer dependencies...
    "%SYSTEM_CONDA%" install -y -p "%ENV_DIR%" -c conda-forge --override-channels spacy sudachipy sudachidict-core
    if errorlevel 1 (
        echo [FAILED] Japanese tokenizer dependency install failed.
        pause
        exit /b 1
    )
    
) else (
    echo [INFO] Conda not found. Proceeding with portable Miniconda install.
    echo.
    echo [1/4] Downloading Miniconda... ^(about 80MB^)
    if not exist "%INSTALLER%" (
        powershell -NoProfile -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%INSTALLER%'}"
        if errorlevel 1 (
            echo [FAILED] Miniconda download failed.
            pause
            exit /b 1
        )
    ) else (
        echo    Already downloaded. Skipping.
    )
    echo [OK] Download complete!
    echo.

    echo [2/4] Installing Miniconda as portable env... ^(2-5 min^)
    if not exist "%ENV_DIR%\Scripts\conda.exe" (
        start /wait "" "%INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%ENV_DIR%
    )
    if not exist "%ENV_DIR%\Scripts\conda.exe" (
        echo [WARN] Conda not found at requested path. Checking default locations...
        set "FOUND_CONDA="
        for %%p in ("%USERPROFILE%\miniconda3\Scripts\conda.exe" "%USERPROFILE%\Miniconda3\Scripts\conda.exe" "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" "%LOCALAPPDATA%\Miniconda3\Scripts\conda.exe") do (
            if not defined FOUND_CONDA if exist "%%~p" set "FOUND_CONDA=%%~p"
        )
        if defined FOUND_CONDA (
            for %%d in ("!FOUND_CONDA!\..\..") do set "ENV_DIR=%%~fd"
            set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
            echo [INFO] Using detected Conda path: !FOUND_CONDA!
        ) else (
            echo [FAILED] Miniconda install failed.
            pause
            exit /b 1
        )
    )
    echo [OK] Miniconda installed!
    echo.

    echo [3/4] Installing Montreal Forced Aligner...
    "%ENV_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels montreal-forced-aligner colorama
    if errorlevel 1 (
        echo [FAILED] MFA install failed.
        pause
        exit /b 1
    )
    
    echo [3.5/4] Installing extra Python dependencies...
    "%ENV_DIR%\Scripts\conda.exe" run -p "%ENV_DIR%" pip install --no-cache-dir eunjeon jamo textgrid
    if errorlevel 1 (
        echo [FAILED] Extra Python dependency install failed.
        echo [HINT] If the log mentions "Microsoft Visual C++ 14.0 or greater is required",
        echo        install C++ Build Tools from:
        echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
        pause
        exit /b 1
    )

    echo [3.8/4] Installing Japanese tokenizer dependencies...
    "%ENV_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels spacy sudachipy sudachidict-core
    if errorlevel 1 (
        echo [FAILED] Japanese tokenizer dependency install failed.
        pause
        exit /b 1
    )
    
    echo Cleaning up installer...
    if exist "%INSTALLER%" del "%INSTALLER%" >nul 2>nul
)

echo [OK] MFA installed successfully!
echo.

echo [Final] Downloading Korean acoustic model... ^(1-2 min^)
if exist "%MFA_EXE%" (
    "%MFA_EXE%" model download acoustic korean_mfa --ignore_cache
) else (
    echo [WARNING] mfa.exe not found at %MFA_EXE%. Model download skipped.
)
echo.

echo ====================================================
echo   Setup complete!
echo   Next step:
echo   1) Launch UTAU_Auto_OTO.exe ^(배포본 사용 시^)
echo   2) Or run run.bat ^(소스 폴더 실행 시^)
echo   3) Then click "3. 음성 정렬" to continue
echo ====================================================
echo.
pause
exit /b 0
