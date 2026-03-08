@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title UTAU Auto OTO - MFA Setup

if /i "%~1"=="--help" (
    echo Usage: setup_mfa.bat
    echo Installs a shared MFA environment in C:\Users\Public\UTAU_Auto_OTO_v3\.env.
    exit /b 0
)

echo ====================================================
echo   UTAU Auto OTO - MFA Portable Environment Setup
echo   This script only needs to run once.
echo ====================================================
echo.
echo [INFO] Initial setup can take a long time.
echo        Depending on network speed and PC performance, it may take 10-20 minutes or longer.
echo        Please do not close the installer while it is working.
echo.

set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
set "PUBLIC_ROOT=%PUBLIC%"
if not defined PUBLIC_ROOT set "PUBLIC_ROOT=C:\Users\Public"
set "ENV_DIR=%PUBLIC_ROOT%\UTAU_Auto_OTO_v3\.env"
set "INSTALLER=%APP_DIR%\Miniconda3-latest-Windows-x86_64.exe"
set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
set "MFA_PYTHON_VERSION=3.10"

set "APP_DIR_NONASCII=0"
for /f %%i in ('powershell -NoProfile -Command "$p=$env:APP_DIR; if($p -match '[^\x00-\x7F]'){'1'} else {'0'}"') do set "APP_DIR_NONASCII=%%i"
if not exist "%PUBLIC_ROOT%\UTAU_Auto_OTO_v3" mkdir "%PUBLIC_ROOT%\UTAU_Auto_OTO_v3" >nul 2>nul
echo [INFO] MFA environment path: %ENV_DIR%
if "!APP_DIR_NONASCII!"=="1" (
    echo [INFO] App path contains non-ASCII characters. Shared MFA env will still be used.
)

set "SYSTEM_CONDA="
where conda >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where conda') do (
        set "SYSTEM_CONDA=%%i"
        goto :found_conda
    )
)
:found_conda

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
if defined SYSTEM_CONDA goto :install_with_system_conda
goto :install_portable_miniconda

:existing_env_ready
echo [OK] MFA is already installed.
echo      Path: %MFA_EXE%
echo.
call :install_japanese_support
if errorlevel 1 exit /b 1
echo.
echo Checking Korean acoustic model...
"%MFA_EXE%" model download acoustic korean_mfa --ignore_cache
echo.
echo Done. You can now launch UTAU_Auto_OTO.exe.
pause
exit /b 0

:install_with_system_conda
echo [INFO] Existing Conda found: %SYSTEM_CONDA%
echo        Skipping Miniconda download.
echo.
echo [1/3] Creating local MFA environment... ^(5-10 min^)
"%SYSTEM_CONDA%" create -y -p "%ENV_DIR%" -c conda-forge --override-channels python=%MFA_PYTHON_VERSION% montreal-forced-aligner colorama
if errorlevel 1 (
    echo [FAILED] MFA install failed.
    pause
    exit /b 1
)
echo [2/3] Installing Japanese tokenizer dependencies...
call :install_japanese_support
if errorlevel 1 exit /b 1
goto :install_done

:install_portable_miniconda
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
echo [OK] Download complete.
echo.
echo [2/4] Installing Miniconda as portable env... ^(2-5 min^)
if not exist "%ENV_DIR%\Scripts\conda.exe" (
    start /wait "" "%INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%ENV_DIR%
)
if not exist "%ENV_DIR%\Scripts\conda.exe" (
    echo [WARN] Conda was not found at the requested path. Checking default locations...
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
echo [OK] Miniconda installed.
echo.
echo [3/4] Installing Montreal Forced Aligner...
"%ENV_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels python=%MFA_PYTHON_VERSION% montreal-forced-aligner colorama
if errorlevel 1 (
    echo [FAILED] MFA install failed.
    pause
    exit /b 1
)
echo [4/4] Installing Japanese tokenizer dependencies...
call :install_japanese_support
if errorlevel 1 exit /b 1
echo Cleaning up installer...
if exist "%INSTALLER%" del "%INSTALLER%" >nul 2>nul

:install_done
echo [OK] MFA installed successfully.
echo.
echo [Final] Downloading Korean acoustic model... ^(1-2 min^)
if exist "%MFA_EXE%" (
    "%MFA_EXE%" model download acoustic korean_mfa --ignore_cache
) else (
    echo [WARNING] mfa.exe not found at %MFA_EXE%. Model download skipped.
)
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

:install_japanese_support
echo Checking Japanese tokenizer dependencies...
if exist "%ENV_DIR%\Scripts\conda.exe" (
    "%ENV_DIR%\Scripts\conda.exe" install -y -p "%ENV_DIR%" -c conda-forge --override-channels spacy sudachipy sudachidict-core
    if errorlevel 1 (
        echo [FAILED] Japanese tokenizer dependency install failed.
        pause
        exit /b 1
    )
    goto :eof
)
if defined SYSTEM_CONDA (
    "%SYSTEM_CONDA%" install -y -p "%ENV_DIR%" -c conda-forge --override-channels spacy sudachipy sudachidict-core
    if errorlevel 1 (
        echo [FAILED] Japanese tokenizer dependency install failed.
        pause
        exit /b 1
    )
    goto :eof
)
echo [WARN] Conda command was not found. Skipping Japanese tokenizer dependency install.
goto :eof

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
