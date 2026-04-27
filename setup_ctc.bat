@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title UTAU Auto OTO - CTC Runtime Setup

set "NON_INTERACTIVE=0"
set "FORCE_CLEAN=0"
set "RUNTIME_ROOT_OVERRIDE="
set "APP_DIR="
set "SETUP_CTC_SELF=%~f0"
set "CTC_ENV_PATH_FILE="

REM Keep host python state clean.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONEXECUTABLE="
set "__PYVENV_LAUNCHER__="
set "VIRTUAL_ENV="
set "CONDA_DEFAULT_ENV="
set "CONDA_PROMPT_MODIFIER="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="--runtime-root" goto :capture_runtime_root

set "ARG_RAW=%~1"
if /i "%ARG_RAW:~0,15%"=="--runtime-root=" goto :capture_runtime_root_inline

if /i "%~1"=="--non-interactive" set "NON_INTERACTIVE=1"
if /i "%~1"=="--yes" set "NON_INTERACTIVE=1"
if /i "%~1"=="--clean" set "FORCE_CLEAN=1"

shift
goto :parse_args

:capture_runtime_root
shift
if "%~1"=="" goto :missing_runtime_root
set "RUNTIME_ROOT_CANDIDATE=%~1"
if /i "%RUNTIME_ROOT_CANDIDATE:~0,2%"=="--" goto :missing_runtime_root
set "RUNTIME_ROOT_OVERRIDE=%~1"
shift
goto :parse_args

:capture_runtime_root_inline
set "RUNTIME_ROOT_OVERRIDE=%ARG_RAW:~15%"
if "%RUNTIME_ROOT_OVERRIDE%"=="" goto :missing_runtime_root
shift
goto :parse_args

:missing_runtime_root
echo [FAILED] --runtime-root requires a valid path value.
echo          Example: --runtime-root "C:\Users\%USERNAME%\AppData\Local\UTAU_Auto_OTO_v3"
exit /b 1

:show_help
echo Usage: setup_ctc.bat [--runtime-root PATH] [--clean] [--non-interactive]
echo Installs dedicated CTC runtime into .env_ctc (no MFA package changes).
echo.
echo   --runtime-root PATH       Explicit runtime root path
echo   --runtime-root=PATH       Same as above ^(inline form^)
echo   --clean                   Remove existing .env_ctc before install
echo   --non-interactive         Run without prompts
echo.
echo Optional env overrides:
echo   UTOA_CTC_TORCH_INDEX_URL  Torch wheel index ^(default: https://download.pytorch.org/whl/cpu^)
echo   UTOA_CTC_TORCH_PACKAGES   Torch package list ^(default: torch torchaudio^)
exit /b 0

:args_done
if defined RUNTIME_ROOT_OVERRIDE (
    for %%I in ("%RUNTIME_ROOT_OVERRIDE%") do set "APP_DIR=%%~fI"
) else (
    for %%I in ("%SETUP_CTC_SELF%") do set "APP_DIR=%%~dpI"
)

if defined APP_DIR if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"
if not defined APP_DIR set "APP_DIR=%CD%"

if not exist "%APP_DIR%" (
    echo [FAILED] Runtime root does not exist: %APP_DIR%
    exit /b 1
)

set "APP_CODE_DIR=%APP_DIR%"
set "HAS_CTC_SOURCE=0"
if exist "%APP_DIR%\UTAU_Auto_OTO\core\ctc_runner.py" set "APP_CODE_DIR=%APP_DIR%\UTAU_Auto_OTO"
if exist "%APP_CODE_DIR%\core\ctc_runner.py" set "HAS_CTC_SOURCE=1"

if "%HAS_CTC_SOURCE%"=="0" (
    echo [WARN] Could not find core\ctc_runner.py under runtime root.
    echo        runtime_root=%APP_DIR%
    echo        code_root=%APP_CODE_DIR%
    echo [INFO] Binary portable payload detected. setup_ctc will continue with dependency-only verification.
)

set "ENV_LINK_DIR=%APP_DIR%\.env_ctc"
set "CTC_ENV_PATH_FILE=%APP_DIR%\.ctc_env_path"
call :resolve_ctc_env_dir
if errorlevel 1 exit /b 1
set "ENV_PY=%ENV_DIR%\Scripts\python.exe"

echo ====================================================
echo UTAU Auto OTO - CTC runtime setup
echo ====================================================
echo [INFO] Runtime root: %APP_DIR%
echo [INFO] Code root   : %APP_CODE_DIR%
echo [INFO] CTC source  : %HAS_CTC_SOURCE%
echo [INFO] CTC env dir : %ENV_DIR%
if /i not "%ENV_DIR%"=="%ENV_LINK_DIR%" (
    echo [INFO] CTC env link: %ENV_LINK_DIR% ^(junction^)
)

if "%FORCE_CLEAN%"=="1" (
    call :remove_env_dir
    if errorlevel 1 exit /b 1
)

if not exist "%ENV_PY%" (
    call :create_ctc_env
    if errorlevel 1 exit /b 1
)

call :bootstrap_pip
if errorlevel 1 exit /b 1

call :install_base_packages
if errorlevel 1 exit /b 1

call :install_torch_packages
if errorlevel 1 exit /b 1

call :verify_ctc_runtime
if errorlevel 1 exit /b 1

call :ensure_ctc_env_link
if errorlevel 1 exit /b 1

echo.
echo [OK] CTC runtime setup completed.
echo [INFO] CTC env python: %ENV_PY%
echo [INFO] You can keep MFA runtime separate at: %APP_DIR%\.env
if not "%NON_INTERACTIVE%"=="1" pause
exit /b 0

:remove_env_dir
if defined CTC_ENV_PATH_FILE if exist "%CTC_ENV_PATH_FILE%" del /q "%CTC_ENV_PATH_FILE%" >nul 2>nul

if /i not "%ENV_DIR%"=="%ENV_LINK_DIR%" (
    if exist "%ENV_LINK_DIR%" (
        echo [INFO] Removing CTC env link: %ENV_LINK_DIR%
        rmdir "%ENV_LINK_DIR%" >nul 2>nul
        if exist "%ENV_LINK_DIR%" (
            echo [FAILED] Failed to remove CTC env link: %ENV_LINK_DIR%
            if not "%NON_INTERACTIVE%"=="1" pause
            exit /b 1
        )
    )
)

if exist "%ENV_DIR%" (
    echo [INFO] Removing existing CTC env: %ENV_DIR%
    rmdir /s /q "%ENV_DIR%" >nul 2>nul
    if exist "%ENV_DIR%" (
        echo [FAILED] Failed to remove CTC env: %ENV_DIR%
        if not "%NON_INTERACTIVE%"=="1" pause
        exit /b 1
    )
)
goto :eof

:create_ctc_env
echo [INFO] Creating CTC venv...
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -m venv "%ENV_DIR%" >nul 2>nul
)
if exist "%ENV_PY%" goto :eof

where python >nul 2>nul
if not errorlevel 1 (
    python -m venv "%ENV_DIR%"
)
if exist "%ENV_PY%" goto :eof

set "BASE_PY="
if exist "%APP_DIR%\.env\python.exe" set "BASE_PY=%APP_DIR%\.env\python.exe"
if not defined BASE_PY if exist "%APP_DIR%\.env\Scripts\python.exe" set "BASE_PY=%APP_DIR%\.env\Scripts\python.exe"
if not defined BASE_PY if exist "%APP_DIR%\UTAU_Auto_OTO\.env\python.exe" set "BASE_PY=%APP_DIR%\UTAU_Auto_OTO\.env\python.exe"
if not defined BASE_PY if exist "%APP_DIR%\UTAU_Auto_OTO\.env\Scripts\python.exe" set "BASE_PY=%APP_DIR%\UTAU_Auto_OTO\.env\Scripts\python.exe"
if defined BASE_PY (
    if exist "%BASE_PY%" (
        echo [INFO] Using base runtime python fallback: %BASE_PY%
        "%BASE_PY%" -m venv "%ENV_DIR%"
    )
)
if exist "%ENV_PY%" goto :eof

call :bootstrap_base_python_via_setup_mfa
if errorlevel 1 goto :ctc_env_post_bootstrap

set "BASE_PY="
if exist "%APP_DIR%\.env\python.exe" set "BASE_PY=%APP_DIR%\.env\python.exe"
if not defined BASE_PY if exist "%APP_DIR%\.env\Scripts\python.exe" set "BASE_PY=%APP_DIR%\.env\Scripts\python.exe"
if not defined BASE_PY if exist "%APP_DIR%\UTAU_Auto_OTO\.env\python.exe" set "BASE_PY=%APP_DIR%\UTAU_Auto_OTO\.env\python.exe"
if not defined BASE_PY if exist "%APP_DIR%\UTAU_Auto_OTO\.env\Scripts\python.exe" set "BASE_PY=%APP_DIR%\UTAU_Auto_OTO\.env\Scripts\python.exe"
if defined BASE_PY if exist "%BASE_PY%" (
    echo [INFO] Retrying CTC venv creation with bootstrapped MFA python...
    "%BASE_PY%" -m venv "%ENV_DIR%"
)

:ctc_env_post_bootstrap
if exist "%ENV_PY%" goto :eof

echo [FAILED] Failed to create CTC venv: %ENV_DIR%
echo        No usable Python runtime was found.
echo        Try running setup_mfa.bat first, then retry setup_ctc.bat.
if not "%NON_INTERACTIVE%"=="1" pause
exit /b 1
goto :eof

:resolve_ctc_env_dir
set "ENV_DIR=%ENV_LINK_DIR%"
set "CTC_ENV_DIR_SHORT=%LOCALAPPDATA%\UTAU_Auto_OTO_v3\ctc_env"
if not defined LOCALAPPDATA set "CTC_ENV_DIR_SHORT=%TEMP%\UTAU_Auto_OTO_ctc_env"

if defined CTC_ENV_PATH_FILE if exist "%CTC_ENV_PATH_FILE%" (
    set /p CTC_ENV_FILE_DIR=<"%CTC_ENV_PATH_FILE%"
    if defined CTC_ENV_FILE_DIR (
        if exist "%CTC_ENV_FILE_DIR%\Scripts\python.exe" (
            set "ENV_DIR=%CTC_ENV_FILE_DIR%"
            exit /b 0
        )
    )
)

set "CTC_PATH_PROBE=%ENV_DIR%\Lib\site-packages\torch\include\ATen\native\transformers\cuda\mem_eff_attention\iterators\predicated_tile_access_iterator_residual_last.h"
call :is_path_too_long "%CTC_PATH_PROBE%" CTC_ENV_PATH_TOO_LONG
if "%CTC_ENV_PATH_TOO_LONG%"=="1" (
    echo [WARN] CTC env path may be too long for torch install reliability.
    echo [INFO] Switching CTC env to short path: %CTC_ENV_DIR_SHORT%
    set "ENV_DIR=%CTC_ENV_DIR_SHORT%"
)
exit /b 0
goto :eof

:ensure_ctc_env_link
call :write_ctc_env_path_file
if errorlevel 1 exit /b 1

if /i "%ENV_DIR%"=="%ENV_LINK_DIR%" goto :eof
if not exist "%ENV_DIR%\Scripts\python.exe" (
    echo [FAILED] CTC env target is missing: %ENV_DIR%
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)

if exist "%ENV_LINK_DIR%" (
    rmdir "%ENV_LINK_DIR%" >nul 2>nul
    if exist "%ENV_LINK_DIR%" (
        rmdir /s /q "%ENV_LINK_DIR%" >nul 2>nul
    )
)
if exist "%ENV_LINK_DIR%" (
    echo [FAILED] Failed to clear old CTC env link path: %ENV_LINK_DIR%
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)

mklink /J "%ENV_LINK_DIR%" "%ENV_DIR%" >nul 2>nul
if errorlevel 1 (
    echo [WARN] Failed to create CTC env junction. Continuing without junction.
    echo        link=%ENV_LINK_DIR%
    echo        target=%ENV_DIR%
    echo [INFO] CTC env path was saved to: %CTC_ENV_PATH_FILE%
    exit /b 0
)
exit /b 0
goto :eof

:write_ctc_env_path_file
if not defined CTC_ENV_PATH_FILE exit /b 0
if not exist "%APP_DIR%" exit /b 1
if not exist "%ENV_DIR%\Scripts\python.exe" exit /b 1
>"%CTC_ENV_PATH_FILE%" echo %ENV_DIR%
if errorlevel 1 (
    echo [FAILED] Failed to write CTC env path file: %CTC_ENV_PATH_FILE%
    exit /b 1
)
exit /b 0
goto :eof

:is_path_too_long
set "%~2=0"
if "%~1"=="" goto :eof

powershell -NoProfile -Command "$p='%~1'; if ($p.Length -ge 230) { exit 0 } else { exit 1 }" >nul 2>nul
if not errorlevel 1 set "%~2=1"
goto :eof

:resolve_setup_mfa_script
set "SETUP_MFA_PATH="
if exist "%APP_DIR%\setup_mfa.bat" set "SETUP_MFA_PATH=%APP_DIR%\setup_mfa.bat"
if not defined SETUP_MFA_PATH if exist "%APP_CODE_DIR%\setup_mfa.bat" set "SETUP_MFA_PATH=%APP_CODE_DIR%\setup_mfa.bat"
if not defined SETUP_MFA_PATH if exist "%APP_CODE_DIR%\..\setup_mfa.bat" set "SETUP_MFA_PATH=%APP_CODE_DIR%\..\setup_mfa.bat"
if not defined SETUP_MFA_PATH if exist "%APP_DIR%\UTAU_Auto_OTO\setup_mfa.bat" set "SETUP_MFA_PATH=%APP_DIR%\UTAU_Auto_OTO\setup_mfa.bat"
if not defined SETUP_MFA_PATH if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\UTAU_Auto_OTO_v3\setup_mfa.bat" set "SETUP_MFA_PATH=%LOCALAPPDATA%\UTAU_Auto_OTO_v3\setup_mfa.bat"
if not defined SETUP_MFA_PATH if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\UTAU_Auto_OTO\setup_mfa.bat" set "SETUP_MFA_PATH=%LOCALAPPDATA%\UTAU_Auto_OTO\setup_mfa.bat"
goto :eof

:bootstrap_base_python_via_setup_mfa
call :resolve_setup_mfa_script
if not defined SETUP_MFA_PATH (
    echo [WARN] setup_mfa.bat not found. Cannot bootstrap Python runtime automatically.
    exit /b 1
)
echo [INFO] Python launcher missing. Attempting MFA runtime bootstrap first...
call "%SETUP_MFA_PATH%" --install --without-ml --non-interactive --runtime-root "%APP_DIR%"
if errorlevel 1 (
    echo [FAILED] setup_mfa bootstrap failed.
    exit /b 1
)
exit /b 0
goto :eof

:bootstrap_pip
echo [INFO] Upgrading pip/setuptools/wheel...
"%ENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [FAILED] pip bootstrap failed.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
goto :eof

:install_base_packages
echo [INFO] Installing base CTC packages...
"%ENV_PY%" -m pip install --upgrade numpy textgrid
if errorlevel 1 (
    echo [FAILED] Base package install failed: numpy/textgrid
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
goto :eof

:install_torch_packages
set "TORCH_INDEX_URL=%UTOA_CTC_TORCH_INDEX_URL%"
if not defined TORCH_INDEX_URL set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"

set "TORCH_PACKAGES=%UTOA_CTC_TORCH_PACKAGES%"
if not defined TORCH_PACKAGES set "TORCH_PACKAGES=torch torchaudio"

echo [INFO] Installing torch packages...
echo [INFO] index-url: %TORCH_INDEX_URL%
echo [INFO] packages : %TORCH_PACKAGES%
"%ENV_PY%" -m pip install --upgrade %TORCH_PACKAGES% --index-url "%TORCH_INDEX_URL%"
if errorlevel 1 (
    echo [FAILED] Torch package install failed.
    echo        You can override source with UTOA_CTC_TORCH_INDEX_URL.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
goto :eof

:verify_ctc_runtime
echo [INFO] Verifying CTC runtime...
if "%HAS_CTC_SOURCE%"=="1" (
    set "UTOA_CODE_ROOT=%APP_CODE_DIR%"
    "%ENV_PY%" -c "import os,sys; code_root=os.environ.get('UTOA_CODE_ROOT','').strip(); (code_root and code_root not in sys.path) and sys.path.insert(0, code_root); import numpy,textgrid,torch,torchaudio; _=torchaudio.pipelines.MMS_FA; import core.ctc_runner as _cr; print('ok')"
    if errorlevel 1 (
        echo [FAILED] CTC runtime verification failed.
        echo        Required check: torch/torchaudio + MMS_FA + core.ctc_runner import
        if not "%NON_INTERACTIVE%"=="1" pause
        exit /b 1
    )
    set "UTOA_CODE_ROOT="
) else (
    "%ENV_PY%" -c "import numpy,textgrid,torch,torchaudio; _=torchaudio.pipelines.MMS_FA; print('ok_binary')"
    if errorlevel 1 (
        echo [FAILED] CTC runtime verification failed.
        echo        Required check: torch/torchaudio + MMS_FA
        if not "%NON_INTERACTIVE%"=="1" pause
        exit /b 1
    )
)
goto :eof
