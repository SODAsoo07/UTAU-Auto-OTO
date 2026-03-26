@echo off

@echo off

setlocal EnableExtensions DisableDelayedExpansion

chcp 65001 >nul

title UTAU Auto OTO - MFA Setup and Recovery

set "INSTALL_ML=0"

set "DELETE_OLD_AFTER_INSTALL=0"

set "AUTO_ML=0"

set "NON_INTERACTIVE=0"

set "DIRECT_SETUP=0"

set "NO_RECOVERY_SHIM=0"

set "FORCE_CLEAN=0"

set "FORCE_MENU=0"

set "REQUESTED_MODE="

set "WRAPPER_MODE="

set "RECOVERY_LANGUAGE=korean"

set "RUNTIME_ROOT_OVERRIDE="

set "SETUP_MFA_SELF=%~f0"

set "SETUP_MFA_DIR=%~dp0"

set "MFA_RUNTIME_BUNDLE_DIR="

set "HAS_CERTUTIL=1"

set "HAS_TAR=1"

REM Clean host Python environment to avoid python311.dll conflicts.
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
if /i "%~1"=="--language" goto :capture_language

set "ARG_RAW=%~1"

if /i "%ARG_RAW:~0,15%"=="--runtime-root=" goto :capture_runtime_root_inline
if /i "%ARG_RAW:~0,11%"=="--language=" goto :capture_language_inline

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

:capture_language

shift

if "%~1"=="" goto :missing_language

set "LANG_CANDIDATE=%~1"

if /i "%LANG_CANDIDATE:~0,2%"=="--" goto :missing_language

set "RECOVERY_LANGUAGE=%~1"

call :normalize_recovery_language

if errorlevel 1 exit /b 1

shift

goto :parse_args

:capture_language_inline

set "RECOVERY_LANGUAGE=%ARG_RAW:~11%"

if "%RECOVERY_LANGUAGE%"=="" goto :missing_language

call :normalize_recovery_language

if errorlevel 1 exit /b 1

shift

goto :parse_args

:missing_language

echo [FAILED] --language requires a value: korean or japanese.

echo          Example: --language japanese

exit /b 1

:missing_runtime_root

echo [FAILED] --runtime-root requires a valid path value.

echo          Example: --runtime-root "C:\Users\%USERNAME%\AppData\Local\UTAU_Auto_OTO"

exit /b 1

:show_help

echo Usage: setup_mfa.bat [--install ^| --recovery ^| --menu] [--runtime-root PATH] [--with-ml] [--non-interactive]

echo Installs or repairs MFA runtime ^(.env^), micromamba packages, and language/model dependencies.
echo For CTC runtime ^(.env_ctc^) use setup_ctc.bat.

echo.

echo   --install                 Run full install flow

echo   --recovery / --recover    Run runtime recovery flow

echo   --menu                    Show mode selection menu

echo   --runtime-root PATH       Explicit runtime root path

echo   --runtime-root=PATH       Same as above ^(inline form^)

echo   --language LANG           Recovery language: korean ^| japanese

echo   --language=LANG           Same as above ^(inline form^)

echo   --with-ml / --install-ml  Install ML dependencies ^(pandas/pyarrow/lightgbm/onnxruntime^)

echo   --non-interactive         Run without prompts ^(requires valid runtime root^)

echo   --direct-setup            Skip wrapper and run install flow directly

echo   --no-recovery-shim        Disable runtime_recovery.ps1 wrapper

echo   --clean                   Force-clean MFA env before install

exit /b 0

:args_done

for %%I in ("%SETUP_MFA_DIR%") do set "SETUP_MFA_DIR_NORM=%%~fI"

if defined SETUP_MFA_DIR_NORM if "%SETUP_MFA_DIR_NORM:~-1%"=="\" set "SETUP_MFA_DIR_NORM=%SETUP_MFA_DIR_NORM:~0,-1%"

if defined RUNTIME_ROOT_OVERRIDE (

    for %%I in ("%RUNTIME_ROOT_OVERRIDE%") do set "APP_DIR=%%~fI"

) else (

    for %%I in ("%SETUP_MFA_SELF%") do set "APP_DIR=%%~dpI"

)

if defined APP_DIR if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"

if not defined APP_DIR set "APP_DIR=%CD%"

if not defined RUNTIME_ROOT_OVERRIDE if /i "%APP_DIR%"=="%WINDIR%\System32" (

    echo [WARN] setup_mfa.bat was launched from System32. Falling back to LOCALAPPDATA runtime root.

    echo        If this is unintended, run setup_mfa.bat from the installed app folder.

    set "APP_DIR=%LOCALAPPDATA%\UTAU_Auto_OTO_v3"

)

if defined RUNTIME_ROOT_OVERRIDE (

    echo [INFO] runtime-root override: %APP_DIR%

)

call :ensure_runtime_root_layout

if errorlevel 1 exit /b 1

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

if /i "%WRAPPER_MODE%"=="menu" call :select_start_mode

if /i "%WRAPPER_MODE%"=="exit" exit /b 2

:wrapper_dispatch

if /i "%WRAPPER_MODE%"=="install" goto :wrapper_install

if /i "%WRAPPER_MODE%"=="recovery" goto :wrapper_recovery

echo [WARN] : %WRAPPER_MODE%

set "WRAPPER_MODE=install"

goto :wrapper_install

:wrapper_install

call :preflight_install_tools

if errorlevel 1 exit /b 1

set "WRAPPER_ARGS=--direct-setup --no-recovery-shim"

if "%INSTALL_ML%"=="1" set "WRAPPER_ARGS=%WRAPPER_ARGS% --with-ml"

if "%NON_INTERACTIVE%"=="1" set "WRAPPER_ARGS=%WRAPPER_ARGS% --non-interactive"

set "WRAPPER_RUNTIME_ARG="

if defined RUNTIME_ROOT_OVERRIDE set "WRAPPER_RUNTIME_ARG=--runtime-root ""%APP_DIR%"""

if "%FORCE_CLEAN%"=="1" set "WRAPPER_ARGS=%WRAPPER_ARGS% --clean"

echo [INFO] %WRAPPER_ARGS%

if defined RUNTIME_ROOT_OVERRIDE (
    call "%SETUP_MFA_SELF%" %WRAPPER_ARGS% --runtime-root "%APP_DIR%"
) else (
    call "%SETUP_MFA_SELF%" %WRAPPER_ARGS%
)

exit /b %ERRORLEVEL%

:wrapper_recovery

if not defined RUNTIME_RECOVERY_SCRIPT (

    echo [WARN] runtime_recovery.ps1

    if "%NON_INTERACTIVE%"=="1" (

        echo [FAILED] setup_mfa.bat --install

        exit /b 1

    )

    echo [INFO] Processing...

    set "WRAPPER_MODE=install"

    goto :wrapper_install

)

echo [INFO] %RUNTIME_RECOVERY_SCRIPT%

if "%NON_INTERACTIVE%"=="1" (

    if "%INSTALL_ML%"=="1" (

        powershell -NoProfile -ExecutionPolicy Bypass -File "%RUNTIME_RECOVERY_SCRIPT%" -SetupScriptPath "%SETUP_MFA_SELF%" -RuntimeRoot "%APP_DIR%" -Language %RECOVERY_LANGUAGE% -NonInteractive -WithMl

    ) else (

        powershell -NoProfile -ExecutionPolicy Bypass -File "%RUNTIME_RECOVERY_SCRIPT%" -SetupScriptPath "%SETUP_MFA_SELF%" -RuntimeRoot "%APP_DIR%" -Language %RECOVERY_LANGUAGE% -NonInteractive

    )

) else (

    if "%INSTALL_ML%"=="1" (

        powershell -NoProfile -ExecutionPolicy Bypass -File "%RUNTIME_RECOVERY_SCRIPT%" -SetupScriptPath "%SETUP_MFA_SELF%" -RuntimeRoot "%APP_DIR%" -Language %RECOVERY_LANGUAGE% -WithMl

    ) else (

        powershell -NoProfile -ExecutionPolicy Bypass -File "%RUNTIME_RECOVERY_SCRIPT%" -SetupScriptPath "%SETUP_MFA_SELF%" -RuntimeRoot "%APP_DIR%" -Language %RECOVERY_LANGUAGE%

    )

)

set "RECOVERY_RC=%ERRORLEVEL%"

if not "%RECOVERY_RC%"=="0" (

    echo [FAILED] runtime_recovery.ps1 ^(code=%RECOVERY_RC%^)

)

if not "%RECOVERY_RC%"=="0" if not "%NON_INTERACTIVE%"=="1" pause

exit /b %RECOVERY_RC%

:resolve_runtime_recovery_script

set "RUNTIME_RECOVERY_SCRIPT="

if exist "%APP_DIR%\runtime_recovery.ps1" (

    set "RUNTIME_RECOVERY_SCRIPT=%APP_DIR%\runtime_recovery.ps1"

)

if not defined RUNTIME_RECOVERY_SCRIPT if exist "%APP_DIR%\scripts\runtime_recovery.ps1" (

    set "RUNTIME_RECOVERY_SCRIPT=%APP_DIR%\scripts\runtime_recovery.ps1"

)

if not defined RUNTIME_RECOVERY_SCRIPT if defined SETUP_MFA_DIR_NORM if exist "%SETUP_MFA_DIR_NORM%\runtime_recovery.ps1" (

    set "RUNTIME_RECOVERY_SCRIPT=%SETUP_MFA_DIR_NORM%\runtime_recovery.ps1"

)

if not defined RUNTIME_RECOVERY_SCRIPT if defined SETUP_MFA_DIR_NORM if exist "%SETUP_MFA_DIR_NORM%\scripts\runtime_recovery.ps1" (

    set "RUNTIME_RECOVERY_SCRIPT=%SETUP_MFA_DIR_NORM%\scripts\runtime_recovery.ps1"

)

goto :eof

:ensure_runtime_root_layout

set "RUNTIME_PAYLOAD_OK=0"

call :is_valid_runtime_root "%APP_DIR%"

if "%CHECK_VALID%"=="1" set "RUNTIME_PAYLOAD_OK=1"

if "%RUNTIME_PAYLOAD_OK%"=="1" exit /b 0

if defined RUNTIME_ROOT_OVERRIDE (

    echo [FAILED] runtime-root is missing required files:

    echo         "%APP_DIR%\UTAU_Auto_OTO\UTAU_Auto_OTO.exe" ^(release layout^)

    echo         OR "%APP_DIR%\requirements.txt" + "%APP_DIR%\requirements-ml.txt" ^(source layout^)

    if not "%NON_INTERACTIVE%"=="1" pause

)

call :auto_detect_runtime_root

if defined DETECTED_RUNTIME_ROOT (

    echo [INFO] Detected runtime root: %DETECTED_RUNTIME_ROOT%

    if "%NON_INTERACTIVE%"=="1" (

        set "APP_DIR=%DETECTED_RUNTIME_ROOT%"

        exit /b 0

    )

    choice /C YN /N /M "Use this detected path for setup/recovery? [Y/N]: "

    if errorlevel 2 (

        echo [INFO] Detected path was not selected.

    ) else (

        set "APP_DIR=%DETECTED_RUNTIME_ROOT%"

        exit /b 0

    )

)

echo [WARN] setup_mfa.bat is not running from the runtime folder.

echo        Run it from the installed folder or pass --runtime-root.

if "%NON_INTERACTIVE%"=="1" (

    echo [FAILED] Non-interactive mode requires --runtime-root when current folder is not valid.

    exit /b 1

)

set "RUNTIME_ROOT_INPUT="

set /p RUNTIME_ROOT_INPUT=Enter runtime root path [empty to cancel]: 

if not defined RUNTIME_ROOT_INPUT (

    echo [FAILED] Runtime root not provided.

    pause

    exit /b 1

)

for %%I in ("%RUNTIME_ROOT_INPUT%") do set "APP_DIR=%%~fI"

call :is_valid_runtime_root "%APP_DIR%"

if "%CHECK_VALID%"=="1" exit /b 0

echo [FAILED] Invalid runtime root: %APP_DIR%

echo         Required marker not found:

echo         "%APP_DIR%\UTAU_Auto_OTO\UTAU_Auto_OTO.exe" ^(release layout^)

echo         OR requirements.txt + requirements-ml.txt ^(source layout^)

pause

exit /b 1

:auto_detect_runtime_root

set "DETECTED_RUNTIME_ROOT="

call :try_runtime_candidate "%APP_DIR%"

if defined DETECTED_RUNTIME_ROOT goto :eof

if defined SETUP_MFA_DIR_NORM (

    call :try_runtime_candidate "%SETUP_MFA_DIR_NORM%"

    if defined DETECTED_RUNTIME_ROOT goto :eof

)

call :try_runtime_candidate "%CD%"

if defined DETECTED_RUNTIME_ROOT goto :eof

if defined LOCALAPPDATA (

    call :try_runtime_candidate "%LOCALAPPDATA%\UTAU_Auto_OTO_v3"

    if defined DETECTED_RUNTIME_ROOT goto :eof

    call :try_runtime_candidate "%LOCALAPPDATA%\UTAU_Auto_OTO"

    if defined DETECTED_RUNTIME_ROOT goto :eof

)

if defined USERPROFILE (

    call :scan_parent_for_runtime "%USERPROFILE%\Desktop"

    if defined DETECTED_RUNTIME_ROOT goto :eof

    call :scan_parent_for_runtime "%USERPROFILE%\Downloads"

    if defined DETECTED_RUNTIME_ROOT goto :eof

)

if defined PUBLIC (

    call :scan_parent_for_runtime "%PUBLIC%\Desktop"

    if defined DETECTED_RUNTIME_ROOT goto :eof

)

goto :eof

:scan_parent_for_runtime

set "SCAN_PARENT=%~1"

if "%SCAN_PARENT%"=="" goto :eof

if not exist "%SCAN_PARENT%" goto :eof

for /f "delims=" %%D in ('dir /b /ad "%SCAN_PARENT%" 2^>nul') do (

    if not defined DETECTED_RUNTIME_ROOT call :try_runtime_candidate "%SCAN_PARENT%\%%D"

)

goto :eof

:try_runtime_candidate

set "TRY_CANDIDATE=%~1"

if "%TRY_CANDIDATE%"=="" goto :eof

for %%I in ("%TRY_CANDIDATE%") do set "TRY_CANDIDATE=%%~fI"

if not exist "%TRY_CANDIDATE%" goto :eof

call :is_valid_runtime_root "%TRY_CANDIDATE%"

if "%CHECK_VALID%"=="1" (

    set "DETECTED_RUNTIME_ROOT=%TRY_CANDIDATE%"

    goto :eof

)

for %%P in ("%TRY_CANDIDATE%\..") do set "TRY_PARENT=%%~fP"

if not defined TRY_PARENT goto :eof

if /i "%TRY_PARENT%"=="%TRY_CANDIDATE%" goto :eof

call :is_valid_runtime_root "%TRY_PARENT%"

if "%CHECK_VALID%"=="1" (

    set "DETECTED_RUNTIME_ROOT=%TRY_PARENT%"

)

goto :eof

:is_valid_runtime_root

set "CHECK_PATH=%~1"

set "CHECK_VALID=0"

if "%CHECK_PATH%"=="" goto :eof

if not exist "%CHECK_PATH%" goto :eof

if exist "%CHECK_PATH%\UTAU_Auto_OTO\UTAU_Auto_OTO.exe" set "CHECK_VALID=1"

if "%CHECK_VALID%"=="0" if exist "%CHECK_PATH%\requirements.txt" if exist "%CHECK_PATH%\requirements-ml.txt" set "CHECK_VALID=1"

goto :eof

:select_start_mode

echo.

echo ====================================================

echo UTAU Auto OTO - Setup Launcher

echo ====================================================

echo [1] Install MFA runtime (basic)

echo [2] Install MFA runtime + ML packages

echo [3] Recovery mode (interactive)

echo [4] Recovery mode (non-interactive)

echo [0] Exit

choice /C 12340 /N /M "Select mode (1/2/3/4/0): "

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

:normalize_recovery_language

set "RECOVERY_LANGUAGE=%RECOVERY_LANGUAGE:"=%"

if /i "%RECOVERY_LANGUAGE%"=="ko" set "RECOVERY_LANGUAGE=korean"
if /i "%RECOVERY_LANGUAGE%"=="kr" set "RECOVERY_LANGUAGE=korean"
if /i "%RECOVERY_LANGUAGE%"=="ja" set "RECOVERY_LANGUAGE=japanese"
if /i "%RECOVERY_LANGUAGE%"=="jp" set "RECOVERY_LANGUAGE=japanese"

if /i "%RECOVERY_LANGUAGE%"=="korean" exit /b 0
if /i "%RECOVERY_LANGUAGE%"=="japanese" exit /b 0

echo [FAILED] Invalid --language value: %RECOVERY_LANGUAGE%
echo          Allowed values: korean, japanese
exit /b 1

:preflight_install_tools

set "HAS_CERTUTIL=1"

set "HAS_TAR=1"

where powershell >nul 2>nul
if errorlevel 1 (

    echo [FAILED] powershell
    exit /b 1

)

where certutil >nul 2>nul

if errorlevel 1 (

    set "HAS_CERTUTIL=0"

    echo [WARN] certutil was not found. MD5 verification will be skipped.
)

where tar >nul 2>nul

if errorlevel 1 (

    set "HAS_TAR=0"

    echo [WARN] tar was not found. Online micromamba extraction may fail.
)

goto :eof

:direct_setup_start

call :preflight_install_tools

if errorlevel 1 exit /b 1

echo ====================================================

echo UTAU Auto OTO - MFA

echo Step 1/5: Runtime bootstrap

echo ====================================================

echo.

echo [INFO] This process can take 10-20 minutes on first install.

echo [INFO] Please keep this window open until setup completes.

echo.

echo [INFO] MFA runtime preparation started...

echo.

if not exist "%APP_DIR%" mkdir "%APP_DIR%" >nul 2>nul

pushd "%APP_DIR%" >nul 2>nul

set "OLD_PUBLIC_ROOT=%PUBLIC%"

if not defined OLD_PUBLIC_ROOT set "OLD_PUBLIC_ROOT=C:\Users\Public"

set "OLD_ENV_DIR=%OLD_PUBLIC_ROOT%\UTAU_Auto_OTO_v3\.env"

set "OLD_MICROMAMBA_ROOT=%OLD_PUBLIC_ROOT%\UTAU_Auto_OTO_v3\micromamba"

set "MFA_SHARED_ROOT=%OLD_PUBLIC_ROOT%\UTAU_Auto_OTO_v3\.mfa_root_ascii"

set "ENV_DIR=%APP_DIR%\.env"

set "MICROMAMBA_ROOT=%APP_DIR%\micromamba"

set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\Library\bin\micromamba.exe"
set "MICROMAMBA_PORTABLE_EXE=%APP_DIR%\micromamba.exe"

if exist "%MICROMAMBA_PORTABLE_EXE%" (
    set "MICROMAMBA_EXE=%MICROMAMBA_PORTABLE_EXE%"
)

set "MICROMAMBA_ARCHIVE=%TEMP%\utau_auto_oto_micromamba-win-64-latest.tar.bz2"

set "MICROMAMBA_LATEST_URL=https://micro.mamba.pm/api/micromamba/win-64/latest"
set "MICROMAMBA_EXE_URL=https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64"

set "MICROMAMBA_MD5_EXPECTED="

set "MICROMAMBA_MD5_ACTUAL="

set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"

set "MFA_PYTHON_VERSION=3.10"

set "BUNDLE_RESTORED=0"

set "AUTO_CLEAN_DONE=0"

call :resolve_runtime_bundle_dir

if defined MFA_RUNTIME_BUNDLE_DIR (

    echo [INFO] MFA runtime bundle detected: %MFA_RUNTIME_BUNDLE_DIR%

)
echo [INFO] MFA env path: %ENV_DIR%

echo [INFO] MFA micromamba path: %MICROMAMBA_ROOT%

if "%FORCE_CLEAN%"=="1" (

    echo [INFO] --clean option detected. Running pre-install clean...

    call :run_clean_recovery

    if errorlevel 1 exit /b 1

)

if exist "%APP_DIR%\ML_models" set "AUTO_ML=1"

if exist "%APP_DIR%\models_installed\oto_ml" set "AUTO_ML=1"

if exist "%APP_DIR%\requirements-ml.txt" set "AUTO_ML=1"

if exist "%APP_DIR%\UTAU_Auto_OTO\pandas" set "AUTO_ML=1"

if exist "%APP_DIR%\UTAU_Auto_OTO\onnxruntime" set "AUTO_ML=1"

if exist "%APP_DIR%\UTAU_Auto_OTO\lightgbm" set "AUTO_ML=1"

if "%AUTO_ML%"=="1" if "%INSTALL_ML%"=="0" (

    echo [INFO] ML runtime hints detected. Enabling ML dependency install.

    set "INSTALL_ML=1"

)

if exist "%OLD_ENV_DIR%" (

    if /i not "%OLD_ENV_DIR%"=="%ENV_DIR%" (

        call :handle_old_env

    )

)

call :restore_from_runtime_bundle

if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.bat" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"
if exist "%MFA_EXE%" (

    call :evaluate_existing_env_python

    if errorlevel 1 exit /b 1

)

if exist "%MFA_EXE%" goto :existing_env_ready

goto :install_micromamba

:evaluate_existing_env_python

call :get_env_python_version "%ENV_DIR%" MFA_ENV_PYTHON

if not defined MFA_ENV_PYTHON goto :eof

echo [INFO] MFA Python %MFA_ENV_PYTHON%

call :python_requires_rebuild "%MFA_ENV_PYTHON%" MFA_ENV_REBUILD

if not "%MFA_ENV_REBUILD%"=="1" goto :eof

echo [WARN] Python %MFA_ENV_PYTHON% Windows MFA

echo Python %MFA_PYTHON_VERSION% ...

call :remove_env_dir

if errorlevel 1 exit /b 1

goto :eof

:existing_env_ready

echo [OK] MFA

echo %MFA_EXE%

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

echo ..

call :download_acoustic_model korean_mfa

if errorlevel 1 exit /b 1

echo.

echo ..

call :download_acoustic_model japanese_mfa

if errorlevel 1 exit /b 1

call :cleanup_env_caches

if errorlevel 1 exit /b 1

call :cleanup_mfa_root_artifacts

if errorlevel 1 exit /b 1

call :cleanup_old_env_if_requested

echo.

echo . UTAU_Auto_OTO.exe .

if not "%NON_INTERACTIVE%"=="1" pause

exit /b 0

:install_micromamba

echo [INFO] Micromamba

echo.

echo [1/5] Preparing micromamba bootstrap... ^(~15MB^)

if "%HAS_CERTUTIL%"=="1" (

    call :resolve_micromamba_expected_md5

    if errorlevel 1 (

        echo [WARN] Could not resolve micromamba expected MD5 metadata. Continuing without strict checksum matching.

        set "MICROMAMBA_MD5_EXPECTED="

    )

) else (

    echo [WARN] certutil unavailable; skipping micromamba checksum metadata lookup.

    set "MICROMAMBA_MD5_EXPECTED="

)


if not exist "%MICROMAMBA_ARCHIVE%" (

    powershell -NoProfile -Command "& {$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MICROMAMBA_LATEST_URL%' -OutFile '%MICROMAMBA_ARCHIVE%'}"

    if errorlevel 1 (

        echo [FAILED] Micromamba

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

) else (

    echo .

)

call :verify_micromamba_archive

if errorlevel 1 (

    echo [FAILED] Micromamba

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

echo [OK] Completed.

echo.

echo [2/5] Installing and validating micromamba...

if not exist "%MICROMAMBA_EXE%" (

    if exist "%MICROMAMBA_ROOT%" (

        echo [INFO] Micromamba

        rmdir /s /q "%MICROMAMBA_ROOT%" >nul 2>nul

    )

    if "%HAS_TAR%"=="0" (

        echo [WARN] tar was not found. Attempting to download micromamba.exe directly.

        call :download_micromamba_exe

        if errorlevel 1 (

            echo [FAILED] Micromamba

            if not "%NON_INTERACTIVE%"=="1" pause

            exit /b 1

        )

        goto :micromamba_exe_post

    )

    mkdir "%MICROMAMBA_ROOT%" >nul 2>nul

    tar -xjf "%MICROMAMBA_ARCHIVE%" -C "%MICROMAMBA_ROOT%"

    if errorlevel 1 (

        echo [WARN] micromamba archive extraction failed. Attempting to download micromamba.exe directly.

        call :download_micromamba_exe

        if errorlevel 1 (

            echo [FAILED] Micromamba

            if not "%NON_INTERACTIVE%"=="1" pause

            exit /b 1

        )

        goto :micromamba_exe_post

    )

    if not exist "%MICROMAMBA_EXE%" if exist "%MICROMAMBA_ROOT%\bin\micromamba.exe" (

        set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\bin\micromamba.exe"

    )

    if not exist "%MICROMAMBA_EXE%" if exist "%MICROMAMBA_ROOT%\micromamba.exe" (

        set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\micromamba.exe"

    )

)

if not exist "%MICROMAMBA_EXE%" (

    echo [FAILED] Micromamba

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

echo [OK] Micromamba

echo.

echo [3/5] Installing Montreal Forced Aligner... ^(~3-10 min^)

if exist "%ENV_DIR%" if not exist "%MFA_EXE%" call :remove_env_dir

set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"

call :run_mfa_create_with_recovery

if errorlevel 1 (

    echo [FAILED] MFA

    if not "%NON_INTERACTIVE%"=="1" pause

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

echo [4/5] Installing language and audio dependencies...

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

echo ..

if exist "%MICROMAMBA_ARCHIVE%" del "%MICROMAMBA_ARCHIVE%" >nul 2>nul

echo [OK] MFA

echo.

echo [ / .. ^( 1~2 )

call :download_acoustic_model korean_mfa

if errorlevel 1 exit /b 1

call :download_acoustic_model japanese_mfa

if errorlevel 1 exit /b 1

call :cleanup_env_caches

if errorlevel 1 exit /b 1

call :cleanup_mfa_root_artifacts

if errorlevel 1 exit /b 1

call :cleanup_old_env_if_requested

echo.

echo ====================================================

echo .

echo.

echo 1) UTAU_Auto_OTO.exe ^( )

echo 2) run.bat ^( )

echo 3) "3. Voice Alignment"

echo ====================================================

echo.

if not "%NON_INTERACTIVE%"=="1" pause

exit /b 0

:run_mfa_create_with_recovery

set "MFA_CREATE_LOG1=%TEMP%\utau_auto_oto_mfa_create_1_%RANDOM%_%RANDOM%.log"

set "MFA_CREATE_LOG2=%TEMP%\utau_auto_oto_mfa_create_2_%RANDOM%_%RANDOM%.log"

set "MFA_CREATE_LOG3=%TEMP%\utau_auto_oto_mfa_create_3_%RANDOM%_%RANDOM%.log"

call :run_mfa_create_once "%MFA_CREATE_LOG1%"

if not errorlevel 1 exit /b 0

call :is_mfa_lock_error "%MFA_CREATE_LOG1%" LOCK_ERR_1

if not "%LOCK_ERR_1%"=="1" exit /b 1

echo [WARN] Detected MFA env lock during micromamba create.

call :release_env_lock_processes

echo [INFO] Retrying MFA create after lock cleanup...

call :run_mfa_create_once "%MFA_CREATE_LOG2%"

if not errorlevel 1 exit /b 0

call :is_mfa_lock_error "%MFA_CREATE_LOG2%" LOCK_ERR_2

if not "%LOCK_ERR_2%"=="1" exit /b 1

if "%AUTO_CLEAN_DONE%"=="1" exit /b 1

echo [WARN] Lock error persisted. Running automatic clean recovery...

call :run_clean_recovery

if errorlevel 1 exit /b 1

set "AUTO_CLEAN_DONE=1"

echo [INFO] Retrying MFA create after clean recovery...

call :run_mfa_create_once "%MFA_CREATE_LOG3%"

if not errorlevel 1 exit /b 0

exit /b 1

:run_mfa_create_once

set "MFA_CREATE_LOG=%~1"

if exist "%MFA_CREATE_LOG%" del "%MFA_CREATE_LOG%" >nul 2>nul

"%MICROMAMBA_EXE%" create -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge python=%MFA_PYTHON_VERSION% montreal-forced-aligner colorama >"%MFA_CREATE_LOG%" 2>&1

set "MFA_CREATE_RC=%ERRORLEVEL%"

type "%MFA_CREATE_LOG%"

if "%MFA_CREATE_RC%"=="0" exit /b 0

exit /b 1

:is_mfa_lock_error

set "%~2=0"

if "%~1"=="" goto :eof

if not exist "%~1" goto :eof

findstr /I /C:"remove_all" /C:"being used by another process" /C:"The process cannot access the file because it is being used by another process" /C:"access is denied" "%~1" >nul 2>nul

if not errorlevel 1 set "%~2=1"

goto :eof

:release_env_lock_processes

echo [INFO] Attempting to release MFA env lock...

set "UTOA_ENV_DIR_LOCK_TARGET=%ENV_DIR%"

powershell -NoProfile -Command "$ErrorActionPreference='SilentlyContinue'; $target=[Regex]::Escape($env:UTOA_ENV_DIR_LOCK_TARGET); $names=@('python.exe','pythonw.exe','mfa.exe','micromamba.exe','conda.exe','UTAU_Auto_OTO.exe'); Get-CimInstance Win32_Process | Where-Object { $_.Name -in $names -and ((($_.CommandLine) -and $_.CommandLine -match $target) -or (($_.ExecutablePath) -and $_.ExecutablePath -match $target)) } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; Write-Host ('[INFO] Stopped lock process: ' + $_.Name + ' pid=' + $_.ProcessId) } catch {} }; Start-Sleep -Seconds 3"

set "UTOA_ENV_DIR_LOCK_TARGET="

taskkill /F /IM mfa.exe >nul 2>nul

taskkill /F /IM micromamba.exe >nul 2>nul

taskkill /F /IM UTAU_Auto_OTO.exe >nul 2>nul

timeout /t 2 /nobreak >nul

goto :eof

:run_clean_recovery

echo [INFO] Running clean recovery for MFA runtime...

call :release_env_lock_processes

if exist "%ENV_DIR%" (

    call :remove_env_dir

    if errorlevel 1 (

        echo [FAILED] Could not remove locked MFA env during clean recovery.

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

)

:micromamba_exe_post

if exist "%MICROMAMBA_EXE%" (

    "%MICROMAMBA_EXE%" clean -a -y -r "%MICROMAMBA_ROOT%" >nul 2>nul

)

echo [OK] Clean recovery complete.

goto :eof

:bootstrap_python_tools

echo MFA Python ..

if not exist "%ENV_DIR%\python.exe" (

    echo [FAILED] MFA Python

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

call :run_env_python -c "import pip, pkg_resources, wheel" >nul 2>nul

if not errorlevel 1 (

    echo [OK] pip/setuptools/wheel

    call :ensure_seaborn_dependency

    goto :eof

)

echo [INFO] pip/setuptools/wheel

call :run_env_python -m ensurepip --upgrade

if errorlevel 1 (

    echo [WARN] ensurepip . pip

)

echo [INFO] pip module availability check...

call :run_env_python -m pip --version >nul 2>nul

if errorlevel 1 (

    echo [WARN] python -m pip failed. Retrying ensurepip with --default-pip...

    call :run_env_python -m ensurepip --upgrade --default-pip

)

call :ensure_seaborn_dependency

set "PYTOOLS_REPAIR_OK=0"

call :run_env_python -m pip install --upgrade --force-reinstall "setuptools<81" wheel

if not errorlevel 1 set "PYTOOLS_REPAIR_OK=1"

if "%PYTOOLS_REPAIR_OK%"=="0" (

    echo [WARN] force-reinstall path failed. Retrying with normal upgrade...

    call :run_env_python -m pip install --upgrade "setuptools<81" wheel

    if not errorlevel 1 set "PYTOOLS_REPAIR_OK=1"

)

if "%PYTOOLS_REPAIR_OK%"=="0" (

    echo [FAILED] pip/setuptools/wheel

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

call :run_env_python -c "import pip, pkg_resources, wheel" >nul 2>nul

if errorlevel 1 (

    echo [FAILED] Python

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

echo [OK] pip/setuptools/wheel

goto :eof

:ensure_seaborn_dependency

if not exist "%ENV_DIR%\python.exe" goto :eof

call :run_env_python -c "import seaborn" >nul 2>nul

if not errorlevel 1 (

    echo [OK] seaborn

    goto :eof

)

echo [INFO] Installing MFA dependency: seaborn

if exist "%MICROMAMBA_EXE%" (

    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge seaborn >nul 2>nul

    if not errorlevel 1 (

        call :run_env_python -c "import seaborn" >nul 2>nul

        if not errorlevel 1 (

            echo [OK] seaborn

            goto :eof

        )

    ) else (

        echo [WARN] micromamba seaborn install failed. Falling back to pip...

    )

)

call :run_env_python -m pip install --upgrade seaborn >nul 2>nul

call :run_env_python -c "import seaborn" >nul 2>nul

if errorlevel 1 (

    echo [WARN] seaborn install failed. Continuing without seaborn.

) else (

    echo [OK] seaborn

)

goto :eof

:install_japanese_support

echo ..

set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"

call :ensure_mfa_entrypoint

if errorlevel 1 exit /b 1

if exist "%MICROMAMBA_EXE%" (

    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge spacy sudachipy sudachidict-core

    if errorlevel 1 (

        echo [FAILED] Operation failed. Check the log output.

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

    goto :eof

)

echo [WARN] Micromamba

if exist "%ENV_DIR%\python.exe" (

    echo [INFO] Micromamba 繝ｻ・ｸ螟句ｷ晢ｽｧﾂ: pip fallback繝ｻ・ｼ繝ｻ繝ｻ繝ｻ・ｼ繝ｻ・ｸ繝ｻ・ｴ tokenizer 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ菫ｯ・･・ｼ 繝ｻ鄂ｹ蟾｡・托ｽｩ繝ｻ螢ｱ蜈ｱ.

    call :run_env_python -m pip install --upgrade spacy sudachipy sudachidict-core

    if errorlevel 1 (

        echo [WARN] pip fallback 繝ｻ・､繝ｻ菫ｯ蟾｡ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ. 繝ｻ・ｼ繝ｻ・ｸ繝ｻ・ｴ 繝ｻ邁ｿ・ｰ・ｬ 繝ｻ繝ｻ繝ｻ・ｰ螟仰繝ｻ繝ｻ荵ｱ繝ｻ繝ｻ繝ｻ・ｬ繝ｻ鄂ｹ蟾｡繝ｻ・ｩ繝ｻ螢ｱ蜈ｱ.

    ) else (

        echo [OK] Japanese tokenizer deps installed via pip fallback.

    )

)

goto :eof

:install_audio_deps

echo ^(libsndfile^)...

call :verify_audio_deps

if not errorlevel 1 goto :eof

if exist "%MICROMAMBA_EXE%" (

    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge libsndfile pysoundfile

    if errorlevel 1 (

        echo [FAILED] Operation failed. Check the log output.

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

    call :verify_audio_deps

    if errorlevel 1 (

        echo [FAILED] libsndfile/soundfile verification failed after micromamba install.

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

    goto :eof

)

if exist "%ENV_DIR%\Scripts\conda.exe" (

    "%ENV_DIR%\Scripts\conda.exe" install -y --solver classic -p "%ENV_DIR%" -c conda-forge libsndfile pysoundfile

    if errorlevel 1 (

        echo [FAILED] Operation failed. Check the log output.

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

    call :verify_audio_deps

    if errorlevel 1 (

        echo [FAILED] libsndfile/soundfile verification failed after conda install.

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

    goto :eof

)

if exist "%ENV_DIR%\python.exe" (

    call :run_env_python -m pip install --upgrade soundfile

    call :verify_audio_deps

    if not errorlevel 1 goto :eof

)

echo [WARN] micromamba/conda/pip

goto :eof

:verify_audio_deps

if not exist "%ENV_DIR%\python.exe" exit /b 1

setlocal EnableExtensions DisableDelayedExpansion

set "OLD_PATH=%PATH%"

set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%OLD_PATH%"

call :run_env_python -c "import soundfile" >nul 2>nul

set "VERIFY_RC=%ERRORLEVEL%"

endlocal & set "VERIFY_RC=%VERIFY_RC%"

if "%VERIFY_RC%"=="0" exit /b 0

exit /b 1

:install_textgrid

echo textgrid ..

if not exist "%ENV_DIR%\python.exe" (

    echo [FAILED] MFA Python

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

call :run_env_python -c "import textgrid" >nul 2>nul

if not errorlevel 1 goto :eof

echo [INFO] textgrid

call :run_env_python -m pip install --upgrade "textgrid>=1.5"

if errorlevel 1 (

    echo [FAILED] textgrid

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

goto :eof

:verify_textgrid

if not exist "%ENV_DIR%\python.exe" (

    echo [FAILED] MFA Python

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

call :run_env_python -c "import textgrid" >nul 2>nul

if errorlevel 1 (

    echo [FAILED] textgrid import

    echo setup_mfa.bat

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

goto :eof

:install_ml_requirements

echo ML ..

if not exist "%ENV_DIR%\python.exe" (

    echo [FAILED] MFA Python

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

set "REQ_BASE=%APP_DIR%\requirements.txt"
set "REQ_ML=%APP_DIR%\requirements-ml.txt"

if exist "%MICROMAMBA_EXE%" (

    echo [INFO] micromamba ML

    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge pandas pyarrow lightgbm onnxruntime

    if errorlevel 1 (

        echo [WARN] micromamba ML . pip

    ) else (

        echo [OK] micromamba ML

    )

)

if exist "%REQ_BASE%" (

    echo [INFO] requirements.txt

    call :run_env_python -m pip install --upgrade -r "%REQ_BASE%"

    if errorlevel 1 (

        echo [WARN] requirements.txt install failed. Continuing to ML runtime verification.

    )

) else (

    echo [WARN] requirements.txt not found at "%REQ_BASE%". Skipping optional base pip sync.

)

if exist "%REQ_ML%" (

    echo [INFO] requirements-ml.txt

    call :run_env_python -m pip install --upgrade -r "%REQ_ML%"

    if errorlevel 1 (

        if exist "%MICROMAMBA_EXE%" (

            echo [WARN] requirements-ml.txt pip install failed after micromamba ML install. Continuing to verification.

        ) else (

            echo [FAILED] ML

            echo lightgbm Microsoft Visual C++ Build Tools .

            if not "%NON_INTERACTIVE%"=="1" pause

            exit /b 1

        )

    )

    goto :ml_install_done

)

if exist "%MICROMAMBA_EXE%" (

    echo [WARN] requirements-ml.txt not found at "%REQ_ML%". Using micromamba-installed ML packages only.

) else (

    echo [INFO] requirements-ml.txt missing. Installing minimal ML packages via pip.

    call :run_env_python -m pip install --upgrade pandas pyarrow lightgbm onnxruntime

    if errorlevel 1 (

        echo [FAILED] ML

        echo lightgbm Microsoft Visual C++ Build Tools .

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

)

:ml_install_done

echo [OK] ML

goto :eof

:verify_ml_runtime

if not exist "%ENV_DIR%\python.exe" (

    echo [FAILED] MFA Python

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

call :run_env_python -c "import pandas, pyarrow, lightgbm, onnxruntime" >nul 2>nul

if errorlevel 1 (

    echo [FAILED] ML import . pandas/pyarrow/lightgbm/onnxruntime

    if exist "%MICROMAMBA_EXE%" (

        echo : "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge pandas pyarrow lightgbm onnxruntime

    ) else (

        echo setup_mfa.bat --with-ml

    )

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

goto :eof

:cleanup_env_caches

echo ..

if exist "%MICROMAMBA_EXE%" (

    "%MICROMAMBA_EXE%" clean -a -y -r "%MICROMAMBA_ROOT%" >nul 2>nul

)

if exist "%ENV_DIR%\python.exe" (

    call :run_env_python -m pip cache purge >nul 2>nul

)

echo [OK] Completed.

goto :eof

:cleanup_mfa_root_artifacts

echo [INFO] Cleaning stale MFA root artifacts...

if not defined MFA_SHARED_ROOT goto :cleanup_mfa_root_artifacts_done

if not exist "%MFA_SHARED_ROOT%" mkdir "%MFA_SHARED_ROOT%" >nul 2>nul

if exist "%ENV_DIR%\.mfa_root_ascii\pretrained_models" (

    call :sync_bundle_tree "%ENV_DIR%\.mfa_root_ascii\pretrained_models" "%MFA_SHARED_ROOT%\pretrained_models"

)

if exist "%ENV_DIR%\.mfa_root_ascii\extracted_models" if not exist "%MFA_SHARED_ROOT%\extracted_models" (

    call :sync_bundle_tree "%ENV_DIR%\.mfa_root_ascii\extracted_models" "%MFA_SHARED_ROOT%\extracted_models"

)

if exist "%ENV_DIR%\.mfa_root_ascii" (

    rmdir /s /q "%ENV_DIR%\.mfa_root_ascii" >nul 2>nul

    if exist "%ENV_DIR%\.mfa_root_ascii" (

        echo [WARN] Failed to remove legacy MFA root from env: "%ENV_DIR%\.mfa_root_ascii"

    ) else (

        echo [OK] Removed legacy MFA root from env.

    )

)

if exist "%ENV_DIR%\.mfa_root_ascii_p*" for /d %%D in ("%ENV_DIR%\.mfa_root_ascii_p*") do (

    rmdir /s /q "%%~fD" >nul 2>nul

)

if exist "%APP_DIR%\.mfa_root_ascii_p*" for /d %%D in ("%APP_DIR%\.mfa_root_ascii_p*") do (

    rmdir /s /q "%%~fD" >nul 2>nul

)

call :prune_mfa_shared_root_cache_dirs

:cleanup_mfa_root_artifacts_done

echo [OK] MFA root artifact cleanup completed.

goto :eof

:prune_mfa_shared_root_cache_dirs

if not defined MFA_SHARED_ROOT goto :eof

if not exist "%MFA_SHARED_ROOT%" goto :eof

setlocal EnableDelayedExpansion

for /d %%D in ("%MFA_SHARED_ROOT%\*") do (

    set "MFA_ROOT_NAME=%%~nxD"
    set "MFA_ROOT_KEEP=0"
    if /I "!MFA_ROOT_NAME!"=="pretrained_models" set "MFA_ROOT_KEEP=1"
    if /I "!MFA_ROOT_NAME!"=="extracted_models" set "MFA_ROOT_KEEP=1"
    if /I "!MFA_ROOT_KEEP!"=="0" (

        rmdir /s /q "%%~fD" >nul 2>nul

    )

)

endlocal

goto :eof

:install_korean_support

echo ..

call :ensure_mfa_entrypoint

if errorlevel 1 exit /b 1

if not exist "%ENV_DIR%\python.exe" (

    echo [FAILED] MFA Python

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

call :check_korean_tokenizer_ready

if "%KOREAN_TOKENIZER_OK%"=="1" goto :patch_korean_support

echo [INFO] Installing Korean tokenizer dependency: jamo

call :run_env_python -m pip install --upgrade jamo

if errorlevel 1 (

    echo [FAILED] Korean tokenizer dependency install failed: jamo

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

call :check_korean_tokenizer_ready

if "%KOREAN_TOKENIZER_OK%"=="1" goto :patch_korean_support

echo [INFO] Installing Korean tokenizer backend: python-mecab-ko ^(wheel-only^)

call :run_env_python -m pip uninstall -y mecab-python3 >nul 2>nul

call :run_env_python -m pip install --upgrade --force-reinstall --only-binary=:all: python-mecab-ko python-mecab-ko-dic

if not errorlevel 1 (

    call :check_korean_tokenizer_ready

    if "%KOREAN_TOKENIZER_OK%"=="1" goto :patch_korean_support

    call :ensure_mecab_module_shim

    call :check_korean_tokenizer_ready

    if "%KOREAN_TOKENIZER_OK%"=="1" goto :patch_korean_support

) else (

    echo [WARN] python-mecab-ko install failed or wheel unavailable.

)

if /i "%UTOA_ENABLE_MECAB_PYTHON3_FALLBACK%"=="1" (

    echo [INFO] Installing Korean tokenizer backend fallback: mecab-python3 ^(wheel-only^)

    call :run_env_python -m pip uninstall -y python-mecab-ko python-mecab-ko-dic >nul 2>nul

    call :run_env_python -m pip install --upgrade --only-binary=:all: mecab-python3

    if not errorlevel 1 (

        call :check_korean_tokenizer_ready

        if "%KOREAN_TOKENIZER_OK%"=="1" goto :patch_korean_support

    ) else (

        echo [WARN] mecab-python3 install failed or wheel unavailable.

    )

) else (

    echo [INFO] Skipping mecab-python3 fallback by default.
    echo [INFO] Set UTOA_ENABLE_MECAB_PYTHON3_FALLBACK=1 to enable fallback.

)

echo [FAILED] Korean tokenizer backend install failed.
echo        Tried: python-mecab-ko ^(wheel-only^), mecab-python3 ^(wheel-only^).
echo        No source build is attempted on Windows.

if not "%NON_INTERACTIVE%"=="1" pause

exit /b 1

:ensure_mecab_module_shim

if not exist "%ENV_DIR%\python.exe" goto :eof

call :run_env_python -c "import importlib.util as u, pathlib, sysconfig; has_mecab=u.find_spec('mecab') is not None; has_mecab_caps=u.find_spec('MeCab') is not None; purelib=sysconfig.get_paths().get('purelib'); target=pathlib.Path(purelib) / 'mecab.py' if purelib else None; (not has_mecab and has_mecab_caps and target and not target.exists()) and target.write_text('from MeCab import *\\n', encoding='utf-8')" >nul 2>nul

goto :eof

:patch_korean_support

set "UTOA_APP_DIR=%APP_DIR%"
set "UTOA_MFA_EXE=%MFA_EXE%"

call :run_env_python -c "import os,sys; app=os.environ.get('UTOA_APP_DIR',''); cand=[os.path.join(app,'UTAU_Auto_OTO'), app]; [sys.path.insert(0,p) for p in cand if p and p not in sys.path]; from core.mfa_runner import patch_mfa_korean_support; patch_mfa_korean_support(os.environ.get('UTOA_MFA_EXE',''))" >nul 2>nul

if not errorlevel 1 exit /b 0

echo [WARN] MFA Korean patch step failed. Continuing.
exit /b 0

:check_korean_tokenizer_ready

set "KOREAN_TOKENIZER_OK=0"

if not exist "%ENV_DIR%\python.exe" goto :eof

call :run_env_python -c "import sys,jamo,subprocess; cmds=['from mecab import MeCab','from mecab import MeCab; MeCab()','from mecab import Tagger','from mecab import Tagger; Tagger()','import MeCab as M','import MeCab as M; M.Tagger()','import mecab_ko']; ok=any(subprocess.run([sys.executable,'-c',c],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 for c in cmds); sys.exit(0 if ok else 1)" >nul 2>nul

if not errorlevel 1 set "KOREAN_TOKENIZER_OK=1"

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

echo [INFO] Micromamba MD5 : %MICROMAMBA_MD5_EXPECTED%

exit /b 0

:download_micromamba_exe

if not exist "%MICROMAMBA_ROOT%" mkdir "%MICROMAMBA_ROOT%" >nul 2>nul

powershell -NoProfile -Command ^

 "$ErrorActionPreference='Stop';" ^

 " [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;" ^

 " $url='%MICROMAMBA_EXE_URL%';" ^

 " $dst=Join-Path '%MICROMAMBA_ROOT%' 'micromamba.exe';" ^

 " Invoke-WebRequest -Uri $url -OutFile $dst -TimeoutSec 120 -UseBasicParsing;" ^

 " if (-not (Test-Path -LiteralPath $dst)) { throw 'micromamba.exe download failed' }"

if errorlevel 1 exit /b 1

if not exist "%MICROMAMBA_ROOT%\micromamba.exe" exit /b 1

set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\micromamba.exe"

exit /b 0

:verify_micromamba_archive

if not exist "%MICROMAMBA_ARCHIVE%" (

    echo [FAILED] Micromamba : %MICROMAMBA_ARCHIVE%

    exit /b 1

)

set "MICROMAMBA_MD5_ACTUAL="

if "%HAS_CERTUTIL%"=="0" (

    echo [WARN] certutil unavailable. Skipping micromamba archive MD5 verification.

    exit /b 0

)

for /f "tokens=1" %%h in ('certutil -hashfile "%MICROMAMBA_ARCHIVE%" MD5 ^| findstr /R /I "^[0-9A-F][0-9A-F]"') do (

    if not defined MICROMAMBA_MD5_ACTUAL set "MICROMAMBA_MD5_ACTUAL=%%h"

)

if not defined MICROMAMBA_MD5_ACTUAL (

    echo [FAILED] Micromamba MD5

    exit /b 1

)

echo [INFO] Micromamba MD5: %MICROMAMBA_MD5_ACTUAL%

if defined MICROMAMBA_MD5_EXPECTED (

    if /I not "%MICROMAMBA_MD5_ACTUAL%"=="%MICROMAMBA_MD5_EXPECTED%" (

        echo [FAILED] Micromamba

        echo %MICROMAMBA_MD5_EXPECTED%

        echo %MICROMAMBA_MD5_ACTUAL%

        del "%MICROMAMBA_ARCHIVE%" >nul 2>nul

        exit /b 1

    )

    echo [OK] Micromamba

) else (

    echo [WARN] Expected micromamba MD5 was not resolved. Proceeding without strict checksum comparison.

)

exit /b 0

:handle_old_env

echo.

echo [WARN] Legacy shared MFA environment was found:

echo        %OLD_ENV_DIR%

echo.

echo Choose how to handle the legacy environment:

echo [M] Migrate: keep now, delete after successful install

echo [D] Delete now

echo [K] Keep as-is (recommended only for troubleshooting)

choice /C MDK /N /M "Select action (M/D/K): "

if errorlevel 3 goto :keep_old_env

if errorlevel 2 goto :delete_old_env_now

if errorlevel 1 goto :migrate_old_env

goto :eof

:migrate_old_env

set "DELETE_OLD_AFTER_INSTALL=1"

echo [INFO] Legacy environment will be removed after successful install.

goto :eof

:delete_old_env_now

call :remove_dir "%OLD_ENV_DIR%"

call :remove_dir "%OLD_MICROMAMBA_ROOT%"

goto :eof

:keep_old_env

echo [INFO] Keeping legacy environment unchanged.

goto :eof

:cleanup_old_env_if_requested

if not "%DELETE_OLD_AFTER_INSTALL%"=="1" goto :eof

if /i "%OLD_ENV_DIR%"=="%ENV_DIR%" goto :eof

call :remove_dir "%OLD_ENV_DIR%"

call :remove_dir "%OLD_MICROMAMBA_ROOT%"

echo [OK] MFA

goto :eof

:remove_dir

set "TARGET_DIR=%~1"

if "%TARGET_DIR%"=="" goto :eof

if not exist "%TARGET_DIR%" goto :eof

echo [INFO] %TARGET_DIR%

rmdir /s /q "%TARGET_DIR%" >nul 2>nul

if exist "%TARGET_DIR%" (

    echo [WARN] : %TARGET_DIR%

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

echo [FAILED] MFA

if not "%NON_INTERACTIVE%"=="1" pause

exit /b 1

:download_acoustic_model

set "MODEL_NAME=%~1"

if not defined MODEL_NAME (

    echo [FAILED] Operation failed. Check the log output.

    if not "%NON_INTERACTIVE%"=="1" pause

    exit /b 1

)

set "MODEL_CACHE_PATH=%MFA_SHARED_ROOT%\pretrained_models\acoustic\%MODEL_NAME%.zip"

if exist "%MODEL_CACHE_PATH%" (

    echo [INFO] Acoustic model already cached: %MODEL_NAME%

    exit /b 0

)

if exist "%ENV_DIR%\python.exe" (

    set "CONDA_PREFIX=%ENV_DIR%"

    set "MFA_ROOT_DIR=%MFA_SHARED_ROOT%"

    set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%PATH%"

    call :run_env_python -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME%

    if errorlevel 1 (

        echo [WARN] Model download failed once. Retrying with --ignore_cache...

        call :run_env_python -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME% --ignore_cache

    )

    exit /b %errorlevel%

)

if exist "%MICROMAMBA_EXE%" (

    set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"

    set "MFA_ROOT_DIR=%MFA_SHARED_ROOT%"

    "%MICROMAMBA_EXE%" run -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" python -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME%

    if errorlevel 1 (

        echo [WARN] Model download failed once. Retrying with --ignore_cache...

        "%MICROMAMBA_EXE%" run -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" python -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME% --ignore_cache

    )

    exit /b %errorlevel%

)

echo [FAILED] MFA

if not "%NON_INTERACTIVE%"=="1" pause

exit /b 1

:resolve_runtime_bundle_dir

set "MFA_RUNTIME_BUNDLE_DIR="

if exist "%APP_DIR%\mfa_runtime_bundle\bundle_manifest.json" set "MFA_RUNTIME_BUNDLE_DIR=%APP_DIR%\mfa_runtime_bundle"

if not defined MFA_RUNTIME_BUNDLE_DIR if exist "%APP_DIR%\UTAU_Auto_OTO\mfa_runtime_bundle\bundle_manifest.json" set "MFA_RUNTIME_BUNDLE_DIR=%APP_DIR%\UTAU_Auto_OTO\mfa_runtime_bundle"

if not defined MFA_RUNTIME_BUNDLE_DIR if defined SETUP_MFA_DIR_NORM if exist "%SETUP_MFA_DIR_NORM%\mfa_runtime_bundle\bundle_manifest.json" set "MFA_RUNTIME_BUNDLE_DIR=%SETUP_MFA_DIR_NORM%\mfa_runtime_bundle"

if not defined MFA_RUNTIME_BUNDLE_DIR if defined SETUP_MFA_DIR_NORM if exist "%SETUP_MFA_DIR_NORM%\..\mfa_runtime_bundle\bundle_manifest.json" (

    for %%I in ("%SETUP_MFA_DIR_NORM%\..\mfa_runtime_bundle") do set "MFA_RUNTIME_BUNDLE_DIR=%%~fI"

)

goto :eof

:restore_from_runtime_bundle

if not defined MFA_RUNTIME_BUNDLE_DIR goto :eof

if not exist "%MFA_RUNTIME_BUNDLE_DIR%\bundle_manifest.json" goto :eof

if not exist "%MFA_EXE%" if exist "%MFA_RUNTIME_BUNDLE_DIR%\.env\Scripts\mfa.exe" (

    echo [INFO] Restoring bundled MFA environment...

    call :sync_bundle_tree "%MFA_RUNTIME_BUNDLE_DIR%\.env" "%ENV_DIR%"

    if errorlevel 1 (

        echo [WARN] Failed to restore bundled MFA environment. Falling back to online install.

    ) else (

        set "BUNDLE_RESTORED=1"

    )

)

if not exist "%MICROMAMBA_EXE%" if exist "%MFA_RUNTIME_BUNDLE_DIR%\micromamba\Library\bin\micromamba.exe" (

    echo [INFO] Restoring bundled micromamba runtime...

    call :sync_bundle_tree "%MFA_RUNTIME_BUNDLE_DIR%\micromamba" "%MICROMAMBA_ROOT%"

    if errorlevel 1 (

        echo [WARN] Failed to restore bundled micromamba runtime.

    ) else (

        set "BUNDLE_RESTORED=1"

    )

)

if not exist "%MFA_SHARED_ROOT%\pretrained_models\acoustic" if exist "%MFA_RUNTIME_BUNDLE_DIR%\.mfa_root_ascii\pretrained_models\acoustic" (

    echo [INFO] Restoring bundled MFA acoustic model cache...

    call :sync_bundle_tree "%MFA_RUNTIME_BUNDLE_DIR%\.mfa_root_ascii" "%MFA_SHARED_ROOT%"

    if not errorlevel 1 set "BUNDLE_RESTORED=1"

)

if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.exe" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"

if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.bat" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"

if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.cmd" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.cmd"

if not exist "%MICROMAMBA_EXE%" if exist "%MICROMAMBA_ROOT%\Library\bin\micromamba.exe" set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\Library\bin\micromamba.exe"

if not exist "%MICROMAMBA_EXE%" if exist "%MICROMAMBA_ROOT%\bin\micromamba.exe" set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\bin\micromamba.exe"

if "%BUNDLE_RESTORED%"=="1" echo [OK] Bundled MFA runtime restore completed.

goto :eof

:sync_bundle_tree

set "BUNDLE_SRC=%~1"

set "BUNDLE_DST=%~2"

if "%BUNDLE_SRC%"=="" exit /b 1

if "%BUNDLE_DST%"=="" exit /b 1

if not exist "%BUNDLE_SRC%" exit /b 1

if exist "%BUNDLE_DST%" rmdir /s /q "%BUNDLE_DST%" >nul 2>nul

mkdir "%BUNDLE_DST%" >nul 2>nul

robocopy "%BUNDLE_SRC%" "%BUNDLE_DST%" /E /COPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >nul

set "ROBOCOPY_RC=%ERRORLEVEL%"

if %ROBOCOPY_RC% GEQ 8 exit /b 1

exit /b 0

:remove_env_dir

if not exist "%ENV_DIR%" goto :eof

echo [INFO] %ENV_DIR%

setlocal EnableDelayedExpansion
set "REMOVE_RETRY=0"
:remove_env_retry_loop
rmdir /s /q "%ENV_DIR%" >nul 2>nul

if exist "%ENV_DIR%" (

    set /a REMOVE_RETRY+=1

    if !REMOVE_RETRY! GEQ 4 (

        endlocal
        echo [FAILED] MFA

        if not "%NON_INTERACTIVE%"=="1" pause

        exit /b 1

    )

    echo [WARN] MFA env delete retry !REMOVE_RETRY!/3 ...

    call :release_env_lock_processes

    timeout /t 2 /nobreak >nul

    goto :remove_env_retry_loop


)

endlocal
goto :eof

:run_env_python

if not exist "%ENV_DIR%\python.exe" exit /b 1

setlocal EnableExtensions DisableDelayedExpansion

set "OLD_PATH=%PATH%"
set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\bin;%OLD_PATH%"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONEXECUTABLE="
set "__PYVENV_LAUNCHER__="
set "PYTHONNOUSERSITE=1"

"%ENV_DIR%\python.exe" %*

set "RUN_RC=%ERRORLEVEL%"

endlocal & exit /b %RUN_RC%

:get_env_python_version

set "%~2="

if exist "%~1\python.exe" (

    setlocal EnableExtensions DisableDelayedExpansion
    set "OLD_PATH=%PATH%"
    set "PATH=%~1;%~1\Scripts;%~1\Library\bin;%~1\Library\usr\bin;%~1\Library\mingw-w64\bin;%~1\bin;%OLD_PATH%"
    for /f "usebackq delims=" %%i in (`"%~1\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2^>nul`) do set "PY_VER=%%i"
    endlocal & set "%~2=%PY_VER%"

    goto :eof

)

if exist "%~1\Scripts\python.exe" (

    setlocal EnableExtensions DisableDelayedExpansion
    set "OLD_PATH=%PATH%"
    set "PATH=%~1;%~1\Scripts;%~1\Library\bin;%~1\Library\usr\bin;%~1\Library\mingw-w64\bin;%~1\bin;%OLD_PATH%"
    for /f "usebackq delims=" %%i in (`"%~1\Scripts\python.exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2^>nul`) do set "PY_VER=%%i"
    endlocal & set "%~2=%PY_VER%"

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

set "REQ_MAJOR="
set "REQ_MINOR="
for /f "tokens=1,2 delims=." %%a in ("%MFA_PYTHON_VERSION%") do (
    set /a REQ_MAJOR=%%a
    set /a REQ_MINOR=%%b
)

if not defined REQ_MAJOR (
    if %PY_MAJOR% GTR 3 (
        set "%~2=1"
        goto :eof
    )
    if %PY_MAJOR% EQU 3 if %PY_MINOR% GEQ 13 set "%~2=1"
    goto :eof
)

if not "%PY_MAJOR%"=="%REQ_MAJOR%" (
    set "%~2=1"
    goto :eof
)
if not "%PY_MINOR%"=="%REQ_MINOR%" set "%~2=1"

goto :eof
