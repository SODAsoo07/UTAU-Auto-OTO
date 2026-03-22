@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title UTAU Auto OTO - MFA Setup
@echo off

set "INSTALL_ML=0"
set "INSTALL_CUDA_RUNTIME=0"
set "DELETE_OLD_AFTER_INSTALL=0"
set "AUTO_ML=0"
set "INTERACTIVE=1"
set "PREFER_BUNDLED_RUNTIME=1"
set "ALLOW_DEGRADED_KO=1"
set "RETRY_COUNT_NETWORK=3"
set "RETRY_COUNT_PIP=3"
set "RETRY_COUNT_CREATE=2"
set "RETRY_COUNT_MODEL=2"
set "RETRY_WAIT_SECONDS=5"
set "MIN_FREE_SPACE_MB=4096"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="--non-interactive" set "INTERACTIVE=0"
if /i "%~1"=="--yes" set "INTERACTIVE=0"
if /i "%~1"=="--with-ml" set "INSTALL_ML=1"
if /i "%~1"=="--install-ml" set "INSTALL_ML=1"
if /i "%~1"=="--with-cuda" (
    set "INSTALL_ML=1"
    set "INSTALL_CUDA_RUNTIME=1"
)
if /i "%~1"=="--install-cuda" (
    set "INSTALL_ML=1"
    set "INSTALL_CUDA_RUNTIME=1"
)
if /i "%~1"=="--skip-bundled-runtime" set "PREFER_BUNDLED_RUNTIME=0"
if /i "%~1"=="--strict-korean-deps" set "ALLOW_DEGRADED_KO=0"
shift
goto :parse_args

:show_help
echo Usage: setup_mfa.bat [--with-ml] [--with-cuda] [--non-interactive] [--skip-bundled-runtime] [--strict-korean-deps]
echo Installs a local MFA environment using Micromamba.
echo Default runtime root: %%LOCALAPPDATA%%\UTAU_Auto_OTO_v3
echo Optional:
echo   --with-ml / --install-ml  Install ML dependencies (pandas/sklearn/lightgbm/pytorch).
echo   --with-cuda / --install-cuda  Also force CUDA-enabled torch runtime check/install.
echo   --non-interactive / --yes  Do not wait for key input ^(for installer automation^).
echo   --skip-bundled-runtime  Ignore bundled MFA runtime payload and force online bootstrap.
echo   --strict-korean-deps  Fail setup when Korean tokenizer deps are unavailable ^(default: degraded continue^).
exit /b 0

:args_done

echo ====================================================
echo   UTAU Auto OTO - MFA Lightweight Environment Setup
echo   This script only needs to run once.
echo ====================================================
echo.
echo [INFO] Initial install/recovery can take a while depending on network speed.
echo.
echo [INFO] First-time setup or repair can take 10-20 minutes.
echo [INFO] Keep this window open until setup is complete.
echo [INFO] Acoustic model download for Korean/Japanese may continue after base install.
echo.

set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
set "RUNTIME_ROOT_FROM_USER=0"
set "RUNTIME_ROOT=%UTOA_MFA_SHARED_ROOT%"
if defined RUNTIME_ROOT set "RUNTIME_ROOT_FROM_USER=1"
if not defined RUNTIME_ROOT (
    if defined LOCALAPPDATA (
        set "RUNTIME_ROOT=%LOCALAPPDATA%\UTAU_Auto_OTO_v3"
    ) else (
        set "RUNTIME_ROOT=%USERPROFILE%\AppData\Local\UTAU_Auto_OTO_v3"
    )
)
set "ASCII_RUNTIME_FALLBACK=%PUBLIC%\UTAU_Auto_OTO_v3"
if not defined ASCII_RUNTIME_FALLBACK set "ASCII_RUNTIME_FALLBACK=%SystemDrive%\Users\Public\UTAU_Auto_OTO_v3"
if not defined ASCII_RUNTIME_FALLBACK set "ASCII_RUNTIME_FALLBACK=C:\Users\Public\UTAU_Auto_OTO_v3"
if "%RUNTIME_ROOT_FROM_USER%"=="0" (
    call :path_requires_ascii_fallback "%RUNTIME_ROOT%"
    if not errorlevel 1 (
        echo [WARN] Auto-selected runtime path contains non-ASCII or shell-sensitive characters.
        echo        Falling back to ASCII-safe path: %ASCII_RUNTIME_FALLBACK%
        set "RUNTIME_ROOT=%ASCII_RUNTIME_FALLBACK%"
    )
) else (
    call :path_requires_ascii_fallback "%RUNTIME_ROOT%"
    if not errorlevel 1 (
        echo [WARN] User-provided runtime path may be unstable for MFA launcher:
        echo        %RUNTIME_ROOT%
        echo        Recommended ASCII path: %ASCII_RUNTIME_FALLBACK%
    )
)
if not exist "%RUNTIME_ROOT%" mkdir "%RUNTIME_ROOT%" >nul 2>nul
set "OLD_ENV_DIR="
set "OLD_MICROMAMBA_ROOT="
set "OLD_APP_ENV_DIR=%APP_DIR%\.env"
set "OLD_APP_MICROMAMBA_ROOT=%APP_DIR%\micromamba"
if exist "%OLD_APP_ENV_DIR%" (
    if /i not "%OLD_APP_ENV_DIR%"=="%RUNTIME_ROOT%\.env" (
        set "OLD_ENV_DIR=%OLD_APP_ENV_DIR%"
        set "OLD_MICROMAMBA_ROOT=%OLD_APP_MICROMAMBA_ROOT%"
    )
)
set "OLD_PUBLIC_ROOT=%PUBLIC%"
if not defined OLD_PUBLIC_ROOT set "OLD_PUBLIC_ROOT=%SystemDrive%\Users\Public"
if not defined OLD_PUBLIC_ROOT set "OLD_PUBLIC_ROOT=C:\Users\Public"
if not defined OLD_ENV_DIR set "OLD_ENV_DIR=%OLD_PUBLIC_ROOT%\UTAU_Auto_OTO_v3\.env"
if not defined OLD_MICROMAMBA_ROOT set "OLD_MICROMAMBA_ROOT=%OLD_PUBLIC_ROOT%\UTAU_Auto_OTO_v3\micromamba"
set "ENV_DIR=%RUNTIME_ROOT%\.env"
set "MICROMAMBA_ROOT=%RUNTIME_ROOT%\micromamba"
set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\Library\bin\micromamba.exe"
set "MICROMAMBA_ARCHIVE=%RUNTIME_ROOT%\micromamba-win-64-latest.tar.bz2"
set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
set "KO_WHEEL_DIR=%APP_DIR%\mfa_ko_wheels"
set "BUNDLED_RUNTIME_ROOT=%APP_DIR%\mfa_runtime_bundle"
set "BUNDLED_ENV_DIR=%BUNDLED_RUNTIME_ROOT%\.env"
set "BUNDLED_MICROMAMBA_ROOT=%BUNDLED_RUNTIME_ROOT%\micromamba"
set "BUNDLED_MFA_EXE=%BUNDLED_ENV_DIR%\Scripts\mfa.exe"
if not exist "%BUNDLED_MFA_EXE%" if exist "%BUNDLED_ENV_DIR%\Scripts\mfa.bat" set "BUNDLED_MFA_EXE=%BUNDLED_ENV_DIR%\Scripts\mfa.bat"
if not exist "%BUNDLED_MFA_EXE%" if exist "%BUNDLED_ENV_DIR%\Scripts\mfa.cmd" set "BUNDLED_MFA_EXE=%BUNDLED_ENV_DIR%\Scripts\mfa.cmd"
set "MFA_PYTHON_VERSION=3.10"
set "VC_REDIST_EXE=%RUNTIME_ROOT%\vc_redist.x64.exe"
set "MICROMAMBA_URL_API=https://micro.mamba.pm/api/micromamba/win-64/latest"
set "MICROMAMBA_URL_GITHUB_EXE=https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64"

echo [INFO] MFA runtime root: %RUNTIME_ROOT%
echo [INFO] MFA environment path: %ENV_DIR%
echo [INFO] MFA Micromamba path: %MICROMAMBA_ROOT%
if exist "%KO_WHEEL_DIR%" echo [INFO] Korean offline wheel bundle path: %KO_WHEEL_DIR%
echo [INFO] Setup options: ML=%INSTALL_ML%, CUDA=%INSTALL_CUDA_RUNTIME%, INTERACTIVE=%INTERACTIVE%, PREFER_BUNDLED_RUNTIME=%PREFER_BUNDLED_RUNTIME%, ALLOW_DEGRADED_KO=%ALLOW_DEGRADED_KO%
if exist "%BUNDLED_MFA_EXE%" (
    echo [INFO] Bundled MFA runtime payload detected: %BUNDLED_RUNTIME_ROOT%
)
call :run_step "Preflight checks (writable path + disk space)" :preflight_checks
if errorlevel 1 exit /b 1

if exist "%APP_DIR%\ML_models" set "AUTO_ML=1"
if exist "%APP_DIR%\models_installed\oto_ml" set "AUTO_ML=1"
if "%AUTO_ML%"=="1" if "%INSTALL_ML%"=="0" (
    echo [INFO] ML bundle assets detected. Enabling ML dependencies.
    set "INSTALL_ML=1"
)
if "%INSTALL_ML%"=="1" (
    where /q nvidia-smi
    if not errorlevel 1 (
        if "%INSTALL_CUDA_RUNTIME%"=="0" (
            echo [INFO] NVIDIA GPU detected. CUDA runtime bootstrap will be checked automatically.
            set "INSTALL_CUDA_RUNTIME=1"
        )
    )
)
if defined OLD_ENV_DIR if exist "%OLD_ENV_DIR%" (
    if /i not "%OLD_ENV_DIR%"=="%ENV_DIR%" (
        call :run_step "Legacy environment decision" :handle_old_env
        if errorlevel 1 exit /b 1
    )
)

if "%PREFER_BUNDLED_RUNTIME%"=="1" (
    call :run_step "Bundled runtime restore check" :try_restore_bundled_runtime
    if errorlevel 1 exit /b 1
)

if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.bat" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"

if exist "%MFA_EXE%" (
    call :check_existing_python_version
    if errorlevel 1 exit /b 1
)

if exist "%MFA_EXE%" (
    call :resolve_env_python_exe
    if not errorlevel 1 (
        call :run_step "Validate MFA launcher entrypoint" :ensure_mfa_entrypoint
        if errorlevel 1 (
            echo [WARN] Existing MFA launcher is not healthy. Rebuilding the local MFA env...
            call :run_step "Remove broken MFA env" :remove_env_dir
            if errorlevel 1 exit /b 1
            set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
        )
    )
)
call :resolve_env_python_exe
if not errorlevel 1 (
    call :run_step "Verify MFA python package import" :check_mfa_python_module
    if errorlevel 1 (
        echo [WARN] Existing MFA python package is not importable. Rebuilding the local MFA env...
        call :run_step "Remove non-importable MFA env" :remove_env_dir
        if errorlevel 1 exit /b 1
        set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
    )
)

if exist "%MFA_EXE%" (
    call :resolve_env_python_exe
    if not errorlevel 1 goto :existing_env_ready
)
call :run_step "Ensure VC++ runtime prerequisite" :ensure_vc_runtime
if errorlevel 1 exit /b 1
goto :install_micromamba

:check_existing_python_version
call :get_env_python_version "%ENV_DIR%" MFA_ENV_PYTHON
if not defined MFA_ENV_PYTHON goto :eof
echo [INFO] Existing MFA Python version: %MFA_ENV_PYTHON%
call :python_requires_rebuild "%MFA_ENV_PYTHON%" MFA_ENV_REBUILD
if "%MFA_ENV_REBUILD%"=="1" (
    echo [WARN] Python %MFA_ENV_PYTHON% is not compatible with the Windows MFA dependency flow.
    echo        Rebuilding the environment with Python %MFA_PYTHON_VERSION%...
    call :run_step "Remove incompatible MFA env" :remove_env_dir
    if errorlevel 1 exit /b 1
)
goto :eof

:existing_env_ready
echo [OK] MFA is already installed.
echo      Path: %MFA_EXE%
echo.
call :run_step "Bootstrap Python package tools" :bootstrap_python_tools
if errorlevel 1 exit /b 1
call :run_step "Install textgrid module" :install_textgrid
if errorlevel 1 exit /b 1
call :run_step "Verify textgrid import" :verify_textgrid
if errorlevel 1 exit /b 1
call :run_step "Install audio runtime dependencies" :install_audio_deps
if errorlevel 1 exit /b 1
call :run_step "Install Korean tokenizer support" :install_korean_support
if errorlevel 1 exit /b 1
call :run_step "Install Japanese tokenizer support" :install_japanese_support
if errorlevel 1 exit /b 1
if "%INSTALL_ML%"=="1" (
    call :run_step "Install optional ML dependencies" :install_ml_requirements
    if errorlevel 1 exit /b 1
    call :run_step "Verify optional ML runtime imports" :verify_ml_runtime
    if errorlevel 1 exit /b 1
    call :run_step "Check CUDA runtime bootstrap for NVIDIA" :ensure_cuda_runtime_for_nvidia
)
echo.
echo Checking Korean acoustic model...
call :run_step "Download Korean acoustic model" :download_acoustic_model korean_mfa
if errorlevel 1 exit /b 1
echo.
echo Checking Japanese acoustic model...
call :run_step "Download Japanese acoustic model" :download_acoustic_model japanese_mfa
if errorlevel 1 exit /b 1
call :run_step "Cleanup legacy environment (if requested)" :cleanup_old_env_if_requested
echo.
echo Done. You can now launch UTAU_Auto_OTO.exe.
call :maybe_pause
exit /b 0

:install_micromamba
echo [INFO] Using Micromamba bootstrap to reduce installation cost.
echo.
echo [1/5] Downloading Micromamba... ^(about 15MB^)
if not exist "%MICROMAMBA_ARCHIVE%" (
    call :run_step_with_retry "Download Micromamba payload" %RETRY_COUNT_NETWORK% :download_micromamba
    if errorlevel 1 (
        echo [FAILED] Micromamba download failed.
        call :maybe_pause
        exit /b 1
    )
) else (
    echo    Already downloaded. Skipping.
)
echo [OK] Download complete.
echo.

echo [2/5] Extracting Micromamba...
if not exist "%MICROMAMBA_EXE%" (
    if exist "%MICROMAMBA_ROOT%" (
        echo [INFO] Removing previous Micromamba root...
        rmdir /s /q "%MICROMAMBA_ROOT%" >nul 2>nul
    )
    mkdir "%MICROMAMBA_ROOT%" >nul 2>nul
    tar -xjf "%MICROMAMBA_ARCHIVE%" -C "%MICROMAMBA_ROOT%"
    if errorlevel 1 (
        echo [WARN] Micromamba extraction failed.
        echo [INFO] Falling back to direct micromamba.exe download...
        if not exist "%MICROMAMBA_ROOT%\Library\bin" mkdir "%MICROMAMBA_ROOT%\Library\bin" >nul 2>nul
        call :run_step_with_retry "Download Micromamba executable fallback" %RETRY_COUNT_NETWORK% :download_micromamba_exe_only
        if errorlevel 1 (
            echo [FAILED] Micromamba executable fallback download failed.
            call :maybe_pause
            exit /b 1
        )
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
    call :maybe_pause
    exit /b 1
)
echo [OK] Micromamba ready.
echo.

echo [3/5] Installing Montreal Forced Aligner... ^(3-10 min^)
call :run_step "Create MFA base environment (python + mfa)" :install_base_mfa_env
if errorlevel 1 exit /b 1
call :run_step "Ensure MFA launcher entrypoint" :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
call :run_step "Bootstrap Python package tools" :bootstrap_python_tools
if errorlevel 1 exit /b 1
call :run_step "Install textgrid module" :install_textgrid
if errorlevel 1 exit /b 1
call :run_step "Verify textgrid import" :verify_textgrid
if errorlevel 1 exit /b 1

echo [4/5] Installing Korean/Japanese tokenizer dependencies...
call :run_step "Install Korean tokenizer support" :install_korean_support
if errorlevel 1 exit /b 1
call :run_step "Install Japanese tokenizer support" :install_japanese_support
if errorlevel 1 exit /b 1
call :run_step "Install audio runtime dependencies" :install_audio_deps
if errorlevel 1 exit /b 1
if "%INSTALL_ML%"=="1" (
    call :run_step "Install optional ML dependencies" :install_ml_requirements
    if errorlevel 1 exit /b 1
    call :run_step "Verify optional ML runtime imports" :verify_ml_runtime
    if errorlevel 1 exit /b 1
    call :run_step "Check CUDA runtime bootstrap for NVIDIA" :ensure_cuda_runtime_for_nvidia
)

echo Cleaning up installer cache...
if exist "%MICROMAMBA_ARCHIVE%" del "%MICROMAMBA_ARCHIVE%" >nul 2>nul

echo [OK] MFA installed successfully.
echo.
echo [Final] Downloading Korean/Japanese acoustic models... ^(1-2 min^)
call :run_step "Download Korean acoustic model" :download_acoustic_model korean_mfa
if errorlevel 1 exit /b 1
call :run_step "Download Japanese acoustic model" :download_acoustic_model japanese_mfa
if errorlevel 1 exit /b 1
call :run_step "Cleanup legacy environment (if requested)" :cleanup_old_env_if_requested
echo.
echo ====================================================
echo   Setup complete.
echo   Next step:
echo   1) Launch UTAU_Auto_OTO.exe ^(release build^)
echo   2) Or run run.bat ^(source checkout^)
echo   3) Then click "3. Voice Alignment" to continue
echo ====================================================
echo.
call :maybe_pause
exit /b 0

:preflight_checks
call :check_writable_path "%RUNTIME_ROOT%"
if errorlevel 1 (
    echo [FAILED] Runtime root is not writable: %RUNTIME_ROOT%
    echo        Check folder permissions or choose another path via UTOA_MFA_SHARED_ROOT.
    call :maybe_pause
    exit /b 1
)
call :check_free_space_mb "%RUNTIME_ROOT%" %MIN_FREE_SPACE_MB%
if errorlevel 1 (
    echo [FAILED] Not enough free disk space on runtime drive.
    echo        Required at least %MIN_FREE_SPACE_MB% MB free.
    call :maybe_pause
    exit /b 1
)
goto :eof

:check_writable_path
set "CHECK_PATH=%~1"
if "%CHECK_PATH%"=="" exit /b 1
if not exist "%CHECK_PATH%" mkdir "%CHECK_PATH%" >nul 2>nul
if not exist "%CHECK_PATH%" exit /b 1
set "WRITE_PROBE=%CHECK_PATH%\.__utoa_write_probe_%RANDOM%%RANDOM%.tmp"
>"%WRITE_PROBE%" echo probe 2>nul
if not errorlevel 1 if exist "%WRITE_PROBE%" (
    del "%WRITE_PROBE%" >nul 2>nul
    exit /b 0
)
set "UTOA_WRITABLE_PATH=%CHECK_PATH%"
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $p=$env:UTOA_WRITABLE_PATH; if([string]::IsNullOrWhiteSpace($p)){exit 1}; if(!(Test-Path -LiteralPath $p)){New-Item -ItemType Directory -Path $p -Force | Out-Null}; $probe = Join-Path $p ('.__utoa_write_probe_' + [guid]::NewGuid().ToString('N') + '.tmp'); Set-Content -LiteralPath $probe -Value 'probe' -Encoding Ascii -NoNewline; Remove-Item -LiteralPath $probe -Force; exit 0" >nul 2>nul
set "UTOA_WRITABLE_PATH="
if errorlevel 1 exit /b 1
exit /b 0

:check_free_space_mb
set "CHECK_SPACE_PATH=%~1"
set "REQUIRED_MB=%~2"
if "%CHECK_SPACE_PATH%"=="" exit /b 1
if "%REQUIRED_MB%"=="" set "REQUIRED_MB=1024"
for %%I in ("%CHECK_SPACE_PATH%") do set "CHECK_DRIVE=%%~dI"
if not defined CHECK_DRIVE exit /b 0
set "FREE_MB="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $d='%CHECK_DRIVE:~0,1%'; $free=(Get-PSDrive -Name $d).Free; [int][math]::Floor($free/1MB)" 2^>nul`) do set "FREE_MB=%%F"
if not defined FREE_MB (
    echo [WARN] Could not determine free disk space. Continuing without strict check.
    exit /b 0
)
echo %FREE_MB% | findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo [WARN] Free disk space probe returned non-numeric output. Continuing without strict check.
    exit /b 0
)
echo [INFO] Free disk space on %CHECK_DRIVE%: %FREE_MB% MB
if %FREE_MB% LSS %REQUIRED_MB% exit /b 1
exit /b 0

:install_base_mfa_env
if exist "%ENV_DIR%" (
    echo [INFO] Existing MFA env directory detected. Recreating from clean state...
    call :run_step "Remove existing env before recreate" :remove_env_dir
    if errorlevel 1 exit /b 1
)
set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
set /a "CREATE_TRY=1"
:install_base_mfa_env_retry
"%MICROMAMBA_EXE%" create -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge python=%MFA_PYTHON_VERSION% montreal-forced-aligner colorama
if not errorlevel 1 goto :install_base_mfa_env_ok
if %CREATE_TRY% GEQ %RETRY_COUNT_CREATE% (
    echo [FAILED] MFA install failed after %CREATE_TRY% attempts.
    call :maybe_pause
    exit /b 1
)
set /a "CREATE_TRY+=1"
echo [WARN] MFA base environment creation failed. Retrying ^(%CREATE_TRY%/%RETRY_COUNT_CREATE%^)...
call :sleep_seconds %RETRY_WAIT_SECONDS%
goto :install_base_mfa_env_retry
:install_base_mfa_env_ok
exit /b 0

:try_restore_bundled_runtime
if "%PREFER_BUNDLED_RUNTIME%"=="0" exit /b 0
if exist "%MFA_EXE%" exit /b 0
if not exist "%BUNDLED_RUNTIME_ROOT%" exit /b 0
if not exist "%BUNDLED_ENV_DIR%" exit /b 0
call :resolve_python_in_dir "%BUNDLED_ENV_DIR%" BUNDLED_ENV_PYTHON_EXE
if errorlevel 1 (
    echo [INFO] Bundled runtime payload is not complete. Falling back to online bootstrap.
    exit /b 0
)
if not exist "%BUNDLED_MFA_EXE%" (
    echo [WARN] Bundled runtime folder exists but MFA entrypoint is missing. Skipping bundled restore.
    exit /b 0
)
echo [INFO] Restoring MFA runtime from bundled payload...
call :run_step "Copy bundled runtime payload to local runtime root" :copy_tree "%BUNDLED_RUNTIME_ROOT%" "%RUNTIME_ROOT%"
if errorlevel 1 (
    echo [FAILED] Bundled runtime restore failed.
    call :maybe_pause
    exit /b 1
)
set "MFA_EXE=%ENV_DIR%\Scripts\mfa.exe"
if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.bat" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"
if not exist "%MFA_EXE%" if exist "%ENV_DIR%\Scripts\mfa.cmd" set "MFA_EXE=%ENV_DIR%\Scripts\mfa.cmd"
if not exist "%MFA_EXE%" (
    echo [WARN] Bundled runtime restore completed but MFA entrypoint is still missing.
    echo        Falling back to online Micromamba bootstrap.
    exit /b 0
)
if exist "%MICROMAMBA_ROOT%\Library\bin\micromamba.exe" set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\Library\bin\micromamba.exe"
if not exist "%MICROMAMBA_EXE%" if exist "%MICROMAMBA_ROOT%\bin\micromamba.exe" set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\bin\micromamba.exe"
if not exist "%MICROMAMBA_EXE%" if exist "%MICROMAMBA_ROOT%\micromamba.exe" set "MICROMAMBA_EXE=%MICROMAMBA_ROOT%\micromamba.exe"
echo [OK] Bundled MFA runtime restored.
exit /b 0

:copy_tree
set "SRC_DIR=%~1"
set "DST_DIR=%~2"
if "%SRC_DIR%"=="" exit /b 1
if "%DST_DIR%"=="" exit /b 1
if not exist "%SRC_DIR%" exit /b 1
if not exist "%DST_DIR%" mkdir "%DST_DIR%" >nul 2>nul
where /q robocopy.exe
if not errorlevel 1 (
    robocopy "%SRC_DIR%" "%DST_DIR%" /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >nul
    if errorlevel 8 exit /b 1
    exit /b 0
)
xcopy "%SRC_DIR%\*" "%DST_DIR%\" /E /I /Y /Q >nul
if errorlevel 1 exit /b 1
exit /b 0

:ensure_vc_runtime
set "VC_DLL_MSVCP=%SystemRoot%\System32\msvcp140.dll"
set "VC_DLL_VCRUNTIME=%SystemRoot%\System32\vcruntime140_1.dll"
set "VC_REDIST_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
if exist "%VC_DLL_MSVCP%" if exist "%VC_DLL_VCRUNTIME%" (
    echo [OK] VC++ runtime detected.
    exit /b 0
)
echo [INFO] VC++ runtime is missing. Installing Microsoft VC++ 2015-2022 x64...
if not exist "%VC_REDIST_EXE%" (
    set "VC_DL_OK=0"
    call :download_with_powershell "%VC_REDIST_URL%" "%VC_REDIST_EXE%"
    if not errorlevel 1 if exist "%VC_REDIST_EXE%" (
        for %%I in ("%VC_REDIST_EXE%") do if %%~zI GTR 0 set "VC_DL_OK=1"
    )
    if "%VC_DL_OK%"=="0" (
        echo [WARN] PowerShell VC++ runtime download failed. Trying curl...
        where /q curl.exe
        if not errorlevel 1 (
            curl.exe -L --fail --retry 2 --retry-delay 2 -o "%VC_REDIST_EXE%" "%VC_REDIST_URL%"
            if not errorlevel 1 if exist "%VC_REDIST_EXE%" (
                for %%I in ("%VC_REDIST_EXE%") do if %%~zI GTR 0 set "VC_DL_OK=1"
            )
        )
    )
    if "%VC_DL_OK%"=="0" (
        echo [FAILED] VC++ runtime download failed.
        echo        Network/DNS issue may be blocking aka.ms.
        echo        Install manually: %VC_REDIST_URL%
        call :maybe_pause
        exit /b 1
    )
)
if not exist "%VC_REDIST_EXE%" (
    echo [FAILED] VC++ runtime installer file was not found: %VC_REDIST_EXE%
    call :maybe_pause
    exit /b 1
)
for %%I in ("%VC_REDIST_EXE%") do set "VC_REDIST_SIZE=%%~zI"
if "%VC_REDIST_SIZE%"=="0" (
    echo [FAILED] VC++ runtime installer file is empty: %VC_REDIST_EXE%
    call :maybe_pause
    exit /b 1
)
if %VC_REDIST_SIZE% LSS 1000000 (
    echo [FAILED] VC++ runtime installer file looks invalid ^(too small^): %VC_REDIST_EXE%
    echo        This often means download was blocked or replaced by an error page.
    del "%VC_REDIST_EXE%" >nul 2>nul
    echo        Retry with network enabled or install manually: https://aka.ms/vs/17/release/vc_redist.x64.exe
    call :maybe_pause
    exit /b 1
)
"%VC_REDIST_EXE%" /install /quiet /norestart
set "VC_REDIST_RC=%errorlevel%"
if not "%VC_REDIST_RC%"=="0" if not "%VC_REDIST_RC%"=="1638" if not "%VC_REDIST_RC%"=="3010" (
    echo [WARN] Silent VC++ runtime install failed with code %VC_REDIST_RC%. Retrying with passive mode...
    "%VC_REDIST_EXE%" /install /passive /norestart
    set "VC_REDIST_RC=%errorlevel%"
)
if not "%VC_REDIST_RC%"=="0" if not "%VC_REDIST_RC%"=="1638" if not "%VC_REDIST_RC%"=="3010" (
    echo [FAILED] VC++ runtime install failed. exit code=%VC_REDIST_RC%
    echo        Install manually: https://aka.ms/vs/17/release/vc_redist.x64.exe
    call :maybe_pause
    exit /b 1
)
if not exist "%VC_DLL_MSVCP%" (
    echo [FAILED] VC++ runtime install finished but msvcp140.dll is still missing.
    call :maybe_pause
    exit /b 1
)
if not exist "%VC_DLL_VCRUNTIME%" (
    echo [FAILED] VC++ runtime install finished but vcruntime140_1.dll is still missing.
    call :maybe_pause
    exit /b 1
)
echo [OK] VC++ runtime installed.
exit /b 0

:download_micromamba
if exist "%MICROMAMBA_ARCHIVE%" (
    for %%I in ("%MICROMAMBA_ARCHIVE%") do if %%~zI GTR 0 exit /b 0
)
if exist "%MICROMAMBA_EXE%" (
    for %%I in ("%MICROMAMBA_EXE%") do if %%~zI GTR 0 exit /b 0
)
echo [INFO] Download source: micro.mamba.pm
call :download_with_powershell "%MICROMAMBA_URL_API%" "%MICROMAMBA_ARCHIVE%"
if not errorlevel 1 if exist "%MICROMAMBA_ARCHIVE%" (
    for %%I in ("%MICROMAMBA_ARCHIVE%") do if %%~zI GTR 0 exit /b 0
)
echo [WARN] PowerShell download from micro.mamba.pm failed. Trying curl...
where /q curl.exe
if not errorlevel 1 (
    curl.exe -L --fail --retry 2 --retry-delay 2 -o "%MICROMAMBA_ARCHIVE%" "%MICROMAMBA_URL_API%"
    if not errorlevel 1 if exist "%MICROMAMBA_ARCHIVE%" (
        for %%I in ("%MICROMAMBA_ARCHIVE%") do if %%~zI GTR 0 exit /b 0
    )
)
echo [WARN] micro.mamba.pm download failed. Trying GitHub mirror...
call :download_micromamba_exe_only
if not errorlevel 1 exit /b 0
exit /b 1

:download_micromamba_exe_only
if not exist "%MICROMAMBA_ROOT%\Library\bin" mkdir "%MICROMAMBA_ROOT%\Library\bin" >nul 2>nul
if exist "%MICROMAMBA_EXE%" (
    for %%I in ("%MICROMAMBA_EXE%") do if %%~zI GTR 0 exit /b 0
)
call :download_with_powershell "%MICROMAMBA_URL_GITHUB_EXE%" "%MICROMAMBA_EXE%"
if not errorlevel 1 if exist "%MICROMAMBA_EXE%" (
    for %%I in ("%MICROMAMBA_EXE%") do if %%~zI GTR 0 exit /b 0
)
where /q curl.exe
if not errorlevel 1 (
    curl.exe -L --fail --retry 2 --retry-delay 2 -o "%MICROMAMBA_EXE%" "%MICROMAMBA_URL_GITHUB_EXE%"
    if not errorlevel 1 if exist "%MICROMAMBA_EXE%" (
        for %%I in ("%MICROMAMBA_EXE%") do if %%~zI GTR 0 exit /b 0
    )
)
exit /b 1

:bootstrap_python_tools
echo Checking MFA Python package tools...
call :resolve_env_python_exe
if errorlevel 1 (
    echo [FAILED] MFA Python runtime was not found.
    call :maybe_pause
    exit /b 1
)
"%ENV_PYTHON_EXE%" -c "import importlib.util as u; import pip, wheel, sys; sys.exit(0 if u.find_spec('setuptools') else 1)" >nul 2>nul
if not errorlevel 1 (
    echo [OK] pip/setuptools/wheel are ready.
    goto :eof
)
echo [INFO] Repairing pip/setuptools/wheel...
"%ENV_DIR%\Scripts\conda.exe" install -y --solver classic -p "%ENV_DIR%" pip setuptools wheel >nul 2>nul
"%ENV_PYTHON_EXE%" -m ensurepip --upgrade
if errorlevel 1 (
    echo [WARN] ensurepip did not complete cleanly. Trying pip repair anyway...
)
call :run_env_pip install --upgrade setuptools wheel
if errorlevel 1 (
    echo [FAILED] pip/setuptools/wheel repair failed.
    call :maybe_pause
    exit /b 1
)
"%ENV_PYTHON_EXE%" -c "import importlib.util as u; import pip, wheel, sys; sys.exit(0 if u.find_spec('setuptools') else 1)" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] Python package tools are still unavailable after repair.
    call :maybe_pause
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
        call :maybe_pause
        exit /b 1
    )
    goto :eof
)
echo [WARN] Micromamba command was not found. Skipping Japanese tokenizer dependency install.
goto :eof

:install_audio_deps
if exist "%ENV_DIR%\Library\bin\libsndfile.dll" goto :eof
echo Checking audio runtime dependencies ^(libsndfile^)...
if exist "%MICROMAMBA_EXE%" (
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge libsndfile pysoundfile
    if errorlevel 1 (
        echo [FAILED] Audio dependency install failed.
        call :maybe_pause
        exit /b 1
    )
    goto :eof
)
if exist "%ENV_DIR%\Scripts\conda.exe" (
    "%ENV_DIR%\Scripts\conda.exe" install -y --solver classic -p "%ENV_DIR%" -c conda-forge libsndfile pysoundfile
    if errorlevel 1 (
        echo [FAILED] Audio dependency install failed.
        call :maybe_pause
        exit /b 1
    )
    goto :eof
)
echo [WARN] No micromamba/conda found. Skipping audio dependency install.
goto :eof

:install_textgrid
echo Checking textgrid module...
call :resolve_env_python_exe
if errorlevel 1 (
    echo [FAILED] MFA Python runtime was not found.
    call :maybe_pause
    exit /b 1
)
"%ENV_PYTHON_EXE%" -c "import textgrid" >nul 2>nul
if not errorlevel 1 goto :eof
echo [INFO] Installing textgrid module...
call :run_env_pip install --upgrade "textgrid>=1.5"
if errorlevel 1 (
    echo [FAILED] textgrid install failed.
    call :maybe_pause
    exit /b 1
)
goto :eof

:verify_textgrid
call :resolve_env_python_exe
if errorlevel 1 (
    echo [FAILED] MFA Python runtime was not found.
    call :maybe_pause
    exit /b 1
)
"%ENV_PYTHON_EXE%" -c "import textgrid" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] textgrid import failed.
    echo        Re-run setup_mfa.bat to repair the environment.
    call :maybe_pause
    exit /b 1
)
goto :eof

:install_ml_requirements
echo Installing optional ML dependencies...
call :resolve_env_python_exe
if errorlevel 1 (
    echo [FAILED] MFA Python runtime was not found.
    call :maybe_pause
    exit /b 1
)
if not exist "%APP_DIR%\requirements.txt" (
    echo [FAILED] requirements.txt not found in %APP_DIR%
    call :maybe_pause
    exit /b 1
)
if not exist "%APP_DIR%\requirements-ml.txt" (
    echo [FAILED] requirements-ml.txt not found in %APP_DIR%
    call :maybe_pause
    exit /b 1
)
if exist "%MICROMAMBA_EXE%" (
    echo [INFO] Installing ML runtime packages via micromamba...
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge pandas scikit-learn lightgbm pytorch
    if errorlevel 1 (
        echo [WARN] Micromamba ML install failed. Falling back to pip.
    ) else (
        echo [OK] Micromamba ML packages installed.
    )
)
echo [INFO] Installing from requirements.txt
call :run_env_pip install --upgrade -r "%APP_DIR%\requirements.txt"
if errorlevel 1 (
    echo [FAILED] requirements.txt install failed.
    call :maybe_pause
    exit /b 1
)
if not exist "%MICROMAMBA_EXE%" (
    echo [INFO] Installing from requirements-ml.txt via pip (micromamba not found)
    call :run_env_pip install --upgrade -r "%APP_DIR%\requirements-ml.txt"
    if errorlevel 1 (
        echo [FAILED] ML dependency install failed.
        echo        You may need Microsoft Visual C++ Build Tools for lightgbm.
        call :maybe_pause
        exit /b 1
    )
)
echo [OK] ML dependencies installed.
goto :eof

:verify_ml_runtime
call :resolve_env_python_exe
if errorlevel 1 (
    echo [FAILED] MFA Python runtime was not found.
    call :maybe_pause
    exit /b 1
)
"%ENV_PYTHON_EXE%" -c "import pandas, sklearn, lightgbm, torch" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] ML runtime import failed. Missing pandas/sklearn/lightgbm/torch.
    if exist "%MICROMAMBA_EXE%" (
        echo        Try: "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge pandas scikit-learn lightgbm pytorch
    ) else (
        echo        Re-run setup_mfa.bat --with-ml
    )
    call :maybe_pause
    exit /b 1
)
goto :eof

:ensure_cuda_runtime_for_nvidia
if not "%INSTALL_ML%"=="1" goto :eof
if not "%INSTALL_CUDA_RUNTIME%"=="1" goto :eof
call :resolve_env_python_exe
if errorlevel 1 goto :eof
where /q nvidia-smi
if errorlevel 1 goto :eof

echo Checking CUDA runtime for NVIDIA GPU...
set "PYTHONPATH=%APP_DIR%;%PYTHONPATH%"
"%ENV_PYTHON_EXE%" -m core.cuda_runtime_bootstrap --auto-install --python-exe "%ENV_PYTHON_EXE%" --quiet
set "CUDA_BOOTSTRAP_RC=%errorlevel%"
if "%CUDA_BOOTSTRAP_RC%"=="0" (
    echo [OK] CUDA runtime check complete.
    goto :eof
)
echo [WARN] CUDA runtime bootstrap returned code %CUDA_BOOTSTRAP_RC%.
echo        MFA/ML setup can continue, but GPU inference may still use CPU.
echo        Re-run setup_mfa.bat --with-cuda after network/pip issues are resolved.
goto :eof

:install_korean_support
echo Checking Korean tokenizer dependencies...
call :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
call :resolve_env_python_exe
if errorlevel 1 (
    echo [FAILED] MFA Python runtime was not found.
    call :maybe_pause
    exit /b 1
)
call :check_korean_tokenizer_deps
if not errorlevel 1 goto :patch_korean_support
echo [INFO] Installing Korean tokenizer dependencies...
if exist "%KO_WHEEL_DIR%" (
    dir /b "%KO_WHEEL_DIR%\*.whl" >nul 2>nul
    if not errorlevel 1 (
        echo [INFO] Trying bundled Korean wheels first ^(offline mode^)...
        call :run_env_pip install --upgrade --no-index --find-links "%KO_WHEEL_DIR%" python-mecab-ko jamo python-mecab-ko-dic
        call :check_korean_tokenizer_deps
        if not errorlevel 1 goto :patch_korean_support
        echo [WARN] Bundled Korean wheel install failed. Falling back to online channels...
    ) else (
        echo [INFO] Korean wheel bundle folder exists but no wheel files were found.
    )
)
if exist "%MICROMAMBA_EXE%" (
    echo [INFO] Trying conda-forge packages first ^(python-mecab-ko, jamo^)...
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge python-mecab-ko jamo
    call :check_korean_tokenizer_deps
    if not errorlevel 1 goto :patch_korean_support
)
echo [INFO] Trying pip wheel packages first ^(python-mecab-ko, jamo; no source build^)...
call :run_env_pip install --upgrade --only-binary=:all: python-mecab-ko jamo
call :check_korean_tokenizer_deps
if not errorlevel 1 goto :patch_korean_support
if errorlevel 1 (
    if "%ALLOW_DEGRADED_KO%"=="1" (
        echo [WARN] Korean tokenizer dependency install failed ^(python-mecab-ko/jamo^). Continuing in degraded mode.
        echo        Alignment can run, but Korean accuracy may be lower.
        echo        Manual Build Tools install link: https://visualstudio.microsoft.com/visual-cpp-build-tools/
        exit /b 0
    )
    echo [FAILED] Korean tokenizer dependency install failed ^(python-mecab-ko/jamo^).
    echo        Build Tools auto-install is intentionally not included.
    echo        Reason: very large package size, admin elevation, and unstable silent provisioning.
    echo        Try:
    echo        1^) Re-run with network access and conda-forge available ^(recommended^)
    echo        2^) Manual install: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    call :maybe_pause
    exit /b 1
)
:patch_korean_support
set "PYTHONPATH=%APP_DIR%"
call :resolve_env_python_exe
if errorlevel 1 exit /b 1
"%ENV_PYTHON_EXE%" -c "from core.mfa_runner import patch_mfa_korean_support; patch_mfa_korean_support(r'%MFA_EXE%')" >nul 2>nul
if errorlevel 1 (
    echo [WARN] Korean MFA patch step failed. Alignment may still run, but Korean tokenizer support could be incomplete.
    exit /b 0
)
exit /b 0

:check_korean_tokenizer_deps
call :resolve_env_python_exe
if errorlevel 1 exit /b 1
"%ENV_PYTHON_EXE%" -c "import jamo" >nul 2>nul
if errorlevel 1 exit /b 1
"%ENV_PYTHON_EXE%" -c "import eunjeon" >nul 2>nul
if not errorlevel 1 exit /b 0
"%ENV_PYTHON_EXE%" -c "from mecab import MeCab" >nul 2>nul
if not errorlevel 1 exit /b 0
exit /b 1

:check_mfa_python_module
call :resolve_env_python_exe
if errorlevel 1 exit /b 1
"%ENV_PYTHON_EXE%" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('montreal_forced_aligner.command_line.mfa') else 1)" >nul 2>nul
if not errorlevel 1 exit /b 0
"%ENV_PYTHON_EXE%" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('montreal_forced_aligner') else 1)" >nul 2>nul
if not errorlevel 1 exit /b 0
exit /b 1

:handle_old_env
echo.
echo [WARN] Legacy MFA environment detected:
echo        %OLD_ENV_DIR%
echo.
echo Choose how to handle the legacy environment:
echo   [M] Migrate ^(rebuild local env, then delete old^)
echo   [D] Delete old now
echo   [K] Keep old ^(no deletion^)
if "%INTERACTIVE%"=="0" (
    echo [INFO] Non-interactive mode: keeping legacy env and continuing.
    goto :keep_old_env
)
choice /C MDK /N /M "Select M/D/K: "
if errorlevel 3 goto :keep_old_env
if errorlevel 2 goto :delete_old_env_now
if errorlevel 1 goto :migrate_old_env
goto :eof

:migrate_old_env
set "DELETE_OLD_AFTER_INSTALL=1"
echo [INFO] Will delete legacy env after successful local install.
goto :eof

:delete_old_env_now
call :remove_dir "%OLD_ENV_DIR%"
call :remove_dir "%OLD_MICROMAMBA_ROOT%"
goto :eof

:keep_old_env
echo [INFO] Keeping legacy env. Local install will continue.
goto :eof

:cleanup_old_env_if_requested
if not "%DELETE_OLD_AFTER_INSTALL%"=="1" goto :eof
if /i "%OLD_ENV_DIR%"=="%ENV_DIR%" goto :eof
call :remove_dir "%OLD_ENV_DIR%"
call :remove_dir "%OLD_MICROMAMBA_ROOT%"
echo [OK] Legacy MFA environment removed.
goto :eof

:remove_dir
set "TARGET_DIR=%~1"
if "%TARGET_DIR%"=="" goto :eof
if not exist "%TARGET_DIR%" goto :eof
echo [INFO] Removing %TARGET_DIR%
rmdir /s /q "%TARGET_DIR%" >nul 2>nul
if exist "%TARGET_DIR%" (
    echo [WARN] Failed to remove %TARGET_DIR%
)
goto :eof

:ensure_mfa_entrypoint
if defined MFA_EXE if exist "%MFA_EXE%" (
    call :probe_mfa_entrypoint "%MFA_EXE%"
    if not errorlevel 1 goto :eof
    echo [WARN] Existing MFA launcher is broken: %MFA_EXE%
)

if exist "%ENV_DIR%\Scripts\mfa.bat" (
    call :probe_mfa_entrypoint "%ENV_DIR%\Scripts\mfa.bat"
    if not errorlevel 1 (
        set "MFA_EXE=%ENV_DIR%\Scripts\mfa.bat"
        goto :eof
    )
)
if exist "%ENV_DIR%\Scripts\mfa.cmd" (
    call :probe_mfa_entrypoint "%ENV_DIR%\Scripts\mfa.cmd"
    if not errorlevel 1 (
        set "MFA_EXE=%ENV_DIR%\Scripts\mfa.cmd"
        goto :eof
    )
)

call :resolve_env_python_exe
if errorlevel 1 (
    echo [FAILED] MFA Python runtime was not found.
    call :maybe_pause
    exit /b 1
)

if not exist "%ENV_DIR%\Scripts" mkdir "%ENV_DIR%\Scripts" >nul 2>nul
set "MFA_BAT_PATH=%ENV_DIR%\Scripts\mfa.bat"
set "MFA_SCRIPT_PATH=%ENV_DIR%\Scripts\mfa-script.py"
set "MFA_ALT_SCRIPT_PATH=%ENV_DIR%\Scripts\mfa.py"
>"%MFA_BAT_PATH%" echo @echo off
>>"%MFA_BAT_PATH%" echo set "SCRIPT_DIR=%%~dp0"
>>"%MFA_BAT_PATH%" echo set "ENV_DIR=%%SCRIPT_DIR%%.."
>>"%MFA_BAT_PATH%" echo for %%%%I in ("%%ENV_DIR%%") do set "ENV_DIR=%%%%~fI"
>>"%MFA_BAT_PATH%" echo set "CONDA_PREFIX=%%ENV_DIR%%"
>>"%MFA_BAT_PATH%" echo set "PATH=%%ENV_DIR%%;%%ENV_DIR%%\Library\mingw-w64\bin;%%ENV_DIR%%\Library\usr\bin;%%ENV_DIR%%\Library\bin;%%ENV_DIR%%\Scripts;%%ENV_DIR%%\bin;%%PATH%%"
>>"%MFA_BAT_PATH%" echo set "MFA_SCRIPT_PATH=%%SCRIPT_DIR%%mfa-script.py"
>>"%MFA_BAT_PATH%" echo set "MFA_ALT_SCRIPT_PATH=%%SCRIPT_DIR%%mfa.py"
>>"%MFA_BAT_PATH%" echo set "ENV_PY=%%ENV_DIR%%\python.exe"
>>"%MFA_BAT_PATH%" echo if not exist "%%ENV_PY%%" set "ENV_PY=%%ENV_DIR%%\Scripts\python.exe"
>>"%MFA_BAT_PATH%" echo if not exist "%%ENV_PY%%" set "ENV_PY=%%ENV_DIR%%\bin\python"
>>"%MFA_BAT_PATH%" echo set "_UTOA_MFA_EXIT=1"
>>"%MFA_BAT_PATH%" echo if exist "%%MFA_SCRIPT_PATH%%" ^(
>>"%MFA_BAT_PATH%" echo   "%%ENV_PY%%" "%%MFA_SCRIPT_PATH%%" %%*
>>"%MFA_BAT_PATH%" echo   set "_UTOA_MFA_EXIT=%%ERRORLEVEL%%"
>>"%MFA_BAT_PATH%" echo ^) else if exist "%%MFA_ALT_SCRIPT_PATH%%" ^(
>>"%MFA_BAT_PATH%" echo   "%%ENV_PY%%" "%%MFA_ALT_SCRIPT_PATH%%" %%*
>>"%MFA_BAT_PATH%" echo   set "_UTOA_MFA_EXIT=%%ERRORLEVEL%%"
>>"%MFA_BAT_PATH%" echo ^)
>>"%MFA_BAT_PATH%" echo if not "%%_UTOA_MFA_EXIT%%"=="0" ^(
>>"%MFA_BAT_PATH%" echo   "%%ENV_PY%%" -m montreal_forced_aligner.command_line.mfa %%*
>>"%MFA_BAT_PATH%" echo   set "_UTOA_MFA_EXIT=%%ERRORLEVEL%%"
>>"%MFA_BAT_PATH%" echo ^)
>>"%MFA_BAT_PATH%" echo if not "%%_UTOA_MFA_EXIT%%"=="0" ^(
>>"%MFA_BAT_PATH%" echo   "%%ENV_PY%%" -m montreal_forced_aligner %%*
>>"%MFA_BAT_PATH%" echo   set "_UTOA_MFA_EXIT=%%ERRORLEVEL%%"
>>"%MFA_BAT_PATH%" echo ^)
>>"%MFA_BAT_PATH%" echo exit /b %%_UTOA_MFA_EXIT%%
set "MFA_EXE=%MFA_BAT_PATH%"
call :probe_mfa_entrypoint "%MFA_EXE%"
if errorlevel 1 (
    echo [FAILED] MFA wrapper creation failed.
    call :maybe_pause
    exit /b 1
)
echo [OK] MFA wrapper generated: %MFA_EXE%
goto :eof

:probe_mfa_entrypoint
set "MFA_PROBE_TARGET=%~1"
if "%MFA_PROBE_TARGET%"=="" exit /b 1
if not exist "%MFA_PROBE_TARGET%" exit /b 1
set "MFA_PROBE_LOG=%TEMP%\utoa_mfa_probe_%RANDOM%%RANDOM%.log"
cmd /c ""%MFA_PROBE_TARGET%" --help >"%MFA_PROBE_LOG%" 2>&1"
set "MFA_PROBE_RC=%errorlevel%"
if exist "%MFA_PROBE_LOG%" (
    findstr /I /C:"failed to create process" /C:"unable to create process" /C:"fatal error in launcher" /C:"no python at" "%MFA_PROBE_LOG%" >nul
    if not errorlevel 1 (
        del "%MFA_PROBE_LOG%" >nul 2>nul
        exit /b 1
    )
    del "%MFA_PROBE_LOG%" >nul 2>nul
)
if "%MFA_PROBE_RC%"=="0" exit /b 0
exit /b 1

:download_acoustic_model
set "MODEL_NAME=%~1"
if not defined MODEL_NAME (
    echo [FAILED] Acoustic model name is missing.
    call :maybe_pause
    exit /b 1
)
call :resolve_env_python_exe
if errorlevel 1 goto :download_acoustic_model_mamba_entry
set "CONDA_PREFIX=%ENV_DIR%"
set "MFA_ROOT_DIR=%RUNTIME_ROOT%\.mfa_root_ascii"
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Library\usr\bin;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%PATH%"
set /a "MODEL_TRY=1"
:download_acoustic_model_python_retry
"%ENV_PYTHON_EXE%" -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME% --ignore_cache
if not errorlevel 1 exit /b 0
"%ENV_PYTHON_EXE%" -m montreal_forced_aligner model download acoustic %MODEL_NAME% --ignore_cache
if not errorlevel 1 exit /b 0
if %MODEL_TRY% GEQ %RETRY_COUNT_MODEL% goto :download_acoustic_model_mamba_entry
set /a "MODEL_TRY+=1"
echo [WARN] Acoustic model download failed. Retrying ^(%MODEL_TRY%/%RETRY_COUNT_MODEL%^)...
call :sleep_seconds %RETRY_WAIT_SECONDS%
goto :download_acoustic_model_python_retry

:download_acoustic_model_mamba_entry
if not exist "%MICROMAMBA_EXE%" goto :download_acoustic_model_fail
set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
set "MFA_ROOT_DIR=%RUNTIME_ROOT%\.mfa_root_ascii"
set /a "MODEL_TRY=1"
:download_acoustic_model_mamba_retry
"%MICROMAMBA_EXE%" run -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" python -m montreal_forced_aligner.command_line.mfa model download acoustic %MODEL_NAME% --ignore_cache
if not errorlevel 1 exit /b 0
"%MICROMAMBA_EXE%" run -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" python -m montreal_forced_aligner model download acoustic %MODEL_NAME% --ignore_cache
if not errorlevel 1 exit /b 0
if %MODEL_TRY% GEQ %RETRY_COUNT_MODEL% goto :download_acoustic_model_fail
set /a "MODEL_TRY+=1"
echo [WARN] Acoustic model download failed via micromamba run. Retrying ^(%MODEL_TRY%/%RETRY_COUNT_MODEL%^)...
call :sleep_seconds %RETRY_WAIT_SECONDS%
goto :download_acoustic_model_mamba_retry

:download_acoustic_model_fail
echo [FAILED] No runnable MFA environment was found for model download.
call :maybe_pause
exit /b 1

:remove_env_dir
if not exist "%ENV_DIR%" goto :eof
echo [INFO] Removing existing env: %ENV_DIR%
rmdir /s /q "%ENV_DIR%" >nul 2>nul
if exist "%ENV_DIR%" (
    echo [FAILED] Could not remove the old MFA environment.
    call :maybe_pause
    exit /b 1
)
goto :eof

:resolve_env_python_exe
call :resolve_python_in_dir "%ENV_DIR%" ENV_PYTHON_EXE
if not errorlevel 1 exit /b 0
set "ENV_PYTHON_EXE=%ENV_DIR%\python.exe"
exit /b 1

:resolve_python_in_dir
set "%~2="
if "%~1"=="" exit /b 1
if exist "%~1\python.exe" (
    set "%~2=%~1\python.exe"
    exit /b 0
)
if exist "%~1\Scripts\python.exe" (
    set "%~2=%~1\Scripts\python.exe"
    exit /b 0
)
if exist "%~1\bin\python" (
    set "%~2=%~1\bin\python"
    exit /b 0
)
exit /b 1

:path_requires_ascii_fallback
set "UTOA_PATH_CHECK=%~1"
if "%UTOA_PATH_CHECK%"=="" exit /b 1
powershell -NoProfile -Command "$p=$env:UTOA_PATH_CHECK; if([string]::IsNullOrWhiteSpace($p)){exit 1}; $unsafe=($p -match '[^\x00-\x7F]') -or ($p -match '[!&|<>()^]'); if($unsafe){exit 0}else{exit 1}" >nul 2>nul
set "UTOA_PATH_CHECK="
exit /b %errorlevel%

:download_with_powershell
set "UTOA_PS_DL_URL=%~1"
set "UTOA_PS_DL_OUT=%~2"
if "%UTOA_PS_DL_URL%"=="" exit /b 1
if "%UTOA_PS_DL_OUT%"=="" exit /b 1
set "UTOA_PS_DL_RC=1"
set /a "UTOA_PS_DL_TRY=1"
:download_with_powershell_retry
powershell -NoProfile -Command "& {$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri $env:UTOA_PS_DL_URL -OutFile $env:UTOA_PS_DL_OUT -ErrorAction Stop; exit 0 }"
set "UTOA_PS_DL_RC=%errorlevel%"
if "%UTOA_PS_DL_RC%"=="0" goto :download_with_powershell_done
if %UTOA_PS_DL_TRY% GEQ %RETRY_COUNT_NETWORK% goto :download_with_powershell_done
set /a "UTOA_PS_DL_TRY+=1"
echo [WARN] Download failed via PowerShell. Retrying ^(%UTOA_PS_DL_TRY%/%RETRY_COUNT_NETWORK%^)...
call :sleep_seconds %RETRY_WAIT_SECONDS%
goto :download_with_powershell_retry
:download_with_powershell_done
set "UTOA_PS_DL_URL="
set "UTOA_PS_DL_OUT="
exit /b %UTOA_PS_DL_RC%

:run_env_pip
call :resolve_env_python_exe
if errorlevel 1 exit /b 1
set "RUN_ENV_PIP_RC=1"
set /a "RUN_ENV_PIP_TRY=1"
:run_env_pip_python_retry
"%ENV_PYTHON_EXE%" -m pip %*
set "RUN_ENV_PIP_RC=%errorlevel%"
if "%RUN_ENV_PIP_RC%"=="0" exit /b 0
if %RUN_ENV_PIP_TRY% GEQ %RETRY_COUNT_PIP% goto :run_env_pip_python_done
set /a "RUN_ENV_PIP_TRY+=1"
echo [WARN] pip command failed via python -m pip. Retrying ^(%RUN_ENV_PIP_TRY%/%RETRY_COUNT_PIP%^)...
call :sleep_seconds %RETRY_WAIT_SECONDS%
goto :run_env_pip_python_retry
:run_env_pip_python_done
if not exist "%ENV_DIR%\Scripts\pip.exe" exit /b %RUN_ENV_PIP_RC%
set /a "RUN_ENV_PIP_EXE_TRY=1"
:run_env_pip_exe_retry
"%ENV_DIR%\Scripts\pip.exe" %*
set "RUN_ENV_PIP_RC=%errorlevel%"
if "%RUN_ENV_PIP_RC%"=="0" exit /b 0
if %RUN_ENV_PIP_EXE_TRY% GEQ %RETRY_COUNT_PIP% exit /b %RUN_ENV_PIP_RC%
set /a "RUN_ENV_PIP_EXE_TRY+=1"
echo [WARN] pip.exe command failed. Retrying ^(%RUN_ENV_PIP_EXE_TRY%/%RETRY_COUNT_PIP%^)...
call :sleep_seconds %RETRY_WAIT_SECONDS%
goto :run_env_pip_exe_retry
exit /b %RUN_ENV_PIP_RC%

:sleep_seconds
set "WAIT_SECONDS=%~1"
if "%WAIT_SECONDS%"=="" set "WAIT_SECONDS=1"
if "%WAIT_SECONDS%"=="0" goto :eof
timeout /t %WAIT_SECONDS% /nobreak >nul 2>nul
goto :eof

:run_step_with_retry
set "RETRY_STEP_TITLE=%~1"
set "RETRY_STEP_MAX=%~2"
if "%RETRY_STEP_MAX%"=="" set "RETRY_STEP_MAX=1"
if %RETRY_STEP_MAX% LSS 1 set "RETRY_STEP_MAX=1"
shift
shift
if "%~1"=="" exit /b 1
set /a "RETRY_STEP_TRY=1"
:run_step_with_retry_loop
call :run_step "%RETRY_STEP_TITLE% ^(attempt %RETRY_STEP_TRY%/%RETRY_STEP_MAX%^) " %1 %2 %3 %4 %5 %6 %7 %8 %9
if not errorlevel 1 exit /b 0
if %RETRY_STEP_TRY% GEQ %RETRY_STEP_MAX% exit /b 1
set /a "RETRY_STEP_TRY+=1"
echo [WARN] Step failed. Retrying: %RETRY_STEP_TITLE% ^(%RETRY_STEP_TRY%/%RETRY_STEP_MAX%^)...
call :sleep_seconds %RETRY_WAIT_SECONDS%
goto :run_step_with_retry_loop

:run_step
setlocal EnableDelayedExpansion
set "STEP_TITLE=%~1"
shift
if "%~1"=="" (
    endlocal
    exit /b 0
)
echo [STEP][%TIME%] START: !STEP_TITLE!
call %1 %2 %3 %4 %5 %6 %7 %8 %9
set "STEP_RC=!errorlevel!"
if "!STEP_RC!"=="0" (
    echo [STEP][%TIME%] DONE : !STEP_TITLE!
) else (
    echo [STEP][%TIME%] FAIL : !STEP_TITLE! ^(code=!STEP_RC!^)
)
for %%R in (!STEP_RC!) do (
    endlocal
    exit /b %%R
)

:maybe_pause
if "%INTERACTIVE%"=="0" goto :eof
pause
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

