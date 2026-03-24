@echo off
@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title UTAU Auto OTO - MFA 繝ｻ・､繝ｻ繝ｻ

set "INSTALL_ML=0"
set "DELETE_OLD_AFTER_INSTALL=0"
set "AUTO_ML=0"
set "NON_INTERACTIVE=0"
set "DIRECT_SETUP=0"
set "NO_RECOVERY_SHIM=0"
set "FORCE_MENU=0"
set "REQUESTED_MODE="
set "WRAPPER_MODE="
set "RUNTIME_ROOT_OVERRIDE="
set "SETUP_MFA_SELF=%~f0"
set "SETUP_MFA_DIR=%~dp0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="--runtime-root" goto :capture_runtime_root
set "ARG_RAW=%~1"
if /i "!ARG_RAW:~0,15!"=="--runtime-root=" goto :capture_runtime_root_inline
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

:capture_runtime_root
shift
if "%~1"=="" goto :missing_runtime_root
set "RUNTIME_ROOT_CANDIDATE=%~1"
if /i "!RUNTIME_ROOT_CANDIDATE:~0,2!"=="--" goto :missing_runtime_root
set "RUNTIME_ROOT_OVERRIDE=%~1"
shift
goto :parse_args

:capture_runtime_root_inline
set "RUNTIME_ROOT_OVERRIDE=!ARG_RAW:~15!"
if "!RUNTIME_ROOT_OVERRIDE!"=="" goto :missing_runtime_root
shift
goto :parse_args

:missing_runtime_root
echo [FAILED] --runtime-root 繝ｻ・､繝ｻ繝ｻ繝ｻﾂ繝ｻ繝ｻ繝ｻ・ｽ繝ｻ鄂ｹ・･・ｼ 繝ｻﾂ繝ｻ蛟｣邏・繝ｻ・ｼ繝ｻ・ｸ繝ｻ繝ｻ
echo          繝ｻ繝ｻ --runtime-root "C:\Users\%USERNAME%\AppData\Local\UTAU_Auto_OTO_v3"
exit /b 1

:show_help
echo 繝ｻ・ｬ繝ｻ・ｩ繝ｻ繝ｻ setup_mfa.bat [--install ^| --recovery ^| --menu] [--runtime-root 繝ｻ・ｽ繝ｻ蠖・[--with-ml] [--non-interactive]
echo 繝ｻ・､蠅橸ｽｬ繝ｻ・ｽ蟄厄ｽｸ 蟆橸ｽｴ繝ｻ閹ｿ(.env^)繝ｻ繝ｻMicromamba 繝ｻ・ｰ繝ｻ繝ｻ繝ｻ諛搾ｽｻ・ｬ MFA 蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻ・､繝ｻ蛟・ｮ偵・螢ｱ蜈ｱ.
echo 繝ｻ・ｵ繝ｻ繝ｻ
echo   --install                 繝ｻ・､繝ｻ繝ｻ繝ｻ・ｨ繝ｻ繝ｻ繝ｻ闖ｩ・ｰ繝ｻ繝ｻ・､蠏ゅ・
echo   --recovery / --recover    繝ｻ・ｵ繝ｻ・ｬ 繝ｻ・ｨ繝ｻ繝ｻ繝ｻ闖ｩ・ｰ繝ｻ繝ｻ・､蠏ゅ・
echo   --menu                    繝ｻ諛肴｢ 繝ｻ鄂ｷ菫ｺ 繝ｻ闖ｩ・ｰ繝ｻ蟯ｺ諛堺ｺｨ
echo   --runtime-root 繝ｻ・ｽ繝ｻ繝ｻ      繝ｻ・､繝ｻ繝ｻ繝ｻ・ｵ繝ｻ・ｬ 繝ｻﾂ繝ｻ繝ｻ繝ｻ・ｨ蟄厄ｽｸ 繝ｻ・ｽ繝ｻ繝ｻ繝ｻ闖ｩ・ｰ繝ｻ繝ｻﾂ繝ｻ繝ｻ
echo   --runtime-root=繝ｻ・ｽ繝ｻ繝ｻ      繝ｻ繝ｻ繝ｻ・ｵ繝ｻ蛟第八 繝ｻ・ｱ蠍ｸ・ｸ 蠍ｸ蛟｣繝ｻ
echo   --with-ml / --install-ml  ML 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ^ (pandas/sklearn/lightgbm/pytorch^) 繝ｻ・､繝ｻ繝ｻ
echo   --non-interactive         繝ｻ繝ｻ蜉・ｶｹ陲ｫ繝ｻ 繝ｻ・､蠏る° (繝ｻ蟆門ｾ・繝ｻ繝ｻ・ｸ・ｰ: env 繝ｻ貅｢諢阪・・ｴ 繝ｻ・ｵ繝ｻ・ｬ, 繝ｻ繝ｻ諢阪・・ｴ 繝ｻ・､繝ｻ骭ｲ)
echo   --direct-setup            繝ｻ・ｴ繝ｻﾂ 繝ｻ・ｬ繝ｻ・ｩ^: 繝ｻ・ｰ繝ｻ・ｴ 繝ｻ・､繝ｻ繝ｻ繝ｻ閧･・ｧ繝ｻ繝ｻ繝ｻ・ｰ繝ｻ繝ｻ・､蠏ゅ・
echo   --no-recovery-shim        繝ｻ・ｵ繝ｻ・ｬ 繝ｻ蛟・ｩ・繝ｻ繝ｻ蜉医・・ｱ蠍ｹ閹ｿ (繝ｻ・､繝ｻ繝ｻ繝ｻ・ｨ繝ｻ繝ｻ繝ｻ闖ｩ・ｰ蠑ｯ)
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
    echo [WARN] setup_mfa.bat繝ｻ・ｴ System32 繝ｻ・ｽ繝ｻ諛堺ｹｱ繝ｻ繝ｻ繝ｻ・､蠏る寉謐ｮ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
    echo        MFA 繝ｻ蜊鍋ｾ・繝ｻ・ｽ繝ｻ鄂ｹ・･・ｼ LOCALAPPDATA繝ｻ繝ｻ繝ｻ繝ｻ蜉搾ｨ托ｽｩ繝ｻ螢ｱ蜈ｱ.
    set "APP_DIR=%LOCALAPPDATA%\UTAU_Auto_OTO_v3"
)
if defined RUNTIME_ROOT_OVERRIDE (
    echo [INFO] runtime-root override 繝ｻ・ｬ繝ｻ・ｩ: %APP_DIR%
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
    set "WRAPPER_RUNTIME_ARG="
    if defined RUNTIME_ROOT_OVERRIDE set "WRAPPER_RUNTIME_ARG=--runtime-root ""!APP_DIR!"""
    echo [INFO] 繝ｻ・､繝ｻ繝ｻ繝ｻ・ｨ繝ｻ繝ｻ繝ｻ・､蠏ゅ・ !WRAPPER_ARGS!
    call "%SETUP_MFA_SELF%" !WRAPPER_ARGS! !WRAPPER_RUNTIME_ARG!
    exit /b !ERRORLEVEL!
)

if /i "%WRAPPER_MODE%"=="recovery" (
    if not defined RUNTIME_RECOVERY_SCRIPT (
        echo [WARN] runtime_recovery.ps1繝ｻ・ｼ 繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
        if "%NON_INTERACTIVE%"=="1" (
            echo [FAILED] 繝ｻ繝ｻ蜉・ｶｹ陲ｫ繝ｻ 繝ｻ・ｵ繝ｻ・ｬ繝ｻ・ｼ 繝ｻ諛肴｢・托｣ｰ 繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ. setup_mfa.bat --install 繝ｻ繝ｻ繝ｻ・､繝ｻ菫ｯ・･・ｼ 繝ｻ・､蠏るｧ慕ｴ・繝ｻ・ｼ繝ｻ・ｸ繝ｻ繝ｻ
            exit /b 1
        )
        echo [INFO] 繝ｻ・､繝ｻ繝ｻ繝ｻ・ｨ繝ｻ鄂ｹ・｡繝ｻ繝ｻ繝ｻ蜉搾ｨ托ｽｩ繝ｻ螢ｱ蜈ｱ.
        set "WRAPPER_MODE=install"
        goto :wrapper_dispatch
    )
    echo [INFO] 繝ｻ・ｵ繝ｻ・ｬ 繝ｻ・ｨ繝ｻ繝ｻ繝ｻ・､蠏ゅ・ !RUNTIME_RECOVERY_SCRIPT!
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
        echo [FAILED] runtime_recovery.ps1 繝ｻ・､蠏ゅ・繝ｻ・､逕ｯ・ｨ ^(code=!RECOVERY_RC!^)
    )
    if not "!RECOVERY_RC!"=="0" if not "%NON_INTERACTIVE%"=="1" pause
    exit /b !RECOVERY_RC!
)

echo [WARN] 繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ譬ｪ 繝ｻ・､蠏ゅ・繝ｻ・ｨ繝ｻ諛肴｡ｿ繝ｻ螢ｱ蜈ｱ: %WRAPPER_MODE%
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
if not defined RUNTIME_RECOVERY_SCRIPT if defined SETUP_MFA_DIR_NORM if exist "%SETUP_MFA_DIR_NORM%\runtime_recovery.ps1" (
    set "RUNTIME_RECOVERY_SCRIPT=%SETUP_MFA_DIR_NORM%\runtime_recovery.ps1"
)
if not defined RUNTIME_RECOVERY_SCRIPT if defined SETUP_MFA_DIR_NORM if exist "%SETUP_MFA_DIR_NORM%\scripts\runtime_recovery.ps1" (
    set "RUNTIME_RECOVERY_SCRIPT=%SETUP_MFA_DIR_NORM%\scripts\runtime_recovery.ps1"
)
goto :eof

:ensure_runtime_root_layout
set "RUNTIME_PAYLOAD_OK=0"
if exist "%APP_DIR%\requirements.txt" if exist "%APP_DIR%\requirements-ml.txt" set "RUNTIME_PAYLOAD_OK=1"
if "%RUNTIME_PAYLOAD_OK%"=="1" goto :eof

if defined RUNTIME_ROOT_OVERRIDE (
    echo [FAILED] runtime-root is missing required files:
    echo         "%APP_DIR%\requirements.txt"
    echo         "%APP_DIR%\requirements-ml.txt"
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)

call :auto_detect_runtime_root
if defined DETECTED_RUNTIME_ROOT (
    echo [INFO] Detected runtime root: !DETECTED_RUNTIME_ROOT!
    if "%NON_INTERACTIVE%"=="1" (
        set "APP_DIR=!DETECTED_RUNTIME_ROOT!"
        goto :eof
    )
    choice /C YN /N /M "Use this detected path for setup/recovery? [Y/N]: "
    if errorlevel 2 (
        echo [INFO] Detected path was not selected.
    ) else (
        set "APP_DIR=!DETECTED_RUNTIME_ROOT!"
        goto :eof
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
if exist "%APP_DIR%\requirements.txt" if exist "%APP_DIR%\requirements-ml.txt" goto :eof
echo [FAILED] Invalid runtime root: %APP_DIR%
echo         requirements.txt / requirements-ml.txt not found.
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
if exist "%TRY_CANDIDATE%\requirements.txt" if exist "%TRY_CANDIDATE%\requirements-ml.txt" (
    set "DETECTED_RUNTIME_ROOT=%TRY_CANDIDATE%"
)
goto :eof

:select_start_mode
echo.
echo ====================================================
echo   UTAU Auto OTO - 繝ｻ・､蠏ゅ・繝ｻ・ｨ繝ｻ繝ｻ繝ｻ・ｰ螟九・
echo ====================================================
echo   [1] 繝ｻ・ｰ繝ｻ・ｸ 繝ｻ・､繝ｻ繝ｻ^(繝ｻ隴ｷ譽与)
echo   [2] 繝ｻ・､繝ｻ繝ｻ+ ML 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ
echo   [3] 繝ｻ・ｵ繝ｻ・ｬ 繝ｻ鄂ｷ菫ｺ 繝ｻ・､蠏ゅ・^(繝ｻ・ｸ繝ｻ繝ｻ繝ｻ繝ｻ蜊ｿ/繝ｻ・ｰ螟九・繝ｻ・ｵ繝ｻ・ｬ^)
echo   [4] 繝ｻ蟆門ｾ・繝ｻ・ｵ繝ｻ・ｬ ^(繝ｻ繝ｻ蜉・ｶｹ陲ｫ繝ｻ^)
echo   [0] 繝ｻ繝ｻ・｣繝ｻ
choice /C 12340 /N /M "繝ｻ・ｰ螟九・^(1/2/3/4/0^): "
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
    echo [FAILED] powershell 繝ｻ繝ｻ・ｰ・ｹ繝ｻ繝ｻ繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    exit /b 1
)
where certutil >nul 2>nul
if errorlevel 1 (
    echo [FAILED] certutil 繝ｻ繝ｻ・ｰ・ｹ繝ｻ繝ｻ繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    exit /b 1
)
where tar >nul 2>nul
if errorlevel 1 (
    echo [FAILED] tar 繝ｻ繝ｻ・ｰ・ｹ繝ｻ繝ｻ繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    exit /b 1
)
goto :eof

:direct_setup_start

call :preflight_install_tools
if errorlevel 1 exit /b 1

echo ====================================================
echo   UTAU Auto OTO - MFA 繝ｻ・ｽ繝ｻ繝ｻ蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ・､繝ｻ繝ｻ
echo   繝ｻ・ｴ 繝ｻ・､蠅橸ｽｬ繝ｻ・ｽ蟄厄ｽｸ繝ｻ繝ｻ繝ｻ諛搾ｽｴ繝ｻ1蟾占ｪ､・ｧ繝ｻ繝ｻ・､蠏るｧ戊村繝ｻ・ｴ 繝ｻ・ｩ繝ｻ螢ｱ蜈ｱ.
echo ====================================================
echo.
echo [繝ｻ螢ｱ縺望 繝ｻ・､繝ｻ蛟台ｹｱ繝ｻ繝ｻ繝ｻ・ｸ螂難ｽｰ繝ｻ・ｷ 繝ｻ・ｰ繝ｻ・ｰ繝ｻ・ｴ ・代・蝗茨ｨ托ｽｩ繝ｻ螢ｱ蜈ｱ.
echo        PC 蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻ・ｰ繝ｻ・ｼ 10~20繝ｻ繝ｻ繝ｻ邁ｿ蟾｡ 繝ｻ隴ｷ蝗医・・ｰ 繝ｻ繝ｻ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
echo        繝ｻ・､繝ｻ繝ｻ繝ｻ蜊謎ｹｱ繝ｻ繝ｻ繝ｻ・ｽ繝ｻ繝ｻ繝ｻ・ｫ繝ｻﾂ 繝ｻ蟾晉ｬｦ 繝ｻ・ｼ繝ｻ・ｸ繝ｻ繝ｻ
echo        繝ｻ繝ｻ・｣繝ｻ蠑｡繝ｻMFA 繝ｻ邁ｿ・ｰ・ｬ 繝ｻ・ｰ繝ｻ・･繝ｻ繝ｻ繝ｻ鄂ｷ・｡繝ｻ繝ｻ・ｬ繝ｻ・ｩ・托｣ｰ 繝ｻ繝ｻ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
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

echo [INFO] MFA 蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ・ｽ繝ｻ繝ｻ %ENV_DIR%
echo [INFO] MFA Micromamba 繝ｻ・ｽ繝ｻ繝ｻ %MICROMAMBA_ROOT%

if exist "%APP_DIR%\ML_models" set "AUTO_ML=1"
if exist "%APP_DIR%\models_installed\oto_ml" set "AUTO_ML=1"
if "%AUTO_ML%"=="1" if "%INSTALL_ML%"=="0" (
    echo [INFO] ML 繝ｻ螢ｱ雎・繝ｻ蟾昴￡繝ｻ・ｴ 繝ｻ蟾晢ｽｧﾂ繝ｻ蛟醍押 ML 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ菫ｯ・･・ｼ 蠍ｹ諛坂筏蠍ｹ陲ｫ魄偵・螢ｱ蜈ｱ.
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
        echo [INFO] 繝ｻ・ｰ繝ｻ・ｴ MFA Python 繝ｻ繝ｻ・ｰ繝ｻ !MFA_ENV_PYTHON!
        call :python_requires_rebuild "!MFA_ENV_PYTHON!" MFA_ENV_REBUILD
        if "!MFA_ENV_REBUILD!"=="1" (
            echo [WARN] Python !MFA_ENV_PYTHON! 繝ｻ繝ｻ・ｰ繝ｻ謐ｩ Windows MFA 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 蠖ｧ蟆厄ｽｦ繝ｻ・ｳ・ｼ 蠍ｸ・ｸ蠍ｹ菫ｯ謐ｮ繝ｻﾂ 繝ｻ蝟懈・繝ｻ螢ｱ蜈ｱ.
            echo        Python %MFA_PYTHON_VERSION% 繝ｻ・ｰ繝ｻﾂ繝ｻ・ｼ繝ｻ繝ｻ蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻ・ｬ繝ｻ・ｬ繝ｻ・ｱ・托ｽｩ繝ｻ螢ｱ蜈ｱ...
            call :remove_env_dir
            if errorlevel 1 exit /b 1
        )
    )
)

if exist "%MFA_EXE%" goto :existing_env_ready
goto :install_micromamba

:existing_env_ready
echo [OK] MFA繝ｻﾂ 繝ｻ・ｴ繝ｻ・ｸ 繝ｻ・､繝ｻ菫ｯ謐ｮ繝ｻ・ｴ 繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
echo      繝ｻ・ｽ繝ｻ繝ｻ %MFA_EXE%
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
echo ・第㈲・ｵ・ｭ繝ｻ・ｴ 繝ｻ驢千ｮ・繝ｻ・ｨ繝ｻ・ｸ 蠍ｹ闖ｩ謾､ 繝ｻ繝ｻ..
call :download_acoustic_model korean_mfa
if errorlevel 1 exit /b 1
echo.
echo 繝ｻ・ｼ繝ｻ・ｸ繝ｻ・ｴ 繝ｻ驢千ｮ・繝ｻ・ｨ繝ｻ・ｸ 蠍ｹ闖ｩ謾､ 繝ｻ繝ｻ..
call :download_acoustic_model japanese_mfa
if errorlevel 1 exit /b 1
call :cleanup_env_caches
if errorlevel 1 exit /b 1
call :cleanup_old_env_if_requested
echo.
echo 繝ｻ繝ｻ・｣隱､謐ｮ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ. 繝ｻ・ｴ繝ｻ繝ｻUTAU_Auto_OTO.exe繝ｻ・ｼ 繝ｻ・､蠏るｧ戊・ 繝ｻ繝ｻ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
if not "%NON_INTERACTIVE%"=="1" pause
exit /b 0

:install_micromamba
echo [INFO] 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ蝨ｸ繝ｻ繝ｻ繝ｻ繝ｻ謫ｽ繝ｻ・ｰ 繝ｻ繝ｻ邏・Micromamba 繝ｻﾂ蟄厄ｽｸ繝ｻ・､蟄厄ｽｸ繝ｻ・ｩ繝ｻ繝ｻ繝ｻ・ｬ繝ｻ・ｩ・托ｽｩ繝ｻ螢ｱ蜈ｱ.
echo.
echo [1/5] Micromamba 繝ｻ・､繝ｻ・ｴ繝ｻ鄂ｹ邉・繝ｻ繝ｻ.. ^(繝ｻ・ｽ 15MB^)
call :resolve_micromamba_expected_md5
if errorlevel 1 (
    echo [WARN] 繝ｻ・ｰ繝ｻ・ｰ 繝ｻﾂ繝ｻ・･・代・Micromamba ・托ｽｴ繝ｻ繝ｻ繝ｻ陲ｫ繝繝ｻ・ｰ繝ｻ・ｴ螂難ｽｰ繝ｻ・ｼ 繝ｻﾂ繝ｻ・ｸ繝ｻ・､繝ｻﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    if "%NON_INTERACTIVE%"=="1" (
        echo [FAILED] 繝ｻ繝ｻ蜉・ｶｹ陲ｫ繝ｻ 繝ｻ・ｨ繝ｻ諛堺ｹｱ繝ｻ鄂ｹ譬ｪ ・托ｽｴ繝ｻ繝ｻ繝ｻﾂ繝ｻ繝ｻ繝ｻ陲ｫ繝繝ｻ・ｰ繝ｻ・ｴ螂難ｽｰ繝ｻﾂ ・代・蝗茨ｨ托ｽｩ繝ｻ螢ｱ蜈ｱ.
        exit /b 1
    )
    choice /C YN /N /M "繝ｻ・ｰ繝ｻ・ｰ ・托ｽｴ繝ｻ繝ｻ繝ｻ陲ｫ繝繝ｻ・ｰ繝ｻ・ｴ螂難ｽｰ 繝ｻ繝ｻ謫ｽ 繝ｻ繝ｻ繝ｻ・托｣ｰ繝ｻ隴ｷ蝗・ [Y/N]: "
    if errorlevel 2 (
        echo [FAILED] 繝ｻ・ｬ繝ｻ・ｩ繝ｻ蟾昜ｹｱ 繝ｻ蛟・ｴ・繝ｻ轢ｧ蜊ｿ繝ｻ蛟第擂繝ｻ・ｵ繝ｻ螢ｱ蜈ｱ.
        exit /b 1
    )
)
if not exist "%MICROMAMBA_ARCHIVE%" (
    powershell -NoProfile -Command "& {$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MICROMAMBA_LATEST_URL%' -OutFile '%MICROMAMBA_ARCHIVE%'}"
    if errorlevel 1 (
        echo [FAILED] Micromamba 繝ｻ・､繝ｻ・ｴ繝ｻ鄂ｹ邉悶・繝ｻ繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
        if not "%NON_INTERACTIVE%"=="1" pause
        exit /b 1
    )
) else (
    echo    繝ｻ・ｴ繝ｻ・ｸ 繝ｻ・､繝ｻ・ｴ繝ｻ鄂ｹ邉悶・蛟醍押 繝ｻ貅｢迚・繝ｻ・ｴ繝ｻ螢ｱ諤舌・螢ｱ蜈ｱ.
)
call :verify_micromamba_archive
if errorlevel 1 (
    echo [FAILED] Micromamba ・托ｽｴ繝ｻ繝ｻ繝ｻﾂ繝ｻ譎ｧ荵ｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
echo [OK] 繝ｻ・､繝ｻ・ｴ繝ｻ鄂ｹ邉悶・ﾂ 繝ｻ繝ｻ・｣隱､謐ｮ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
echo.

echo [2/5] Micromamba 繝ｻ闖ｩ・ｶ繝ｻ・托ｽｴ繝ｻ繝ｻ繝ｻ繝ｻ..
if not exist "%MICROMAMBA_EXE%" (
    if exist "%MICROMAMBA_ROOT%" (
        echo [INFO] 繝ｻ・ｰ繝ｻ・ｴ Micromamba 繝ｻ・ｨ蟄厄ｽｸ繝ｻ・ｼ 繝ｻ諛具ｽｱ・ｰ・托ｽｩ繝ｻ螢ｱ蜈ｱ...
        rmdir /s /q "%MICROMAMBA_ROOT%" >nul 2>nul
    )
    mkdir "%MICROMAMBA_ROOT%" >nul 2>nul
    tar -xjf "%MICROMAMBA_ARCHIVE%" -C "%MICROMAMBA_ROOT%"
    if errorlevel 1 (
        echo [FAILED] Micromamba 繝ｻ闖ｩ・ｶ繝ｻ・托ｽｴ繝ｻ諛堺ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
        if not "%NON_INTERACTIVE%"=="1" pause
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
    echo [FAILED] 繝ｻ闖ｩ・ｶ繝ｻ・托ｽｴ繝ｻ繝ｻ蠑｡繝ｻMicromamba 繝ｻ・､蠏ゅ・逕ｯ隴ｷ謾ｵ繝ｻ繝ｻ繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
echo [OK] Micromamba 繝ｻﾂ繝ｻ繝ｻ繝ｻ繝ｻ・｣繝ｻ
echo.

echo [3/5] Montreal Forced Aligner 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ.. ^(繝ｻ・ｽ 3~10繝ｻﾐｭ)
if exist "%ENV_DIR%" if not exist "%MFA_EXE%" call :remove_env_dir
set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
"%MICROMAMBA_EXE%" create -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge python=%MFA_PYTHON_VERSION% montreal-forced-aligner colorama
if errorlevel 1 (
    echo [FAILED] MFA 繝ｻ・､繝ｻ蛟台ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
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

echo [4/5] ・第㈲・ｵ・ｭ繝ｻ・ｴ/繝ｻ・ｼ繝ｻ・ｸ繝ｻ・ｴ 螂晢｣ｰ蠅橸ｽｬ繝ｻ蛟第匿繝ｻﾂ 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ..
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

echo 繝ｻ・､繝ｻ繝ｻ繝ｻ蟾昜ｺｨ 繝ｻ邁ｿ・ｦ・ｬ 繝ｻ繝ｻ..
if exist "%MICROMAMBA_ARCHIVE%" del "%MICROMAMBA_ARCHIVE%" >nul 2>nul

echo [OK] MFA 繝ｻ・､繝ｻ菫ｾ・ｰﾂ 繝ｻ繝ｻ・｣隱､謐ｮ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
echo.
echo [繝ｻ諛搾ｽ｢繝ｻ ・第㈲・ｵ・ｭ繝ｻ・ｴ/繝ｻ・ｼ繝ｻ・ｸ繝ｻ・ｴ 繝ｻ驢千ｮ・繝ｻ・ｨ繝ｻ・ｸ 繝ｻ・､繝ｻ・ｴ繝ｻ鄂ｹ邉・繝ｻ繝ｻ.. ^(繝ｻ・ｽ 1~2繝ｻﾐｭ)
call :download_acoustic_model korean_mfa
if errorlevel 1 exit /b 1
call :download_acoustic_model japanese_mfa
if errorlevel 1 exit /b 1
call :cleanup_env_caches
if errorlevel 1 exit /b 1
call :cleanup_old_env_if_requested
echo.
echo ====================================================
echo   繝ｻ・､繝ｻ菫ｾ・ｰﾂ 繝ｻ繝ｻ・｣隱､謐ｮ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
echo   繝ｻ・､繝ｻ繝ｻ繝ｻ・ｨ繝ｻ繝ｻ
echo   1) UTAU_Auto_OTO.exe 繝ｻ・､蠏ゅ・^(繝ｻ・ｴ繝ｻ・ｬ繝ｻ繝ｻ繝ｻ隱､邉某)
echo   2) 繝ｻ蟆匁ｪ run.bat 繝ｻ・､蠏ゅ・^(繝ｻ隴ｷ萓・繝ｻ・ｴ蠅橸ｽｬ繝ｻ繝ｻ蟇タ)
echo   3) 繝ｻ・ｴ蠑｡繝ｻ"3. Voice Alignment"繝ｻ・ｼ 繝ｻ隱､豢ｳ 繝ｻ繝ｻ蜩ｩ
echo ====================================================
echo.
if not "%NON_INTERACTIVE%"=="1" pause
exit /b 0

:bootstrap_python_tools
echo MFA Python 逕ｯ・ｨ蠅ｲ・､繝ｻﾂ 繝ｻ繝ｻ・ｵ・ｬ 繝ｻ蟆ゑｽｲﾂ 繝ｻ繝ｻ..
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 繝ｻ・ｰ螟仰繝ｻ繝ｻ謠・繝ｻ・ｾ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import pip, pkg_resources, wheel" >nul 2>nul
if not errorlevel 1 (
    echo [OK] pip/setuptools/wheel繝ｻ・ｴ 繝ｻﾂ繝ｻ繝ｻ謐ｮ繝ｻ・ｴ 繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
    goto :eof
)
echo [INFO] pip/setuptools/wheel 繝ｻ・ｵ繝ｻ・ｬ 繝ｻ繝ｻ..
"%ENV_DIR%\python.exe" -m ensurepip --upgrade
if errorlevel 1 (
    echo [WARN] ensurepip繝ｻ・ｴ 繝ｻ繝ｻ・ｰ繝ｻ譴ｭ 繝ｻ譎､・倥・ﾂ 繝ｻ蝟懆村繝ｻ・ｵ繝ｻ螢ｱ蜈ｱ. pip 繝ｻ・ｵ繝ｻ・ｬ繝ｻ・ｼ 繝ｻ繝ｻ繝ｻ 繝ｻ鄂ｹ蟾｡・托ｽｩ繝ｻ螢ｱ蜈ｱ...
)
echo [INFO] pip module availability check...
"%ENV_DIR%\python.exe" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [WARN] python -m pip failed. Retrying ensurepip with --default-pip...
    "%ENV_DIR%\python.exe" -m ensurepip --upgrade --default-pip
)
set "PYTOOLS_REPAIR_OK=0"
"%ENV_DIR%\python.exe" -m pip install --upgrade --force-reinstall "setuptools<81" wheel
if not errorlevel 1 set "PYTOOLS_REPAIR_OK=1"
if "!PYTOOLS_REPAIR_OK!"=="0" (
    echo [WARN] force-reinstall path failed. Retrying with normal upgrade...
    "%ENV_DIR%\python.exe" -m pip install --upgrade "setuptools<81" wheel
    if not errorlevel 1 set "PYTOOLS_REPAIR_OK=1"
)
if "!PYTOOLS_REPAIR_OK!"=="0" (
    echo [FAILED] pip/setuptools/wheel 繝ｻ・ｵ繝ｻ・ｬ繝ｻ繝ｻ繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import pip, pkg_resources, wheel" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] 繝ｻ・ｵ繝ｻ・ｬ 蠑｡繝ｻ荵ｱ繝ｻ繝ｻPython 逕ｯ・ｨ蠅ｲ・､繝ｻﾂ 繝ｻ繝ｻ・ｵ・ｬ繝ｻ・ｼ 繝ｻ・ｬ繝ｻ・ｩ・托｣ｰ 繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
echo [OK] pip/setuptools/wheel 繝ｻ・ｵ繝ｻ・ｬ 繝ｻ繝ｻ・｣繝ｻ
goto :eof

:install_japanese_support
echo 繝ｻ・ｼ繝ｻ・ｸ繝ｻ・ｴ 螂晢｣ｰ蠅橸ｽｬ繝ｻ蛟第匿繝ｻﾂ 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ蟆ゑｽｲﾂ 繝ｻ繝ｻ..
set "MAMBA_ROOT_PREFIX=%MICROMAMBA_ROOT%"
call :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
if exist "%MICROMAMBA_EXE%" (
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge spacy sudachipy sudachidict-core
    if errorlevel 1 (
        echo [FAILED] 繝ｻ・ｼ繝ｻ・ｸ繝ｻ・ｴ 螂晢｣ｰ蠅橸ｽｬ繝ｻ蛟第匿繝ｻﾂ 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ蛟台ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
        if not "%NON_INTERACTIVE%"=="1" pause
        exit /b 1
    )
    goto :eof
)
echo [WARN] Micromamba 繝ｻ繝ｻ・ｰ・ｹ繝ｻ繝ｻ繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ・托ｽｴ 繝ｻ・ｼ繝ｻ・ｸ繝ｻ・ｴ 螂晢｣ｰ蠅橸ｽｬ繝ｻ蛟第匿繝ｻﾂ 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ菫ｯ・･・ｼ 繝ｻ・ｴ繝ｻ螢ｱ諤舌・螢ｱ蜈ｱ.
goto :eof

:install_audio_deps
if exist "%ENV_DIR%\Library\bin\libsndfile.dll" goto :eof
echo 繝ｻ・､繝ｻ閧･荳ｶ 繝ｻ・ｰ螟仰繝ｻ繝ｻ繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ蟆ゑｽｲﾂ 繝ｻ繝ｻ^(libsndfile^)...
if exist "%MICROMAMBA_EXE%" (
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge libsndfile pysoundfile
    if errorlevel 1 (
        echo [FAILED] 繝ｻ・､繝ｻ閧･荳ｶ 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ蛟台ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
        if not "%NON_INTERACTIVE%"=="1" pause
        exit /b 1
    )
    goto :eof
)
if exist "%ENV_DIR%\Scripts\conda.exe" (
    "%ENV_DIR%\Scripts\conda.exe" install -y --solver classic -p "%ENV_DIR%" -c conda-forge libsndfile pysoundfile
    if errorlevel 1 (
        echo [FAILED] 繝ｻ・､繝ｻ閧･荳ｶ 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ蛟台ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
        if not "%NON_INTERACTIVE%"=="1" pause
        exit /b 1
    )
    goto :eof
)
echo [WARN] micromamba/conda繝ｻ・ｼ 繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ・托ｽｴ 繝ｻ・､繝ｻ閧･荳ｶ 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ菫ｯ・･・ｼ 繝ｻ・ｴ繝ｻ螢ｱ諤舌・螢ｱ蜈ｱ.
goto :eof

:install_textgrid
echo textgrid 繝ｻ・ｨ繝ｻ繝ｻ繝ｻ蟆ゑｽｲﾂ 繝ｻ繝ｻ..
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 繝ｻ・ｰ螟仰繝ｻ繝ｻ謠・繝ｻ・ｾ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import textgrid" >nul 2>nul
if not errorlevel 1 goto :eof
echo [INFO] textgrid 繝ｻ・ｨ繝ｻ繝ｻ繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ..
"%ENV_DIR%\python.exe" -m pip install --upgrade "textgrid>=1.5"
if errorlevel 1 (
    echo [FAILED] textgrid 繝ｻ・､繝ｻ蛟台ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
goto :eof

:verify_textgrid
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 繝ｻ・ｰ螟仰繝ｻ繝ｻ謠・繝ｻ・ｾ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import textgrid" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] textgrid import 繝ｻﾂ繝ｻ・ｬ繝ｻ繝ｻ繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    echo        蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ・ｵ繝ｻ・ｬ繝ｻ・ｼ 繝ｻ繝ｻ邏・setup_mfa.bat繝ｻ・ｼ 繝ｻ・､繝ｻ繝ｻ繝ｻ・､蠏るｧ慕ｴ・繝ｻ・ｼ繝ｻ・ｸ繝ｻ繝ｻ
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
goto :eof

:install_ml_requirements
echo 繝ｻ・ｰ螟九・ML 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ..
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 繝ｻ・ｰ螟仰繝ｻ繝ｻ謠・繝ｻ・ｾ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
if not exist "%APP_DIR%\requirements.txt" (
    echo [FAILED] %APP_DIR%繝ｻ蟾昴・ requirements.txt繝ｻ・ｼ 繝ｻ・ｾ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
if not exist "%APP_DIR%\requirements-ml.txt" (
    echo [FAILED] %APP_DIR%繝ｻ蟾昴・ requirements-ml.txt繝ｻ・ｼ 繝ｻ・ｾ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
if exist "%MICROMAMBA_EXE%" (
    echo [INFO] micromamba繝ｻ繝ｻML 繝ｻ・ｰ螟仰繝ｻ繝ｻ逕ｯ・ｨ蠅ｲ・､繝ｻﾂ 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ..
    "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge pandas lightgbm onnxruntime
    if errorlevel 1 (
        echo [WARN] micromamba ML 繝ｻ・､繝ｻ蛟台ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ. pip繝ｻ繝ｻ蟆橸ｽｴ繝ｻ・ｱ・托ｽｩ繝ｻ螢ｱ蜈ｱ.
    ) else (
        echo [OK] micromamba ML 逕ｯ・ｨ蠅ｲ・､繝ｻﾂ 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ・｣繝ｻ
    )
)
echo [INFO] requirements.txt 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ
"%ENV_DIR%\python.exe" -m pip install --upgrade -r "%APP_DIR%\requirements.txt"
if errorlevel 1 (
    echo [FAILED] requirements.txt 繝ｻ・､繝ｻ蛟台ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
if not exist "%MICROMAMBA_EXE%" (
    echo [INFO] pip繝ｻ繝ｻrequirements-ml.txt 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ^(micromamba繝ｻ・ｼ 繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ・托ｽｨ^)
    "%ENV_DIR%\python.exe" -m pip install --upgrade -r "%APP_DIR%\requirements-ml.txt"
    if errorlevel 1 (
        echo [FAILED] ML 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ蛟台ｹｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
        echo        lightgbm 繝ｻ隱､邉悶・・ｼ 繝ｻ繝ｻ邏・Microsoft Visual C++ Build Tools繝ｻﾂ ・代・蝗茨ｨ托｣ｰ 繝ｻ繝ｻ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
        if not "%NON_INTERACTIVE%"=="1" pause
        exit /b 1
    )
)
echo [OK] ML 繝ｻ・ｰ螟仰繝ｻ繝ｻ繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ・､繝ｻ繝ｻ繝ｻ繝ｻ・｣繝ｻ
goto :eof

:verify_ml_runtime
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 繝ｻ・ｰ螟仰繝ｻ繝ｻ謠・繝ｻ・ｾ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import pandas, lightgbm, onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo [FAILED] ML 繝ｻ・ｰ螟仰繝ｻ繝ｻimport 繝ｻﾂ繝ｻ・ｬ繝ｻ繝ｻ繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ. pandas/lightgbm/onnxruntime繝ｻﾂ ・代・蝗茨ｨ托ｽｩ繝ｻ螢ｱ蜈ｱ.
    if exist "%MICROMAMBA_EXE%" (
        echo        繝ｻ鄂ｹ蟾｡: "%MICROMAMBA_EXE%" install -y -r "%MICROMAMBA_ROOT%" -p "%ENV_DIR%" -c conda-forge pandas lightgbm onnxruntime
    ) else (
        echo        setup_mfa.bat --with-ml 繝ｻ繝ｻ繝ｻ・､繝ｻ繝ｻ繝ｻ・､蠏るｧ慕ｴ・繝ｻ・ｼ繝ｻ・ｸ繝ｻ繝ｻ
    )
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
goto :eof

:cleanup_env_caches
echo 繝ｻ諛搾ｽ｢繝ｻ繝ｻ・､繝ｻ繝ｻ繝ｻ・ｩ繝ｻ繝ｻ繝ｻ闖ｩ繝ｻ繝ｻ・ｼ 繝ｻ繝ｻ邏・逕ｯ・ｨ蠅ｲ・､繝ｻﾂ 繝ｻ蟾昜ｺｨ 繝ｻ邁ｿ・ｦ・ｬ 繝ｻ繝ｻ..
if exist "%MICROMAMBA_EXE%" (
    "%MICROMAMBA_EXE%" clean -a -y -r "%MICROMAMBA_ROOT%" >nul 2>nul
)
if exist "%ENV_DIR%\python.exe" (
    "%ENV_DIR%\python.exe" -m pip cache purge >nul 2>nul
)
echo [OK] 繝ｻ蟾昜ｺｨ 繝ｻ邁ｿ・ｦ・ｬ 繝ｻ繝ｻ・｣繝ｻ
goto :eof

:install_korean_support
echo ・第㈲・ｵ・ｭ繝ｻ・ｴ 螂晢｣ｰ蠅橸ｽｬ繝ｻ蛟第匿繝ｻﾂ 繝ｻ蛟托ｽ｡・ｴ繝ｻ・ｱ 繝ｻ蟆ゑｽｲﾂ 繝ｻ繝ｻ..
call :ensure_mfa_entrypoint
if errorlevel 1 exit /b 1
if not exist "%ENV_DIR%\python.exe" (
    echo [FAILED] MFA Python 繝ｻ・ｰ螟仰繝ｻ繝ｻ謠・繝ｻ・ｾ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import sys,importlib,jamo; ok=bool(importlib.util.find_spec('eunjeon') or importlib.util.find_spec('mecab')); sys.exit(0 if ok else 1)" >nul 2>nul
if not errorlevel 1 goto :patch_korean_support
echo [INFO] Installing Korean tokenizer dependency: jamo
"%ENV_DIR%\python.exe" -m pip install --upgrade jamo
if errorlevel 1 (
    echo [FAILED] Korean tokenizer dependency install failed: jamo
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
"%ENV_DIR%\python.exe" -c "import sys,importlib,jamo; ok=bool(importlib.util.find_spec('eunjeon') or importlib.util.find_spec('mecab')); sys.exit(0 if ok else 1)" >nul 2>nul
if not errorlevel 1 goto :patch_korean_support
echo [INFO] Installing Korean tokenizer backend: eunjeon
"%ENV_DIR%\python.exe" -m pip install --upgrade eunjeon
if errorlevel 1 (
    echo [FAILED] Korean tokenizer backend install failed: eunjeon
    echo        Microsoft Visual C++ Build Tools may be required for this package.
    if not "%NON_INTERACTIVE%"=="1" pause
    exit /b 1
)
:patch_korean_support
set "PYTHONPATH=%APP_DIR%"
"%ENV_DIR%\python.exe" -c "from core.mfa_runner import patch_mfa_korean_support; patch_mfa_korean_support(r'%MFA_EXE%')" >nul 2>nul
if errorlevel 1 (
    echo [WARN] ・第㈲・ｵ・ｭ繝ｻ・ｴ MFA 逕ｯ・ｨ繝ｻ繝ｻ繝ｻ・ｨ繝ｻ繝ｻ荵ｱ 繝ｻ・､逕ｯ・ｨ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ. 繝ｻ邁ｿ・ｰ・ｬ繝ｻﾂ 繝ｻ蜻ｷ譴・托｣ｰ 繝ｻ繝ｻ繝ｻ貅｢諢阪・繝ｻ・第㈲・ｵ・ｭ繝ｻ・ｴ 螂晢｣ｰ蠅橸ｽｬ繝ｻ蛟第匿繝ｻﾂ 繝ｻﾂ繝ｻ蟾晄匿 繝ｻ貅｢蛻九・繝ｻ閻ｹ 繝ｻ繝ｻ繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
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
echo [INFO] 繝ｻ・ｰ繝ｻ・ｰ Micromamba MD5 繝ｻ陲ｫ繝繝ｻ・ｰ繝ｻ・ｴ螂難ｽｰ: %MICROMAMBA_MD5_EXPECTED%
exit /b 0

:verify_micromamba_archive
if not exist "%MICROMAMBA_ARCHIVE%" (
    echo [FAILED] Micromamba 繝ｻ繝ｻ・ｹ・ｴ繝ｻ・ｴ繝ｻ繝ｻ逕ｯ隴ｷ謾ｵ繝ｻ・ｴ 繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ: %MICROMAMBA_ARCHIVE%
    exit /b 1
)
set "MICROMAMBA_MD5_ACTUAL="
for /f "tokens=1" %%h in ('certutil -hashfile "%MICROMAMBA_ARCHIVE%" MD5 ^| findstr /R /I "^[0-9A-F][0-9A-F]"') do (
    if not defined MICROMAMBA_MD5_ACTUAL set "MICROMAMBA_MD5_ACTUAL=%%h"
)
if not defined MICROMAMBA_MD5_ACTUAL (
    echo [FAILED] Micromamba MD5 ・托ｽｴ繝ｻ鄂ｹ・･・ｼ 繝ｻ繝ｻ縺抵ｨ大托ｽｧﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    exit /b 1
)
echo [INFO] Micromamba 繝ｻ繝ｻ・ｹ・ｴ繝ｻ・ｴ繝ｻ繝ｻMD5: %MICROMAMBA_MD5_ACTUAL%
if defined MICROMAMBA_MD5_EXPECTED (
    if /I not "%MICROMAMBA_MD5_ACTUAL%"=="%MICROMAMBA_MD5_EXPECTED%" (
        echo [FAILED] Micromamba ・托ｽｴ繝ｻ繝ｻ繝ｻ貅｢謾ｵ繝ｻ菫ｾ・ｰﾂ 繝ｻ蟾晢ｽｧﾂ繝ｻ蛟第擂繝ｻ・ｵ繝ｻ螢ｱ蜈ｱ.
        echo         繝ｻ・ｰ繝ｻﾂ繝ｻ繝ｻ %MICROMAMBA_MD5_EXPECTED%
        echo         繝ｻ・､繝ｻ諛具ｽｰ繝ｻ %MICROMAMBA_MD5_ACTUAL%
        del "%MICROMAMBA_ARCHIVE%" >nul 2>nul
        exit /b 1
    )
    echo [OK] Micromamba ・托ｽｴ繝ｻ繝ｻ繝ｻﾂ繝ｻ繝ｻ螂晢ｽｵ繝ｻ・ｼ.
) else (
    echo [WARN] 繝ｻ・ｰ繝ｻ・ｰ Micromamba ・托ｽｴ繝ｻ繝ｻ繝ｻ陲ｫ繝繝ｻ・ｰ繝ｻ・ｴ螂難ｽｰ繝ｻ・ｼ 繝ｻ・ｬ繝ｻ・ｩ・托｣ｰ 繝ｻ繝ｻ繝ｻ繝ｻ諷｣繝ｻ螢ｱ蜈ｱ.
    if "%NON_INTERACTIVE%"=="1" exit /b 1
    choice /C YN /N /M "繝ｻﾂ繝ｻ譎､鬆・Micromamba ・托ｽｴ繝ｻ繝ｻ繝ｻ繝ｻ謫ｽ 繝ｻ繝ｻ繝ｻ・托｣ｰ繝ｻ隴ｷ蝗・ [Y/N]: "
    if errorlevel 2 exit /b 1
)
exit /b 0

:handle_old_env
echo.
echo [WARN] 繝ｻ荳・ｱ・ｰ繝ｻ繝ｻMFA 蠍ｹ菫ｾ・ｲ・ｽ繝ｻ・ｴ 繝ｻ蟾晢ｽｧﾂ繝ｻ蛟第擂繝ｻ・ｵ繝ｻ螢ｱ蜈ｱ:
echo        %OLD_ENV_DIR%
echo.
echo 繝ｻ荳・ｱ・ｰ繝ｻ繝ｻ蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ菫ｯ・ｦ・ｬ 繝ｻ・ｩ繝ｻ闖ｩ謠・繝ｻ・ｰ螟区勣闡ｺ繝ｻ・ｸ繝ｻ繝ｻ
echo   [M] 繝ｻ貅｢謫ｽ繝ｻ・ｸ繝ｻ貅｢謫ｽ繝ｻ繝ｻ^(繝ｻ諛搾ｽｻ・ｬ 蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ・ｬ繝ｻ・ｬ繝ｻ・ｱ 蠑｡繝ｻ繝ｻ・ｰ繝ｻ・ｴ 蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ・ｭ繝ｻ蠑ｯ)
echo   [D] 繝ｻ・ｰ繝ｻ・ｴ 蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻﾂ繝ｻ繝ｻ繝ｻ鄂ｷ・｡繝ｻ繝ｻ・ｭ繝ｻ繝ｻ
echo   [K] 繝ｻ・ｰ繝ｻ・ｴ 蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ・ｰ繝ｻﾂ ^(繝ｻ・ｭ繝ｻ諛ｦ闡ｺ繝ｻﾂ 繝ｻ蝟懈炊^)
choice /C MDK /N /M "繝ｻ・ｰ螟九・^(M/D/K^): "
if errorlevel 3 goto :keep_old_env
if errorlevel 2 goto :delete_old_env_now
if errorlevel 1 goto :migrate_old_env
goto :eof

:migrate_old_env
set "DELETE_OLD_AFTER_INSTALL=1"
echo [INFO] 繝ｻ諛搾ｽｻ・ｬ 繝ｻ・､繝ｻ菫ｾ・ｰﾂ 繝ｻ・ｱ繝ｻ・ｵ・台ｿｯ・ｩ・ｴ 繝ｻ荳・ｱ・ｰ繝ｻ繝ｻ蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻ・ｭ繝ｻ諛ｦ魄偵・螢ｱ蜈ｱ.
goto :eof

:delete_old_env_now
call :remove_dir "%OLD_ENV_DIR%"
call :remove_dir "%OLD_MICROMAMBA_ROOT%"
goto :eof

:keep_old_env
echo [INFO] 繝ｻ荳・ｱ・ｰ繝ｻ繝ｻ蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻ・ｰ繝ｻﾂ・代・繝ｻ繝ｻ繝ｻ諛搾ｽｻ・ｬ 繝ｻ・､繝ｻ菫ｯ・･・ｼ 繝ｻ繝ｻ繝ｻ・托ｽｩ繝ｻ螢ｱ蜈ｱ.
goto :eof

:cleanup_old_env_if_requested
if not "%DELETE_OLD_AFTER_INSTALL%"=="1" goto :eof
if /i "%OLD_ENV_DIR%"=="%ENV_DIR%" goto :eof
call :remove_dir "%OLD_ENV_DIR%"
call :remove_dir "%OLD_MICROMAMBA_ROOT%"
echo [OK] 繝ｻ荳・ｱ・ｰ繝ｻ繝ｻMFA 蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻ諛具ｽｱ・ｰ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
goto :eof

:remove_dir
set "TARGET_DIR=%~1"
if "%TARGET_DIR%"=="" goto :eof
if not exist "%TARGET_DIR%" goto :eof
echo [INFO] 繝ｻ諛具ｽｱ・ｰ 繝ｻ繝ｻ %TARGET_DIR%
rmdir /s /q "%TARGET_DIR%" >nul 2>nul
if exist "%TARGET_DIR%" (
    echo [WARN] 繝ｻ諛具ｽｱ・ｰ 繝ｻ・､逕ｯ・ｨ: %TARGET_DIR%
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
echo [FAILED] 蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ譎ｧ笏ｳ 蠑｡繝ｻMFA 繝ｻ・､蠏ゅ・逕ｯ隴ｷ謾ｵ繝ｻ繝ｻ繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
if not "%NON_INTERACTIVE%"=="1" pause
exit /b 1

:download_acoustic_model
set "MODEL_NAME=%~1"
if not defined MODEL_NAME (
    echo [FAILED] 繝ｻ驢千ｮ・繝ｻ・ｨ繝ｻ・ｸ 繝ｻ・ｴ繝ｻ繝ｻ謫ｽ 繝ｻ繝ｻ迚・繝ｻ貅｢諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
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
echo [FAILED] 繝ｻ・ｨ繝ｻ・ｸ 繝ｻ・､繝ｻ・ｴ繝ｻ鄂ｹ邉悶・繝ｻ繝ｻ・ｬ繝ｻ・ｩ・托｣ｰ 繝ｻ・､蠏ゅ・繝ｻﾂ繝ｻ・･・代・MFA 蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻ・ｾ繝ｻﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
if not "%NON_INTERACTIVE%"=="1" pause
exit /b 1

:remove_env_dir
if not exist "%ENV_DIR%" goto :eof
echo [INFO] 繝ｻ・ｰ繝ｻ・ｴ 蠍ｹ菫ｾ・ｲ・ｽ 繝ｻ諛具ｽｱ・ｰ 繝ｻ繝ｻ %ENV_DIR%
rmdir /s /q "%ENV_DIR%" >nul 2>nul
if exist "%ENV_DIR%" (
    echo [FAILED] 繝ｻ・ｰ繝ｻ・ｴ MFA 蠍ｹ菫ｾ・ｲ・ｽ繝ｻ繝ｻ繝ｻ諛具ｽｱ・ｰ・大托ｽｧﾂ 繝ｻ・ｻ蠏よｺ｢諷｣繝ｻ螢ｱ蜈ｱ.
    if not "%NON_INTERACTIVE%"=="1" pause
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
