@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "APP_DIR=%~dp0"
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"
set "DIAG_SCRIPT=%APP_DIR%\startup_diagnose.ps1"

echo ==========================================
echo UTAU Auto OTO - Startup Diagnose
echo ==========================================

if not exist "%DIAG_SCRIPT%" (
  echo [ERROR] startup_diagnose.ps1 was not found.
  echo        Expected: "%DIAG_SCRIPT%"
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%DIAG_SCRIPT%" -AppRoot "%APP_DIR%" -OpenFolder
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [OK] Startup diagnosis completed.
) else (
  echo [WARN] Startup diagnosis reported failures. Check logs\startup_diagnose_*.txt
)
pause
exit /b %RC%
