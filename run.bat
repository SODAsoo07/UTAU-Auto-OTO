@echo off
chcp 65001 >nul
title UTAU Auto OTO Generator
cd /d "%~dp0"

echo [Checking core dependencies...]
set "HAS_SYS_PY=0"
where python >nul 2>nul && set "HAS_SYS_PY=1"

if exist ".env\python.exe" (
    .env\Scripts\pip.exe install -r requirements.txt -q --disable-pip-version-check
) else if "%HAS_SYS_PY%"=="1" (
    python -m pip install -r requirements.txt -q --user --no-warn-script-location --disable-pip-version-check
)

if exist ".env\python.exe" (
    .env\python.exe main.py
) else if "%HAS_SYS_PY%"=="1" (
    python main.py
) else (
    echo.
    echo [ERROR] Python is not installed and .env is missing.
    echo         Run setup_mfa.bat first, or launch UTAU_Auto_OTO.exe.
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo [ERROR] Program execution failed.
    echo        Check files in the logs folder.
    pause
)
