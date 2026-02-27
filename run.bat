@echo off
chcp 65001 >nul
title UTAU Auto OTO Generator
cd /d "%~dp0"

echo [Checking core dependencies...]
python -m pip install -r requirements.txt -q --user --no-warn-script-location
if exist ".env\python.exe" (
    .env\Scripts\pip.exe install -r requirements.txt -q --user --no-warn-script-location
)

REM 포터블 환경이 있으면 그걸 사용
if exist ".env\python.exe" (
    .env\python.exe main.py
) else (
    python main.py
)

if errorlevel 1 (
    echo.
    echo ❌ 프로그램 실행 중 오류가 발생했습니다.
    echo    logs 폴더의 로그 파일을 확인해 주세요.
    pause
)
