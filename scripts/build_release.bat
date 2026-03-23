@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0.."
set "ROOT=%ROOT:~0,-1%"
set "CHANNEL=%~1"
if "%CHANNEL%"=="" set "CHANNEL=stable"
if /I "%CHANNEL%"=="default" set "CHANNEL=stable"

if /I not "%CHANNEL%"=="stable" if /I not "%CHANNEL%"=="preview" (
  echo [ERROR] Invalid channel: %CHANNEL%
  echo Usage: scripts\build_release.bat [stable^|default^|preview]
  exit /b 1
)

set "RELEASE_DIR=UTAU_Auto_OTO_Release_%CHANNEL%"
set "RELEASE_ZIP=UTAU_Auto_OTO_Release_%CHANNEL%.zip"
set "PORTABLE_ZIP=portable_output\UTAU_Auto_OTO_portable_with_models_%CHANNEL%.zip"
set "MFA_BUNDLE_ARGS=--bundle-mfa-models --require-mfa-models"
if /I "%UTOA_DISABLE_MFA_BUNDLE%"=="1" set "MFA_BUNDLE_ARGS="

echo ==========================================
echo   UTAU Auto OTO - Build and Package
echo   Channel: %CHANNEL%
if defined MFA_BUNDLE_ARGS (
echo   MFA bundle: enabled ^(acoustic models only^)
) else (
echo   MFA bundle: disabled ^(UTOA_DISABLE_MFA_BUNDLE=1^)
)
echo ==========================================

pushd "%ROOT%"

echo [1/4] Build app (onedir)
python build.py --name UTAU_Auto_OTO --backend nuitka --channel %CHANNEL% %MFA_BUNDLE_ARGS%
if errorlevel 1 goto :error

echo [2/4] Create portable zip (without models)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path '%RELEASE_DIR%\*' -DestinationPath '%RELEASE_ZIP%' -Force"
if errorlevel 1 goto :error

echo [3/4] Create portable zip (with ML_models)
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build_portable_with_models.ps1" -Channel "%CHANNEL%" -SourceDir "%RELEASE_DIR%" -OutputZip "%PORTABLE_ZIP%"
if errorlevel 1 goto :error

echo [4/4] Build installer (optional)
where /q ISCC.exe
if errorlevel 1 (
  echo [WARN] Inno Setup not found. Skipping installer build.
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build_installer.ps1" -Channel "%CHANNEL%" -SourceDir "%RELEASE_DIR%"
  if errorlevel 1 goto :error
)

popd
echo.
echo [DONE] Artifacts:
echo   - %RELEASE_ZIP%
echo   - %PORTABLE_ZIP%
echo   - installer_output\UTAU_Auto_OTO_Setup_*_%CHANNEL%.exe (if Inno Setup is installed)
pause
exit /b 0

:error
echo [ERROR] Build failed.
popd
pause
exit /b 1
