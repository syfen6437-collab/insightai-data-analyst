@echo off
setlocal
cd /d "%~dp0"

rem Keep one startup path so the API-key prompt and port handling match start.ps1.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
  echo.
  echo 启动失败，请查看本目录 startup.log。
  pause
  exit /b 1
)
exit /b 0
