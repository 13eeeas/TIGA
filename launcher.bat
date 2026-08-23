@echo off
setlocal
title TIGA Hunt — Launcher

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Run setup.bat first.
    pause & exit /b 1
)

call .venv\Scripts\activate.bat

:: Start services if not already running
python -c "from tools.launcher_util import ensure_services, launcher_url; ensure_services(); print(launcher_url())" > "%TEMP%\tiga_launcher_url.txt"
set /p LAUNCHER=<"%TEMP%\tiga_launcher_url.txt"
del "%TEMP%\tiga_launcher_url.txt" 2>nul

timeout /t 2 /nobreak >nul

start "" "%LAUNCHER%"
echo.
echo TIGA launcher opened in your browser.
echo Bookmark %LAUNCHER% for one-click access.
echo.
pause
