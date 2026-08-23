@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — Open Admin

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Run setup.bat first.
    pause & exit /b 1
)

call .venv\Scripts\activate.bat
python -c "from tools.launcher_util import ensure_services, admin_url; ensure_services(); print(admin_url())" > "%TEMP%\tiga_admin_url.txt"
set /p ADMIN_URL=<"%TEMP%\tiga_admin_url.txt"
del "%TEMP%\tiga_admin_url.txt" 2>nul
timeout /t 3 /nobreak >nul
start "" "%ADMIN_URL%"
