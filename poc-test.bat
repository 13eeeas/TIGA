@echo off
setlocal
title TIGA Hunt — POC Test (one-click)

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Run START-HERE.bat first ^(install + configure^).
    pause & exit /b 1
)

call .venv\Scripts\activate.bat

python tools\office_setup.py check >nul 2>&1
if errorlevel 2 (
    echo [WARN] index_roots not set — running configure...
    python tools\office_setup.py configure
    if errorlevel 1 pause & exit /b 1
)

echo.
echo ============================================================
echo   TIGA POC Test
echo   Choose projects -^> index -^> stress retrieval -^> export zip
echo ============================================================
echo.

python tiga.py poc-test run

echo.
echo Export: tiga_work\poc_test\exports\
pause
