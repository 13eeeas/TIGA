@echo off
setlocal
title TIGA Hunt — POC Test (one-click)

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Run setup.bat first.
    pause & exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo ============================================================
echo   TIGA POC Test
echo   Choose projects -^> index -^> stress retrieval -^> export zip
echo ============================================================
echo.

python tiga.py poc-test run

echo.
pause
