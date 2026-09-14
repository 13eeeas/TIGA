@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — Safe Update

cd /d "%~dp0"

echo.

:: Prefer a system / Codex Python for the updater itself (stdlib only).
:: The office .venv is used later only for pip, so a broken venv cannot
:: prevent the next update.

set "PYTHON_CMD="
set "CODEX_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if exist "%CODEX_PYTHON%" (
    "%CODEX_PYTHON%" --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%CODEX_PYTHON%"
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    where py >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"
)

if not defined PYTHON_CMD (
    echo [ERROR] Python not found. Install from https://www.python.org
    echo         Tick "Add Python to PATH". tiga_work data was not changed.
    if /I not "%TIGA_UPDATE_NOPAUSE%"=="1" pause
    exit /b 1
)

:: Strip wrapper-only --nopause so argparse in safe_update.py stays clean.
set "FORWARD="
set "DO_PAUSE=1"
if /I "%TIGA_UPDATE_NOPAUSE%"=="1" set "DO_PAUSE=0"

:argloop
if "%~1"=="" goto run
if /I "%~1"=="--nopause" (
    set "DO_PAUSE=0"
) else (
    set "FORWARD=!FORWARD! %1"
)
shift
goto argloop

:run
%PYTHON_CMD% "%~dp0tools\safe_update.py" !FORWARD!
set "UPDATE_EXIT=!ERRORLEVEL!"

if not !UPDATE_EXIT! equ 0 (
    echo.
    echo [ERROR] Update did not succeed. See rollback notes above.
    echo         Office source edits and tiga_work data were not discarded.
)

if "!DO_PAUSE!"=="1" pause
exit /b !UPDATE_EXIT!
