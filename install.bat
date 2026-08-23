@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — Install and Run

echo.
echo ============================================================
echo   TIGA Hunt — One-Click Install and Run
echo ============================================================
echo.

cd /d "%~dp0"

:: First-time setup (creates .venv, deps, Ollama models, config)
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] First run — running setup...
    call setup.bat
    if errorlevel 1 (
        echo [ERROR] Setup failed.
        pause & exit /b 1
    )
) else (
    echo [OK] Virtual environment ready
)

:: Launch TIGA
echo.
echo [INFO] Starting TIGA Hunt...
call run.bat
