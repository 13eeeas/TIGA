@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  TIGA Hunt — Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://www.python.org
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER%

:: Create virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 ( echo [ERROR] Failed to create venv & pause & exit /b 1 )
    echo [OK] .venv created
) else (
    echo [OK] .venv already exists
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install dependencies
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed & pause & exit /b 1 )
echo [OK] Dependencies installed

:: Init config
if not exist "tiga_work\config.yaml" (
    echo Creating default config...
    python tiga.py init
) else (
    echo [OK] Config already exists at tiga_work\config.yaml
)

echo.
echo ============================================
echo  MANUAL STEPS STILL REQUIRED:
echo ============================================
echo.
echo  1. Install Ollama from: https://ollama.com
echo     (run the installer, then come back here)
echo.
echo  2. Pull required models:
echo     ollama pull nomic-embed-text
echo     ollama pull mistral
echo.
echo  3. Edit tiga_work\config.yaml
echo     Set index_roots to your archive directories
echo.
echo  4. Run TIGA Hunt:
echo     run.bat
echo.
pause
