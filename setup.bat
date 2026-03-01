@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — One-Click Setup

echo.
echo ============================================================
echo   TIGA Hunt — One-Click Installer
echo ============================================================
echo.

:: ---------------------------------------------------------------------------
:: 1. Python check
:: ---------------------------------------------------------------------------

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Download from: https://www.python.org/downloads/
    echo         Make sure to tick "Add Python to PATH" during install.
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER%

:: ---------------------------------------------------------------------------
:: 2. Virtual environment
:: ---------------------------------------------------------------------------

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 ( echo [ERROR] Failed to create .venv & pause & exit /b 1 )
    echo [OK] .venv created
) else (
    echo [OK] .venv already exists
)

call .venv\Scripts\activate.bat

:: ---------------------------------------------------------------------------
:: 3. pip + dependencies
:: ---------------------------------------------------------------------------

echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo Installing TIGA dependencies (this may take a few minutes)...
pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed & pause & exit /b 1 )
echo [OK] Dependencies installed

:: Optional: sentence-transformers for cross-encoder reranker
echo Installing sentence-transformers (cross-encoder reranker, optional)...
pip install sentence-transformers --quiet
if not errorlevel 1 (
    echo [OK] sentence-transformers installed ^(reranker enabled^)
) else (
    echo [WARN] sentence-transformers skipped — reranker will be disabled
)

:: ---------------------------------------------------------------------------
:: 4. Ollama check + install
:: ---------------------------------------------------------------------------

where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] Ollama not found. Downloading Ollama installer...
    echo        This requires an internet connection.
    echo.
    :: Download and run Ollama installer silently
    set "OLLAMA_INSTALLER=%TEMP%\OllamaSetup.exe"
    powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile '!OLLAMA_INSTALLER!' -UseBasicParsing" 2>nul
    if exist "!OLLAMA_INSTALLER!" (
        echo [INFO] Running Ollama installer...
        "!OLLAMA_INSTALLER!" /S
        timeout /t 5 /nobreak >nul
        where ollama >nul 2>&1
        if not errorlevel 1 (
            echo [OK] Ollama installed
        ) else (
            echo [WARN] Ollama installer ran but 'ollama' not in PATH yet.
            echo        You may need to restart your terminal after setup.
        )
    ) else (
        echo [WARN] Could not download Ollama. Install manually from: https://ollama.com
    )
) else (
    for /f "tokens=*" %%v in ('ollama --version 2^>^&1') do set OL_VER=%%v
    echo [OK] Ollama found: !OL_VER!
)

:: ---------------------------------------------------------------------------
:: 5. Pull Ollama models
:: ---------------------------------------------------------------------------

where ollama >nul 2>&1
if not errorlevel 1 (
    echo.
    echo Pulling nomic-embed-text (embedding model, ~274 MB)...
    ollama pull nomic-embed-text
    if errorlevel 1 (
        echo [WARN] Could not pull nomic-embed-text — run manually: ollama pull nomic-embed-text
    ) else (
        echo [OK] nomic-embed-text ready
    )

    echo Pulling mistral (chat model, ~4 GB)...
    ollama pull mistral
    if errorlevel 1 (
        echo [WARN] Could not pull mistral — run manually: ollama pull mistral
    ) else (
        echo [OK] mistral ready
    )
) else (
    echo [WARN] Ollama not available — skipping model pull.
    echo        Install Ollama from https://ollama.com and then run:
    echo          ollama pull nomic-embed-text
    echo          ollama pull mistral
)

:: ---------------------------------------------------------------------------
:: 6. Create default config (if missing)
:: ---------------------------------------------------------------------------

if not exist "tiga_work\config.yaml" (
    echo.
    echo Creating default config.yaml...
    python tiga.py init
    if errorlevel 1 (
        echo [WARN] Could not create config.yaml automatically.
        echo        Run: python tiga.py init
    ) else (
        echo [OK] config.yaml created at tiga_work\config.yaml
    )
) else (
    echo [OK] config.yaml already exists
)

:: ---------------------------------------------------------------------------
:: 7. Work directory structure
:: ---------------------------------------------------------------------------

echo Creating work directories...
python -c "from config import cfg; cfg.ensure_dirs(); print('[OK] Work dirs ready')" 2>nul
if errorlevel 1 (
    echo [WARN] Could not create work dirs (config may need editing first)
)

:: ---------------------------------------------------------------------------
:: Done
:: ---------------------------------------------------------------------------

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo  NEXT STEPS:
echo.
echo  1. Edit your archive paths in:
echo       tiga_work\config.yaml
echo     Set index_roots to your project drive, e.g.:
echo       index_roots:
echo         - "Z:/Projects"
echo.
echo  2. Start TIGA Hunt:
echo       run.bat
echo     (or: python tiga.py serve)
echo.
echo  3. Open your browser to: http://localhost:7860
echo.
echo  4. (Optional) Run a quick folder scan before indexing:
echo       python tiga.py scan "Z:/Projects" --phases
echo.
echo  5. (Optional) Run the WOHA project seeder:
echo       python tiga.py scrape-woha
echo.
pause
