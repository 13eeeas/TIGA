@echo off
setlocal

echo ============================================
echo  TIGA Hunt — Starting services
echo ============================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause & exit /b 1
)

call .venv\Scripts\activate.bat

:: Index (incremental — skips unchanged files)
echo Checking for new/changed files to index...
python tiga.py index

echo.
echo Starting API server on port 7860...
start "TIGA-Server" cmd /k "call .venv\Scripts\activate.bat && python tiga.py serve"

timeout /t 3 /nobreak >nul

echo Starting UI on port 8501...
start "TIGA-UI" cmd /k "call .venv\Scripts\activate.bat && python tiga.py ui"

echo.
echo ============================================
echo  TIGA Hunt is running
echo  UI:  http://localhost:8501
echo  API: http://localhost:7860
echo  Press Ctrl+C in each window to stop
echo ============================================
echo.
pause
