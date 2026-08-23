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
echo Starting API server...
start "TIGA-Server" cmd /k "call .venv\Scripts\activate.bat && python tiga.py serve"

timeout /t 3 /nobreak >nul

echo Starting Admin panel...
start "TIGA-Admin" cmd /k "call .venv\Scripts\activate.bat && python tiga.py ui"

timeout /t 2 /nobreak >nul

for /f "delims=" %%u in ('python -c "from tools.launcher_util import launcher_url; print(launcher_url())"') do set LAUNCHER=%%u
start "" "%LAUNCHER%"

echo.
echo ============================================
echo  TIGA Hunt is running
echo  Launcher: %LAUNCHER%
echo  Hunt:     http://localhost:7860/
echo  Admin:    http://localhost:7861/
echo  Press Ctrl+C in each service window to stop
echo ============================================
echo.
pause
