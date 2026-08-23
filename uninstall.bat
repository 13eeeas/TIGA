@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — Uninstall

cd /d "%~dp0"

echo.
echo ============================================================
echo   TIGA Hunt — Uninstall
echo ============================================================
echo.
echo This will:
echo   - Stop TIGA server and admin processes
echo   - Remove desktop shortcuts (TIGA Hunt, TIGA Admin, Uninstall TIGA)
echo.
echo Your indexed data in tiga_work\ is kept by default.
echo.

choice /C YN /M "Continue with uninstall?"
if errorlevel 2 goto :cancel

set REMOVE_VENV=N
set REMOVE_DATA=N

choice /C YN /M "Also remove Python virtual environment (.venv)?"
if not errorlevel 2 set REMOVE_VENV=Y

choice /C YN /M "Also DELETE all local index data (tiga_work)? THIS CANNOT BE UNDONE"
if not errorlevel 2 set REMOVE_DATA=Y

if not exist ".venv\Scripts\activate.bat" (
    echo [WARN] .venv not found — running uninstall steps without venv.
    goto :manual
)

call .venv\Scripts\activate.bat

if "%REMOVE_VENV%"=="Y" if "%REMOVE_DATA%"=="Y" (
    python tiga.py uninstall --venv --data --yes
) else if "%REMOVE_VENV%"=="Y" (
    python tiga.py uninstall --venv --yes
) else if "%REMOVE_DATA%"=="Y" (
    python tiga.py uninstall --data --yes
) else (
    python tiga.py uninstall --yes
)
goto :done

:manual
echo Removing desktop shortcuts manually...
powershell -NoProfile -Command ^
  "$d=[Environment]::GetFolderPath('Desktop');" ^
  "'TIGA Hunt','TIGA Admin','Uninstall TIGA' | ForEach-Object { Remove-Item (Join-Path $d ($_.lnk)) -ErrorAction SilentlyContinue }"
goto :done

:cancel
echo Uninstall cancelled.
pause
exit /b 0

:done
echo.
echo ============================================================
echo   Uninstall complete
echo ============================================================
echo.
echo TIGA shortcuts removed. The install folder was not deleted.
echo You can delete this folder manually if you no longer need it:
echo   %~dp0
echo.
pause
