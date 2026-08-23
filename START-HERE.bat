@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — Start Here

cd /d "%~dp0"

echo.
echo ============================================================
echo   TIGA Hunt — Setup + POC Test (one click)
echo ============================================================
echo.
echo   1. Install dependencies ^(first time only^)
echo   2. Point at your project folders on NAS/drive
echo   3. Index, stress-test retrieval, export results zip
echo.

:: ---------------------------------------------------------------------------
:: Step 1 — Install
:: ---------------------------------------------------------------------------

if not exist ".venv\Scripts\activate.bat" (
    echo [Step 1/3] Installing TIGA ^(may take 10-20 min first time^)...
    call "%~dp0setup.bat" --nopause
    if errorlevel 1 (
        echo.
        echo Setup failed. Fix errors above and run START-HERE.bat again.
        pause & exit /b 1
    )
) else (
    echo [Step 1/3] Install — already done
)

call .venv\Scripts\activate.bat

:: ---------------------------------------------------------------------------
:: Step 2 — Configure NAS paths
:: ---------------------------------------------------------------------------

echo.
echo [Step 2/3] Project folders...
python tools\office_setup.py configure
if errorlevel 1 (
    echo.
    echo Configure cancelled or failed. Edit tiga_work\config.yaml then run poc-test.bat
    pause & exit /b 1
)

:: ---------------------------------------------------------------------------
:: Step 3 — POC test
:: ---------------------------------------------------------------------------

echo.
echo [Step 3/3] POC retrieval test ^(pick projects, index, stress, export^)...
echo.
python tiga.py poc-test run
set TEST_EXIT=!ERRORLEVEL!

echo.
echo ============================================================
if !TEST_EXIT! equ 0 (
    echo   POC test complete
) else (
    echo   POC test finished — review scores above
)
echo ============================================================
echo.
echo   Export zip:  tiga_work\poc_test\exports\
echo   Daily use:   launcher.bat
echo   Re-test:     poc-test.bat
echo.
pause
exit /b !TEST_EXIT!
