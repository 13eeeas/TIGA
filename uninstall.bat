@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — Uninstall

echo.
echo ============================================================
echo   TIGA Hunt — Uninstall
echo ============================================================
echo.

cd /d "%~dp0"

set REMOVE_DATA=0
choice /C YN /M "Also remove local data (tiga_work — indexes, config, logs)?"
if !errorlevel! equ 1 set REMOVE_DATA=1

echo.
echo Stopping TIGA processes...
taskkill /F /FI "WINDOWTITLE eq TIGA*" 2>nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq TIGA*" 2>nul

echo Removing virtual environment...
if exist ".venv" rmdir /s /q ".venv"

if !REMOVE_DATA! equ 1 (
    echo Removing local data...
    if exist "tiga_work" rmdir /s /q "tiga_work"
)

echo.
echo ============================================================
echo   TIGA Hunt uninstalled.
echo   Source files remain in: %CD%
echo   Delete this folder manually if you want a full removal.
echo ============================================================
echo.
pause
