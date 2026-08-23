@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — Bootstrap Installer

echo.
echo ============================================================
echo   TIGA Hunt — Bootstrap Installer
echo   Clones from GitHub, installs, and starts TIGA
echo ============================================================
echo.

set "DEFAULT_DIR=%USERPROFILE%\TIGA"
set "INSTALL_DIR=%DEFAULT_DIR%"

:: Allow custom install path
set /p CUSTOM_DIR="Install location [%DEFAULT_DIR%]: "
if not "!CUSTOM_DIR!"=="" set "INSTALL_DIR=!CUSTOM_DIR!"

:: Git check
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Install from: https://git-scm.com/download/win
    pause & exit /b 1
)

if exist "!INSTALL_DIR!\.git" (
    echo [INFO] TIGA already installed at !INSTALL_DIR!
    cd /d "!INSTALL_DIR!"
    call install.bat
    exit /b 0
)

if exist "!INSTALL_DIR!" (
    echo [ERROR] Folder exists but is not a TIGA install: !INSTALL_DIR!
    pause & exit /b 1
)

echo.
echo Cloning TIGA from GitHub to !INSTALL_DIR! ...
git clone https://github.com/13eeeas/TIGA.git "!INSTALL_DIR!"
if errorlevel 1 (
    echo [ERROR] git clone failed. Check network connection.
    pause & exit /b 1
)

cd /d "!INSTALL_DIR!"
echo.
echo [OK] Clone complete. Running install...
call install.bat
