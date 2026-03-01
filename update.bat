@echo off
setlocal enabledelayedexpansion
title TIGA Hunt — Update

echo.
echo ============================================================
echo   TIGA Hunt — Update from GitHub
echo ============================================================
echo.

:: ---------------------------------------------------------------------------
:: Check we are in a git repo
:: ---------------------------------------------------------------------------

git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not a git repository. Run this from the TIGA folder.
    pause & exit /b 1
)

:: ---------------------------------------------------------------------------
:: Show current state
:: ---------------------------------------------------------------------------

for /f "tokens=*" %%b in ('git branch --show-current 2^>^&1') do set CUR_BRANCH=%%b
for /f "tokens=*" %%h in ('git rev-parse --short HEAD 2^>^&1') do set CUR_HASH=%%h
echo [INFO] Current branch: %CUR_BRANCH% @ %CUR_HASH%

:: ---------------------------------------------------------------------------
:: Check for local uncommitted changes
:: ---------------------------------------------------------------------------

git diff --quiet 2>nul && git diff --cached --quiet 2>nul
if errorlevel 1 (
    echo.
    echo [WARN] You have uncommitted local changes:
    git status --short
    echo.
    choice /C YN /M "Stash local changes before updating?"
    if !errorlevel! equ 1 (
        git stash push -m "auto-stash before TIGA update %date% %time%"
        echo [OK] Changes stashed. Run 'git stash pop' to restore them.
        set STASHED=1
    ) else (
        echo [INFO] Proceeding without stashing. Merge conflicts may occur.
        set STASHED=0
    )
) else (
    set STASHED=0
)

:: ---------------------------------------------------------------------------
:: Fetch latest changes
:: ---------------------------------------------------------------------------

echo.
echo Fetching latest changes from origin...
git fetch origin
if errorlevel 1 (
    echo [ERROR] Could not reach remote. Check network/VPN.
    if !STASHED! equ 1 (
        echo [INFO] Restoring stashed changes...
        git stash pop
    )
    pause & exit /b 1
)

:: ---------------------------------------------------------------------------
:: Check if already up to date
:: ---------------------------------------------------------------------------

for /f "tokens=*" %%h in ('git rev-parse HEAD 2^>^&1') do set LOCAL_H=%%h
for /f "tokens=*" %%h in ('git rev-parse origin/%CUR_BRANCH% 2^>^&1') do set REMOTE_H=%%h

if "%LOCAL_H%"=="%REMOTE_H%" (
    echo [OK] Already up to date — no changes to pull.
    goto :post_update
)

:: Show what changed
echo.
echo Changes coming in:
git log --oneline HEAD..origin/%CUR_BRANCH%
echo.

:: ---------------------------------------------------------------------------
:: Pull
:: ---------------------------------------------------------------------------

git pull origin %CUR_BRANCH%
if errorlevel 1 (
    echo [ERROR] git pull failed — possible merge conflict.
    echo         Resolve conflicts manually, then run: git pull origin %CUR_BRANCH%
    if !STASHED! equ 1 (
        echo [INFO] Your stashed changes are still in the stash.
        echo        Run: git stash pop  (after resolving conflicts)
    )
    pause & exit /b 1
)

for /f "tokens=*" %%h in ('git rev-parse --short HEAD 2^>^&1') do set NEW_HASH=%%h
echo.
echo [OK] Updated to %NEW_HASH%

:: ---------------------------------------------------------------------------
:: Restore stash
:: ---------------------------------------------------------------------------

if !STASHED! equ 1 (
    echo.
    echo Restoring your stashed local changes...
    git stash pop
    if errorlevel 1 (
        echo [WARN] Stash pop had conflicts. Resolve manually: git stash pop
    ) else (
        echo [OK] Local changes restored.
    )
)

:post_update

:: ---------------------------------------------------------------------------
:: Reinstall dependencies (in case requirements.txt changed)
:: ---------------------------------------------------------------------------

echo.
echo Updating dependencies...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt --quiet
    if not errorlevel 1 ( echo [OK] Dependencies up to date ) else ( echo [WARN] pip install had issues )
) else (
    echo [WARN] No .venv found — run setup.bat first.
)

:: ---------------------------------------------------------------------------
:: Done
:: ---------------------------------------------------------------------------

echo.
echo ============================================================
echo   Update Complete!
echo ============================================================
echo.
echo  Restart TIGA Hunt to apply changes:
echo    run.bat
echo.
pause
