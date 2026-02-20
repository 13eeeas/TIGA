@echo off
set TIGA_WORK_DIR=%~dp0tiga_work_tianmu

echo Starting TIGA Hunt...

start "TIGA Server" cmd /k ".venv\Scripts\python tiga.py serve"

timeout /t 3 /nobreak >nul

start "TIGA Admin" cmd /k ".venv\Scripts\python tiga.py ui"

timeout /t 4 /nobreak >nul

start http://localhost:7860
