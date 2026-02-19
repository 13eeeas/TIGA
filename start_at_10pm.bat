@echo off
cd /d "C:\Users\shawnlam\iCloudDrive\TIGA Hunt"

powershell -NoProfile -ExecutionPolicy Bypass -File "wait_until_10pm.ps1"

echo.
echo [%date% %time%] Starting overnight index of all projects...
echo ================================================================
call .venv\Scripts\activate.bat
echo [%date% %time%] Started >> tiga_work\index.log
python tiga.py rebuild >> tiga_work\index.log 2>&1
echo [%date% %time%] Finished >> tiga_work\index.log
echo.
echo ================================================================
echo Done! Check tiga_work\index.log for results.
python tiga.py status
pause
