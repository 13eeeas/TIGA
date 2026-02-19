@echo off
cd /d "C:\Users\shawnlam\iCloudDrive\TIGA Hunt"
call .venv\Scripts\activate.bat
echo [%date% %time%] TIGA overnight index started >> tiga_work\index.log
python tiga.py rebuild >> tiga_work\index.log 2>&1
echo [%date% %time%] TIGA overnight index finished >> tiga_work\index.log
