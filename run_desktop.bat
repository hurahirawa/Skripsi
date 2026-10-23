@echo off
rem Jalankan dashboard sebagai jendela desktop (pywebview); server berhenti saat jendela ditutup.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo .venv tidak ditemukan. Buat dulu: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python scripts\desktop_app.py %*
