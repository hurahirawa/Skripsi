@echo off
rem Jalankan dashboard lokal: server Streamlit headless + buka browser otomatis setelah server siap.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo .venv tidak ditemukan. Buat dulu: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if not defined PORT set PORT=8501
start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 90;$i++){try{if((Invoke-WebRequest -UseBasicParsing http://localhost:%PORT%/_stcore/health).Content -eq 'ok'){Start-Process http://localhost:%PORT%;break}}catch{};Start-Sleep 1}"
streamlit run app\Home.py --server.headless true --server.port %PORT% --browser.gatherUsageStats false
