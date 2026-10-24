@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo Firebreak R1 - Install Python dependencies
echo ============================================
echo.
python --version
if errorlevel 1 (
  echo ERROR: Python was not found. Install Python 3.11 or 3.12 and enable Add Python to PATH.
  pause
  exit /b 1
)
echo.
python -m pip install --upgrade pip
if errorlevel 1 goto failed
echo.
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Checking key libraries...
python -c "import streamlit,numpy,pandas,scipy,requests,matplotlib,rasterio; print('All core imports OK'); print('streamlit',streamlit.__version__); print('rasterio',rasterio.__version__)"
if errorlevel 1 goto failed
echo.
echo SUCCESS: required libraries are installed for this Python interpreter.
pause
exit /b 0
:failed
echo.
echo INSTALL FAILED. Copy the full error text and send it for diagnosis.
pause
exit /b 1
