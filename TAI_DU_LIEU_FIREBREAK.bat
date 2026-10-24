@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo Firebreak - tai du lieu DEM va WorldCover cho 5 vung nghien cuu
py -c "import requests" >nul 2>&1
if errorlevel 1 (
  echo Dang cai thu vien requests...
  py -m pip install requests
  if errorlevel 1 goto error
)
py download_firebreak_data.py
set RC=%ERRORLEVEL%
echo.
if not "%RC%"=="0" (
  echo Co tile tai that bai. Hay chup lai toan bo cua so nay va gui ket qua.
  pause
  exit /b %RC%
)
echo Da tai xong cac file co the truy cap. Vui long tiep tuc bang validate_firebreak_data.py.
pause
exit /b 0
:error
echo Khong chay duoc Python launcher `py` hoac cai requests that bai.
echo Hay mo PowerShell tai thu muc du an va chay: python -m pip install requests
pause
exit /b 1
