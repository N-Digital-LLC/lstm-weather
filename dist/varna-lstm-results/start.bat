@echo off
setlocal
cd /d "%~dp0"
set "PORT=8099"
set "URL=http://localhost:%PORT%/"

echo ============================================================
echo   Varna Weather LSTM - results explorer
echo ============================================================
echo.
echo Starting a small local web server and opening your browser.
echo If the page does not open automatically, go to:  %URL%
echo.
echo KEEP THIS WINDOW OPEN while you browse.
echo Close it (or press Ctrl+C) when you are done.
echo.

REM Open the browser a few seconds after the server starts (give it time to bind).
start "" cmd /c "timeout /t 3 /nobreak >nul & start "" "%URL%""

where py >nul 2>nul
if %errorlevel%==0 (
  py -m http.server %PORT%
  goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
  python -m http.server %PORT%
  goto :eof
)

where npx >nul 2>nul
if %errorlevel%==0 (
  npx --yes serve -l %PORT% .
  goto :eof
)

echo.
echo ------------------------------------------------------------
echo  Could not find Python or Node.js on this computer.
echo  Easiest fix: install Python from
echo      https://www.python.org/downloads/
echo  (tick "Add python.exe to PATH" during setup), then
echo  double-click start.bat again.
echo  See README.txt for other options.
echo ------------------------------------------------------------
pause
