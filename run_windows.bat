@echo off
setlocal EnableExtensions
REM UDI - one-click setup and start for Windows. Errors are shown, never hidden: the script stops at the first failing step.
REM Usage:  run_windows.bat                       (install, build, start)
REM         run_windows.bat --retrain-reference   (also rebuild the optional domain reference models - slow, not needed to run UDI)
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 ( echo [ERROR] Python 3.10+ was not found on PATH. Install it from python.org and tick "Add python to PATH". & goto :fail )
where node >nul 2>nul
if errorlevel 1 ( echo [ERROR] Node.js 18+ was not found on PATH. Install it from nodejs.org. & goto :fail )
where npm >nul 2>nul
if errorlevel 1 ( echo [ERROR] npm was not found on PATH. Reinstall Node.js. & goto :fail )

echo === 1/6 Python virtual environment ===
cd backend
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
  if errorlevel 1 ( echo [ERROR] Could not create the virtual environment. & goto :fail )
)
call .venv\Scripts\activate.bat
if errorlevel 1 ( echo [ERROR] Could not activate the virtual environment. & goto :fail )

echo === 2/6 Installing backend packages ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed. See the messages above. & goto :fail )

echo === 3/6 Environment file ===
if not exist .env (
  copy .env.example .env >nul
  echo Created backend\.env from .env.example.
)
findstr /C:"YOUR_GEMINI_API_KEY" .env >nul
if not errorlevel 1 (
  echo.
  echo [ACTION NEEDED] Open backend\.env and set GEMINI_API_KEY and GEMINI_MODEL to your own values, save, then run this script again.
  echo UDI starts without a key, but it cannot answer questions until Gemini is configured.
  echo.
  pause
)

echo === 4/6 Optional domain reference models ===
if /I "%~1"=="--retrain-reference" (
  python sample_data\generate_samples.py
  if errorlevel 1 ( echo [ERROR] Sample generation failed. & goto :fail )
  python train_models.py
  if errorlevel 1 ( echo [ERROR] Reference model training failed. & goto :fail )
) else (
  echo Skipped. The shipped reference models are used only for validation/demo. Pass --retrain-reference to rebuild them.
)

echo === 5/6 Frontend install and build ===
cd ..\frontend
call npm install
if errorlevel 1 ( echo [ERROR] npm install failed. & goto :fail )
call npm run build
if errorlevel 1 ( echo [ERROR] npm run build failed. & goto :fail )

echo === 6/6 Starting UDI ===
cd ..\backend
echo Open http://localhost:5000 in your browser. Press Ctrl+C to stop.
python app.py
if errorlevel 1 ( echo [ERROR] The backend stopped with an error. & goto :fail )
goto :eof

:fail
echo.
echo Setup stopped. Fix the error above and run run_windows.bat again.
pause
exit /b 1
