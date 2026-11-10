@echo off
setlocal EnableExtensions
REM UDI 2.0 - one-click setup and start for Windows. The script stops at the first failing step and shows the error.
REM Usage:  run_windows.bat                       (install, build, start)
REM         run_windows.bat --retrain-reference   (also rebuild the optional domain reference models - slow)
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 ( echo [ERROR] Python 3.11+ was not found on PATH. Install it from python.org and tick "Add python to PATH". & goto :fail )
where npm >nul 2>nul
if errorlevel 1 ( echo [ERROR] Node.js 18+ / npm was not found on PATH. Install it from nodejs.org. & goto :fail )

echo === 1/5 Python virtual environment and packages ===
cd backend
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
  if errorlevel 1 ( echo [ERROR] Could not create the virtual environment. & goto :fail )
)
call .venv\Scripts\activate.bat
if errorlevel 1 ( echo [ERROR] Could not activate the virtual environment. & goto :fail )
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed. See the messages above. & goto :fail )

echo === 2/5 Environment file ===
if not exist .env (
  copy .env.example .env >nul
  echo Created backend\.env from .env.example.
)
findstr /R /C:"^GEMINI_API_KEY=..*" .env >nul
if errorlevel 1 echo [INFO] GEMINI_API_KEY is empty: UDI will plan questions with its local intent model. Add a key to backend\.env to use Gemini.

echo === 3/5 Local planner model ===
python train_planner.py
if errorlevel 1 ( echo [ERROR] Training the local planner failed. & goto :fail )

if /I "%~1"=="--retrain-reference" (
  echo === optional: domain reference models ===
  python sample_data\generate_samples.py
  if errorlevel 1 ( echo [ERROR] Sample generation failed. & goto :fail )
  python train_models.py
  if errorlevel 1 ( echo [ERROR] Reference model training failed. & goto :fail )
)

echo === 4/5 Frontend install and build ===
cd ..\frontend
call npm install
if errorlevel 1 ( echo [ERROR] npm install failed. & goto :fail )
call npm run build
if errorlevel 1 ( echo [ERROR] npm run build failed. & goto :fail )

echo === 5/5 Starting UDI ===
cd ..\backend
echo Open http://localhost:5000 in your browser. Press Ctrl+C to stop.
python serve.py
if errorlevel 1 ( echo [ERROR] The backend stopped with an error. & goto :fail )
goto :eof

:fail
echo.
echo Setup stopped. Fix the error above and run run_windows.bat again.
pause
exit /b 1
