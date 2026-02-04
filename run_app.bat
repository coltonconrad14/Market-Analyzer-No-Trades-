@echo off
REM Simple launcher script for Market Analyzer GUI (Windows)

echo 🚀 Launching Market Analyzer GUI...
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Launch the app
python app.py

pause
