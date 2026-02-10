@echo off

REM Stock Predictor - Quick Start Script for Windows

echo 🚀 Stock Predictor - Quick Start
echo ==================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version
echo.

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully
echo.

REM Run Streamlit
echo 🎉 Starting Stock Predictor...
echo.
echo 💡 Tip: The app will open in your browser at http://localhost:8501
echo 💡 To stop the app, press Ctrl + C
echo.

streamlit run app.py

pause
