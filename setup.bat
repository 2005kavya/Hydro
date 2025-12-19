@echo off
echo 🚰 HydroAlert Auto-Setup for Windows
echo ======================================

echo.
echo 📦 Installing packages and setting up environment...
python setup.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Setup failed! Please check the errors above.
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo 🚀 Starting HydroAlert servers...
echo.

echo 📡 Starting Backend Server (FastAPI)...
start "HydroAlert Backend" cmd /k "cd backend && python -m uvicorn main:app --reload --port 8000"

echo ⏳ Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul

echo 🌐 Starting Frontend Server (Streamlit)...
start "HydroAlert Frontend" cmd /k "cd frontend && streamlit run streamlit_app.py"

echo.
echo ✅ Both servers are starting up!
echo.
echo 📱 Your HydroAlert app will be available at:
echo    Frontend: http://localhost:8501
echo    Backend:  http://localhost:8000
echo.
echo 🎯 Open your browser and navigate to: http://localhost:8501
echo.
echo 💡 Keep both terminal windows open while using the app.
echo.
pause
