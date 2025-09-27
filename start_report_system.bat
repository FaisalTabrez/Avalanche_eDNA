@echo off
echo Starting eDNA Report Management System...
echo.

cd /d "c:\Volume D\Avalanche"

echo [1/2] Starting FastAPI Server on port 8000...
start "FastAPI Server" cmd /k "set PYTHONPATH=c:/Volume D/Avalanche && python -m uvicorn src.api.report_management_api:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Streamlit Dashboard on port 8501...
start "Streamlit Dashboard" cmd /k "set PYTHONPATH=c:/Volume D/Avalanche && python -m streamlit run src/dashboards/report_management_dashboard.py --server.port 8501"

echo.
echo ===== eDNA Report Management System Started =====
echo FastAPI Server: http://127.0.0.1:8000
echo Streamlit Dashboard: http://localhost:8501
echo API Documentation: http://127.0.0.1:8000/docs
echo.
echo Both services are now running in separate windows.
echo Close those windows to stop the services.
pause