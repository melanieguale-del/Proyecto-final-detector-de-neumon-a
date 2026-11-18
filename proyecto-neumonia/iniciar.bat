@echo off
rem abre la API y la interfaz en dos ventanas
start "API" cmd /k "venv\Scripts\activate && uvicorn backend.main:app --port 8000"
timeout /t 4 >nul
start "Interfaz" cmd /k "venv\Scripts\activate && streamlit run frontend/app.py"
