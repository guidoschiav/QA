@echo off
cd /d "%~dp0"

echo Iniciando Dataset Auditor...
echo.

call .venv\Scripts\activate.bat

echo La app se va a abrir en tu navegador en unos segundos.
echo Para cerrar la app, cerrá esta ventana.
echo.

streamlit run app.py --server.maxUploadSize 500

pause
