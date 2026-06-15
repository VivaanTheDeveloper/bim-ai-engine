@echo off
cd /d "%~dp0"
echo Initializing Enterprise BIM AI Blueprint Engine...
call venv\Scripts\activate
streamlit run src/app_ui.py --server.headless true
pause