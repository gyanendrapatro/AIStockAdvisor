@echo off
REM Launches the AIStockAdvisor Streamlit UI and opens it in your browser.
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run "src\stock_advisor\dashboard\app.py"
) else (
    streamlit run "src\stock_advisor\dashboard\app.py"
)

pause
