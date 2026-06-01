@echo off
cd C:\Users\Administrator\Documents\trae_projects\RAG-QA-System---2405030149-main
call venv\Scripts\activate.bat
python -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false