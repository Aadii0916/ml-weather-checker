@echo off
title AI Weather Checker Launcher
echo =========================================================
echo   Starting AI Weather Checker Web Application...
echo =========================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
