@echo off
title Dual Language Translator
echo.
echo ================================================================
echo          DUAL LANGUAGE TRANSLATOR - Internship Project
echo ================================================================
echo.
echo Starting application...
echo Please wait while the ML models are loading.
echo (First run may take a few minutes to download models)
echo.
echo The application will open in your web browser automatically.
echo.
echo ================================================================
echo.

cd /d "%~dp0"
python main.py

echo.
echo Application stopped.
pause
