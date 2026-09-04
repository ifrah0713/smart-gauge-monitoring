@echo off
title Smart Gauge Monitor - Remote Access
echo ============================================
echo  Smart Industrial Gauge Monitoring System
echo  Remote Dashboard Access via ngrok
echo ============================================
echo.

REM Check if ngrok is installed
where ngrok >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ngrok not found!
    echo.
    echo Please install ngrok:
    echo 1. Go to https://ngrok.com/download
    echo 2. Download and extract ngrok.exe
    echo 3. Place ngrok.exe in this folder
    echo 4. Run: ngrok authtoken YOUR_TOKEN
    echo.
    pause
    exit /b 1
)

echo [INFO] Starting ngrok tunnel on port 5000...
echo [INFO] Make sure pipeline is running first!
echo.
echo [INFO] Your remote URL will appear below:
echo ============================================
ngrok http 5000
pause
