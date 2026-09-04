@echo off
title Smart Gauge Monitor - Full System Launcher
color 0A

echo ============================================================
echo   SMART INDUSTRIAL GAUGE MONITORING SYSTEM
echo   Alibaba Cloud AI Hackathon Pakistan 2026
echo   Ifrah Gohar ^| IST Islamabad
echo ============================================================
echo.
echo   This will start:
echo     [1] Pipeline  (localhost:5000)
echo     [2] ngrok     (public remote URL)
echo.
echo   Make sure ngrok is installed and authenticated.
echo   https://ngrok.com/download
echo ============================================================
echo.

REM ── Start pipeline in a new window ──────────────────────────
echo [STEP 1] Starting pipeline...
start "Gauge Monitor Pipeline" cmd /k "python pipeline_main_software_v2.py"

REM ── Wait for Flask to be ready ───────────────────────────────
echo [STEP 2] Waiting 8 seconds for pipeline to start...
timeout /t 8 /nobreak >nul

REM ── Start ngrok ──────────────────────────────────────────────
echo [STEP 3] Starting ngrok tunnel...
echo.
echo ============================================================
echo   Copy the Forwarding URL below and share it!
echo   Example: https://abcd-123.ngrok-free.app
echo ============================================================
echo.

ngrok http 5000

pause
