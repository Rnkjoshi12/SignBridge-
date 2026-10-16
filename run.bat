@echo off
title SignBridge - Real-Time Sign Language Assistant
cd /d "%~dp0"
echo ========================================================
echo Starting SignBridge with Custom YOLOv5 Sign Language Model
echo ========================================================
if exist "C:\Users\Lenovo\AppData\Local\Programs\Python\Python313\python.exe" (
    "C:\Users\Lenovo\AppData\Local\Programs\Python\Python313\python.exe" main.py %*
) else (
    py -3.13 main.py %*
)
if errorlevel 1 (
    echo.
    echo Application exited with error code %errorlevel%.
    pause
)
