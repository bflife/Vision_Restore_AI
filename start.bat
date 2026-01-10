@echo off
TITLE Vision-Restore AI v2.1.0
echo Starting Vision-Restore AI...
echo --------------------------------
echo Mode: Normal (GPU/NPU Detected)
echo Version: 2.1.0 (Pulse Mode Active)
echo Python: 3.10
echo --------------------------------

REM Try to run with Python 3.10 launcher
py -3.10 run_gui.py %*
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Launcher failed. Trying default 'python'...
    python run_gui.py %*
)

pause
