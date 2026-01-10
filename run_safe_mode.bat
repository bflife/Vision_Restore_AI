@echo off
TITLE Vision-Restore AI - SAFE MODE (CPU ONLY)
echo ===================================================
echo   VISION-RESTORE AI: SAFE MODE
echo ===================================================
echo.
echo [INFO] Forcing CPU Mode (--gpu-id -1)...
echo [INFO] This bypasses GPU hardware issues.
echo [INFO] Processing will be slower but STABLE.
echo.

py -3.10 run_gui.py --gpu-id -1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to launch in safe mode.
    pause
)
pause
