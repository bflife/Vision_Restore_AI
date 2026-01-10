@echo off
echo ===============================================================================
echo   Vision-Restore AI - AMD/Intel GPU Support Installer
echo ===============================================================================
echo.
echo This script will install the necessary components for AMD and Intel GPU support
echo using DirectML. This allows the AI to run on your GPU instead of CPU.
echo.
echo Requirements:
echo - Windows 10 Version 1709 or later, or Windows 11
echo - DirectX 12 capable GPU
echo.
pause

echo.
echo [1/2] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/2] Installing torch-directml...
python -m pip install torch-directml

echo.
echo ===============================================================================
echo   Installation Complete!
echo ===============================================================================
echo.
echo Please restart Vision-Restore AI.
echo You should see "DirectML detected" or "privateuseone" in the logs.
echo.
pause
