@echo off
echo ========================================
echo  StreamDiffusionTD v0.3.1 Installation
echo  Daydream Fork with StreamV2V
echo ========================================
echo.

:: Prerequisite checks
echo Checking prerequisites...

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 not found via py launcher.
    echo Install Python 3.11 from https://python.org and ensure the py launcher is available.
    pause
    exit /b 1
)

git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git not found in PATH.
    echo Install Git from https://git-scm.com/ (required for pip git+ packages).
    pause
    exit /b 1
)

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo WARNING: C++ compiler (cl.exe) not found. Some packages may require it to build.
    echo If installation fails, install Visual Studio Build Tools from:
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
)

echo Prerequisites OK. Starting installation...
echo.

cd /d "%~dp0"
cd StreamDiffusion-installer

py -3.11 -m sd_installer --base-folder "%~dp0." install --cuda cu128 --no-cache

pause
