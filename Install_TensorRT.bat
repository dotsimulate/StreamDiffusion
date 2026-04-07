@echo off
echo ========================================
echo  StreamDiffusionTD TensorRT Installation
echo ========================================
echo.

cd /d "%~dp0"

:: Check venv exists before trying to activate
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at venv\Scripts\activate.bat
    echo Run Install_StreamDiffusion.bat first to create the environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "venv\Scripts\activate.bat"

if "%VIRTUAL_ENV%" == "" (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated: %VIRTUAL_ENV%

echo.
echo Installing TensorRT via CLI...
cd StreamDiffusion-installer
python -m sd_installer install-tensorrt

echo.
echo TensorRT installation finished
pause
