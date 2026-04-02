@echo off
echo ========================================
echo  StreamDiffusionTD TensorRT Installation
echo ========================================
echo.
cd /d "D:/Users/alexk/FORKNI/STREAM_DIFFUSION/STREAM_DIFFUSION_LIVEPEER/StreamDiffusion"

echo Attempting to activate virtual environment...
call "venv\Scripts\activate.bat"

if "%VIRTUAL_ENV%" == "" (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
) else (
    echo Virtual environment activated.
)

echo.
echo Installing TensorRT via CLI...
cd /d "D:/Users/alexk/FORKNI/STREAM_DIFFUSION/STREAM_DIFFUSION_LIVEPEER/StreamDiffusion\StreamDiffusion-installer"
python -m sd_installer install-tensorrt

echo.
echo TensorRT installation finished
pause
