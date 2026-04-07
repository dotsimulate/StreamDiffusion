@echo off
cd /d %~dp0

:: Load runtime environment variables if set_env.bat exists
if exist "%~dp0set_env.bat" call "%~dp0set_env.bat"

if exist venv (
    call venv\Scripts\activate.bat
    venv\Scripts\python.exe StreamDiffusionTD\td_main.py
) else (
    call .venv\Scripts\activate.bat
    .venv\Scripts\python.exe StreamDiffusionTD\td_main.py
)
pause
