
            @echo off
            cd /d %~dp0
            
            if exist venv (
                call venv\Scripts\activate.bat
                venv\Scripts\python.exe streamdiffusionTD\td_main.py
            ) else (
                call .venv\Scripts\activate.bat
                .venv\Scripts\python.exe streamdiffusionTD\td_main.py
            )
            pause
            