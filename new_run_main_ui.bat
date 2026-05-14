@echo off
setlocal

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [run] Failed to activate venv at venv\Scripts\activate.bat
    pause
    exit /b 1
)

python core\check_requirements.py
if errorlevel 1 (
    echo [run] Dependency check/install failed. See log above.
    pause
    exit /b 1
)

python core\fetch_data.py
if errorlevel 1 (
    echo [run] Data fetch failed. See log above.
    pause
    exit /b 1
)

python new_main_ui.py
endlocal
