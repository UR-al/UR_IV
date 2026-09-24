@echo off
setlocal EnableExtensions

cd /d "%~dp0"
if errorlevel 1 (
    echo [run] Failed to enter app directory: %~dp0
    pause
    exit /b 1
)

powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$n='Local\AIStudioPro.UR_IV.Update'; try{$m=[Threading.Mutex]::OpenExisting($n)}catch [Threading.WaitHandleCannotBeOpenedException]{exit 0}; $owned=$false; try{try{$owned=$m.WaitOne(120000)}catch [Threading.AbandonedMutexException]{$owned=$true}; if(-not $owned){exit 2}}finally{if($owned){$m.ReleaseMutex()};$m.Dispose()}"
if errorlevel 1 (
    echo [run] An application update is still running. Try again in a moment.
    pause
    exit /b 1
)

set "VENV_DIR=%~dp0venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_DIR%\" (
    call :venv_is_usable
    if errorlevel 1 (
        echo [run] Existing virtual environment belongs to another Python installation.
        call :preserve_incompatible_venv
        if errorlevel 1 (
            echo [run] Failed to preserve the incompatible virtual environment.
            pause
            exit /b 1
        )
    )
)

if not exist "%VENV_PYTHON%" (
    call :create_venv
    if errorlevel 1 (
        echo [run] Failed to create the virtual environment: %VENV_DIR%
        echo [run] Install Python 3.10 or 3.11, then run this file again.
        pause
        exit /b 1
    )
)

"%VENV_PYTHON%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [run] pip is missing. Installing pip into the virtual environment...
    "%VENV_PYTHON%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo [run] Failed to install pip into: %VENV_DIR%
        pause
        exit /b 1
    )
    "%VENV_PYTHON%" -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo [run] pip is still unavailable after ensurepip completed.
        pause
        exit /b 1
    )
)

if not exist logs mkdir logs
if not exist logs\last_crash.log if exist config\last_crash.log move config\last_crash.log logs\last_crash.log >nul
set "CRASH_LOG=logs\last_crash.log"
set "PREVIOUS_CRASH_LOG="
call :preserve_previous_crash_log

echo [run] Starting AI Studio Pro...
"%VENV_PYTHON%" "new_main_ui.py"
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo.
    echo ============================================================
    echo [run] The app exited unexpectedly. Check the crash log below.
    echo ============================================================
    if exist "%CRASH_LOG%" (
        call :type_utf8 "%CRASH_LOG%"
    ) else (
        echo [run] No crash log was created for this run. Use the traceback above.
    )
    if defined PREVIOUS_CRASH_LOG echo [run] Previous crash log preserved at: %PREVIOUS_CRASH_LOG%
    echo ============================================================
    pause
)

endlocal & exit /b %APP_EXIT%

:type_utf8
REM The crash log is UTF-8 (Korean messages/paths), but a new console starts in the OEM code page
REM (cp949 on Korean Windows). Show it under UTF-8 and restore the previous code page afterwards.
set "TYPE_PREV_CP="
for /f "tokens=2 delims=:." %%c in ('"%SystemRoot%\System32\chcp.com"') do set /a "TYPE_PREV_CP=%%c" >nul 2>&1
"%SystemRoot%\System32\chcp.com" 65001 >nul
type "%~1"
if defined TYPE_PREV_CP "%SystemRoot%\System32\chcp.com" %TYPE_PREV_CP% >nul
exit /b 0

:create_venv
echo [run] Virtual environment not found. Creating: %VENV_DIR%

where py >nul 2>&1
if errorlevel 1 goto :create_venv_with_python

call :try_py_version -3.11
if not errorlevel 1 exit /b 0

call :try_py_version -3.10
if not errorlevel 1 exit /b 0

echo [run] Python 3.10/3.11 was not found; trying the default Python launcher...
py -3 -m venv "%VENV_DIR%"
if exist "%VENV_PYTHON%" exit /b 0

:create_venv_with_python
where python >nul 2>&1
if errorlevel 1 exit /b 1

echo [run] Using python from PATH to create the virtual environment...
python -m venv "%VENV_DIR%"
if not exist "%VENV_PYTHON%" exit /b 1
exit /b 0

:try_py_version
py %~1 -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 1

echo [run] Using Python %~1 to create the virtual environment...
py %~1 -m venv "%VENV_DIR%"
if errorlevel 1 exit /b 1
if not exist "%VENV_PYTHON%" exit /b 1
exit /b 0

:venv_is_usable
if not exist "%VENV_PYTHON%" exit /b 1
"%VENV_PYTHON%" -c "import sys" >nul 2>&1
exit /b %ERRORLEVEL%

:preserve_incompatible_venv
:choose_venv_backup
set "VENV_BACKUP=%VENV_DIR%.incompatible-%RANDOM%-%RANDOM%"
if exist "%VENV_BACKUP%" goto :choose_venv_backup
move "%VENV_DIR%" "%VENV_BACKUP%" >nul
if errorlevel 1 exit /b 1
echo [run] Preserved the old environment at: %VENV_BACKUP%
exit /b 0

:preserve_previous_crash_log
if not exist "%CRASH_LOG%" exit /b 0

:choose_crash_log_backup
set "PREVIOUS_CRASH_LOG=logs\last_crash.previous-%RANDOM%-%RANDOM%.log"
if exist "%PREVIOUS_CRASH_LOG%" goto :choose_crash_log_backup
move "%CRASH_LOG%" "%PREVIOUS_CRASH_LOG%" >nul
if errorlevel 1 (
    echo [run] WARNING: Could not preserve the previous crash log.
    set "PREVIOUS_CRASH_LOG="
    exit /b 1
)
exit /b 0
