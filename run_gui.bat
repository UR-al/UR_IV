@echo off
setlocal
REM This file and the crash log it shows (logs\last_crash*.log) are UTF-8, but a new console starts
REM in the OEM code page (cp949 on Korean Windows) and garbles the Korean messages below.
REM Switch to UTF-8 before any of them. Everything the user may interrupt with Ctrl+C (the update
REM wait, the app itself, the final pause) runs under the caller's code page instead: answering Y to
REM "Terminate batch job (Y/N)?" skips :finish and would leave the caller's console in 65001. The app
REM also gets the same console code page as with new_run_main_ui.bat. Only ASCII lines may sit
REM between a restore and the next switch back to 65001.
set "URIV_PREV_CP="
for /f "tokens=2 delims=:." %%c in ('"%SystemRoot%\System32\chcp.com"') do set /a "URIV_PREV_CP=%%c" >nul 2>&1
"%SystemRoot%\System32\chcp.com" 65001 >nul
set "URIV_EXIT=0"
set "URIV_PAUSE="
REM ── 로컬 창 모드 (PyQt 창 안의 Vue) ──
REM 웹 모드는 run_WEB_gui.bat 참고.

cd /d "%~dp0"
if errorlevel 1 (
    set "URIV_EXIT=1"
    goto :finish
)
if defined URIV_PREV_CP "%SystemRoot%\System32\chcp.com" %URIV_PREV_CP% >nul
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$n='Local\AIStudioPro.UR_IV.Update'; try{$m=[Threading.Mutex]::OpenExisting($n)}catch [Threading.WaitHandleCannotBeOpenedException]{exit 0}; $owned=$false; try{try{$owned=$m.WaitOne(120000)}catch [Threading.AbandonedMutexException]{$owned=$true}; if(-not $owned){exit 2}}finally{if($owned){$m.ReleaseMutex()};$m.Dispose()}"
set "URIV_WAIT=%ERRORLEVEL%"
"%SystemRoot%\System32\chcp.com" 65001 >nul
if %URIV_WAIT% GEQ 1 (
    echo [run] 앱 업데이트가 진행 중입니다. 잠시 후 다시 실행하세요.
    set "URIV_PAUSE=1"
    set "URIV_EXIT=1"
    goto :finish
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [run] venv 활성화 실패: venv\Scripts\activate.bat
    set "URIV_PAUSE=1"
    set "URIV_EXIT=1"
    goto :finish
)

REM 앱은 호출한 콘솔의 코드 페이지로 돈다(Ctrl+C → 일괄 작업 끝내기 Y 가 :finish 를 건너뛰어도 그대로다).
if defined URIV_PREV_CP "%SystemRoot%\System32\chcp.com" %URIV_PREV_CP% >nul
python new_main_ui.py
set "URIV_EXIT=%ERRORLEVEL%"
"%SystemRoot%\System32\chcp.com" 65001 >nul
REM 0 이 아니면 전부 비정상 종료다 — 네이티브 크래시(0xC0000005 등)는 음수라 `if errorlevel 1` 로는 못 잡는다.
if not "%URIV_EXIT%"=="0" (
    echo.
    echo ============================================================
    echo [run] 앱이 비정상 종료했습니다. 아래 크래시 로그를 확인하세요:
    echo ============================================================
    if exist logs\last_crash.log type logs\last_crash.log
    echo ============================================================
    set "URIV_PAUSE=1"
)

:finish
REM pause 도 Ctrl+C 로 끊길 수 있다 — 코드 페이지를 먼저 되돌린다.
if defined URIV_PREV_CP "%SystemRoot%\System32\chcp.com" %URIV_PREV_CP% >nul
if defined URIV_PAUSE pause
endlocal & exit /b %URIV_EXIT%
