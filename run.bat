@echo off
REM Launch meetandread (meetandread) application
REM Uses py launcher to avoid Windows Store python stub.

set "SCRIPT_DIR=%~dp0"
set "PYTHONPATH=%SCRIPT_DIR%src"
cd /d "%SCRIPT_DIR%"

echo Starting meetandread...
py -m meetandread

if errorlevel 1 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)
pause