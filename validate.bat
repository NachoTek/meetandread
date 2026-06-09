@echo off
REM Validate an existing build without rebuilding
REM Use this after making changes to the build

if not exist "dist\meetandread\meetandread.exe" (
    echo ERROR: Build not found at dist\meetandread\meetandread.exe
    echo Run build-and-validate.bat first
    exit /b 1
)

echo Validating existing build...
python validate_build.py