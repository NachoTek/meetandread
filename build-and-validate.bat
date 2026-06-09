@echo off
REM Quick build and validation script
REM Run this before pushing to GitHub

echo ========================================
echo Building MeetAndRead...
echo ========================================

pyinstaller meetandread.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed
    exit /b 1
)

echo.
echo ========================================
echo Validating build...
echo ========================================

python validate_build.py
if errorlevel 1 (
    echo.
    echo ERROR: Validation failed
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Build is ready for release
echo ========================================
echo.
echo Next steps:
echo   1. Test manually: dist\meetandread\meetandread.exe
echo   2. If good: git tag v0.19.2
echo   3. Push: git push origin v0.19.2
echo.
exit /b 0