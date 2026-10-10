@echo off
rem ============================================================
rem build.bat - rebuild nnunet_gui.exe (needed ONLY after python
rem code changes: gui\*.py, scripts\*.py). Qt Designer .ui edits
rem do NOT require a rebuild.
rem ============================================================
set "PYDIR="
for %%D in ("%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%USERPROFILE%\anaconda3" "C:\ProgramData\miniconda3" "C:\ProgramData\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
    if not defined PYDIR if exist "%%~D\envs\nnunet_env\python.exe" set "PYDIR=%%~D\envs\nnunet_env"
)
if not defined PYDIR (
    echo nnunet_env python not found.
    echo Set up the environment with one command:
    echo   python scripts\setup_env.py
    pause
    exit /b 1
)
set "PYTHON=%PYDIR%\python.exe"
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
"%PYTHON%" -m PyInstaller nnunet_gui_pyside.spec --noconfirm
if errorlevel 1 (
    echo.
    echo BUILD FAILED
    pause
    exit /b 1
)
copy /y dist\nnunet_gui.exe nnunet_gui.exe >nul
echo.
echo DONE: nnunet_gui.exe updated in the project root.
pause
