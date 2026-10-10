@echo off
rem ============================================================
rem designer.bat - open Qt Designer (ships with PySide6) on the
rem gui\ui folder. Edit main_window.ui / postprocess_tab.ui there;
rem changes are picked up on next GUI start (no code generation).
rem ============================================================
set "PYDIR="
for %%D in ("%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%USERPROFILE%\anaconda3" "C:\ProgramData\miniconda3" "C:\ProgramData\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
    if not defined PYDIR if exist "%%~D\envs\nnunet_env\python.exe" set "PYDIR=%%~D\envs\nnunet_env"
)
if not defined PYDIR (
    echo nnunet_env python not found.
    echo Set up the environment with one command:
    echo   python setup_env.py
    pause
    exit /b 1
)
set "DESIGNER=%PYDIR%\Lib\site-packages\PySide6\designer.exe"
if not exist "%DESIGNER%" (
    echo Qt Designer not found: %DESIGNER%
    echo Reinstall PySide6: python -m pip install --no-deps PySide6 PySide6-Essentials shiboken6
    pause
    exit /b 1
)
start "" "%DESIGNER%" "%~dp0gui\ui"
