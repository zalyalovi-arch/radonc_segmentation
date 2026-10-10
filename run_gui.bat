@echo off
rem ============================================================
rem run_gui.bat - run PySide6 GUI with the nnunet_env python
rem (environment is created by setup_env.py in the project root;
rem python path is auto-detected in typical conda locations)
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
set "PYTHON=%PYDIR%\python.exe"
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
rem -m: запуск в пакетном контексте (импорты radonc_segmentation/gui
rem резолвятся без sys.path-хаков; cwd добавляется в sys.path)
"%PYTHON%" -m gui.main
if errorlevel 1 pause
