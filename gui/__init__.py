# gui — PySide6-версия GUI пайплайна H&N.
# Раскладка окон описана в gui/ui/*.ui (правится в Qt Designer),
# логика — в gui/main_window.py и gui/postprocess_tab.py.
import os
import sys
import sysconfig

_FROZEN = getattr(sys, "frozen", False)

if _FROZEN:
    # onefile: данные распакованы в _MEIPASS
    _MEIPASS = getattr(sys, "_MEIPASS", "")
    if _MEIPASS:
        PROJECT_ROOT = os.path.join(_MEIPASS, "gui", "ui")
        SCRIPTS_DIR = os.path.join(_MEIPASS, "scripts")
    else:
        # fallback: exe в корне проекта
        _root = os.path.dirname(os.path.abspath(sys.executable))
        for _ in range(3):
            if os.path.isdir(os.path.join(_root, "scripts")):
                break
            _parent = os.path.dirname(_root)
            if _parent == _root:
                break
            _root = _parent
        PROJECT_ROOT = os.path.join(_root, "gui", "ui")
        SCRIPTS_DIR = os.path.join(_root, "scripts")
else:
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui", "ui")
    SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    # в dev-режиме pyside6-uic.exe живёт в Scripts окружения и не всегда в PATH
    _scripts = sysconfig.get_path("scripts")
    if _scripts and os.path.isdir(_scripts) and _scripts not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _scripts + os.pathsep + os.environ.get("PATH", "")

# .ui и scripts читаются из папки проекта В ОБОИХ режимах
UI_DIR = PROJECT_ROOT
