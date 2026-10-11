# gui — PySide6-версия GUI пайплайна H&N.
# Раскладка окон описана в gui/ui/*.ui (правится в Qt Designer),
# логика — в gui/main_window.py и gui/postprocess_tab.py.
import os
import sys
import sysconfig

_FROZEN = getattr(sys, "frozen", False)

# корень проекта: папка с settings.ini, patients\, models\ и пакетом
# radonc_segmentation (config.py якорит пути так же — dirname² от __file__)
if _FROZEN:
    # «дом» exe — папка, где он лежит (рядом ждём settings.ini, models\, ...);
    # dirname(_MEIPASS) — это %TEMP%, детям туда нельзя (см. child_env)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI_DIR = os.path.join(PROJECT_ROOT, "gui", "ui")

if _FROZEN:
    # onefile: .ui и пакет radonc_segmentation распакованы в _MEIPASS;
    # settings.ini/models/patients ищем рядом с exe (см. config.py)
    _MEIPASS = getattr(sys, "_MEIPASS", "")
    if _MEIPASS:
        UI_DIR = os.path.join(_MEIPASS, "gui", "ui")
        PKG_DIR = os.path.join(_MEIPASS, "radonc_segmentation")
    else:
        # fallback: exe в корне проекта
        _root = os.path.dirname(os.path.abspath(sys.executable))
        for _ in range(3):
            if os.path.isdir(os.path.join(_root, "gui", "ui")):
                break
            _parent = os.path.dirname(_root)
            if _parent == _root:
                break
            _root = _parent
        UI_DIR = os.path.join(_root, "gui", "ui")
        PKG_DIR = os.path.join(_root, "radonc_segmentation")
else:
    PKG_DIR = os.path.join(PROJECT_ROOT, "radonc_segmentation")

# в dev-режиме pyside6-uic.exe живёт в Scripts окружения и не всегда в PATH
if not _FROZEN:
    _scripts = sysconfig.get_path("scripts")
    if _scripts and os.path.isdir(_scripts) and _scripts not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _scripts + os.pathsep + os.environ.get("PATH", "")


def child_env():
    """Окружение для дочерних процессов (скрипты пайплайна запускаются
    python'ом окружения nnunet_env через subprocess).

    В frozen-exe пакет radonc_segmentation лежит распакованным в _MEIPASS —
    передаём его в PYTHONPATH, чтобы дети могли импортировать
    radonc_segmentation.config и запускались через python -m.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if _FROZEN:
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            env["PYTHONPATH"] = meipass + os.pathsep + env.get("PYTHONPATH", "")
        # дети импортируют пакет из _MEIPASS, где settings.ini нет —
        # config.py читает RADONC_SETTINGS_DIR и якорится к папке exe
        env["RADONC_SETTINGS_DIR"] = PROJECT_ROOT
    return env
