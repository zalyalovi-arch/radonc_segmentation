# -*- mode: python ; coding: utf-8 -*-
# ============================================================
# nnunet_gui_pyside.spec — сборка PySide6-версии GUI (exe в корне проекта)
# ------------------------------------------------------------
# Сборка (из корня C:\nnunet_reboot_pyside):
#   build.bat     (= python -m PyInstaller nnunet_gui_pyside.spec --noconfirm,
#                  затем копирование dist\nnunet_gui.exe в корень)
#
# Важно:
# - .ui файлы В СБОРКУ НЕ ВХОДЯТ: exe читает их из gui\ui\ рядом с собой,
#   поэтому правки раскладки из Qt Designer подхватываются без пересборки.
#   (в onefile их кладём в _MEIPASS: datas ниже)
# - pyside6-uic.exe (нативный компилятор .ui из пакета PySide6) кладём в
#   сборку под именем pyside6-uic.exe: loadUiType запускает его из PATH
#   (PATH дополняет gui/__init__.py через sys._MEIPASS).
# - Тяжёлые ML-пакеты исключены: пайплайн запускается python'ом окружения
#   через subprocess (PYTHON_EXE из [gui] CONDA_PYTHON / автопоиск).
#   Пакет radonc_segmentation (config, pipeline, env_check, ...) кладётся
#   в _MEIPASS как ДАННЫЕ (source-файлы), а gui/__init__.child_env()
#   передаёт PYTHONPATH=_MEIPASS дочерним python -m процессам. В exe
#   бандлим только то, что GUI выполняет сам: PySide6 + numpy/scipy
#   (постпроцессинг) + SimpleITK + radonc_segmentation.config.
#   matplotlib ИСКЛЮЧЁН (вкладка «Постпроцессинг» отключена в exe).
# ============================================================
import os
import shutil
import tempfile
import PySide6

ROOT = os.path.dirname(os.path.abspath(SPEC))
PYSIDE_DIR = os.path.dirname(os.path.abspath(PySide6.__file__))

# нативный компилятор .ui -> python (PySide6/uic.exe), переименовываем:
# QtUiTools.loadUiType ищет в PATH именно "pyside6-uic"
_uic_build_dir = os.path.join(tempfile.gettempdir(), "nnunet_gui_build")
os.makedirs(_uic_build_dir, exist_ok=True)
_uic_exe = os.path.join(_uic_build_dir, "pyside6-uic.exe")
shutil.copy2(os.path.join(PYSIDE_DIR, "uic.exe"), _uic_exe)

binaries = [(_uic_exe, ".")]

# Данные для onefile: .ui файлы и пакет radonc_segmentation (исходники,
# включая __init__.py и подпакеты; template-JSONы постпроцессинга внутри)
datas = [
    (os.path.join(ROOT, "gui", "ui"), "gui/ui"),
    (os.path.join(ROOT, "radonc_segmentation"), "radonc_segmentation"),
]

a = Analysis(
    ["gui/main.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "radonc_segmentation",
        "radonc_segmentation.config",
        "radonc_segmentation.postprocess_core",
        "radonc_segmentation.import_dicom",
        "gui.postprocess_tab",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tensorflow", "keras", "torch", "torchvision", "torchaudio",
        "nnunetv2", "nncf", "openvino", "onnx", "skimage", "rt_utils",
        "datasets", "transformers", "tokenizers", "optimum", "graphviz",
        "sklearn", "pandas", "nibabel", "tkinter", "tests",
        "matplotlib", "PIL", "Pillow", "PIL.Image",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="nnunet_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
