# ============================================================
# setup_env.py — настройка НОВОГО компьютера одной командой
# ------------------------------------------------------------
# Запускается ЛЮБЫМ python (только стандартная библиотека):
#   python "C:\путь\к\проекту\setup_env.py"
#
# Что делает:
#   1. Ищет conda (Miniconda/Anaconda) в типовых местах; если нет —
#      предлагает скачать и установить Miniconda автоматически.
#   2. Создаёт окружение nnunet_env (Python 3.10), если его нет.
#   3. Ставит зависимости из requirements.txt и пакет (pip install -e .).
#   4. Проверяет окружение (env_check) и подстраивает настройки
#      под железо этого компьютера (config.autotune_hardware).
#
# Повторный запуск безопасен: шаги пропускаются, если уже сделаны.
# ============================================================
import hashlib
import os
import subprocess
import sys
import urllib.request

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REQ_PATH = os.path.join(PROJECT_DIR, "requirements.txt")

ENV_NAME = "nnunet_env"
PYTHON_VERSION = "3.10"

# Зафиксированная сборка вместо latest + контроль SHA256 (supply-chain):
# клиника ставит инструмент на офисный ПК, установщик проверяется ДО запуска.
# Хеш снят с официального repo.anaconda.com на момент пина (2026-10-11);
# при обновлении пина — скачать сборку и пересчитать.
MINICONDA_URL = ("https://repo.anaconda.com/miniconda/"
                 "Miniconda3-py310_26.7.1-1-Windows-x86_64.exe")
MINICONDA_SHA256 = "1f0df3ddbb963e5bd2746be55bae774940da009f7948ca873cbbbc58683e197b"

CONDA_BASES = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "miniconda3"),
    os.path.join(os.path.expanduser("~"), "miniconda3"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "anaconda3"),
    os.path.join(os.path.expanduser("~"), "anaconda3"),
    r"C:\ProgramData\miniconda3",
    r"C:\ProgramData\anaconda3",
    r"C:\miniconda3",
    r"C:\anaconda3",
]


def find_conda_exe():
    """conda.exe из типовых установок или PATH."""
    candidates = [os.path.join(b, "Scripts", "conda.exe") for b in CONDA_BASES]
    candidates.append(os.path.join(os.path.dirname(sys.executable), "Scripts", "conda.exe"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    try:
        r = subprocess.run(["where", "conda"], capture_output=True, text=True, timeout=15)
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line.lower().endswith("conda.exe") and os.path.isfile(line):
                return line
    except Exception:
        pass
    return None


def env_python(conda_exe):
    """Путь к python окружения ENV_NAME (проверяем, что файл существует)."""
    base = os.path.dirname(os.path.dirname(conda_exe))  # ...\<conda root>
    p = os.path.join(base, "envs", ENV_NAME, "python.exe")
    return p if os.path.isfile(p) else None


def run(cmd, **kw):
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd, **kw)


def install_miniconda():
    print("\n[1/5] conda не найден. Установить Miniconda автоматически?")
    print(f"      (источник: {MINICONDA_URL})")
    answer = input("Введите Д для установки или Enter для выхода: ").strip().lower()
    if answer not in ("д", "d", "да", "yes", "y"):
        print("Установите Miniconda вручную с https://docs.conda.io/en/latest/miniconda.html "
              "и запустите setup_env.py снова.")
        sys.exit(1)
    installer = os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "Miniconda3_setup.exe")
    print(f"Скачиваю: {installer} ...")
    urllib.request.urlretrieve(MINICONDA_URL, installer)
    print("Проверяю SHA256 установщика ...")
    h = hashlib.sha256()
    with open(installer, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest().lower() != MINICONDA_SHA256:
        print("[ОШИБКА] SHA256 скачанного установщика НЕ совпал с ожидаемым.")
        print(f"  ожидалось: {MINICONDA_SHA256}")
        print(f"  получено:  {h.hexdigest()}")
        print("Файл повреждён или подменён — установка прервана. Скачайте установщик")
        print(f"вручную: {MINICONDA_URL}")
        print("и сверьте хеш со страницей Anaconda, затем повторите setup_env.py.")
        sys.exit(1)
    print("  SHA256 OK")
    target = os.path.join(os.path.expanduser("~"), "miniconda3")
    print(f"Тихая установка в {target} (несколько минут)...")
    r = run([installer, "/InstallationType=JustMe", "/AddToPath=0", "/RegisterPython=0",
             "/S", f"/D={target}"])
    if r.returncode != 0 or not os.path.isfile(os.path.join(target, "Scripts", "conda.exe")):
        print("[ОШИБКА] Установка Miniconda не удалась, установите вручную.")
        sys.exit(1)
    return os.path.join(target, "Scripts", "conda.exe")


def main():
    print(f"Проект: {PROJECT_DIR}")
    print(f"requirements.txt: {REQ_PATH}")

    # --- 1. conda ---
    print("\n[1/5] Ищу conda...")
    conda = find_conda_exe()
    if conda:
        print(f"  найден: {conda}")
    else:
        conda = install_miniconda()
        print(f"  установлен: {conda}")

    # --- 2. окружение ---
    print(f"\n[2/5] Проверяю окружение {ENV_NAME} (Python {PYTHON_VERSION})...")
    py = env_python(conda)
    if py:
        print(f"  уже существует: {py}")
    else:
        print(f"  создаю окружение {ENV_NAME} с Python {PYTHON_VERSION}...")
        r = run([conda, "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"])
        if r.returncode != 0:
            print("[ОШИБКА] Не удалось создать окружение (см. вывод выше).")
            sys.exit(1)
        py = env_python(conda)
        if not py:
            print("[ОШИБКА] Окружение создано, но python.exe не найден.")
            sys.exit(1)
        print(f"  создано: {py}")

    # --- 3. зависимости ---
    print("\n[3/5] Проверяю зависимости (requirements.txt + пакет)...")
    chk = run([py, "-m", "radonc_segmentation.env_check"])
    if chk.returncode == 0:
        print("  все пакеты уже установлены — пропускаю установку.")
    else:
        print("  ставлю пакеты (обычный pip, пины requirements.txt совместимы).\n"
              "  Это долго (10-20 минут, ~2 ГБ загрузок)...")
        r = run([py, "-m", "pip", "install", "-r", REQ_PATH])
        if r.returncode != 0:
            print("[ОШИБКА] pip завершился с ошибкой (см. вывод выше).")
            sys.exit(1)
        # сам пакет (radonc_segmentation + gui) — в редактируемом режиме:
        # импорты работают из любой папки, правки кода подхватываются сразу
        r = run([py, "-m", "pip", "install", "-e", PROJECT_DIR, "--no-deps"])
        if r.returncode != 0:
            print("[ОШИБКА] Не удалось установить пакет (pip install -e .).")
            sys.exit(1)
        chk = run([py, "-m", "radonc_segmentation.env_check"])
        if chk.returncode != 0:
            print("[ВНИМАНИЕ] env_check всё ещё находит проблемы — проверьте вывод выше.")

    # --- 4. опциональные пакеты квантизации (импорт .pth -> IR, INT8) ---
    # ~2 ГБ (torch CPU); без них инференс работает, но кнопки «Импорт модели»
    # и «Квантизация» в GUI — нет. Ошибка установки не фатальна.
    quant_req = os.path.join(PROJECT_DIR, "requirements-quantization.txt")
    print("\n[4/5] Пакеты квантизации (requirements-quantization.txt, опционально)...")
    if os.path.isfile(quant_req):
        r = run([py, "-m", "pip", "install", "-r", quant_req])
        if r.returncode != 0:
            print("[ВНИМАНИЕ] Не удалось установить пакеты квантизации — "
                  "пропускаю (можно доустановить позже этой же командой).")
    else:
        print("  requirements-quantization.txt не найден — пропускаю.")

    # --- 5. настройки под железо ---
    print("\n[5/5] Подстраиваю настройки под железо этого компьютера...")
    code = ("import config; config.autotune_hardware(force=True); "
            "print('DEVICE=%s NUM_THREADS=%s ACCUMULATOR=%s' % "
            "(config.DEVICE, config.NUM_THREADS, config.ACCUMULATOR))")
    run([py, "-c", code], cwd=PROJECT_DIR)

    print("\n===== ГОТОВО =====")
    print("Программа настроена. Дальше можно:")
    print('  * запустить GUI:            nnunet_gui.exe (в корне проекта)')
    print(f'  * проверить устройства:     "{py}" -m radonc_segmentation.check_devices')
    print(f'  * смоук-тест инференса:     "{py}" -m radonc_segmentation.infer_ov P110 --folds 0')
    print('  * обработать пациента:      run_pipeline.bat P110')
    print("Python окружения (на случай ручной настройки): " + py)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(1)
