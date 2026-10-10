# ============================================================
# config.py — значения по умолчанию + загрузчик settings.ini
# ------------------------------------------------------------
# Пользовательские настройки живут в settings.ini (корень проекта) —
# правьте его, а не этот файл. Здесь только значения по умолчанию и
# константы модели (aligned-преобработка), которые менять нельзя.
# Приоритет: CLI-флаги > settings.ini > значения ниже.
# ============================================================
import configparser
import os
import re
import sys


class ConfigError(Exception):
    """Базовое исключение ошибок конфигурации."""
    pass


class ModelNotFoundError(ConfigError):
    """Модель не найдена в MODELS_ROOT."""
    pass


class SettingsValidationError(ConfigError):
    """Ошибка валидации значения в settings.ini."""
    pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)

# при запуске из exe (PyInstaller) «дом» — папка, где лежит exe:
# там же ожидаем settings.ini и подпапку scripts
_FROZEN = getattr(sys, "frozen", False)


def _exe_dir():
    return os.path.dirname(os.path.abspath(sys.executable))


def _settings_base_dir():
    return _exe_dir() if _FROZEN else _PROJECT_DIR


# где ищем settings.ini: рядом с exe (если frozen), корень проекта, папка scripts
_SETTINGS_CANDIDATES = []
if _FROZEN:
    _SETTINGS_CANDIDATES.append(os.path.join(_exe_dir(), "settings.ini"))
_SETTINGS_CANDIDATES += [
    os.path.join(_PROJECT_DIR, "settings.ini"),
    os.path.join(_THIS_DIR, "settings.ini"),
]

# путь к фактически загруженному settings.ini (None, если не найден);
# заполняется _load_settings()
SETTINGS_PATH = None

# ---------- Значения по умолчанию (переопределяются settings.ini) ----------
DEVICE = "cpu"
NUM_THREADS = 6
N_ASYNC_REQUESTS = 2
OV_PERFORMANCE_HINT = "LATENCY"

DEFAULT_VARIANT = "legacy"
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
STEP_SIZE_RATIO = 0.5
CROP_AIR = True
AIR_THRESHOLD_HU = -300
CROP_MARGIN_VOXELS = 10
ACCUMULATOR = "disk"  # disk | ram

# модели и данные. ПУСТОЕ ЗНАЧЕНИЕ = подпапка проекта (переносимо между
# компьютерами); в settings.ini можно задать и абсолютный путь.
MODELS_ROOT = ""                     # -> <проект>\models
MODEL_NAME = "headandneck"           # анатомическая модель (подпапка models/, patients/, validation/)
MODEL = "int8"                       # квантизация: подпапка models\<MODEL_NAME>\MODEL

PATIENTS_ROOT = ""                   # -> <проект>\patients
LOG_DIR = ""                         # -> <проект>\logs
WATCH_DIR = ""                       # -> <проект>\inbox
WATCH_POLL_SECONDS = 10
VALIDATION_ROOT = ""                 # -> <проект>\validation

# папки по умолчанию для GUI (пусто = не подставлять)
DEFAULT_IMPORT_DIR = ""
DEFAULT_EXPORT_DIR = ""

CONDA_PYTHON = ""  # пусто = найдём автоматически (см. find_conda_python)

# отпечаток машины из settings.ini [hardware]; не совпал — значит проект
# перенесли на другой компьютер, подстраиваем настройки под новое железо
HW_FINGERPRINT = ""

# ---------- Константы модели (НЕ менять — должны совпадать с обучением) ----------
# Опциональный путь к dataset_fingerprint.json, снятому при обучении
# (переопределяется в settings.ini [model] DATASET_FINGERPRINT). Файл есть
# только на машине, где обучали модель; пусто = использовать зашитые ниже
# константы (их значения сняты с этого fingerprint).
DATASET_FINGERPRINT = ""
TARGET_SPACING_ZYX = (2.0, 1.171875, 1.171875)
CT_CLIP_LO = -990.0
CT_CLIP_HI = 1470.0
CT_MEAN = 50.021175384521484
CT_STD = 215.93724060058594

DEFAULT_PATCH_SIZE = (56, 192, 192)  # (Z, Y, X), уточняется по модели при старте

VALIDATION_IMAGES = ""  # -> <проект>\validation\imagesTr (переопределяется в settings.ini)
VALIDATION_LABELS = ""  # -> <проект>\validation\labelsTr

LABELS = {
    "BrachialPlexus_R": 1,
    "BrachialPlexus_L": 2,
    "SpinalCanal": 3,
    "SpinalCord": 4,
    "Trachea": 5,
    "Esophagus": 6,
    "Thyroid": 7,
    "A_Carotid_R": 8,
    "A_Carotid_L": 9,
    "Larynx": 10,
    "PharynxConst_I": 11,
    "PharynxConst_M": 12,
    "PharynxConst_S": 13,
    "OpticNerve_R": 14,
    "OpticNerve_L": 15,
    "Brain": 16,
    "Submandibular_R": 17,
    "Submandibular_L": 18,
    "Parotid_R": 19,
    "Parotid_L": 20,
    "Lips": 21,
    "Tongue": 22,
    "Mandible": 23,
    "Eye_R": 24,
    "Lens_R": 25,
    "Eye_L": 26,
    "Lens_L": 27,
    "TMJoint_R": 28,
    "TMJoint_L": 29,
    "Cochlea_R": 30,
    "Cochlea_L": 31,
}


# ---------- Модели ----------

def available_models():
    """Модели, пригодные для инференса: "<анатомия>/<квантизация>" по новой
    структуре (models\\<анатомия>\\<квантизация>\fold_*\\*.xml) плюс legacy
    плоские подпапки models\\<имя> с xml/fold_*/."""
    if not os.path.isdir(MODELS_ROOT):
        return []
    out = []
    for d in sorted(os.listdir(MODELS_ROOT)):
        p = os.path.join(MODELS_ROOT, d)
        if not os.path.isdir(p) or d.startswith((".", "_", "~")):
            continue
        for q in sorted(os.listdir(p)):
            qd = os.path.join(p, q)
            if not os.path.isdir(qd) or q.startswith((".", "_", "~")):
                continue
            entries = os.listdir(qd)
            has_xml = any(f.endswith(".xml") for f in entries
                          if os.path.isfile(os.path.join(qd, f)))
            has_folds = any(f.startswith("fold_") for f in entries
                            if os.path.isdir(os.path.join(qd, f)))
            if has_xml or has_folds:
                out.append(f"{d}/{q}")
            elif any(f.endswith(".xml") for f in entries
                     if os.path.isfile(os.path.join(qd, f))):
                out.append(f"{d}/{q}")
    return sorted(out)


def _detect_xml(d):
    """Имя IR-файла модели в папке: quantized_model.xml / model.xml / единственный .xml."""
    for name in ("quantized_model.xml", "model.xml"):
        if os.path.isfile(os.path.join(d, name)):
            return name
    xmls = [f for f in os.listdir(d) if f.lower().endswith(".xml")]
    if len(xmls) == 1:
        return xmls[0]
    raise ConfigError(f"В {d} не найден model.xml/quantized_model.xml (однозначный .xml)")


def resolve_model(name=None):
    """
    Разрешает имя модели из MODELS_ROOT в параметры для движка.
    name = "<анатомия>/<квантизация>" (напр. headandneck/int8) или legacy
    "int8". Ансамбль:  <base>\fold_0..N\\<xml>  -> base_path с {} и список фолдов
    Одиночная:        <base>\\<xml>            -> base_path без {}, folds=[0]
    """
    name = str(name or MODEL).strip()
    candidates = [os.path.join(MODELS_ROOT, name)]
    if "/" not in name:
        candidates.append(os.path.join(MODELS_ROOT, MODEL_NAME, name))
    base = next((c for c in candidates if os.path.isdir(c)), None)
    if base is None:
        raise ModelNotFoundError(
            f"Папка модели не найдена: {candidates[0]}\nДоступные модели: {available_models() or 'нет ни одной (см. README — export_original.py)'}"
        )
    fold_dirs = [d for d in os.listdir(base)
                 if re.fullmatch(r"fold_\d+", d) and os.path.isdir(os.path.join(base, d))]
    if fold_dirs:
        folds = sorted(int(d.split("_")[1]) for d in fold_dirs)
        return {
            "name": name,
            "base_path": os.path.join(base, "fold_{}"),
            "filename": _detect_xml(os.path.join(base, f"fold_{folds[0]}")),
            "folds": folds,
            "cache_dir": base,
            "ensemble": True,
        }
    return {
        "name": name,
        "base_path": base,  # без {} — .format() не меняет строку
        "filename": _detect_xml(base),
        "folds": [0],
        "cache_dir": base,
        "ensemble": False,
    }


def _load_settings():
    """Переопределяет значения выше из settings.ini (если файл найден)."""
    global SETTINGS_PATH
    cp = configparser.ConfigParser(interpolation=None)
    found = cp.read(_SETTINGS_CANDIDATES, encoding="utf-8-sig")
    if not found:
        return
    SETTINGS_PATH = found[0]
    ini_path = found[0]
    g = globals()

    def get(sec, key):
        try:
            return cp.get(sec, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return None

    def set_str(sec, key, target, allow_empty=False):
        v = get(sec, key)
        if v is not None:
            if v.strip() == "" and not allow_empty:
                return
            g[target] = v.strip()

    def set_int(sec, key, target):
        v = get(sec, key)
        if v is not None and v.strip() != "":
            try:
                g[target] = int(v.strip())
            except ValueError:
                raise SettingsValidationError(f"[settings.ini] {sec}.{key}: ожидалось целое число, получено {v!r}")

    def set_float(sec, key, target):
        v = get(sec, key)
        if v is not None and v.strip() != "":
            try:
                g[target] = float(v.strip())
            except ValueError:
                raise SettingsValidationError(f"[settings.ini] {sec}.{key}: ожидалось число, получено {v!r}")

    def set_bool(sec, key, target):
        v = get(sec, key)
        if v is not None and v.strip() != "":
            try:
                g[target] = cp.getboolean(sec, key)
            except ValueError:
                raise SettingsValidationError(f"[settings.ini] {sec}.{key}: ожидалось true/false, получено {v!r}")

    set_str("device", "DEVICE", "DEVICE")
    set_int("device", "NUM_THREADS", "NUM_THREADS")
    set_int("device", "N_ASYNC_REQUESTS", "N_ASYNC_REQUESTS")
    set_str("device", "PERFORMANCE_HINT", "OV_PERFORMANCE_HINT")

    set_str("inference", "VARIANT", "DEFAULT_VARIANT")
    set_float("inference", "STEP_SIZE_RATIO", "STEP_SIZE_RATIO")
    set_bool("inference", "CROP_AIR", "CROP_AIR")
    set_float("inference", "AIR_THRESHOLD_HU", "AIR_THRESHOLD_HU")
    set_int("inference", "CROP_MARGIN_VOXELS", "CROP_MARGIN_VOXELS")
    set_str("inference", "ACCUMULATOR", "ACCUMULATOR")

    folds_raw = get("inference", "FOLDS")
    if folds_raw is not None and folds_raw.strip() != "":
        try:
            g["DEFAULT_FOLDS"] = [int(x) for x in folds_raw.split(",") if x.strip() != ""]
        except ValueError:
            raise SettingsValidationError(f"[settings.ini] inference.FOLDS: ожидались номера через запятую, получено {folds_raw!r}")
        if not g["DEFAULT_FOLDS"]:
            raise SettingsValidationError("[settings.ini] inference.FOLDS: список пуст")

    set_str("model", "MODELS_ROOT", "MODELS_ROOT")
    set_str("model", "MODEL_NAME", "MODEL_NAME")
    set_str("model", "MODEL", "MODEL")
    set_str("model", "DATASET_FINGERPRINT", "DATASET_FINGERPRINT", allow_empty=True)

    set_str("paths", "PATIENTS_ROOT", "PATIENTS_ROOT")
    set_str("paths", "LOG_DIR", "LOG_DIR")
    set_str("paths", "WATCH_DIR", "WATCH_DIR")
    set_int("paths", "WATCH_POLL_SECONDS", "WATCH_POLL_SECONDS")
    set_str("paths", "DEFAULT_IMPORT_DIR", "DEFAULT_IMPORT_DIR", allow_empty=True)
    set_str("paths", "DEFAULT_EXPORT_DIR", "DEFAULT_EXPORT_DIR", allow_empty=True)
    set_str("paths", "VALIDATION_ROOT", "VALIDATION_ROOT", allow_empty=True)

    set_str("gui", "CONDA_PYTHON", "CONDA_PYTHON")
    set_str("hardware", "FINGERPRINT", "HW_FINGERPRINT", allow_empty=True)

    # валидация значений, где есть фиксированный набор
    if str(g["DEVICE"]).lower() not in ("cpu", "gpu", "gpu.0"):
        raise SettingsValidationError(f"[settings.ini] device.DEVICE: допустимо cpu | gpu | gpu.0, получено {g['DEVICE']!r}")
    if str(g["OV_PERFORMANCE_HINT"]).upper() not in ("LATENCY", "THROUGHPUT", "NONE"):
        raise SettingsValidationError(f"[settings.ini] device.PERFORMANCE_HINT: допустимо LATENCY | THROUGHPUT | none, получено {g['OV_PERFORMANCE_HINT']!r}")
    if str(g["ACCUMULATOR"]).lower() not in ("disk", "ram"):
        raise SettingsValidationError(f"[settings.ini] inference.ACCUMULATOR: допустимо disk | ram, получено {g['ACCUMULATOR']!r}")
    if str(g["DEFAULT_VARIANT"]).lower() not in ("legacy", "aligned"):
        raise SettingsValidationError(f"[settings.ini] inference.VARIANT: допустимо legacy | aligned, получено {g['DEFAULT_VARIANT']!r}")

    print(f"[config] настройки прочитаны из: {ini_path}")
    print(f"[config] устройство={g['DEVICE']}, stride={g['STEP_SIZE_RATIO']}, "
          f"фолды={g['DEFAULT_FOLDS']}, кроп={g['CROP_AIR']}, "
          f"async={g['N_ASYNC_REQUESTS']}, накопитель={g['ACCUMULATOR']}, "
          f"вариант={g['DEFAULT_VARIANT']}, модель={g['MODEL']}")


_load_settings()


# ---------- Запись настроек (для GUI) ----------

_SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")


def write_settings(updates):
    """
    Меняет значения в settings.ini, СОХРАНЯЯ комментарии и форматирование
    (построчная замена «КЛЮЧ = значение»). updates: {(секция, ключ): значение}
    в виде строк. Возвращает путь к записанному файлу. Если файла нет —
    создаёт с минимальными секциями.
    """
    target = SETTINGS_PATH or os.path.join(_settings_base_dir(), "settings.ini")
    # нормализуем ключи к нижнему регистру, помня исходное написание
    remaining = {}
    for (sec, key), val in updates.items():
        remaining[(sec.strip().lower(), key.strip().lower())] = (key.strip(), str(val))
    lines = []
    if os.path.exists(target):
        with open(target, "r", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()

    out = []
    section = None
    for line in lines:
        m = _SECTION_RE.match(line)
        if m:
            section = m.group(1).strip().lower()
            out.append(line)
            continue
        stripped = line.strip()
        if stripped and not stripped.startswith((";", "#")) and "=" in stripped:
            key, _eq, _val = stripped.partition("=")
            lookup = (section, key.strip().lower())
            if lookup in remaining:
                orig_key, val = remaining.pop(lookup)
                out.append(f"{key.rstrip()} = {val}")
                continue
        out.append(line)

    # ключей, которых не было в файле, немного: дописываем в конец их секций
    by_section = {}
    for (sec, _lk), (orig_key, val) in remaining.items():
        by_section.setdefault(sec, []).append((orig_key, val))
    for sec, pairs in by_section.items():
        sec_header_idx = [i for i, line in enumerate(out)
                          if _SECTION_RE.match(line) and _SECTION_RE.match(line).group(1).strip().lower() == sec]
        if not sec_header_idx:
            out += ["", f"[{sec}]"]
            out.extend(f"{k} = {v}" for k, v in pairs)
            continue
        start = sec_header_idx[0]
        end = len(out)
        for j in range(start + 1, len(out)):
            if _SECTION_RE.match(out[j]):
                end = j
                break
        for offset, (k, v) in enumerate(pairs):
            out.insert(end + offset, f"{k} = {v}")

    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return target


# ============================================================
# Портабельность: разрешение путей, автодетект python, железо.
# Цель — папку с программой можно копировать между компьютерами:
# пустые/относительные пути считаются от папки проекта, python
# окружения и настройки потоков/ОЗУ подстраиваются автоматически.
# ============================================================

def _resolve_dir(v, default_name):
    """Абсолютный путь — как есть; пустой/относительный — от папки проекта (или exe)."""
    if v:
        return v if os.path.isabs(v) else os.path.normpath(os.path.join(_settings_base_dir(), v))
    return os.path.join(_settings_base_dir(), default_name)


CONDA_ENV_NAME = "nnunet_env"


def candidate_conda_pythons(env_name=None):
    """Типовые расположения conda-окружения на Windows (для переноса между машинами)."""
    env_name = env_name or CONDA_ENV_NAME
    bases = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "miniconda3"),
        os.path.join(os.path.expanduser("~"), "miniconda3"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "anaconda3"),
        os.path.join(os.path.expanduser("~"), "anaconda3"),
        r"C:\ProgramData\miniconda3",
        r"C:\ProgramData\anaconda3",
        r"C:\miniconda3",
        r"C:\anaconda3",
    ]
    return [os.path.join(b, "envs", env_name, "python.exe") for b in bases if b and b.strip()]


def find_conda_python(env_name=None):
    """Первый существующий python окружения из типовых мест (или None)."""
    for p in candidate_conda_pythons(env_name):
        if os.path.isfile(p):
            return p
    return None


def _total_ram_gb():
    """Общий объём ОЗУ, ГБ (ctypes; psutil — запасной вариант)."""
    try:
        import ctypes

        class _MSE(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MSE()
        st.dwLength = ctypes.sizeof(_MSE)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def hardware_fingerprint():
    """Отпечаток машины: хостнейм | CPU | логических ядер | ОЗУ ГБ."""
    import platform
    import socket
    cpu = " ".join((platform.processor() or "?").split())
    return f"{socket.gethostname()}|{cpu}|{os.cpu_count()}|{_total_ram_gb():.0f}"


# текст для GUI, если при загрузке настроек сработала автоподстройка
AUTOTUNE_NOTICE = None

def autotune_hardware(force=False, log=print):
    """Первый запуск на новом компьютере: подбираем NUM_THREADS/ACCUMULATOR/CONDA_PYTHON
    под железо и запоминаем отпечаток машины в settings.ini. Пока отпечаток совпадает,
    ручные правки пользователя не перетираются."""
    global NUM_THREADS, ACCUMULATOR, CONDA_PYTHON, HW_FINGERPRINT, AUTOTUNE_NOTICE
    fp = hardware_fingerprint()
    if not force and HW_FINGERPRINT == fp:
        return False

    ram = _total_ram_gb()
    # логические ядра: на i5-9400T (6C/6T) это 6, на i5-11300H (4C/8T) — 8,
    # что померено быстрее физических 4 на этой memory-bound INT8-нагрузке (~+16%)
    threads = os.cpu_count() or 1
    # под логиты 32 классов нужно ~7 ГБ — только на машинах с запасом ОЗУ
    acc = "ram" if ram >= 15.0 else "disk"
    py = CONDA_PYTHON if (CONDA_PYTHON and os.path.isfile(CONDA_PYTHON)) else find_conda_python()

    updates = {
        ("device", "NUM_THREADS"): str(threads),
        ("inference", "ACCUMULATOR"): acc,
        ("hardware", "FINGERPRINT"): fp,
    }
    if py:
        updates[("gui", "CONDA_PYTHON")] = py
    try:
        write_settings(updates)
    except OSError as e:
        log(f"[config] не удалось записать автоподстройку в settings.ini: {e}")

    NUM_THREADS = threads
    ACCUMULATOR = acc
    HW_FINGERPRINT = fp
    if py:
        CONDA_PYTHON = py
    AUTOTUNE_NOTICE = (
        f"Обнаружен новый компьютер: {threads} логических ядер, {ram:.0f} ГБ ОЗУ — "
        f"настройки подстроены (потоков {threads}, накопитель {acc.upper()}"
        + (f", python: {py}" if py else "") + ")."
    )
    log(f"[config] {AUTOTUNE_NOTICE}")
    return True


def _heal_conda_python(log=print):
    """Если записанный python окружения не существует (другой компьютер/переустановка) —
    ищем его в типовых местах и вписываем найденный путь в settings.ini."""
    global CONDA_PYTHON
    if CONDA_PYTHON and os.path.isfile(CONDA_PYTHON):
        return CONDA_PYTHON
    py = find_conda_python()
    if py:
        try:
            write_settings({("gui", "CONDA_PYTHON"): py})
        except OSError as e:
            log(f"[config] не удалось обновить CONDA_PYTHON в settings.ini: {e}")
        CONDA_PYTHON = py
        log(f"[config] python окружения найден автоматически: {py}")
    return py


# применяем всё это при импорте модуля (то есть при запуске ЛЮБОГО скрипта):
MODELS_ROOT = _resolve_dir(MODELS_ROOT, "models")
PATIENTS_ROOT = _resolve_dir(PATIENTS_ROOT, "patients")
LOG_DIR = _resolve_dir(LOG_DIR, "logs")
WATCH_DIR = _resolve_dir(WATCH_DIR, "inbox")
VALIDATION_ROOT = _resolve_dir(VALIDATION_ROOT, "validation")


# ---------- per-модельные пути (структура: patients/модель/PXXX и т.д.) ----------

def patients_dir():
    """Папка пациентов ТЕКУЩЕЙ анатомической модели."""
    return os.path.join(PATIENTS_ROOT, MODEL_NAME)


def patient_dir(patient_id):
    return os.path.join(patients_dir(), patient_id)


def validation_cases_dir():
    r"""Папка валидационных кейсов текущей модели: validation\<модель>\<кейс>."""
    return os.path.join(VALIDATION_ROOT, MODEL_NAME)


def validation_cases():
    """[(case, img_path, gt_path|None)] текущей модели. Поддерживаются обе
    структуры: новая per-case (<кейс>\nnunet_input\\... + <кейс>\\gt\\...) и
    legacy (imagesTr\\*_0000.nii.gz + labelsTr\\<кейс>.nii.gz)."""
    root = validation_cases_dir()
    out = []
    if os.path.isdir(root):
        for case in sorted(os.listdir(root)):
            d = os.path.join(root, case)
            if not os.path.isdir(d):
                continue
            img = os.path.join(d, "nnunet_input", f"{case}_0000.nii.gz")
            gt = os.path.join(d, "gt", f"{case}.nii.gz")
            if os.path.isfile(img):
                out.append((case, img, gt if os.path.isfile(gt) else None))
                continue
            img = os.path.join(d, "imagesTr", f"{case}_0000.nii.gz")
            gt = os.path.join(d, "labelsTr", f"{case}.nii.gz")
            if os.path.isfile(img):
                out.append((case, img, gt if os.path.isfile(gt) else None))
    return out


def model_dir(quant=None):
    r"""Папка весов модели: models\<анатомия>\<квантизация> (legacy-фолбэк:
    models\<квантизация> для старых установок)."""
    quant = quant or MODEL
    new = os.path.join(MODELS_ROOT, MODEL_NAME, quant)
    if os.path.isdir(new):
        return new
    legacy = os.path.join(MODELS_ROOT, quant)
    if os.path.isdir(legacy):
        return legacy
    return new


def init_config(log=print):
    """
    Явная инициализация конфигурации после создания QApplication.
    Вызывает автоподстройку под железо и поиск python окружения.
    Должен вызываться один раз при старте приложения.
    """
    _heal_conda_python(log=log)
    autotune_hardware(log=log)
    ensure_dirs()


# ---------- Утилиты ----------

def ensure_dirs():
    os.makedirs(patients_dir(), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
