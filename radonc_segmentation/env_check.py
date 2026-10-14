# ============================================================
# env_check.py — проверка окружения против requirements.txt
# ------------------------------------------------------------
# Только стандартная библиотека: работает ЛЮБЫМ python (в т.ч. до
# установки пакетов) и не тянет зависимостей сам.
#
# Использование:
#   python env_check.py            # полная проверка всех пакетов requirements.txt
#   python env_check.py --fast     # только критичные для запуска скриптов пакеты
#   python env_check.py --json     # вывод + JSON-блок в конце (для GUI)
#
# Код возврата: 0 — всё в порядке; 1 — есть проблемы.
#
# В скриптах используется функция ensure_ready(): печатает понятное
# предупреждение и команду установки вместо трейсбека ModuleNotFoundError.
# ============================================================
import argparse
import json
import os
import re
import sys

try:
    from importlib import metadata as _im
except ImportError:  # python < 3.8 — не поддерживаем
    _im = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
REQ_PATH = os.path.join(_PROJECT_DIR, "requirements.txt")
# опциональные пакеты (импорт .pth -> IR, INT8-квантизация): отсутствие
# НЕ мешает инференсу, поэтому проверяются отдельно и без кода ошибки
QUANT_REQ_PATH = os.path.join(_PROJECT_DIR, "requirements-quantization.txt")
# стек нативного nnUNet-движка без torch (для машин без NVIDIA поверх
# torch-CPU); на GPU-машинах его заменяет requirements-gpu.txt
NATIVE_REQ_PATH = os.path.join(_PROJECT_DIR, "requirements-native.txt")
# опциональные пакеты нативного GPU-инференса (torch CUDA + nnunetv2)
GPU_REQ_PATH = os.path.join(_PROJECT_DIR, "requirements-gpu.txt")

# Дистрибутивы, без которых скрипты падают сразу (имена как в requirements.txt).
# Этого набора достаточно, чтобы «предупредить заранее» любой входной скрипт.
CRITICAL = ["numpy", "scipy", "nibabel", "openvino", "tqdm",
            "SimpleITK", "pydicom", "rt-utils"]

# Поддерживаемые версии python (проект собран на 3.10)
_PY_MIN, _PY_MAX = (3, 10), (3, 12)

_INSTALL_HINT = (
    "Установка в текущее окружение:\n"
    "  python -m pip install -r requirements.txt\n"
    "Полная настройка нового компьютера одной командой:\n"
    "  python scripts\\setup_env.py"
)

_REQ_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$")


def parse_requirements(path=REQ_PATH):
    """[(имя, спецификация версии или '')] из requirements.txt; опции/-index-url пропускаются."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            line = line.split(";", 1)[0].strip()  # environment markers не проверяем
            m = _REQ_RE.match(line)
            if not m:
                continue
            name = m.group(1)
            spec = (m.group(3) or "").strip().split("#", 1)[0].strip()
            out.append((name, spec))
    return out


def _norm(name):
    """Канонизация имени дистрибутива (PEP 503): underscores/dots -> дефисы, lower."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_version(dist_name):
    if _im is None:
        return None
    try:
        return _im.version(dist_name)
    except Exception:
        # PackageNotFoundError и прочее — считаем «не установлен»
        return None


def _cmp_versions(a, b):
    """Сравнение версий без пакета packaging: по числовым сегментам, лексически для строк.
    Возвращает -1/0/1. Локальные суффиксы (+cpu) учитываются только при точном равенстве."""
    def core(v):
        return re.split(r"[.+-]", v)
    ca, cb = core(a), core(b)
    for x, y in zip(ca, cb):
        xn, yn = _num(x), _num(y)
        if xn is not None and yn is not None:
            if xn != yn:
                return -1 if xn < yn else 1
        else:
            if x != y:
                return -1 if x < y else 1
    return (len(ca) > len(cb)) - (len(ca) < len(cb))


def _num(s):
    try:
        return int(s)
    except ValueError:
        return None


def _base_version(v):
    """Версия без локального суффикса сборки: '2.8.0+cu126' -> '2.8.0'."""
    return v.split("+", 1)[0]


def _spec_ok(spec, installed):
    """Грубая проверка 'spec' (поддерживаются ==, !=, >=, <=, >, <) против установленной.
    Пин с суффиксом сборки (==2.8.0+cpu / ==2.8.0+cu126) сравнивается по базовой
    версии: в окружении стоит одна сборка torch под железо, а квантизация и
    нативный движок работают с любой — суффикс лишь фиксирует, какую ставил pip."""
    if not spec:
        return True
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(==|!=|>=|<=|>|<)\s*(.+)$", part)
        if not m:
            return True  # непонятная спецификация — не блокируем
        op, ver = m.group(1), m.group(2).strip()
        if op in ("==", "!=") and "+" in ver:
            same_base = _base_version(installed) == _base_version(ver)
            if (op == "==") != same_base:
                return False
        elif op == "==":
            if installed != ver:
                return False
        elif op == "!=":
            if installed == ver:
                return False
        else:
            c = _cmp_versions(installed, ver)
            if {"<": c < 0, "<=": c <= 0, ">": c > 0, ">=": c >= 0}[op] is False:
                return False
    return True


def check(critical=None):
    """Возвращает (problems, checked, python_note).
    problems: [{"name", "required", "installed", "status": missing|version}]
    """
    problems = []
    checked = 0

    if _im is None:
        problems.append({"name": "python", "required": ">=3.8",
                         "installed": sys.version, "status": "version"})
        return problems, checked, "python слишком старый (нет importlib.metadata)"

    py_note = ""
    if sys.version_info[:2] < _PY_MIN or sys.version_info[:2] > _PY_MAX:
        py_note = (f"python {sys.version.split()[0]}: проект собран на 3.10, "
                   f"поддерживаются 3.{_PY_MIN[1]}–3.{_PY_MAX[1]} — возможны несовместимости колёс")

    reqs = parse_requirements()
    wanted = []
    if critical is None:
        wanted = reqs  # полная проверка
    else:
        by_norm = {_norm(n): (n, s) for n, s in reqs}
        for name in critical:
            if _norm(name) in by_norm:
                wanted.append(by_norm[_norm(name)])
            else:
                wanted.append((name, ""))  # нет в requirements — проверяем только наличие

    for name, spec in wanted:
        checked += 1
        installed = _installed_version(name)
        if installed is None:
            problems.append({"name": name, "required": spec or "?",
                             "installed": None, "status": "missing"})
        elif spec and not _spec_ok(spec, installed):
            problems.append({"name": name, "required": spec,
                             "installed": installed, "status": "version"})
    return problems, checked, py_note


def check_optional(path=QUANT_REQ_PATH):
    """(problems, total) по опциональному requirements-файлу. Проблемы носят
    информационный характер: без них не работает только соответствующая
    функция (квантизация / нативный движок), инференс работает."""
    if not os.path.isfile(path):
        return [], 0
    problems = []
    reqs = parse_requirements(path)
    for name, spec in reqs:
        installed = _installed_version(name)
        if installed is None:
            problems.append({"name": name, "required": spec or "?",
                             "installed": None, "status": "missing"})
        elif spec and not _spec_ok(spec, installed):
            problems.append({"name": name, "required": spec,
                             "installed": installed, "status": "version"})
    return problems, len(reqs)


def ensure_ready(critical=None, exit_on_problem=True, log=print):
    """Быстрая проверка перед запуском скрипта. Возвращает список проблем (пустой — ОК).
    При проблемах печатает понятное сообщение; по умолчанию завершает скрипт."""
    problems, _checked, py_note = check(critical=critical if critical is not None else CRITICAL)
    if py_note:
        log(f"[ОКРУЖЕНИЕ] ВНИМАНИЕ: {py_note}")
    if not problems:
        return []
    missing = [p["name"] for p in problems if p["status"] == "missing"]
    mismatched = [f"{p['name']} (нужно {p['required']}, стоит {p['installed']})"
                  for p in problems if p["status"] == "version"]
    msg = "[ОКРУЖЕНИЕ] Проблемы с зависимостями — работа остановлена.\n"
    if missing:
        msg += "  Отсутствуют: " + ", ".join(missing) + "\n"
    if mismatched:
        msg += "  Неверные версии: " + ", ".join(mismatched) + "\n"
    msg += _INSTALL_HINT
    log(msg)
    if exit_on_problem:
        sys.exit(1)
    return problems


def main():
    ap = argparse.ArgumentParser(description="Проверка окружения против requirements.txt")
    ap.add_argument("--fast", action="store_true",
                    help="проверить только критичные пакеты (быстрый стартовый чек)")
    ap.add_argument("--json", action="store_true", help="дописать JSON-блок в конец (для GUI)")
    args = ap.parse_args()

    problems, checked, py_note = check(critical=CRITICAL if args.fast else None)
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    if py_note:
        print(f"ВНИМАНИЕ: {py_note}")
    print(f"requirements.txt: {REQ_PATH}")
    print(f"Проверено пакетов: {checked}; проблем: {len(problems)}")
    for p in problems:
        if p["status"] == "missing":
            print(f"  ОТСУТСТВУЕТ     {p['name']}  (нужно {p['required']})")
        else:
            print(f"  НЕ ТА ВЕРСИЯ   {p['name']}: нужно {p['required']}, установлено {p['installed']}")
    if not problems:
        print("Все пакеты на месте.")
    else:
        print()
        print(_INSTALL_HINT)

    # опциональные наборы — информационно, на код возврата не влияют
    q_problems, q_total = check_optional(QUANT_REQ_PATH)
    if q_total:
        if q_problems:
            print(f"\nОпционально (квантизация/импорт .pth): не хватает {len(q_problems)} из {q_total}: "
                  + ", ".join(p["name"] for p in q_problems))
            print("  Установка: python -m pip install -r requirements-quantization.txt\n"
                  "  (без них инференс работает, но «Импорт модели»/«Квантизация» в GUI — нет)")
        else:
            print(f"Опциональные пакеты квантизации ({q_total}): все на месте.")

    # torch — одна сборка на окружение (cpu или cu126); пины с +суффиксом
    # проверяются по базовой версии, поэтому любая установленная сборка годится
    torch_ver = _installed_version("torch")
    if torch_ver is None:
        print("\nОпционально (torch): не установлен — «Импорт модели»/«Квантизация» "
              "и нативный движок не заработают.")
        print("  Установка: python -m pip install -r requirements-torch-cpu.txt\n"
              "  (на машине с NVIDIA GPU — requirements-gpu.txt: тот же torch, CUDA-сборка)")
    else:
        print(f"\ntorch: {torch_ver}"
              + (" (CUDA-сборка)" if "+cu" in torch_ver else " (CPU-сборка)"))

    # стек нативного nnUNet-движка (без torch) и GPU-набор — тоже информационно
    for path, title, hint in (
        (NATIVE_REQ_PATH, "нативный nnUNet-движок (CPU-инференс, КРАЙНЕ медленно)",
         "  Установка: python -m pip install -r requirements-native.txt"),
        (GPU_REQ_PATH, "нативный GPU-инференс (NVIDIA)",
         "  Установка: python -m pip install -r requirements-gpu.txt"),
    ):
        problems_n, total_n = check_optional(path)
        if not total_n:
            continue
        if problems_n:
            print(f"Опционально ({title}): не хватает {len(problems_n)} из {total_n}: "
                  + ", ".join(p["name"] for p in problems_n))
            print(hint)
        else:
            print(f"Опционально ({title}): все на месте.")

    if args.json:
        print("__JSON__" + json.dumps({
            "python": sys.version.split()[0],
            "python_note": py_note,
            "checked": checked,
            "problems": problems,
            "ok": not problems,
        }, ensure_ascii=False))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
