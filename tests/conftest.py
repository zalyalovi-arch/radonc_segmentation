# conftest.py — общий конфиг pytest
# Единственное легитимное место добавления пути в sys.path: pytest
# выполняется из произвольного cwd, поэтому корень проекта добавляем
# явно (в обычных запусках это не нужно: python -m кладёт cwd в sys.path,
# а pip install -e . ставит пакеты radonc_segmentation и gui в окружение).
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
