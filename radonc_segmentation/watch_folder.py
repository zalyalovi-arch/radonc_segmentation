# ============================================================
# watch_folder.py — автоматическая обработка папки-«входящей»
# ------------------------------------------------------------
# Кладёте подпапку с экспортированными DICOM в WATCH_DIR (по умолчанию
# C:\nnunet\inbox) — скрипт сам прогоняет полный пайплайн
# (import -> NIfTI -> инференс -> RTSTRUCT -> экспорт) и кладёт
# rtstruct.dcm обратно в вашу подпапку (в подпапку rtstruct\).
#
#   python watch_folder.py
#
# Маркеры в подпапке пациента: .processing (в работе), .done (готово),
# .failed (ошибка, текст — рядом в .failed.txt). Обработку можно
# повторить, удалив маркер.
# Работает в одном окне консоли; остановка — Ctrl+C.
# ============================================================
import os
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
import config
import env_check

# предупреждаем о проблемах окружения ДО старта цикла (см. env_check.py)
env_check.ensure_ready()


def has_dicom(folder):
    for _root, _dirs, files in os.walk(folder):
        if any(f.lower().endswith(".dcm") for f in files):
            return True
    return False


def acquire_lock(watch_dir):
    """Простейшая защита от второго экземпляра. Возвращает True, если лок взят."""
    lock_path = os.path.join(watch_dir, "watch.lock")
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                pid = int(f.read().strip() or 0)
            running = False
            try:
                import psutil
                running = psutil.pid_exists(pid)
            except ImportError:
                running = True  # без psutil считаем, что жив (безопаснее)
            if running:
                print(f"Уже запущен другой watch_folder (pid {pid}). "
                      f"Если это не так, удалите {lock_path}")
                return False
            print(f"Найден устаревший lock (pid {pid} не работает) — перехватываю.")
            os.remove(lock_path)
        except (ValueError, OSError):
            os.remove(lock_path)
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    return True


def main():
    watch_dir = config.WATCH_DIR
    os.makedirs(watch_dir, exist_ok=True)
    if not acquire_lock(watch_dir):
        sys.exit(1)

    print(f"Слежу за папкой: {watch_dir} (опрос каждые {config.WATCH_POLL_SECONDS} с)")
    print("Положите ПОДПАПКУ с DICOM — RTSTRUCT вернётся в неё же (rtstruct\\rtstruct.dcm).")
    print("Остановка: Ctrl+C\n")

    try:
        while True:
            try:
                for name in sorted(os.listdir(watch_dir)):
                    folder = os.path.join(watch_dir, name)
                    if not os.path.isdir(folder):
                        continue
                    markers = [os.path.join(folder, m) for m in (".done", ".failed", ".processing")]
                    if any(os.path.exists(m) for m in markers):
                        continue
                    if not has_dicom(folder):
                        continue

                    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{stamp}] Новый пациент: {name} — запускаю пайплайн...")
                    processing = os.path.join(folder, ".processing")
                    with open(processing, "w", encoding="utf-8") as f:
                        f.write(stamp)

                    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "pipeline.py"), folder]
                    r = subprocess.run(cmd, cwd=SCRIPTS_DIR)

                    os.remove(processing)
                    if r.returncode == 0:
                        with open(os.path.join(folder, ".done"), "w", encoding="utf-8") as f:
                            f.write(stamp)
                        print(f"[{time.strftime('%H:%M:%S')}] {name}: готово, "
                              f"RTSTRUCT в {os.path.join(folder, 'rtstruct')}\n")
                    else:
                        with open(os.path.join(folder, ".failed"), "w", encoding="utf-8") as f:
                            f.write(stamp)
                        with open(os.path.join(folder, ".failed.txt"), "w", encoding="utf-8") as f:
                            f.write(f"pipeline.py завершился с кодом {r.returncode}\n"
                                    f"Смотрите вывод консоли. Повтор: удалите .failed и .failed.txt\n")
                        print(f"[{time.strftime('%H:%M:%S')}] {name}: ОШИБКА "
                              f"(код {r.returncode})\n")
            except FileNotFoundError:
                pass  # папку могли удалить прямо во время листинга
            time.sleep(config.WATCH_POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        lock_path = os.path.join(watch_dir, "watch.lock")
        try:
            os.remove(lock_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
