# ============================================================
# workers.py — фоновые задачи GUI
# ------------------------------------------------------------
# JobWorker выполняет в рабочем потоке либо последовательность
# команд (скрипты пайплайна), либо функцию in_process (импорт
# DICOM, батч-постпроцессинг). Вывод скриптов и итоговый код
# возврата передаются в GUI-поток сигналами — окно не замирает.
# ============================================================
import os
import re
import subprocess
import sys
import traceback

from PySide6.QtCore import QObject, Signal, Slot

# не мигать консольными окнами на Windows; на POSIX creationflags игнорируются
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class JobWorker(QObject):
    """Не создаётся напрямую из GUI-логики — см. MainWindow._start_job.

    in_process (если задана) вызывается как in_process(log) в рабочем
    потоке; log — функция, безопасно выводящая строку в лог GUI
    (через сигнал). Вернула int — это код возврата задачи.
    """

    line = Signal(str)             # строка лога
    finished_job = Signal(str, int)  # (заголовок, код возврата)

    def __init__(self, title, commands=None, in_process=None, cwd=None, env=None):
        super().__init__()
        self.title = title
        self.commands = list(commands or [])
        self.in_process = in_process
        self.cwd = cwd
        self.env = env
        self.proc = None           # текущий запущенный процесс (для stop())

    @Slot()
    def run(self):
        rc = 0
        try:
            if self.in_process is not None:
                ret = self.in_process(lambda s: self.line.emit(s))
                if isinstance(ret, int):
                    rc = ret
            else:
                env = self.env if self.env is not None else os.environ.copy()
                env.setdefault("PYTHONIOENCODING", "utf-8")
                for cmd in self.commands:
                    self.line.emit("$ " + subprocess.list2cmdline(cmd) + "\n")
                    self.proc = subprocess.Popen(
                        cmd, cwd=self.cwd, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        bufsize=0, creationflags=CREATE_NO_WINDOW,
                    )
                    # Бинарное чтение с разбивкой по \r И \n: tqdm-прогресс-бары
                    # (квантизация NNCF, инференс) шлют обновления через \r без
                    # \n — построчный text-read держал бы лог GUI пустым до конца
                    # всей фазы, и было непонятно, идёт процесс или завис.
                    buf = b""
                    while True:
                        chunk = self.proc.stdout.read(512)
                        if not chunk:
                            break
                        buf += chunk
                        parts = re.split(b"[\r\n]", buf)
                        buf = parts.pop()
                        for part in parts:
                            if part.strip():
                                self.line.emit(part.decode("utf-8", errors="replace") + "\n")
                    if buf.strip():
                        self.line.emit(buf.decode("utf-8", errors="replace") + "\n")
                    self.proc.wait()
                    if self.proc.returncode != 0:
                        rc = self.proc.returncode
                        break
        except Exception as e:
            self.line.emit(f"ОШИБКА: {e}\n")
            self.line.emit(f"TRACEBACK:\n{traceback.format_exc()}\n")
            rc = -1
        self.finished_job.emit(self.title, rc)

    def stop(self):
        """Убивает ДЕРЕВО текущего запущенного процесса (кнопкой stop или при
        закрытии GUI), чтобы скрипт-пайплайн не остался работать бесхозным.
        kill() самого процесса недостаточно: скрипты (pipeline.py,
        validate_model.py) сами spawn'ят внуков (infer_ov и др.) — осиротевший
        внук висит с переполненным pipe stdout и никогда не завершается."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                if sys.platform == "win32":
                    # /T — дерево процессов (дети и внуки), /F — принудительно
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                                   capture_output=True, timeout=15)
                else:
                    self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:
                pass
