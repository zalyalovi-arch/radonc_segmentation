# ============================================================
# workers.py — фоновые задачи GUI
# ------------------------------------------------------------
# JobWorker выполняет в рабочем потоке либо последовательность
# команд (скрипты пайплайна), либо функцию in_process (импорт
# DICOM, батч-постпроцессинг). Вывод скриптов и итоговый код
# возврата передаются в GUI-поток сигналами — окно не замирает.
# ============================================================
import os
import subprocess
import traceback

from PySide6.QtCore import QObject, Signal, Slot

CREATE_NO_WINDOW = 0x08000000  # не мигать консольными окнами на Windows


class JobWorker(QObject):
    """Не создаётся напрямую из GUI-логики — см. MainWindow._start_job.

    in_process (если задана) вызывается как in_process(log) в рабочем
    потоке; log — функция, безопасно выводящая строку в лог GUI
    (через сигнал). Вернула int — это код возврата задачи.
    """

    line = Signal(str)             # строка лога
    finished_job = Signal(str, int)  # (заголовок, код возврата)

    def __init__(self, title, commands=None, in_process=None, cwd=None):
        super().__init__()
        self.title = title
        self.commands = list(commands or [])
        self.in_process = in_process
        self.cwd = cwd
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
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                for cmd in self.commands:
                    self.line.emit("$ " + subprocess.list2cmdline(cmd) + "\n")
                    self.proc = subprocess.Popen(
                        cmd, cwd=self.cwd, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        bufsize=1, creationflags=CREATE_NO_WINDOW,
                    )
                    for ln in self.proc.stdout:
                        self.line.emit(ln)
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
        """Убивает текущий запущенный процесс (при закрытии GUI), чтобы
        скрипт-пайплайн не остался работать бесхозным процессом."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
