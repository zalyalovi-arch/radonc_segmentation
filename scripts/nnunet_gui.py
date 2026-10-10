# nnunet_gui.py — GUI пайплайна оконтуринга H&N (v3)
# ------------------------------------------------------------
# Вкладка «Работа»: импорт DICOM (или пересчёт по ID), инференс+RTSTRUCT,
# экспорт, управление watch-папкой. Вкладка «Настройки»: редактор
# settings.ini (те же параметры, что в файле; сохранение не трогает
# комментарии). Долгие операции — в рабочем потоке, окно не замирает.
#
# exe должен лежать в КОРНЕ проекта (рядом settings.ini и scripts\):
#   C:\nnunet reboot\nnunet_gui.exe
# Пересборка (после правок; тяжёлые пакеты окружения исключены — GUI нужны
# только tkinter и config, иначе анализ PyInstaller падает по RecursionError):
#   conda activate nnunet_env
#   cd "C:\nnunet reboot\scripts"
#   python -m PyInstaller nnunet_gui.spec
#   copy /y ..\dist\nnunet_gui.exe ..\nnunet_gui.exe   (из папки scripts)

import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

_IS_FROZEN = getattr(sys, "frozen", False)
if _IS_FROZEN:
    _EXE_DIR = os.path.dirname(os.path.abspath(sys.executable))
    SCRIPTS_DIR = os.path.join(_EXE_DIR, "scripts")
    if not os.path.isdir(SCRIPTS_DIR):
        SCRIPTS_DIR = _EXE_DIR  # exe положили внутрь scripts
else:
    SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, SCRIPTS_DIR)

import config

# вкладка «Постпроцессинг» требует numpy/matplotlib — в exe-сборке они
# исключены, там вкладка недоступна (запускайте GUI из окружения:
# python scripts\nnunet_gui.py)
try:
    import postprocess_viewer
    _POST_OK = True
except Exception as _post_err:
    _POST_OK = False
    _POST_ERR = _post_err

# Python, которым запускать скрипты: в exe-сборке — интерпретатор conda-окружения
# (путь настраивается на вкладке «Настройки», при переносе проекта ищется сам),
# при запуске из окружения — текущий.
if _IS_FROZEN:
    if os.path.isfile(config.CONDA_PYTHON):
        PYTHON_EXE = config.CONDA_PYTHON
    else:
        PYTHON_EXE = config.find_conda_python()
else:
    PYTHON_EXE = sys.executable

CREATE_NO_WINDOW = 0x08000000 if _IS_FROZEN else 0


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Автоконтуринг головы и шеи — H&N")
        self.geometry("1240x820")
        self.minsize(1150, 720)

        self.log_queue = queue.Queue()
        self.worker = None
        self.patient_id = None
        self._on_job_done = None
        self.watch_proc = None

        nb = ttk.Notebook(self)
        self.tab_work = ttk.Frame(nb, padding=8)
        self.tab_quant = ttk.Frame(nb, padding=8)
        self.tab_settings = ttk.Frame(nb, padding=8)
        nb.add(self.tab_work, text="  Работа  ")
        nb.add(self.tab_quant, text="  Квантизация  ")
        if _POST_OK:
            self.tab_post = postprocess_viewer.PostprocessTab(nb, log=self._log)
            nb.add(self.tab_post, text="  Постпроцессинг  ")
        nb.add(self.tab_settings, text="  Настройки  ")
        # лог строится и пакуется ПЕРВЫМ (внизу, фиксированная высота) —
        # иначе вкладка с matplotlib-канвасом выдавливает его в ноль
        self._build_log_section()
        self.log_frame.pack_configure(side="bottom", fill="x", expand=False)
        nb.pack(fill="both", expand=True)

        self._build_import_section()
        self._build_run_section()
        self._build_watch_section()
        self._build_export_section()
        self._build_quant_tab()
        self._build_settings_tab()

        # папки по умолчанию из settings.ini -> поля главного окна
        if config.DEFAULT_IMPORT_DIR:
            self.entry_dicom.insert(0, config.DEFAULT_IMPORT_DIR)
        if config.DEFAULT_EXPORT_DIR:
            self.entry_export.insert(0, config.DEFAULT_EXPORT_DIR)

        self.after(100, self._poll_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._startup_checks()

    # ---------- служебное ----------
    def _startup_checks(self):
        problems = []
        if not os.path.isfile(os.path.join(SCRIPTS_DIR, "pipeline.py")):
            problems.append(f"Не найден pipeline.py в {SCRIPTS_DIR}.\n"
                            f"Положите exe в корень проекта (рядом с settings.ini и папкой scripts).")
        if PYTHON_EXE is None:
            problems.append(
                "Не найден python окружения nnunet_env ни в одном типовом месте.\n"
                "Настройте компьютер одной командой:\n"
                "  python scripts\\setup_env.py\n"
                "или укажите путь на вкладке «Настройки» (CONDA_PYTHON).")
        if problems:
            messagebox.showwarning("Проблема конфигурации", "\n\n".join(problems))

        # зависимости окружения проверяем ТЕМ python, которым будут запускаться скрипты
        if PYTHON_EXE:
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                r = subprocess.run(
                    [PYTHON_EXE, os.path.join(SCRIPTS_DIR, "env_check.py")],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=300, creationflags=CREATE_NO_WINDOW)
                out = ((r.stdout or "") + (r.stderr or "")).strip()
                if r.returncode != 0:
                    self._log(out + "\n")
                    messagebox.showwarning(
                        "Проверка окружения",
                        "Пакеты из requirements.txt установлены не полностью:\n\n"
                        + out[-1800:] + "\n\nУстановка: python scripts\\setup_env.py")
                else:
                    self._log("[GUI] Окружение проверено: все пакеты requirements.txt на месте.\n")
            except Exception as e:
                self._log(f"[GUI] Не удалось проверить окружение: {e}\n")

        if config.AUTOTUNE_NOTICE:
            self._log(f"[GUI] {config.AUTOTUNE_NOTICE}\n")
            messagebox.showinfo("Новый компьютер", config.AUTOTUNE_NOTICE)

        self._log(f"Скрипты: {SCRIPTS_DIR}\nPython: {PYTHON_EXE or 'НЕ НАЙДЕН'}\n"
                  f"Настройки: {config.SETTINGS_PATH or 'settings.ini не найден (значения по умолчанию)'}\n")

    def _python_cmd(self, script, *args):
        if PYTHON_EXE is None:
            raise RuntimeError("Python окружения не настроен (вкладка «Настройки» → CONDA_PYTHON)")
        return [PYTHON_EXE, os.path.join(SCRIPTS_DIR, script), *args]

    def _start_job(self, title, cmd_iterable, on_done=None, in_process=None):
        if self._busy():
            messagebox.showwarning("Занято", "Дождитесь завершения текущей операции.")
            return
        self._set_buttons(running=True)
        self._log(f"\n>>> {title}\n")
        self._on_job_done = on_done

        def runner():
            rc = 0
            try:
                if in_process is not None:
                    in_process()
                else:
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    for cmd in cmd_iterable:
                        self.log_queue.put("$ " + " ".join(cmd) + "\n")
                        proc = subprocess.Popen(
                            cmd, cwd=SCRIPTS_DIR, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1, creationflags=CREATE_NO_WINDOW,
                        )
                        for line in proc.stdout:
                            self.log_queue.put(line)
                        proc.wait()
                        if proc.returncode != 0:
                            rc = proc.returncode
                            break
            except Exception as e:
                self.log_queue.put(f"ОШИБКА: {e}\n")
                rc = -1
            self.log_queue.put(f"__DONE__{title}\t{rc}")

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def _busy(self):
        return self.worker is not None and self.worker.is_alive()

    def _poll_log(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item.startswith("__DONE__"):
                    title, rc = item[len("__DONE__"):].strip().split("\t")
                    ok = rc == "0"
                    self._set_buttons(running=False)
                    self._log(f"<<< {title}: {'ГОТОВО' if ok else f'ОШИБКА (код {rc})'}\n")
                    if self._on_job_done:
                        self._on_job_done(ok, title)
                else:
                    self.log_text.insert("end", item)
                    self.log_text.see("end")
        except queue.Empty:
            pass
        # статус watch-процесса
        if self.watch_proc is not None and self.watch_proc.poll() is not None:
            rc = self.watch_proc.returncode
            self.watch_proc = None
            self.lbl_watch_status.configure(
                text=f"_watch-папка: остановлена (код {rc})", fg="gray")
            self.btn_watch_start.configure(state="normal")
            self.btn_watch_stop.configure(state="disabled")
        self.after(100, self._poll_log)

    def _log(self, s):
        self.log_text.insert("end", s)
        self.log_text.see("end")

    def _set_buttons(self, running):
        state = "disabled" if running else "normal"
        for b in (self.btn_import, self.btn_run, self.btn_export, self.btn_run_id):
            b.configure(state=state)

    def _on_close(self):
        if self._busy() or (self.watch_proc is not None and self.watch_proc.poll() is None):
            if not messagebox.askokcancel("Выход", "Операция/watch ещё выполняется. Выйти?"):
                return
        if self.watch_proc is not None and self.watch_proc.poll() is None:
            try:
                self.watch_proc.terminate()
            except OSError:
                pass
        self.destroy()

    # ================= ВКЛАДКА «РАБОТА» =================

    def _build_import_section(self):
        frame = tk.LabelFrame(self.tab_work, text="1) Импорт DICOM (или укажите ID уже импортированного пациента)", padx=10, pady=10)
        frame.pack(fill="x", padx=6, pady=4)

        tk.Label(frame, text="Папка DICOM:").grid(row=0, column=0, sticky="w")
        self.entry_dicom = tk.Entry(frame, width=66)
        self.entry_dicom.grid(row=0, column=1, padx=5, pady=2, sticky="we")
        tk.Button(frame, text="Обзор...", command=lambda: self._browse("entry_dicom", "Выберите папку с DICOM")
                  ).grid(row=0, column=2, padx=5)
        self.btn_import = tk.Button(frame, text="Импортировать", command=self._import)
        self.btn_import.grid(row=0, column=3, padx=5)

        tk.Label(frame, text="ID пациента:").grid(row=1, column=0, sticky="w")
        self.entry_pid = tk.Entry(frame, width=12)
        self.entry_pid.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.btn_run_id = tk.Button(frame, text="Выбрать", command=self._select_pid)
        self.btn_run_id.grid(row=1, column=2, sticky="w")
        tk.Label(frame, text="(например P110 — для пересчёта существующего)",
                 fg="gray").grid(row=1, column=3, sticky="w")
        frame.columnconfigure(1, weight=1)

        self.lbl_patient = tk.Label(frame, text="Пациент не выбран", fg="gray")
        self.lbl_patient.grid(row=2, column=0, columnspan=4, sticky="w")

    def _browse(self, attr, title):
        folder = filedialog.askdirectory(title=title)
        if folder:
            getattr(self, attr).delete(0, "end")
            getattr(self, attr).insert(0, folder)

    def _import(self):
        folder = self.entry_dicom.get().strip()
        if not folder:
            messagebox.showerror("Ошибка", "Выберите папку с DICOM.")
            return

        def do_import():
            import import_dicom
            pid, base_dir = import_dicom.import_dicom(folder)
            self.patient_id = pid

        def after(ok, title):
            if ok:
                self.entry_pid.delete(0, "end")
                self.entry_pid.insert(0, self.patient_id)
                self.lbl_patient.configure(text=f"Пациент: {self.patient_id}", fg="black")
                messagebox.showinfo("Импорт", f"Импортирован пациент {self.patient_id}.")
            else:
                messagebox.showerror("Импорт", "Ошибка импорта — см. лог.")

        self._start_job("Импорт DICOM", None, on_done=after, in_process=do_import)

    def _select_pid(self):
        pid = self.entry_pid.get().strip().upper()
        if not pid:
            messagebox.showerror("Ошибка", "Введите ID пациента (например P110).")
            return
        if not os.path.isdir(config.patient_dir(pid)):
            messagebox.showerror("Ошибка", f"Папка пациента не найдена: {config.patient_dir(pid)}")
            return
        self.patient_id = pid
        self.lbl_patient.configure(text=f"Пациент: {pid}", fg="black")

    def _build_run_section(self):
        frame = tk.LabelFrame(self.tab_work, text="2) Инференс + RTSTRUCT (настройки по умолчанию — на вкладке «Настройки»)", padx=10, pady=10)
        frame.pack(fill="x", padx=6, pady=4)

        self.btn_run = tk.Button(frame, text="Запустить", command=self._run_pipeline, width=16)
        self.btn_run.grid(row=0, column=0, padx=5, pady=2)

        self.var_crop = tk.BooleanVar(value=bool(config.CROP_AIR))
        tk.Checkbutton(frame, text="кроп воздуха", variable=self.var_crop).grid(row=0, column=1, sticky="w")
        self.var_async = tk.BooleanVar(value=int(config.N_ASYNC_REQUESTS) > 1)
        tk.Checkbutton(frame, text="async-конвейер", variable=self.var_async).grid(row=0, column=2, sticky="w")

        tk.Label(frame, text="Вариант:").grid(row=0, column=3, sticky="e")
        self.var_variant = tk.StringVar(value=str(config.DEFAULT_VARIANT))
        ttk.Combobox(frame, textvariable=self.var_variant, state="readonly",
                     values=["legacy", "aligned"], width=8).grid(row=0, column=4, padx=3)
        tk.Label(frame, text="Stride:").grid(row=0, column=5, sticky="e")
        self.var_stride = tk.StringVar(value=str(config.STEP_SIZE_RATIO))
        tk.Entry(frame, textvariable=self.var_stride, width=5).grid(row=0, column=6, padx=3)

        self.lbl_info = tk.Label(frame, text=f"Модель: {config.MODEL} · устройство: {str(config.DEVICE).upper()} · "
                                         f"накопитель: {str(config.ACCUMULATOR).lower()}" +
                                 ("" if _POST_OK else " · постобработка: недоступна в exe"),
                 fg="gray")
        self.lbl_info.grid(row=1, column=0, columnspan=7, sticky="w", pady=(4, 0))

        row_tpl = tk.Frame(frame)
        row_tpl.grid(row=2, column=0, columnspan=7, sticky="w", pady=(2, 0))
        if _POST_OK:
            import postprocess_core as ppcore
            tk.Label(row_tpl, text="Постобработка:").pack(side="left")
            self.var_post_tpl = tk.StringVar(value="")
            self.cmb_tpl = ttk.Combobox(row_tpl, textvariable=self.var_post_tpl,
                                        values=[""] + ppcore.list_templates(),
                                        state="readonly", width=22)
            self.cmb_tpl.pack(side="left", padx=4)
            tk.Label(row_tpl, text="(шаблоны из вкладки «Постпроцессинг»)",
                     fg="gray").pack(side="left")

    def _run_pipeline(self):
        if not self.patient_id:
            messagebox.showerror("Ошибка", "Импортируйте DICOM или выберите ID пациента.")
            return
        cmd = self._python_cmd("pipeline.py", self.patient_id, "--no-export",
                               "--variant", self.var_variant.get())
        try:
            stride = float(self.var_stride.get())
            if stride != config.STEP_SIZE_RATIO:
                cmd += ["--stride", str(stride)]
        except ValueError:
            messagebox.showerror("Ошибка", f"Stride должен быть числом, получено: {self.var_stride.get()!r}")
            return
        if not self.var_crop.get():
            cmd.append("--no-crop")
        if not self.var_async.get():
            cmd.append("--no-async")
        if _POST_OK and getattr(self, "var_post_tpl", None):
            tpl = self.var_post_tpl.get().strip()
            if tpl:
                import postprocess_core as ppcore
                cmd += ["--postprocess-template", ppcore.template_path(tpl)]

        def after(ok, title):
            if ok:
                messagebox.showinfo("Готово",
                                    f"RTSTRUCT для {self.patient_id} создан.\n"
                                    f"Экспортируйте его в папку DICOM (шаг 4).")
            else:
                messagebox.showerror("Ошибка", "Пайплайн завершился с ошибкой — см. лог.")

        self._start_job(f"Пайплайн {self.patient_id} ({self.var_variant.get()})",
                        [cmd], on_done=after)

    def _build_watch_section(self):
        frame = tk.LabelFrame(self.tab_work, text="3) Watch-папка (авто-обработка подпапок с DICOM)", padx=10, pady=10)
        frame.pack(fill="x", padx=6, pady=4)

        tk.Label(frame, text="Папка:").grid(row=0, column=0, sticky="w")
        self.lbl_watch_dir = tk.Label(frame, text=config.WATCH_DIR)
        self.lbl_watch_dir.grid(row=0, column=1, sticky="w", padx=5)
        self.btn_watch_start = tk.Button(frame, text="Запустить", command=self._watch_start, width=12)
        self.btn_watch_start.grid(row=0, column=2, padx=5)
        self.btn_watch_stop = tk.Button(frame, text="Остановить", command=self._watch_stop,
                                        width=12, state="disabled")
        self.btn_watch_stop.grid(row=0, column=3, padx=5)
        self.lbl_watch_status = tk.Label(frame, text="watch-папка: не запущена", fg="gray")
        self.lbl_watch_status.grid(row=0, column=4, sticky="w", padx=10)
        frame.columnconfigure(1, weight=1)

    def _watch_start(self):
        if self.watch_proc is not None and self.watch_proc.poll() is None:
            return
        try:
            cmd = self._python_cmd("watch_folder.py")
        except RuntimeError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self._log(f"\n>>> watch-папка: {config.WATCH_DIR}\n$ " + " ".join(cmd) + "\n")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.watch_proc = subprocess.Popen(
            cmd, cwd=SCRIPTS_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._stream_watch, daemon=True).start()
        self.lbl_watch_status.configure(text="watch-папка: работает (опрос каждые "
                                             f"{config.WATCH_POLL_SECONDS} с)", fg="green")
        self.btn_watch_start.configure(state="disabled")
        self.btn_watch_stop.configure(state="normal")

    def _stream_watch(self):
        proc = self.watch_proc
        try:
            for line in proc.stdout:
                self.log_queue.put(line)
        except Exception:
            pass

    def _watch_stop(self):
        if self.watch_proc is None:
            return
        try:
            self.watch_proc.terminate()
            self.watch_proc.wait(timeout=10)
        except Exception:
            try:
                self.watch_proc.kill()
            except Exception:
                pass
        self._log("<<< watch-папка: остановлена пользователем\n")

    def _build_export_section(self):
        frame = tk.LabelFrame(self.tab_work, text="4) Экспорт RTSTRUCT в исходную папку", padx=10, pady=10)
        frame.pack(fill="x", padx=6, pady=4)

        tk.Label(frame, text="Папка DICOM:").grid(row=0, column=0, sticky="w")
        self.entry_export = tk.Entry(frame, width=66)
        self.entry_export.grid(row=0, column=1, padx=5, pady=2, sticky="we")
        tk.Button(frame, text="Обзор...", command=lambda: self._browse("entry_export", "Куда положить RTSTRUCT")
                  ).grid(row=0, column=2, padx=5)
        self.btn_export = tk.Button(frame, text="Экспортировать", command=self._export)
        self.btn_export.grid(row=0, column=3, padx=5)
        frame.columnconfigure(1, weight=1)

    def _export(self):
        if not self.patient_id:
            messagebox.showerror("Ошибка", "Сначала выберите пациента.")
            return
        folder = self.entry_export.get().strip()
        if not folder:
            messagebox.showerror("Ошибка", "Укажите папку для экспорта.")
            return
        cmd = self._python_cmd("export_rtstruct.py", self.patient_id, folder)

        def after(ok, title):
            if ok:
                messagebox.showinfo("Экспорт",
                                    f"RTSTRUCT для {self.patient_id} выгружен в:\n"
                                    f"{os.path.join(folder, 'rtstruct')}")
            else:
                messagebox.showerror("Экспорт", "Ошибка экспорта — см. лог.")

        self._start_job(f"Экспорт {self.patient_id}", [cmd], on_done=after)

    # ================= ВКЛАДКА «КВАНТИЗАЦИЯ» =================

    def _build_quant_tab(self):
        f = self.tab_quant

        tk.Label(f, text="Создание новой (квантованной) версии модели из имеющейся.\n"
                         f"Папка моделей: {config.MODELS_ROOT}\n"
                         "Источник только читается; результат — новая подпапка, доступная для инференса.",
                 fg="gray", justify="left").pack(anchor="w", pady=(0, 8))

        box = tk.LabelFrame(f, text="Параметры квантизации", padx=10, pady=8)
        box.pack(fill="x")

        tk.Label(box, text="Модель-источник:").grid(row=0, column=0, sticky="w")
        self.var_quant_source = tk.StringVar(
            value=os.path.join(config.MODELS_ROOT, config.MODEL))
        tk.Entry(box, textvariable=self.var_quant_source, width=60).grid(row=0, column=1, sticky="we", padx=4, pady=2)
        tk.Button(box, text="Обзор...", command=lambda: self._browse_var(self.var_quant_source, "Папка модели-источника")).grid(row=0, column=2, padx=4)

        tk.Label(box, text="Имя новой модели:").grid(row=1, column=0, sticky="w")
        self.var_quant_name = tk.StringVar(value="")
        tk.Entry(box, textvariable=self.var_quant_name, width=30).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        tk.Label(box, text="(латиницей, напр. int8_exp1)", fg="gray").grid(row=1, column=2, sticky="w")

        tk.Label(box, text="Режим квантизации:").grid(row=2, column=0, sticky="w")
        self.var_quant_mode = tk.StringVar(value="int8")
        ttk.Combobox(box, textvariable=self.var_quant_mode, state="readonly",
                     values=["int8", "int4", "int8_weights", "int4_weights"], width=14).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        tk.Label(box, text="int8 — полное PTQ-квантизация с калибровкой (как рабочая модель — рекомендую);\n"
                           "int4 — PTQ с 4-битными весами и калибровкой (эксперимент: качество\n"
                           "просядет заметнее, ускорения на CPU не будет); биты: --weights-bits/--act-bits;\n"
                           "int8_weights / int4_weights — сжатие только весов, без калибровки\n"
                           "(для этой UNet-модели int4_weights фактически = int8_weights)",
                 fg="gray", justify="left").grid(row=3, column=1, columnspan=2, sticky="w")
        box.columnconfigure(1, weight=1)

        btns = tk.Frame(f)
        btns.pack(fill="x", pady=8)
        tk.Button(btns, text="Квантизовать", command=self._run_quantize, width=18).pack(side="left", padx=4)
        tk.Button(btns, text="Список моделей", command=self._quant_list, width=16).pack(side="left", padx=4)
        tk.Label(btns, text="После квантизации выберите новую модель на вкладке «Настройки»",
                 fg="gray").pack(side="left", padx=10)

        # --- Проверка качества/скорости против GT ---
        bench = tk.LabelFrame(f, text="Проверка Dice против GT (баланс качество/скорость)", padx=10, pady=8)
        bench.pack(fill="x", pady=(10, 0))
        tk.Label(bench, text="Прогоняет случаи из валидационного набора через обе модели, считает Dice\n"
                             "по органам против GT и время инференса. Полный прогон — ~10-15 мин\n"
                             "на случай × модель; для быстрой проверки укажите фолды = 0.",
                 fg="gray", justify="left").grid(row=0, column=0, columnspan=4, sticky="w")

        self.bench_a_combo = ttk.Combobox(bench, state="readonly", width=20,
                                          values=config.available_models() or ["original"])
        self.bench_a_combo.set("original" if "original" in config.available_models() else str(config.MODEL))
        self.bench_b_combo = ttk.Combobox(bench, state="readonly", width=20,
                                          values=config.available_models() or [str(config.MODEL)])
        self.bench_b_combo.set(str(config.MODEL))

        tk.Label(bench, text="Модель A (эталон):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.bench_a_combo.grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))
        tk.Label(bench, text="Модель B (сравниваемая):").grid(row=2, column=0, sticky="w")
        self.bench_b_combo.grid(row=2, column=1, sticky="w", padx=4)

        tk.Label(bench, text="Случаев:").grid(row=1, column=2, sticky="e", pady=(6, 0))
        self.var_bench_n = tk.StringVar(value="2")
        tk.Entry(bench, textvariable=self.var_bench_n, width=4).grid(row=1, column=3, sticky="w", pady=(6, 0))
        tk.Label(bench, text="Фолды:").grid(row=2, column=2, sticky="e")
        self.var_bench_folds = tk.StringVar(value="")
        tk.Entry(bench, textvariable=self.var_bench_folds, width=4).grid(row=2, column=3, sticky="w")
        tk.Label(bench, text="(пусто = все)", fg="gray").grid(row=3, column=3, sticky="w")

        tk.Button(bench, text="Запустить сравнение", command=self._run_benchmark,
                  width=20).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        bench.columnconfigure(1, weight=1)

    def _browse_var(self, var, title):
        folder = filedialog.askdirectory(title=title, initialdir=var.get() or config.MODELS_ROOT)
        if folder:
            var.set(folder)

    def _run_quantize(self):
        source = self.var_quant_source.get().strip()
        name = self.var_quant_name.get().strip()
        if not source or not os.path.isdir(source):
            messagebox.showerror("Ошибка", f"Папка источника не найдена:\n{source}")
            return
        if not name or not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            messagebox.showerror("Ошибка", "Имя новой модели: латиница/цифры/_/- (напр. int8_exp1)")
            return
        cmd = self._python_cmd(os.path.join("quantization", "quantize_ov.py"),
                               "--source", source, "--name", name,
                               "--mode", self.var_quant_mode.get())

        def after(ok, title):
            if ok:
                self._refresh_model_combo()
                messagebox.showinfo(
                    "Квантизация",
                    f"Модель {name!r} создана.\n"
                    f"Выберите её на вкладке «Настройки» и сравните качество:\n"
                    f"python compare_dice.py <seg> <seg_новой_модели>")
            else:
                messagebox.showerror("Квантизация", "Ошибка квантизации — см. лог.")

        self._start_job(f"Квантизация {name} ({self.var_quant_mode.get()})", [cmd], on_done=after)

    def _run_benchmark(self):
        try:
            n = int(self.var_bench_n.get())
        except ValueError:
            messagebox.showerror("Ошибка", f"Случаев должно быть числом: {self.var_bench_n.get()!r}")
            return
        models = f"{self.bench_a_combo.get()},{self.bench_b_combo.get()}"
        cmd = self._python_cmd(os.path.join("quantization", "benchmark_dice.py"),
                               "--models", models, "--n", str(n))
        if self.var_bench_folds.get().strip():
            cmd += ["--folds", self.var_bench_folds.get().strip()]

        def after(ok, title):
            if ok:
                messagebox.showinfo(
                    "Бенчмарк",
                    f"Сравнение готово.\nТаблица и CSV: {os.path.join(config.LOG_DIR, 'benchmark')}\n"
                    f"(полные цифры — в логе ниже)")
            else:
                messagebox.showerror("Бенчмарк", "Ошибка бенчмарка — см. лог.")

        self._start_job(f"Бенчмарк Dice: {models} × {n} случ.", [cmd], on_done=after)

    def _quant_list(self):
        try:
            cmd = self._python_cmd(os.path.join("quantization", "quantize_ov.py"), "--list")
        except RuntimeError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self._start_job("Список моделей", [cmd])

    # ================= ВКЛАДКА «НАСТРОЙКИ» =================

    def _build_settings_tab(self):
        f = self.tab_settings

        tk.Label(f, text=f"Файл настроек: {config.SETTINGS_PATH or 'settings.ini (не найден — будет создан)'}\n"
                         f"Изменения применяются к НОВЫМ запускам; комментарии в файле сохраняются.",
                 fg="gray", justify="left").pack(anchor="w")

        body = tk.Frame(f)
        body.pack(fill="both", expand=True)

        # реестр: (секция, ключ) -> (переменная, тип)
        self.sv = {}

        def row(parent, r, label, section, key, value, kind="str", width=14, choices=None):
            tk.Label(parent, text=label).grid(row=r, column=1, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(value))
            if choices:
                w = ttk.Combobox(parent, textvariable=var, values=choices, state="readonly", width=width)
            else:
                w = tk.Entry(parent, textvariable=var, width=width)
            w.grid(row=r, column=2, sticky="w", padx=4, pady=2)
            self.sv[(section, key)] = (var, kind)

        # --- устройство ---
        box1 = tk.LabelFrame(body, text="Устройство", padx=8, pady=6)
        box1.grid(row=0, column=0, sticky="nw", padx=6, pady=4)
        row(box1, 0, "Устройство (cpu / gpu)", "device", "DEVICE", str(config.DEVICE).lower(),
            choices=["cpu", "gpu", "gpu.0"])
        row(box1, 1, "Потоков CPU", "device", "NUM_THREADS", config.NUM_THREADS, kind="int")
        row(box1, 2, "Async-запросов", "device", "N_ASYNC_REQUESTS", config.N_ASYNC_REQUESTS, kind="int")
        row(box1, 3, "PERFORMANCE_HINT", "device", "PERFORMANCE_HINT", config.OV_PERFORMANCE_HINT,
            choices=["LATENCY", "THROUGHPUT", "none"])

        # --- инференс ---
        box2 = tk.LabelFrame(body, text="Инференс", padx=8, pady=6)
        box2.grid(row=0, column=1, sticky="nw", padx=6, pady=4)
        row(box2, 0, "Вариант (legacy / aligned)", "inference", "VARIANT", config.DEFAULT_VARIANT,
            choices=["legacy", "aligned"])
        row(box2, 1, "Фолды (пусто = все)", "inference", "FOLDS", ",".join(map(str, config.DEFAULT_FOLDS)))
        row(box2, 2, "Шаг окна (stride)", "inference", "STEP_SIZE_RATIO", config.STEP_SIZE_RATIO, kind="float")
        row(box2, 3, "Кроп воздуха", "inference", "CROP_AIR", "true" if config.CROP_AIR else "false",
            kind="bool", choices=["true", "false"])
        row(box2, 4, "Накопитель логитов", "inference", "ACCUMULATOR", config.ACCUMULATOR,
            choices=["disk", "ram"])

        # --- модель ---
        box4 = tk.LabelFrame(body, text="Модель для инференса")
        box4.grid(row=1, column=1, sticky="nw", padx=6, pady=4)
        tk.Label(box4, text="Анатомическая модель\n(папки patients/validation/models)").grid(row=0, column=0, sticky="w", padx=4)
        self.model_name_var = tk.StringVar(value=str(config.MODEL_NAME))
        tk.Entry(box4, textvariable=self.model_name_var, width=20).grid(row=0, column=1, padx=4)
        self.sv[("model", "MODEL_NAME")] = (self.model_name_var, "str")
        tk.Label(box4, text="Квантизация:").grid(row=1, column=0, sticky="w", padx=4)
        self.model_combo = ttk.Combobox(box4, state="readonly", width=24,
                                        values=config.available_models())
        self.model_combo.set(str(config.MODEL))
        self.model_combo.grid(row=1, column=1, sticky="w", padx=4)
        self.sv[("model", "MODEL")] = (self.model_combo, "str")
        tk.Button(box4, text="Обновить список", width=14,
                  command=self._refresh_model_combo).grid(row=2, column=0, columnspan=2, padx=4, pady=2)
        tk.Label(box4, text="int8 — рабочая; original — эталон качества (медленно);\n"
                            "новые — после квантизации",
                 fg="gray", justify="left").grid(row=3, column=0, columnspan=2, sticky="w", padx=4)

        # --- пути и watch ---
        box3 = tk.LabelFrame(body, text="Пути и watch-папка", padx=8, pady=6)
        box3.grid(row=1, column=0, sticky="nw", padx=6, pady=4)
        for i, (label, section, key, value) in enumerate([
            ("Пациенты (пусто = папка проекта\\patients)", "paths", "PATIENTS_ROOT", config.PATIENTS_ROOT),
            ("Логи (пусто = папка проекта\\logs)", "paths", "LOG_DIR", config.LOG_DIR),
            ("Watch-папка (пусто = папка проекта\\inbox)", "paths", "WATCH_DIR", config.WATCH_DIR),
            ("Опрос watch, секунд", "paths", "WATCH_POLL_SECONDS", config.WATCH_POLL_SECONDS),
            ("Python окружения (пусто = автопоиск)", "gui", "CONDA_PYTHON", config.CONDA_PYTHON),
        ]):
            tk.Label(box3, text=label).grid(row=i, column=1, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(value))
            tk.Entry(box3, textvariable=var, width=58).grid(row=i, column=2, sticky="we", padx=4, pady=2)
            self.sv[(section, key)] = (var, "str")
        box3.columnconfigure(2, weight=1)

        # --- папки по умолчанию для главного окна ---
        box5 = tk.LabelFrame(body, text="Папки по умолчанию в окне «Работа»", padx=8, pady=6)
        box5.grid(row=2, column=0, columnspan=2, sticky="we", padx=6, pady=4)
        for i, (label, key, value) in enumerate([
            ("Импорт DICOM (при старте GUI)", "DEFAULT_IMPORT_DIR", config.DEFAULT_IMPORT_DIR),
            ("Экспорт RTSTRUCT (при старте GUI)", "DEFAULT_EXPORT_DIR", config.DEFAULT_EXPORT_DIR),
        ]):
            tk.Label(box5, text=label).grid(row=i, column=1, sticky="w", padx=4, pady=2)
            var = tk.StringVar(value=str(value))
            tk.Entry(box5, textvariable=var, width=58).grid(row=i, column=2, sticky="we", padx=4, pady=2)
            tk.Button(box5, text="Обзор...",
                      command=lambda v=var: self._browse_var(v, "Выберите папку")).grid(row=i, column=3, padx=4)
            self.sv[("paths", key)] = (var, "str")
        box5.columnconfigure(2, weight=1)

        tk.Label(body, text="Порог/запас кропа и пути к модели — только в settings.ini (раздел [inference]/[model]).",
                 fg="gray").grid(row=2, column=0, sticky="w", padx=6)

        btns = tk.Frame(f)
        btns.pack(fill="x", pady=8)
        tk.Button(btns, text="Сохранить настройки", command=self._save_settings, width=24).pack(side="left", padx=4)
        tk.Button(btns, text="Проверить устройства (check_devices)", command=self._check_devices
                  ).pack(side="left", padx=4)

    def _refresh_model_combo(self):
        models = config.available_models()
        self.model_combo.configure(values=models or [str(config.MODEL)])
        if config.MODEL in models:
            self.model_combo.set(str(config.MODEL))
        for combo in (self.bench_a_combo, self.bench_b_combo):
            combo.configure(values=models or [str(config.MODEL)])
        self._log(f"[GUI] Модели в {config.MODELS_ROOT}: {', '.join(models) or 'нет'}\n")

    def _save_settings(self):
        # эти ключи МОЖНО сохранять пустыми — пусто = «папка проекта»
        # (портабельность: путь подстроится под то, куда скопировали проект)
        allow_empty = {("paths", "PATIENTS_ROOT"), ("paths", "LOG_DIR"), ("paths", "WATCH_DIR"),
                       ("gui", "CONDA_PYTHON"), ("hardware", "FINGERPRINT")}
        updates = {}
        for (section, key), (var, kind) in self.sv.items():
            val = var.get().strip()
            try:
                if kind == "int":
                    updates[(section, key)] = str(int(val))
                elif kind == "float":
                    updates[(section, key)] = str(float(val))
                elif kind == "bool":
                    updates[(section, key)] = "true" if val.lower() in ("true", "1", "yes") else "false"
                else:
                    if val == "" and (section, key) not in allow_empty:
                        continue
                    updates[(section, key)] = val
            except ValueError:
                messagebox.showerror("Ошибка", f"Некорректное значение {key}: {val!r}")
                return
        # спец-обработка фолдов: пустое поле = все фолды (в ini пишем пусто)
        if ("inference", "FOLDS") in self.sv and self.sv[("inference", "FOLDS")][0].get().strip() == "":
            updates[("inference", "FOLDS")] = ""

        try:
            path = config.write_settings(updates)
        except OSError as e:
            messagebox.showerror("Ошибка", f"Не удалось записать settings.ini: {e}")
            return
        self._log(f"[GUI] Настройки сохранены: {path}\n")
        for (sec, key), v in sorted(updates.items()):
            self._log(f"  [{sec}] {key} = {v}\n")
        self._log("[GUI] Применяются к новым запускам (watch-папку перезапустите, если работала).\n")
        # обновляем сводку на вкладке «Работа»
        self.lbl_watch_dir.configure(text=self.sv[("paths", "WATCH_DIR")][0].get())
        self.lbl_info.configure(
            text=f"Модель: {self.sv[('model', 'MODEL')][0].get()} · "
                 f"устройство: {str(config.DEVICE).upper()} · "
                 f"накопитель: {str(config.ACCUMULATOR).lower()}")
        messagebox.showinfo("Настройки", f"Сохранено в:\n{path}")

    def _check_devices(self):
        try:
            cmd = self._python_cmd("check_devices.py")
        except RuntimeError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self._start_job("Проверка устройств", [cmd])

    # ---------- Лог ----------
    def _build_log_section(self):
        self.log_frame = tk.LabelFrame(self, text="Лог", padx=5, pady=5)
        self.log_text = tk.Text(self.log_frame, height=10, wrap="word")
        scroll = tk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")


if __name__ == "__main__":
    app = App()
    app.mainloop()
