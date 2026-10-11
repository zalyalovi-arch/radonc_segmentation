# ============================================================
# main_window.py — логика главного окна (PySide6 + QUiLoader)
# ------------------------------------------------------------
# GUI описан в gui/ui/main_window.ui — правится в Qt Designer
# (запуск: designer.bat). Здесь только бэкенд: запуск скриптов
# пайплайна в рабочем потоке, watch-папка, редактор settings.ini,
# проверка окружения. Вкладка «Постпроцессинг» — gui/postprocess_tab.py.
#
# Запуск: run_gui.bat  (python окружения nnunet_env — ему же
# запускаются все скрипты пайплайна, отдельный CONDA_PYTHON не нужен).
# ============================================================
import os
import re
import subprocess
import sys
import threading

from PySide6.QtCore import QFile, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QComboBox, QFileDialog, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPushButton,
                               QWidget)

from radonc_segmentation import config
from gui import PROJECT_ROOT, PKG_DIR, UI_DIR, child_env
from gui.workers import JobWorker

_IS_FROZEN = getattr(sys, "frozen", False)
if _IS_FROZEN:
    # в exe скрипты запускаются python'ом окружения nnunet_env
    # (сам exe python'ом не является) — как в tkinter-версии
    if os.path.isfile(config.CONDA_PYTHON or ""):
        PYTHON_EXE = config.CONDA_PYTHON
    else:
        PYTHON_EXE = config.find_conda_python()
else:
    PYTHON_EXE = sys.executable  # dev-запуск из окружения nnunet_env
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# вкладка «Постпроцессинг» требует numpy/SimpleITK/matplotlib
try:
    from gui.postprocess_tab import PostprocessTab
    from radonc_segmentation import postprocess_core as ppcore
    _POST_OK = True
except Exception as _post_err:
    _POST_OK = False
    _POST_ERR = _post_err


class MainWindow(QMainWindow):
    # сигналы для потокобезопасного обновления GUI из потока-читальника
    # watch-процесса (строка лога / завершение watch)
    _log_signal = Signal(str)
    _watch_done_signal = Signal()

    def __init__(self):
        super().__init__()
        self._load_ui()

        self.worker = None
        self.job_thread = None
        self.on_job_done_cb = None
        self.patient_id = None
        self.watch_proc = None
        self.sv = {}  # (секция, ключ) -> (виджет, тип) — реестр полей настроек

        self._log_signal.connect(self._log)
        self._watch_done_signal.connect(self._watch_finished)

        # явная инициализация конфигурации (автоподстройка, поиск python)
        try:
            config.init_config(log=self._log)
        except config.ConfigError as e:
            self._log(f"[GUI] Ошибка инициализации конфигурации: {e}\n")

        self._setup_post_tab()
        self._tune_layout()
        self._build_settings_fields()
        self._fill_from_config()
        self._connect_signals()
        self._startup_checks()

    # ================= загрузка UI =================

    def _load_ui(self):
        """Загружает .ui через QUiLoader и маппит виджеты по objectName."""
        loader = QUiLoader()
        ui_path = os.path.join(UI_DIR, "main_window.ui")
        ui_file = QFile(ui_path)
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError(f"Не удалось открыть {ui_path}")
        try:
            loader.load(ui_file, self)
        finally:
            ui_file.close()

        # QUiLoader создаёт форму СКРЫТОЙ (WA_WState_Hidden) — win.show()
        # скрытых детей не показывает, окно остаётся пустым без явного show
        cw = self.centralWidget()
        if cw is not None:
            cw.setVisible(True)

        # Кэш виджетов по objectName для быстрого доступа
        self._ui = {}
        for child in self.findChildren(QWidget):
            if child.objectName():
                self._ui[child.objectName()] = child

        # Удобные алиасы для часто используемых виджетов
        self._alias_widgets()

    def _alias_widgets(self):
        """Создаёт атрибуты self.widget_name для совместимости со старым кодом."""
        # Вкладка «Работа»
        self.tabWidget = self._ui_get("tabWidget")
        self.splitterMain = self._ui_get("splitterMain")
        self.groupLog = self._ui_get("groupLog")
        self.logText = self._ui_get("logText")
        self.entryDicom = self._ui_get("entryDicom")
        self.btnBrowseDicom = self._ui_get("btnBrowseDicom")
        self.btnImport = self._ui_get("btnImport")
        self.entryPid = self._ui_get("entryPid")
        self.btnSelectPid = self._ui_get("btnSelectPid")
        self.lblPatient = self._ui_get("lblPatient")
        self.btnRun = self._ui_get("btnRun")
        self.chkCrop = self._ui_get("chkCrop")
        self.chkAsync = self._ui_get("chkAsync")
        self.cmbVariant = self._ui_get("cmbVariant")
        self.entryStride = self._ui_get("entryStride")
        self.lblInfo = self._ui_get("lblInfo")
        self.rowPost = self._ui_get("rowPost")
        self.cmbPostTpl = self._ui_get("cmbPostTpl")
        self.lblWatchDir = self._ui_get("lblWatchDir")
        self.btnWatchStart = self._ui_get("btnWatchStart")
        self.btnWatchStop = self._ui_get("btnWatchStop")
        self.lblWatchStatus = self._ui_get("lblWatchStatus")
        self.entryExport = self._ui_get("entryExport")
        self.btnBrowseExport = self._ui_get("btnBrowseExport")
        self.btnExport = self._ui_get("btnExport")
        self.entryQuantSource = self._ui_get("entryQuantSource")
        self.btnBrowseQuantSource = self._ui_get("btnBrowseQuantSource")
        self.entryQuantName = self._ui_get("entryQuantName")
        self.cmbQuantMode = self._ui_get("cmbQuantMode")
        self.cmbBenchA = self._ui_get("cmbBenchA")
        self.cmbBenchB = self._ui_get("cmbBenchB")
        self.entryBenchN = self._ui_get("entryBenchN")
        self.entryBenchFolds = self._ui_get("entryBenchFolds")
        self.btnBenchmark = self._ui_get("btnBenchmark")
        self.btnQuantize = self._ui_get("btnQuantize")
        self.btnQuantList = self._ui_get("btnQuantList")
        # группа импорта обученной модели (.pth -> IR) и теста на валидации
        self.entryImportSrc = self._ui_get("entryImportSrc")
        self.btnBrowseImportSrc = self._ui_get("btnBrowseImportSrc")
        self.entryImportPlans = self._ui_get("entryImportPlans")
        self.btnBrowseImportPlans = self._ui_get("btnBrowseImportPlans")
        self.entryImportName = self._ui_get("entryImportName")
        self.entryImportFolds = self._ui_get("entryImportFolds")
        self.btnImportModel = self._ui_get("btnImportModel")
        self.cmbTestModel = self._ui_get("cmbTestModel")
        self.cmbTestRef = self._ui_get("cmbTestRef")
        self.cmbTestCase = self._ui_get("cmbTestCase")
        self.entryTestFolds = self._ui_get("entryTestFolds")
        self.btnTestModel = self._ui_get("btnTestModel")
        # Группы настроек (динамические строки создаются кодом)
        self.groupDevice = self._ui_get("groupDevice")
        self.groupInference = self._ui_get("groupInference")
        self.groupPaths = self._ui_get("groupPaths")
        self.groupModel = self._ui_get("groupModel")
        self.groupDefaults = self._ui_get("groupDefaults")
        # layouts внутри групп — получаем через .layout()
        self.gridDevice = self.groupDevice.layout()
        self.gridInference = self.groupInference.layout()
        self.gridPaths = self.groupPaths.layout()
        self.gridModel = self.groupModel.layout()
        self.gridDefaults = self.groupDefaults.layout()
        self.lblSettingsIntro = self._ui_get("lblSettingsIntro")
        self.lblQuantIntro = self._ui_get("lblQuantIntro")
        self.btnSaveSettings = self._ui_get("btnSaveSettings")
        self.btnCheckDevices = self._ui_get("btnCheckDevices")
        # Вкладки
        self.tabWork = self._ui_get("tabWork")
        self.tabQuant = self._ui_get("tabQuant")
        self.tabPost = self._ui_get("tabPost")
        self.tabSettings = self._ui_get("tabSettings")
        # Страница постпроцессинга (layPostPage — layout внутри tabPost)
        self.layPostPage = self.tabPost.layout()

    def _ui_get(self, name):
        """Получить виджет по objectName с понятной ошибкой."""
        w = self._ui.get(name)
        if w is None:
            raise RuntimeError(f"Виджет с objectName='{name}' не найден в main_window.ui")
        return w

    def _fill_from_config(self):
        if config.DEFAULT_IMPORT_DIR:
            self.entryDicom.setText(config.DEFAULT_IMPORT_DIR)
        if config.DEFAULT_EXPORT_DIR:
            self.entryExport.setText(config.DEFAULT_EXPORT_DIR)
        self.chkCrop.setChecked(bool(config.CROP_AIR))
        self.chkAsync.setChecked(int(config.N_ASYNC_REQUESTS) > 1)
        self.cmbVariant.setCurrentText(str(config.DEFAULT_VARIANT))
        self.entryStride.setText(str(config.STEP_SIZE_RATIO))
        self.lblWatchDir.setText(config.WATCH_DIR)
        self.lblQuantIntro.setText(
            f"Полный цикл модели: импорт чекпоинта (.pth) → тест на валидации → квантизация → бенчмарк.\n"
            f"Папка моделей: {config.MODELS_ROOT}\n"
            "Активация — вкладка «Настройки» (MODEL); визуальная проверка — «Постпроцессинг».")
        self.entryQuantSource.setText(os.path.join(config.MODELS_ROOT, config.MODEL))
        self._update_lbl_info()
        self.lblSettingsIntro.setText(
            f"Файл настроек: {config.SETTINGS_PATH or 'settings.ini (не найден — будет создан)'}\n"
            "Изменения применяются к НОВЫМ запускам; комментарии в файле сохраняются.")
        models = config.available_models()
        for combo in (self.cmbBenchA, self.cmbBenchB):
            combo.addItems(models or ["original"])
        self.cmbBenchA.setCurrentText("original" if "original" in models else str(config.MODEL))
        self.cmbBenchB.setCurrentText(str(config.MODEL))
        self._refresh_test_combos()
        self.sv[("model", "MODEL")][0].setCurrentText(str(config.MODEL))
        if _POST_OK:
            self.cmbPostTpl.setPlaceholderText("— без постобработки —")
            self.cmbPostTpl.addItems(ppcore.list_templates())

    def _update_lbl_info(self):
        self.lblInfo.setText(
            f"Модель: {config.MODEL} · устройство: {str(config.DEVICE).upper()} · "
            f"накопитель: {str(config.ACCUMULATOR).lower()}"
            + ("" if _POST_OK else " · постобработка: недоступна"))

    # ---------- служебное ----------

    def _log(self, s):
        self.logText.moveCursor(QTextCursor.End)
        self.logText.insertPlainText(s)
        self.logText.ensureCursorVisible()

    def _browse(self, line_edit, title, initial=None):
        folder = QFileDialog.getExistingDirectory(self, title, initial or "")
        if folder:
            line_edit.setText(folder)

    def _browse_file(self, line_edit, title, initial=None, flt=None):
        f, _flt = QFileDialog.getOpenFileName(self, title, initial or "", flt or "")
        if f:
            line_edit.setText(f)

    def _set_buttons(self, running):
        state = not running
        for b in (self.btnImport, self.btnRun, self.btnExport, self.btnSelectPid):
            b.setEnabled(state)

    def _busy(self):
        return self.job_thread is not None and self.job_thread.isRunning()

    def _python_cmd(self, module, *args):
        if PYTHON_EXE is None:
            raise RuntimeError(
                "Python окружения nnunet_env не найден. Настройте компьютер:\n"
                "  python setup_env.py\n"
                "или укажите путь в settings.ini ([gui] CONDA_PYTHON).")
        return [PYTHON_EXE, "-m", module, *args]

    # ---------- фоновые задачи ----------

    def _start_job(self, title, commands=None, on_done=None, in_process=None):
        if self._busy():
            QMessageBox.warning(self, "Занято", "Дождитесь завершения текущей операции.")
            return
        self._set_buttons(running=True)
        self._log(f"\n>>> {title}\n")
        self.on_job_done_cb = on_done
        self.job_thread = QThread(self)
        self.worker = JobWorker(title, commands=commands, in_process=in_process,
                                cwd=PROJECT_ROOT, env=child_env())
        self.worker.moveToThread(self.job_thread)
        self.worker.line.connect(self._log)
        self.worker.finished_job.connect(self._on_job_finished)
        self.job_thread.started.connect(self.worker.run)
        self.job_thread.start()

    def _on_job_finished(self, title, rc):
        ok = rc == 0
        self._set_buttons(running=False)
        self._log(f"<<< {title}: {'ГОТОВО' if ok else f'ОШИБКА (код {rc})'}\n")
        self.job_thread.quit()
        self.job_thread.wait(20000)
        self.job_thread = None
        self.worker = None
        cb = self.on_job_done_cb
        self.on_job_done_cb = None
        if cb:
            cb(ok, title)

    # ================= инициализация =================

    def _setup_post_tab(self):
        global _POST_OK
        if _POST_OK:
            self.post_tab = PostprocessTab(log=self._log, start_job=self._start_job)
            self.layPostPage.addWidget(self.post_tab)
        else:
            self.tabWidget.removeTab(self.tabWidget.indexOf(self.tabPost))
            self.rowPost.setVisible(False)
            self.lblInfo.setText(f"Модель: {config.MODEL} · постпроцессинг недоступен: {_POST_ERR}")

    def _tune_layout(self):
        # лог снизу фиксированной высоты, вкладки тянутся
        self.splitterMain.setStretchFactor(0, 1)
        self.splitterMain.setStretchFactor(1, 0)
        self.splitterMain.setSizes([620, 180])
        self.btnWatchStop.setEnabled(False)

    def _connect_signals(self):
        # вкладка «Работа»
        self.btnBrowseDicom.clicked.connect(lambda: self._browse(self.entryDicom, "Выберите папку с DICOM"))
        self.btnImport.clicked.connect(self._import)
        self.btnSelectPid.clicked.connect(self._select_pid)
        self.btnRun.clicked.connect(self._run_pipeline)
        self.btnBrowseExport.clicked.connect(lambda: self._browse(self.entryExport, "Куда положить RTSTRUCT"))
        self.btnExport.clicked.connect(self._export)
        self.btnWatchStart.clicked.connect(self._watch_start)
        self.btnWatchStop.clicked.connect(self._watch_stop)
        # вкладка «Квантизация»
        self.btnBrowseQuantSource.clicked.connect(
            lambda: self._browse(self.entryQuantSource, "Папка модели-источника",
                                 initial=config.MODELS_ROOT))
        self.btnQuantize.clicked.connect(self._run_quantize)
        self.btnQuantList.clicked.connect(self._quant_list)
        self.btnBenchmark.clicked.connect(self._run_benchmark)
        self.btnBrowseImportSrc.clicked.connect(self._browse_import_src)
        self.btnBrowseImportPlans.clicked.connect(
            lambda: self._browse_file(self.entryImportPlans, "Файл plans.json",
                                      flt="JSON (*.json);;Все файлы (*.*)"))
        self.btnImportModel.clicked.connect(self._run_import_model)
        self.btnTestModel.clicked.connect(self._run_test_model)
        # вкладка «Настройки»
        self.btnSaveSettings.clicked.connect(self._save_settings)
        self.btnCheckDevices.clicked.connect(self._check_devices)


    # ---------- стартовые проверки ----------

    def _startup_checks(self):
        problems = []
        if not os.path.isfile(os.path.join(PKG_DIR, "pipeline.py")):
            problems.append(f"Не найден radonc_segmentation\\pipeline.py в {PKG_DIR}.\n"
                            "Положите exe в корень проекта (рядом с settings.ini).")
        if PYTHON_EXE is None:
            problems.append(
                "Не найден python окружения nnunet_env ни в одном типовом месте.\n"
                "Настройте компьютер одной командой:\n"
                "  python setup_env.py\n"
                "или укажите путь в settings.ini ([gui] CONDA_PYTHON).")
        if problems:
            QMessageBox.warning(self, "Проблема конфигурации", "\n\n".join(problems))

        if config.AUTOTUNE_NOTICE:
            self._log(f"[GUI] {config.AUTOTUNE_NOTICE}\n")
            QMessageBox.information(self, "Новый компьютер", config.AUTOTUNE_NOTICE)

        self._log(f"Пакет: {PKG_DIR}\nPython: {PYTHON_EXE}\n"
                  f"Настройки: {config.SETTINGS_PATH or 'settings.ini не найден (значения по умолчанию)'}\n")

        # зависимости окружения проверяем тем же python, которым будут
        # запускаться скрипты (в фоне — окно не блокируем)
        # Используем --fast для быстрой проверки только критических пакетов
        out_box = []

        def run_env_check(log):
            r = subprocess.run(
                [PYTHON_EXE, "-m", "radonc_segmentation.env_check", "--fast"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=300, creationflags=CREATE_NO_WINDOW, env=child_env())
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            if out:
                log(out + "\n")
                out_box.append(out)
            return r.returncode

        def after(ok, title):
            if not ok:
                QMessageBox.warning(
                    self, "Проверка окружения",
                    "Пакеты из requirements.txt установлены не полностью (см. лог):\n\n"
                    + (out_box[-1][-1800:] if out_box else "")
                    + "\n\nУстановка: python setup_env.py")
            else:
                self._log("[GUI] Окружение проверено: все пакеты requirements.txt на месте.\n")

        self._start_job("Проверка окружения", in_process=run_env_check, on_done=after)

    # ================= ВКЛАДКА «РАБОТА» =================

    def _import(self):
        folder = self.entryDicom.text().strip()
        if not folder:
            QMessageBox.critical(self, "Ошибка", "Выберите папку с DICOM.")
            return

        def do_import(log):
            from radonc_segmentation import import_dicom
            pid, base_dir = import_dicom.import_dicom(folder)
            self.patient_id = pid

        def after(ok, title):
            if ok:
                self.entryPid.setText(self.patient_id or "")
                self.lblPatient.setStyleSheet("")
                self.lblPatient.setText(f"Пациент: {self.patient_id}")
                QMessageBox.information(self, "Импорт", f"Импортирован пациент {self.patient_id}.")
            else:
                QMessageBox.critical(self, "Импорт", "Ошибка импорта — см. лог.")

        self._start_job("Импорт DICOM", in_process=do_import, on_done=after)

    def _select_pid(self):
        pid = self.entryPid.text().strip().upper()
        if not pid:
            QMessageBox.critical(self, "Ошибка", "Введите ID пациента (например P110).")
            return
        if not os.path.isdir(config.patient_dir(pid)):
            QMessageBox.critical(self, "Ошибка", f"Папка пациента не найдена: {config.patient_dir(pid)}")
            return
        self.patient_id = pid
        self.lblPatient.setStyleSheet("")
        self.lblPatient.setText(f"Пациент: {pid}")

    def _run_pipeline(self):
        if not self.patient_id:
            QMessageBox.critical(self, "Ошибка", "Импортируйте DICOM или выберите ID пациента.")
            return
        try:
            cmd = self._python_cmd("radonc_segmentation.pipeline", self.patient_id, "--no-export",
                                   "--variant", self.cmbVariant.currentText())
        except RuntimeError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return
        try:
            stride = float(self.entryStride.text())
            if stride != config.STEP_SIZE_RATIO:
                cmd += ["--stride", str(stride)]
        except ValueError:
            QMessageBox.critical(self, "Ошибка", f"Stride должен быть числом, получено: {self.entryStride.text()!r}")
            return
        if not self.chkCrop.isChecked():
            cmd.append("--no-crop")
        if not self.chkAsync.isChecked():
            cmd.append("--no-async")
        if _POST_OK:
            tpl = self.cmbPostTpl.currentText().strip()
            if tpl:
                cmd += ["--postprocess-template", ppcore.template_path(tpl)]

        def after(ok, title):
            if ok:
                QMessageBox.information(
                    self, "Готово",
                    f"RTSTRUCT для {self.patient_id} создан.\n"
                    f"Экспортируйте его в папку DICOM (шаг 4).")
            else:
                QMessageBox.critical(self, "Ошибка", "Пайплайн завершился с ошибкой — см. лог.")

        self._start_job(f"Пайплайн {self.patient_id} ({self.cmbVariant.currentText()})",
                        [cmd], on_done=after)

    def _export(self):
        if not self.patient_id:
            QMessageBox.critical(self, "Ошибка", "Сначала выберите пациента.")
            return
        folder = self.entryExport.text().strip()
        if not folder:
            QMessageBox.critical(self, "Ошибка", "Укажите папку для экспорта.")
            return
        try:
            cmd = self._python_cmd("radonc_segmentation.export_rtstruct", self.patient_id, folder)
        except RuntimeError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        def after(ok, title):
            if ok:
                QMessageBox.information(
                    self, "Экспорт",
                    f"RTSTRUCT для {self.patient_id} выгружен в:\n"
                    f"{os.path.join(folder, 'rtstruct')}")
            else:
                QMessageBox.critical(self, "Экспорт", "Ошибка экспорта — см. лог.")

        self._start_job(f"Экспорт {self.patient_id}", [cmd], on_done=after)

    # ---------- watch-папка ----------
    # Запуск через subprocess.Popen (НЕ QProcess): python.exe — консольное
    # приложение, QProcess не передаёт CREATE_NO_WINDOW и из GUI-exe
    # watch получает видимую пустую консоль. Вывод стримится потоком-
    # читальником через сигналы (потокобезопасно для GUI).

    def _watch_running(self):
        return self.watch_proc is not None and self.watch_proc.poll() is None

    def _watch_start(self):
        if self._watch_running():
            return
        cmd = self._python_cmd("radonc_segmentation.watch_folder")
        self._log(f"\n>>> watch-папка: {config.WATCH_DIR}\n$ " + subprocess.list2cmdline(cmd) + "\n")
        self.watch_proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._stream_watch, daemon=True).start()
        poll_sec = int(config.WATCH_POLL_SECONDS) if str(config.WATCH_POLL_SECONDS).strip() else 10
        self.lblWatchStatus.setStyleSheet("color: green")
        self.lblWatchStatus.setText(f"watch-папка: работает (опрос каждые {poll_sec} с)")
        self.btnWatchStart.setEnabled(False)
        self.btnWatchStop.setEnabled(True)

    def _stream_watch(self):
        """Читает вывод watch-процесса в GUI-поток (сигналом _log_direct)."""
        proc = self.watch_proc
        try:
            for ln in proc.stdout:
                self._log_signal.emit(ln)
        except Exception:
            pass
        rc = proc.wait()
        self._log_signal.emit(f"<<< watch-папка: завершена (код {rc})\n")
        self._watch_done_signal.emit()

    def _watch_finished(self):
        """Слот сигнала _watch_done_signal (GUI-поток): кнопки/статус."""
        self.lblWatchStatus.setStyleSheet("color: gray")
        self.lblWatchStatus.setText("watch-папка: остановлена")
        self.btnWatchStart.setEnabled(True)
        self.btnWatchStop.setEnabled(False)

    def _watch_stop(self):
        if not self._watch_running():
            return
        # на Windows terminate() часто игнорируется Python-процессами —
        # сразу kill() (TerminateProcess)
        try:
            self.watch_proc.kill()
            self.watch_proc.wait(timeout=5)
        except Exception:
            pass
        self._log("<<< watch-папка: остановлена пользователем\n")

    # ================= ВКЛАДКА «КВАНТИЗАЦИЯ» =================

    def _browse_import_src(self):
        f, _flt = QFileDialog.getOpenFileName(
            self, "Чекпоинт модели (checkpoint_*.pth или .zip из Kaggle Output)",
            "", "Чекпоинт/архив (*.pth *.zip);;Все файлы (*.*)")
        if f:
            self.entryImportSrc.setText(f)

    def _run_import_model(self):
        src = self.entryImportSrc.text().strip()
        name = self.entryImportName.text().strip()
        if not src or not (os.path.isfile(src) or os.path.isdir(src)):
            QMessageBox.critical(self, "Ошибка",
                                 f"Не найден файл/папка чекпоинта:\n{src}\n"
                                 "Укажите checkpoint_*.pth, .zip из Kaggle Output "
                                 "или папку с fold_N\\.")
            return
        if not name or not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            QMessageBox.critical(self, "Ошибка",
                                 "Имя модели: латиница/цифры/_/- (напр. distilled)")
            return
        try:
            cmd = self._python_cmd("radonc_segmentation.quantization.import_checkpoint",
                                   "--src", src, "--name", name)
        except RuntimeError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return
        plans = self.entryImportPlans.text().strip()
        if plans:
            cmd += ["--plans", plans]
        folds = self.entryImportFolds.text().strip()
        if folds:
            cmd += ["--folds", folds]

        def after(ok, title):
            if ok:
                self._refresh_model_combo()
                self.entryQuantSource.setText(
                    os.path.join(config.MODELS_ROOT, config.MODEL_NAME, name))
                QMessageBox.information(
                    self, "Импорт модели",
                    f"Модель \"{config.MODEL_NAME}/{name}\" экспортирована в OpenVINO IR.\n"
                    "Дальше: «Тест модели на валидации» (ниже) — гейт качества,\n"
                    "затем INT8-квантизация этой модели.")
            else:
                QMessageBox.critical(
                    self, "Импорт модели",
                    "Ошибка импорта/экспорта — см. лог.\n"
                    "Если в логе «Отсутствуют пакеты» — установите зависимости\n"
                    "квантизации: pip install -e .[quantization] (или python setup_env.py).")

        self._start_job(f"Импорт модели {name} (.pth → IR FP32)", [cmd], on_done=after)

    def _refresh_test_combos(self):
        models = config.available_models()
        for combo in (self.cmbTestModel, self.cmbTestRef):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if combo is self.cmbTestRef:
                combo.addItem("— нет —")
            combo.addItems(models or [str(config.MODEL)])
            if keep:
                combo.setCurrentText(keep)
            combo.blockSignals(False)
        cases = [c for c, _img, _gt in config.validation_cases()]
        keep_case = self.cmbTestCase.currentText()
        self.cmbTestCase.blockSignals(True)
        self.cmbTestCase.clear()
        self.cmbTestCase.addItems(cases)
        if keep_case in cases:
            self.cmbTestCase.setCurrentText(keep_case)
        self.cmbTestCase.blockSignals(False)

    def _last_verdict(self):
        """Последний «ВЕРДИКТ: ...» из лога (для окна после теста)."""
        tail = self.logText.toPlainText()[-4000:]
        found = re.findall(r"ВЕРДИКТ: .+", tail)
        return found[-1].strip() if found else None

    def _run_test_model(self):
        model = self.cmbTestModel.currentText().strip()
        if not model or model.startswith("—"):
            QMessageBox.critical(self, "Ошибка",
                                 "Нет модели для теста — сначала импортируйте .pth (выше).")
            return
        case = self.cmbTestCase.currentText().strip()
        if not case:
            QMessageBox.critical(self, "Ошибка",
                                 "Нет валидационных кейсов (validation\\<анатомия>\\<кейс>).")
            return
        try:
            cmd = self._python_cmd("radonc_segmentation.quantization.validate_model",
                                   "--model", model, "--case", case)
        except RuntimeError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return
        ref = self.cmbTestRef.currentText().strip()
        if ref and not ref.startswith("—"):
            cmd += ["--reference", ref]
        folds = self.entryTestFolds.text().strip()
        if folds:
            cmd += ["--folds", folds]

        def after(ok, title):
            if ok:
                if _POST_OK:
                    self.post_tab._refresh_cases()
                verdict = self._last_verdict()
                QMessageBox.information(
                    self, "Тест модели",
                    (verdict or "Тест завершён.") +
                    "\n\nДетали по органам — в логе; сегмент доступен на вкладке «Постпроцессинг».")
            else:
                QMessageBox.critical(self, "Тест модели", "Ошибка теста — см. лог.")

        self._start_job(f"Тест {model} на {case}", [cmd], on_done=after)

    def _run_quantize(self):
        source = self.entryQuantSource.text().strip()
        name = self.entryQuantName.text().strip()
        if not source or not os.path.isdir(source):
            QMessageBox.critical(self, "Ошибка", f"Папка источника не найдена:\n{source}")
            return
        if not name or not re.fullmatch(r"[A-Za-z0-9_./\-]+", name):
            QMessageBox.critical(self, "Ошибка",
                                 "Имя новой модели: латиница/цифры/./_/- (напр. headandneck/distilled_int8)")
            return
        try:
            cmd = self._python_cmd("radonc_segmentation.quantization.quantize_ov",
                                   "--source", source, "--name", name,
                                   "--mode", self.cmbQuantMode.currentText())
        except RuntimeError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        def after(ok, title):
            if ok:
                self._refresh_model_combo()
                QMessageBox.information(
                    self, "Квантизация",
                    f"Модель {name!r} создана.\n"
                    f"Выберите её на вкладке «Настройки» и сравните качество:\n"
                    f"python -m radonc_segmentation.compare_dice <seg> <seg_новой_модели>")
            else:
                QMessageBox.critical(self, "Квантизация", "Ошибка квантизации — см. лог.")

        self._start_job(f"Квантизация {name} ({self.cmbQuantMode.currentText()})", [cmd], on_done=after)

    def _run_benchmark(self):
        try:
            n = int(self.entryBenchN.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", f"Случаев должно быть числом: {self.entryBenchN.text()!r}")
            return
        models = f"{self.cmbBenchA.currentText()},{self.cmbBenchB.currentText()}"
        try:
            cmd = self._python_cmd("radonc_segmentation.quantization.benchmark_dice",
                                   "--models", models, "--n", str(n))
        except RuntimeError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return
        folds = self.entryBenchFolds.text().strip()
        if folds:
            cmd += ["--folds", folds]

        def after(ok, title):
            if ok:
                QMessageBox.information(
                    self, "Бенчмарк",
                    f"Сравнение готово.\nТаблица и CSV: {os.path.join(config.LOG_DIR, 'benchmark')}\n"
                    f"(полные цифры — в логе ниже)")
            else:
                QMessageBox.critical(self, "Бенчмарк", "Ошибка бенчмарка — см. лог.")

        self._start_job(f"Бенчмарк Dice: {models} × {n} случ.", [cmd], on_done=after)

    def _quant_list(self):
        cmd = self._python_cmd("radonc_segmentation.quantization.quantize_ov", "--list")
        self._start_job("Список моделей", [cmd])

    # ================= ВКЛАДКА «НАСТРОЙКИ» =================

    def _settings_row(self, grid, label, section, key, value, kind="str",
                      choices=None, narrow=False, browse=False):
        """Строка «подпись + редактор» в группе настроек; редактор попадает
        в реестр self.sv[(секция, ключ)] для сохранения."""
        r = grid.rowCount()
        grid.addWidget(QLabel(label), r, 0)
        if choices:
            w = QComboBox()
            w.addItems(choices)
            w.setCurrentText(str(value))
        else:
            w = QLineEdit(str(value))
            if narrow:
                w.setMaximumWidth(90)
        grid.addWidget(w, r, 1)
        if browse:
            btn = QPushButton("Обзор...")
            btn.clicked.connect(lambda: self._browse(w, "Выберите папку"))
            grid.addWidget(btn, r, 2)
        self.sv[(section, key)] = (w, kind)

    def _build_settings_fields(self):
        # --- устройство ---
        self._settings_row(self.gridDevice, "Устройство (cpu / gpu)", "device", "DEVICE",
                           str(config.DEVICE).lower(), choices=["cpu", "gpu", "gpu.0"])
        self._settings_row(self.gridDevice, "Потоков CPU", "device", "NUM_THREADS",
                           config.NUM_THREADS, kind="int", narrow=True)
        self._settings_row(self.gridDevice, "Async-запросов", "device", "N_ASYNC_REQUESTS",
                           config.N_ASYNC_REQUESTS, kind="int", narrow=True)
        self._settings_row(self.gridDevice, "PERFORMANCE_HINT", "device", "PERFORMANCE_HINT",
                           config.OV_PERFORMANCE_HINT, choices=["LATENCY", "THROUGHPUT", "none"])

        # --- инференс ---
        self._settings_row(self.gridInference, "Вариант (legacy / aligned)", "inference", "VARIANT",
                           config.DEFAULT_VARIANT, choices=["legacy", "aligned"])
        self._settings_row(self.gridInference, "Фолды (пусто = все)", "inference", "FOLDS",
                           ",".join(map(str, config.DEFAULT_FOLDS)))
        self._settings_row(self.gridInference, "Шаг окна (stride)", "inference", "STEP_SIZE_RATIO",
                           config.STEP_SIZE_RATIO, kind="float", narrow=True)
        self._settings_row(self.gridInference, "Кроп воздуха", "inference", "CROP_AIR",
                           "true" if config.CROP_AIR else "false", kind="bool",
                           choices=["true", "false"])
        self._settings_row(self.gridInference, "Накопитель логитов", "inference", "ACCUMULATOR",
                           config.ACCUMULATOR, choices=["disk", "ram"])

        # --- модель ---
        self._settings_row(self.gridModel, "Анатомическая модель\n(папки patients/validation/models)",
                           "model", "MODEL_NAME", str(config.MODEL_NAME))
        self._settings_row(self.gridModel, "Квантизация:", "model", "MODEL",
                           str(config.MODEL),
                           choices=config.available_models() or [str(config.MODEL)])
        r = self.gridModel.rowCount()
        btn = QPushButton("Обновить список")
        btn.clicked.connect(self._refresh_model_combo)
        self.gridModel.addWidget(btn, r, 0, 1, 2)
        hint = QLabel("int8 — рабочая; original — эталон качества (медленно);\n"
                      "новые — после квантизации")
        hint.setStyleSheet("color: gray")
        self.gridModel.addWidget(hint, r + 1, 0, 1, 2)

        # --- пути и watch ---
        for label, section, key, value in [
            ("Пациенты (пусто = папка проекта\\patients)", "paths", "PATIENTS_ROOT", config.PATIENTS_ROOT),
            ("Логи (пусто = папка проекта\\logs)", "paths", "LOG_DIR", config.LOG_DIR),
            ("Watch-папка (пусто = папка проекта\\inbox)", "paths", "WATCH_DIR", config.WATCH_DIR),
            ("Опрос watch, секунд", "paths", "WATCH_POLL_SECONDS", config.WATCH_POLL_SECONDS),
            ("Python окружения (пусто = автопоиск)", "gui", "CONDA_PYTHON", config.CONDA_PYTHON),
        ]:
            self._settings_row(self.gridPaths, label, section, key, value)

        # --- папки по умолчанию для главного окна ---
        for label, key, value in [
            ("Импорт DICOM (при старте GUI)", "DEFAULT_IMPORT_DIR", config.DEFAULT_IMPORT_DIR),
            ("Экспорт RTSTRUCT (при старте GUI)", "DEFAULT_EXPORT_DIR", config.DEFAULT_EXPORT_DIR),
        ]:
            self._settings_row(self.gridDefaults, label, "paths", key, value, browse=True)

    def _get_setting(self, widget):
        return widget.currentText() if isinstance(widget, QComboBox) else widget.text()

    def _refresh_model_combo(self):
        models = config.available_models()
        combos = (self.sv[("model", "MODEL")][0], self.cmbBenchA, self.cmbBenchB)
        for combo in combos:
            current = combo.currentText()
            combo.clear()
            combo.addItems(models or [str(config.MODEL)])
            if current in models:
                combo.setCurrentText(current)
        self._refresh_test_combos()
        self._log(f"[GUI] Модели в {config.MODELS_ROOT}: {', '.join(models) or 'нет'}\n")

    def _save_settings(self):
        # эти ключи МОЖНО сохранять пустыми — пусто = «папка проекта»
        # (портабельность: путь подстроится под то, куда скопировали проект)
        allow_empty = {("paths", "PATIENTS_ROOT"), ("paths", "LOG_DIR"), ("paths", "WATCH_DIR"),
                       ("gui", "CONDA_PYTHON"), ("hardware", "FINGERPRINT")}
        updates = {}
        for (section, key), (widget, kind) in self.sv.items():
            val = self._get_setting(widget).strip()
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
                QMessageBox.critical(self, "Ошибка", f"Некорректное значение {key}: {val!r}")
                return
        # спец-обработка фолдов: пустое поле = все фолды (в ini пишем пусто)
        if ("inference", "FOLDS") in self.sv and self._get_setting(self.sv[("inference", "FOLDS")][0]).strip() == "":
            updates[("inference", "FOLDS")] = ""

        try:
            path = config.write_settings(updates)
        except OSError as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось записать settings.ini: {e}")
            return
        self._log(f"[GUI] Настройки сохранены: {path}\n")
        for (sec, key), v in sorted(updates.items()):
            self._log(f"  [{sec}] {key} = {v}\n")
        self._log("[GUI] Применяются к новым запускам (watch-папку перезапустите, если работала).\n")
        self.lblWatchDir.setText(self._get_setting(self.sv[("paths", "WATCH_DIR")][0]))
        self._update_lbl_info()
        QMessageBox.information(self, "Настройки", f"Сохранено в:\n{path}")

    def _check_devices(self):
        self._start_job("Проверка устройств", [self._python_cmd("radonc_segmentation.check_devices")])

    # ---------- закрытие ----------

    def closeEvent(self, event):
        busy = self._busy() or self._watch_running()
        if busy:
            answer = QMessageBox.question(
                self, "Выход", "Операция/watch ещё выполняется. Выйти?",
                QMessageBox.Yes | QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        if self._watch_running():
            try:
                # на Windows terminate() часто игнорируется — сразу kill()
                self.watch_proc.kill()
                self.watch_proc.wait(timeout=3)
            except Exception:
                pass
        if self.worker is not None:
            # kill python-процесса текущей задачи — иначе пайплайн остаётся
            # работать бесхозным процессом после закрытия GUI
            self.worker.stop()
        event.accept()
