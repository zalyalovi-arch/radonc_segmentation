# test_main_window.py — standalone test for MainWindow migration
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from PySide6.QtWidgets import QApplication, QMainWindow
app = QApplication([])

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sv = {}
        self.job_thread = None
        self.worker = None
        self.on_job_done_cb = None
        self.patient_id = None
        self.watch_proc = None
        self._load_ui()
        
    def _load_ui(self):
        from PySide6.QtCore import QFile
        from PySide6.QtUiTools import QUiLoader
        from gui import UI_DIR
        import os
        loader = QUiLoader()
        ui_path = os.path.join(UI_DIR, 'main_window.ui')
        ui_file = QFile(ui_path)
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()
        self._ui = {}
        for child in self.findChildren(object):
            if child.objectName():
                self._ui[child.objectName()] = child
        self._alias_widgets()
        
    def _alias_widgets(self):
        self.tabWidget = self._ui_get('tabWidget')
        self.logText = self._ui_get('logText')
        self.entryDicom = self._ui_get('entryDicom')
        self.btnBrowseDicom = self._ui_get('btnBrowseDicom')
        self.btnImport = self._ui_get('btnImport')
        self.entryPid = self._ui_get('entryPid')
        self.btnSelectPid = self._ui_get('btnSelectPid')
        self.lblPatient = self._ui_get('lblPatient')
        self.btnRun = self._ui_get('btnRun')
        self.chkCrop = self._ui_get('chkCrop')
        self.chkAsync = self._ui_get('chkAsync')
        self.cmbVariant = self._ui_get('cmbVariant')
        self.entryStride = self._ui_get('entryStride')
        self.lblInfo = self._ui_get('lblInfo')
        self.rowPost = self._ui_get('rowPost')
        self.cmbPostTpl = self._ui_get('cmbPostTpl')
        self.lblWatchDir = self._ui_get('lblWatchDir')
        self.btnWatchStart = self._ui_get('btnWatchStart')
        self.btnWatchStop = self._ui_get('btnWatchStop')
        self.lblWatchStatus = self._ui_get('lblWatchStatus')
        self.entryExport = self._ui_get('entryExport')
        self.btnBrowseExport = self._ui_get('btnBrowseExport')
        self.btnExport = self._ui_get('btnExport')
        self.entryQuantSource = self._ui_get('entryQuantSource')
        self.btnBrowseQuantSource = self._ui_get('btnBrowseQuantSource')
        self.entryQuantName = self._ui_get('entryQuantName')
        self.cmbQuantMode = self._ui_get('cmbQuantMode')
        self.cmbBenchA = self._ui_get('cmbBenchA')
        self.cmbBenchB = self._ui_get('cmbBenchB')
        self.entryBenchN = self._ui_get('entryBenchN')
        self.entryBenchFolds = self._ui_get('entryBenchFolds')
        self.btnBenchmark = self._ui_get('btnBenchmark')
        self.btnQuantize = self._ui_get('btnQuantize')
        self.btnQuantList = self._ui_get('btnQuantList')
        self.groupDevice = self._ui_get('groupDevice')
        self.groupInference = self._ui_get('groupInference')
        self.groupPaths = self._ui_get('groupPaths')
        self.groupModel = self._ui_get('groupModel')
        self.groupDefaults = self._ui_get('groupDefaults')
        self.gridDevice = self.groupDevice.layout()
        self.gridInference = self.groupInference.layout()
        self.gridPaths = self.groupPaths.layout()
        self.gridModel = self.groupModel.layout()
        self.gridDefaults = self.groupDefaults.layout()
        self.lblSettingsIntro = self._ui_get('lblSettingsIntro')
        self.lblQuantIntro = self._ui_get('lblQuantIntro')
        self.btnSaveSettings = self._ui_get('btnSaveSettings')
        self.btnCheckDevices = self._ui_get('btnCheckDevices')
        self.tabWork = self._ui_get('tabWork')
        self.tabQuant = self._ui_get('tabQuant')
        self.tabPost = self._ui_get('tabPost')
        self.tabSettings = self._ui_get('tabSettings')
        self.layPostPage = self.tabPost.layout()
        self.sv = {}
        self.job_thread = None
        self.worker = None
        self.on_job_done_cb = None
        self.patient_id = None
        self.watch_proc = None
        print('All widgets aliased!')
        
    def _ui_get(self, name):
        w = self._ui.get(name)
        if w is None:
            raise RuntimeError(f'Widget {name} not found')
        return w
    
    def _log(self, s):
        self.logText.moveCursor(self.logText.textCursor().End)
        self.logText.insertPlainText(s)
        self.logText.ensureCursorVisible()
        
    def _browse(self, line_edit, title, initial=None):
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, title, initial or '')
        if folder:
            line_edit.setText(folder)
            
    def _set_buttons(self, running):
        state = not running
        for b in (self.btnImport, self.btnRun, self.btnExport, self.btnSelectPid):
            b.setEnabled(state)
            
    def _busy(self):
        return self.job_thread is not None and self.job_thread.isRunning()
        
    def _python_cmd(self, script, *args):
        import config
        PYTHON_EXE = config.CONDA_PYTHON or sys.executable
        return [PYTHON_EXE, os.path.join(config.SCRIPTS_DIR, script), *args]

    def _setup_post_tab(self):
        global _POST_OK
        try:
            from gui.postprocess_tab import PostprocessTab
            import postprocess_core as ppcore
            _POST_OK = True
        except Exception as e:
            _POST_OK = False
            _POST_ERR = e
        if _POST_OK:
            self.post_tab = PostprocessTab(log=self._log, start_job=self._start_job)
            self.layPostPage.addWidget(self.post_tab)
        else:
            self.tabWidget.removeTab(self.tabWidget.indexOf(self.tabPost))
            self.rowPost.setVisible(False)
            import config
            self.lblInfo.setText(f'Модель: {config.MODEL} · постпроцессинг недоступен: {_POST_ERR}')
        print('_setup_post_tab OK')
        
    def _tune_layout(self):
        self.splitterMain.setStretchFactor(0, 1)
        self.splitterMain.setStretchFactor(1, 0)
        self.splitterMain.setSizes([620, 180])
        self.btnWatchStop.setEnabled(False)
        print('_tune_layout OK')
        
    def _connect_signals(self):
        self.btnBrowseDicom.clicked.connect(lambda: self._browse(self.entryDicom, 'Выберите папку с DICOM'))
        self.btnImport.clicked.connect(self._import)
        self.btnSelectPid.clicked.connect(self._select_pid)
        self.btnRun.clicked.connect(self._run_pipeline)
        self.btnBrowseExport.clicked.connect(lambda: self._browse(self.entryExport, 'Куда положить RTSTRUCT'))
        self.btnExport.clicked.connect(self._export)
        self.btnWatchStart.clicked.connect(self._watch_start)
        self.btnWatchStop.clicked.connect(self._watch_stop)
        self.btnBrowseQuantSource.clicked.connect(lambda: self._browse(self.entryQuantSource, 'Папка модели-источника', initial=config.MODELS_ROOT))
        self.btnQuantize.clicked.connect(self._run_quantize)
        self.btnQuantList.clicked.connect(self._quant_list)
        self.btnBenchmark.clicked.connect(self._run_benchmark)
        self.btnSaveSettings.clicked.connect(self._save_settings)
        self.btnCheckDevices.clicked.connect(self._check_devices)
        print('_connect_signals OK')
        
    def _fill_from_config(self):
        import config
        if config.DEFAULT_IMPORT_DIR:
            self.entryDicom.setText(config.DEFAULT_IMPORT_DIR)
        if config.DEFAULT_EXPORT_DIR:
            self.entryExport.setText(config.DEFAULT_EXPORT_DIR)
        self.chkCrop.setChecked(bool(config.CROP_AIR))
        self.chkAsync.setChecked(int(config.N_ASYNC_REQUESTS) > 1)
        self.cmbVariant.setCurrentText(str(config.DEFAULT_VARIANT))
        self.entryStride.setText(str(config.STEP_SIZE_RATIO))
        self.lblWatchDir.setText(config.WATCH_DIR)
        self.lblQuantIntro.setText(f'Создание новой (квантованной) версии модели из имеющейся.\nПапка моделей: {config.MODELS_ROOT}\nИсточник только читается; результат — новая подпапка, доступная для инференса.')
        self.entryQuantSource.setText(os.path.join(config.MODELS_ROOT, config.MODEL))
        self._update_lbl_info()
        self.lblSettingsIntro.setText(f'Файл настроек: {config.SETTINGS_PATH or "settings.ini (не найден — будет создан)"}\nИзменения применяются к НОВЫМ запускам; комментарии в файле сохраняются.')
        models = config.available_models()
        for combo in (self.cmbBenchA, self.cmbBenchB):
            combo.addItems(models or ['original'])
        self.cmbBenchA.setCurrentText('original' if 'original' in models else str(config.MODEL))
        self.cmbBenchB.setCurrentText(str(config.MODEL))
        self.sv[('model', 'MODEL')] = (self.cmbBenchA, 'str')
        if _POST_OK:
            self.cmbPostTpl.setPlaceholderText('— без постобработки —')
            self.cmbPostTpl.addItems(ppcore.list_templates())
        print('_fill_from_config OK')
        
    def _update_lbl_info(self):
        import config
        self.lblInfo.setText(f'Модель: {config.MODEL} · устройство: {str(config.DEVICE).upper()} · накопитель: {str(config.ACCUMULATOR).lower()}' + ('' if _POST_OK else ' · постобработка: недоступна'))
        
    def _import(self):
        print('_import called')
        
    def _select_pid(self):
        print('_select_pid called')
        
    def _run_pipeline(self):
        print('_run_pipeline called')
        
    def _export(self):
        print('_export called')
        
    def _watch_running(self):
        return self.watch_proc is not None and self.watch_proc.state() != 0
        
    def _watch_start(self):
        print('_watch_start called')
        
    def _watch_read(self):
        pass
        
    def _watch_finished(self, rc, status):
        pass
        
    def _watch_stop(self):
        pass
        
    def _run_quantize(self):
        print('_run_quantize called')
        
    def _run_benchmark(self):
        print('_run_benchmark called')
        
    def _quant_list(self):
        print('_quant_list called')
        
    def _settings_row(self, grid, label, section, key, value, kind='str', choices=None, narrow=False, browse=False):
        from PySide6.QtWidgets import QLabel, QComboBox, QLineEdit, QPushButton
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
            btn = QPushButton('Обзор...')
            btn.clicked.connect(lambda: self._browse(w, 'Выберите папку'))
            grid.addWidget(btn, r, 2)
        self.sv[(section, key)] = (w, kind)
        
    def _build_settings_fields(self):
        import config
        self._settings_row(self.gridDevice, 'Устройство (cpu / gpu)', 'device', 'DEVICE', str(config.DEVICE).lower(), choices=['cpu', 'gpu', 'gpu.0'])
        self._settings_row(self.gridDevice, 'Потоков CPU', 'device', 'NUM_THREADS', config.NUM_THREADS, kind='int', narrow=True)
        self._settings_row(self.gridDevice, 'Async-запросов', 'device', 'N_ASYNC_REQUESTS', config.N_ASYNC_REQUESTS, kind='int', narrow=True)
        self._settings_row(self.gridDevice, 'PERFORMANCE_HINT', 'device', 'PERFORMANCE_HINT', config.OV_PERFORMANCE_HINT, choices=['LATENCY', 'THROUGHPUT', 'none'])
        self._settings_row(self.gridInference, 'Вариант (legacy / aligned)', 'inference', 'VARIANT', config.DEFAULT_VARIANT, choices=['legacy', 'aligned'])
        self._settings_row(self.gridInference, 'Фолды (пусто = все)', 'inference', 'FOLDS', ','.join(map(str, config.DEFAULT_FOLDS)))
        self._settings_row(self.gridInference, 'Шаг окна (stride)', 'inference', 'STEP_SIZE_RATIO', config.STEP_SIZE_RATIO, kind='float', narrow=True)
        self._settings_row(self.gridInference, 'Кроп воздуха', 'inference', 'CROP_AIR', 'true' if config.CROP_AIR else 'false', kind='bool', choices=['true', 'false'])
        self._settings_row(self.gridInference, 'Накопитель логитов', 'inference', 'ACCUMULATOR', config.ACCUMULATOR, choices=['disk', 'ram'])
        self._settings_row(self.gridModel, 'Анатомическая модель\n(папки patients/validation/models)', 'model', 'MODEL_NAME', str(config.MODEL_NAME))
        self._settings_row(self.gridModel, 'Квантизация:', 'model', 'MODEL', str(config.MODEL), choices=config.available_models() or [str(config.MODEL)])
        r = self.gridModel.rowCount()
        from PySide6.QtWidgets import QPushButton, QLabel
        btn = QPushButton('Обновить список')
        btn.clicked.connect(self._refresh_model_combo)
        self.gridModel.addWidget(btn, r, 0, 1, 2)
        hint = QLabel('int8 — рабочая; original — эталон качества (медленно);\nновые — после квантизации')
        hint.setStyleSheet('color: gray')
        self.gridModel.addWidget(hint, r + 1, 0, 1, 2)
        for label, section, key, value in [
            ('Пациенты (пусто = папка проекта\\patients)', 'paths', 'PATIENTS_ROOT', config.PATIENTS_ROOT),
            ('Логи (пусто = папка проекта\\logs)', 'paths', 'LOG_DIR', config.LOG_DIR),
            ('Watch-папка (пусто = папка проекта\\inbox)', 'paths', 'WATCH_DIR', config.WATCH_DIR),
            ('Опрос watch, секунд', 'paths', 'WATCH_POLL_SECONDS', config.WATCH_POLL_SECONDS),
            ('Python окружения (пусто = автопоиск)', 'gui', 'CONDA_PYTHON', config.CONDA_PYTHON),
        ]:
            self._settings_row(self.gridPaths, label, section, key, value)
        for label, key, value in [
            ('Импорт DICOM (при старте GUI)', 'DEFAULT_IMPORT_DIR', config.DEFAULT_IMPORT_DIR),
            ('Экспорт RTSTRUCT (при старте GUI)', 'DEFAULT_EXPORT_DIR', config.DEFAULT_EXPORT_DIR),
        ]:
            self._settings_row(self.gridDefaults, label, 'paths', key, value, browse=True)
        print('_build_settings_fields OK')
        
    def _get_setting(self, widget):
        from PySide6.QtWidgets import QComboBox
        return widget.currentText() if isinstance(widget, QComboBox) else widget.text()
        
    def _refresh_model_combo(self):
        import config
        models = config.available_models()
        print('_refresh_model_combo OK')
        
    def _save_settings(self):
        print('_save_settings called')
        
    def _check_devices(self):
        print('_check_devices called')
        
    def _startup_checks(self):
        import config
        problems = []
        if not os.path.isfile(os.path.join(config.SCRIPTS_DIR, 'pipeline.py')):
            problems.append(f'Не найден pipeline.py в {config.SCRIPTS_DIR}.')
        if config.CONDA_PYTHON is None and not os.path.isfile(sys.executable):
            problems.append('Python не найден.')
        if problems:
            print('Problems:', problems)
        if config.AUTOTUNE_NOTICE:
            print('AUTOTUNE_NOTICE:', config.AUTOTUNE_NOTICE)
        print(f'Скрипты: {config.SCRIPTS_DIR}')
        print('_startup_checks OK')
        
    def _start_job(self, title, commands=None, on_done=None, in_process=None):
        if self._busy():
            print('Busy, ignoring')
            return
        self._set_buttons(running=True)
        self._log(f'\n>>> {title}\n')
        self.on_job_done_cb = on_done
        from PySide6.QtCore import QThread
        from gui.workers import JobWorker
        import config
        self.job_thread = QThread(self)
        self.worker = JobWorker(title, commands=commands, in_process=in_process, cwd=config.SCRIPTS_DIR)
        self.worker.moveToThread(self.job_thread)
        self.worker.line.connect(self._log)
        self.worker.finished_job.connect(self._on_job_finished)
        self.job_thread.started.connect(self.worker.run)
        self.job_thread.start()
        
    def _on_job_finished(self, title, rc):
        ok = rc == 0
        self._set_buttons(running=False)
        self._log(f'<<< {title}: {"ГОТОВО" if ok else f"ОШИБКА (код {rc})"}\n')
        self.job_thread.quit()
        self.job_thread.wait(20000)
        self.job_thread = None
        self.worker = None
        cb = self.on_job_done_cb
        self.on_job_done_cb = None
        if cb:
            cb(ok, title)

import config
import postprocess_core as ppcore
_POST_OK = True

win = TestWindow()
print('TestWindow created')
print('_ui len:', len(win._ui))

# Test all methods
print('Testing _setup_post_tab...')
win._setup_post_tab()

print('Testing _tune_layout...')
win._tune_layout()

print('Testing _connect_signals...')
win._connect_signals()

print('Testing _fill_from_config...')
win._fill_from_config()

print('Testing _update_lbl_info...')
win._update_lbl_info()

print('Testing _build_settings_fields...')
win._build_settings_fields()

print('Testing _refresh_model_combo...')
win._refresh_model_combo()

print('Testing _startup_checks...')
win._startup_checks()

print('ALL TESTS PASSED')
app.quit()