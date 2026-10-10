# ============================================================
# postprocess_tab.py — вкладка GUI «Постпроцессинг» (PySide6 + QUiLoader)
# ------------------------------------------------------------
# Две панели (GT | инференс+постпроцессинг) на аксиальных срезах,
# выбор органа, per-структурные методы с мгновенными Dice/HD95,
# шаблоны JSON (движок — postprocess_core.py).
# Раскладка: gui/ui/postprocess_tab.ui (правится в Qt Designer).
# Подключается из main_window: PostprocessTab(log=..., start_job=...).
# ============================================================
import glob
import os

import numpy as np
import SimpleITK as sitk

from PySide6.QtCore import QFile, Qt, QTimer
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QApplication, QFileDialog, QInputDialog,
                               QMessageBox, QTreeWidgetItem, QWidget)

from radonc_segmentation import config
from radonc_segmentation import postprocess_core as core
from gui import UI_DIR
from gui.mpl_canvas import MplCanvas  # noqa: F401 — нужен для промоушена в .ui

WINDOW_PRESETS = {"мягкая ткань": (400, 40), "кость": (2000, 500), "лёгкие": (1500, -600)}

# описания методов для кнопки «?»
METHODS_HELP = {
    "closing": "Дилатация + эрозия (радиус). Закрывает тонкие разрывы контура; "
               "тонкие выросты может срезать.",
    "closing_z": "Closing линейным элементом только вдоль оси Z — закрывает "
                 "межсрезовые разрывы, не расширяя маску в плоскости XY.",
    "dilate": "Расширение маски на radius вокселей.",
    "erode": "Сжатие маски на radius вокселей.",
    "median": "Медианный фильтр (size) — убирает шум «соль-перец», границы почти не трогает.",
    "fill3d": "Заполняет замкнутые дыры внутри органа. Вложенные органы "
              "(линза в глазе, мозг в канале) НЕ затрагиваются.",
    "fill2d": "Заполнение дыр послойно — каждый срез независимо.",
    "fill_all": "Заполняет ВСЕ замкнутые дыры, включая занятые вложенными "
                "органами: воксели линзы/мозга внутри дыры перезаписываются "
                "меткой органа (осторожно — Dice вложенного органа упадёт).",
    "bridge": "Соединяет N крупнейших связных компонент перемычкой по ближайшим "
              "точкам (перешеек щитовидной железы). Толщина — утолщение перемычки.",
    "gaussian": "Гауссово сглаживание (sigma) с порогом 0.5 — сглаживает ступеньки.",
    "keep_largest": "Оставляет только крупнейшую связную компоненту (ложные острова удаляются).",
    "remove_small": "Удаляет связные компоненты меньше min_voxels вокселей.",
}


def _read_nifti_zyx(path):
    """NIfTI -> (массив (z,y,x) float32, spacing (z,y,x) мм). SimpleITK —
    ориентация гарантированно совпадает с остальным пайплайном."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (z,y,x)
    spacing = tuple(reversed([float(s) for s in img.GetSpacing()]))
    return arr, spacing


def _organ_color(label, n=31):
    import colorsys
    import matplotlib.colors as mcolors
    return mcolors.hsv_to_rgb((((label - 1) % n) / n, 0.85, 0.95))


class PostprocessTab(QWidget):
    def __init__(self, log=lambda s: None, start_job=None, parent=None):
        super().__init__(parent)
        self._load_ui()
        self.log = log
        self.start_job = start_job  # (title, commands, on_done, in_process) главного окна

        self.ct = None          # (z,y,x) float32, HU
        self.spacing = None     # (z,y,x) мм
        self.base_seg = None    # (z,y,x) uint8
        self.gt = None          # (z,y,x) uint8 или None
        self.work_seg = None
        self.chains = {}        # {label: [steps]}
        self.case = None
        self.selected_label = None
        self._cases = []
        self._metrics_cache = {}  # {label: (d0, h0, d1, h1, vol0, vol1)}

        # дебаунс перерисовки при движении слайдера
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)
        self._redraw_timer.timeout.connect(self.redraw)

        self._connect()
        self.cmbMethod.addItems(list(core.METHODS))
        self.cmbMethod.setCurrentText("closing")
        self._adjust_spin(self.cmbMethod.currentText())
        self._refresh_cases()
        self._refresh_templates()
        self._tune_tree_columns()

    # ================= загрузка UI =================

    def _load_ui(self):
        """Загружает .ui через QUiLoader и маппит виджеты по objectName."""
        from gui.mpl_canvas import MplCanvas
        loader = QUiLoader()
        # без регистрации промоутенный MplCanvas инстанцируется как голый QWidget
        loader.registerCustomWidget(MplCanvas)
        ui_file = QFile(os.path.join(UI_DIR, "postprocess_tab.ui"))
        ui_file.open(QFile.ReadOnly)
        try:
            loaded = loader.load(ui_file, self)
        finally:
            ui_file.close()
        # QUiLoader создаёт форму СКРЫТОЙ (WA_WState_Hidden) — без явного
        # show вкладка остаётся пустой
        if loaded is not None and loaded is not self:
            loaded.setVisible(True)

        # Кэш виджетов по objectName
        self._ui = {}
        for child in self.findChildren(QWidget):
            if child.objectName():
                self._ui[child.objectName()] = child

        # Удобные алиасы для часто используемых виджетов
        self._alias_widgets()

    def _alias_widgets(self):
        """Создаёт атрибуты self.widget_name для совместимости со старым кодом."""
        self.cmbCases = self._ui_get("cmbCases")
        self.btnRefreshCases = self._ui_get("btnRefreshCases")
        self.btnLoadCase = self._ui_get("btnLoadCase")
        self.lblCaseInfo = self._ui_get("lblCaseInfo")
        self.cmbTemplates = self._ui_get("cmbTemplates")
        self.btnApplyTemplate = self._ui_get("btnApplyTemplate")
        self.btnBatchApply = self._ui_get("btnBatchApply")
        self.btnSaveTemplate = self._ui_get("btnSaveTemplate")
        self.btnRefreshTemplates = self._ui_get("btnRefreshTemplates")
        self.sliderSlice = self._ui_get("sliderSlice")
        self.cmbWindow = self._ui_get("cmbWindow")
        self.btnResetZoom = self._ui_get("btnResetZoom")
        self.mplCanvas = self._ui_get("mplCanvas")
        self.organTree = self._ui_get("organTree")
        self.cmbMethod = self._ui_get("cmbMethod")
        self.btnMethodHelp = self._ui_get("btnMethodHelp")
        self.spinParam = self._ui_get("spinParam")
        self.btnApply = self._ui_get("btnApply")
        self.btnUndo = self._ui_get("btnUndo")
        self.btnResetAll = self._ui_get("btnResetAll")
        self.lblMetrics = self._ui_get("lblMetrics")
        self.lblChain = self._ui_get("lblChain")

    def _ui_get(self, name):
        """Получить виджет по objectName с понятной ошибкой."""
        w = self._ui.get(name)
        if w is None:
            raise RuntimeError(f"Виджет с objectName='{name}' не найден в postprocess_tab.ui")
        return w

    # ---------- подключение сигналов ----------

    def _connect(self):
        self.btnRefreshCases.clicked.connect(self._refresh_cases)
        self.btnLoadCase.clicked.connect(self._load_case)
        self.sliderSlice.valueChanged.connect(lambda _: self._redraw_timer.start())
        self.sliderSlice.sliderReleased.connect(self.redraw)
        self.cmbWindow.currentTextChanged.connect(lambda _: self.redraw())
        self.btnResetZoom.clicked.connect(self._reset_zoom)
        self.mplCanvas.sliceRequested.connect(self._step_slice)
        self.organTree.itemSelectionChanged.connect(self._on_organ_select)
        self.cmbMethod.currentTextChanged.connect(self._adjust_spin)
        self.btnMethodHelp.clicked.connect(self._show_method_help)
        self.btnApply.clicked.connect(self._apply_method)
        self.btnUndo.clicked.connect(self._reset_organ)
        self.btnResetAll.clicked.connect(self._reset_all)
        self.btnApplyTemplate.clicked.connect(self._apply_template)
        self.btnBatchApply.clicked.connect(self._batch_apply)
        self.btnSaveTemplate.clicked.connect(self._save_template)
        self.btnRefreshTemplates.clicked.connect(self._refresh_templates)

    def _tune_tree_columns(self):
        self.organTree.setColumnWidth(0, 190)
        self.organTree.setColumnWidth(1, 60)
        self.organTree.setColumnWidth(2, 70)
        self.organTree.header().setStretchLastSection(True)

    # ---------- поиск кейсов ----------

    def _scan_cases(self):
        cases = []
        pdir = config.patients_dir()
        if os.path.isdir(pdir):
            for d in sorted(os.listdir(pdir)):
                ct = os.path.join(pdir, d, "nnunet_input", f"{d}_0000.nii.gz")
                segs = sorted(glob.glob(os.path.join(pdir, d, "seg_nnunet", f"{d}*.nii.gz")))
                if os.path.isfile(ct) and segs:
                    cases.append((f"[пациент] {d}", ct, segs[0], None))
        for case, img, gt in config.validation_cases():
            seg = os.path.join(config.validation_cases_dir(), case, "seg_nnunet", f"{case}.nii.gz")
            if os.path.isfile(seg):
                cases.append((f"[валидация] {case}" + (" +GT" if gt else ""), img, seg, gt))
        return cases

    def _refresh_cases(self):
        self._cases = self._scan_cases()
        self.cmbCases.blockSignals(True)
        self.cmbCases.clear()
        self.cmbCases.addItems([c[0] for c in self._cases])
        self.cmbCases.blockSignals(False)

    def _load_case(self):
        idx = self.cmbCases.currentIndex()
        if idx < 0 or idx >= len(self._cases):
            return
        name, ct_path, seg_path, gt_path = self._cases[idx]
        self.busy("Загрузка...")
        try:
            self.ct, self.spacing = _read_nifti_zyx(ct_path)
            self.base_seg, _ = _read_nifti_zyx(seg_path)
            self.base_seg = self.base_seg.astype(np.uint8)
            self.gt = None
            if gt_path:
                self.gt, _ = _read_nifti_zyx(gt_path)
                self.gt = self.gt.astype(np.uint8)
        except Exception as e:
            self.busy(None)
            QMessageBox.critical(self, "Постпроцессинг", f"Ошибка загрузки: {e}")
            return
        self.case = name
        self.chains = {}
        self.mplCanvas.reset_zoom()
        self.work_seg = self.base_seg.copy()
        # инвалидация кэша метрик при загрузке нового кейса
        self._metrics_cache.clear()
        self.sliderSlice.setMaximum(self.ct.shape[0] - 1)
        self.sliderSlice.setValue(self.ct.shape[0] // 2)
        gt_note = ", GT есть" if self.gt is not None else ", GT нет (метрики недоступны)"
        self.lblCaseInfo.setStyleSheet("")
        self.lblCaseInfo.setText(
            f"{name}: {self.ct.shape}, spacing {tuple(round(s, 2) for s in self.spacing)}{gt_note}")
        self._fill_organ_list()
        self.busy(None)
        self.redraw()

    def busy(self, msg):
        if msg:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.lblCaseInfo.setText(msg)
            self.lblCaseInfo.setStyleSheet("color: blue")
        else:
            QApplication.restoreOverrideCursor()

    # ---------- срезы и зум ----------

    def _step_slice(self, delta):
        if self.ct is None:
            return
        z = self.sliderSlice.value() + delta
        self.sliderSlice.setValue(max(0, min(self.ct.shape[0] - 1, z)))

    def _reset_zoom(self):
        self.mplCanvas.reset_zoom()
        self.redraw()

    # ---------- органы ----------

    def _fill_organ_list(self):
        self.organTree.clear()
        if self.base_seg is None:
            return
        for lbl_name in sorted(config.LABELS, key=config.LABELS.get):
            item = QTreeWidgetItem([lbl_name, "—", "—"])
            item.setData(0, Qt.UserRole, config.LABELS[lbl_name])
            self.organTree.addTopLevelItem(item)

    def _on_organ_select(self):
        items = self.organTree.selectedItems()
        if not items:
            return
        self.selected_label = int(items[0].data(0, Qt.UserRole))
        self._update_metrics()

    def _organ_name(self, lbl):
        return next((n for n, v in config.LABELS.items() if v == lbl), f"label_{lbl}")

    def _update_metrics(self):
            if self.base_seg is None or self.selected_label is None:
                return
            lbl = self.selected_label
            name = self._organ_name(lbl)
            base_m = self.base_seg == lbl
            work_m = self.work_seg == lbl
            vol0, vol1 = int(base_m.sum()), int(work_m.sum())
            d0 = d1 = h0 = h1 = None
        
            # Проверяем кэш
            cache_key = (lbl, id(self.base_seg), id(self.work_seg), id(self.gt))
            if cache_key in self._metrics_cache:
                d0, h0, d1, h1, vol0_cached, vol1_cached = self._metrics_cache[cache_key]
                if vol0_cached == vol0 and vol1_cached == vol1:
                    # Кэш валиден
                    pass
                else:
                    # Объёмы изменились — инвалидация
                    self._metrics_cache.pop(cache_key, None)
                    cache_key = None
            else:
                cache_key = None
        
            if self.gt is not None:
                if cache_key is None:
                    gt_m = self.gt == lbl
                    d0 = core.dice(base_m, gt_m)
                    h0 = core.hd95(base_m, gt_m, self.spacing)
                    d1 = core.dice(work_m, gt_m)
                    h1 = core.hd95(work_m, gt_m, self.spacing)
                    # Сохраняем в кэш
                    self._metrics_cache[(lbl, id(self.base_seg), id(self.work_seg), id(self.gt))] = (d0, h0, d1, h1, vol0, vol1)

            def fmt(d, h):
                return ("—" if d is None else f"{d:.3f}") + " / " + ("—" if h is None else f"{h:.1f}")

            if self.gt is not None:
                better = (d1 is not None and d0 is not None and d1 >= d0)
                self.lblMetrics.setStyleSheet("font-weight: bold; color: %s" % ("black" if better else "red"))
                self.lblMetrics.setText(
                    f"{name}\nДО:      {vol0} вокс., Dice {fmt(d0, h0)}\n"
                    f"ПОСЛЕ: {vol1} вокс., Dice {fmt(d1, h1)}")
                # обновить колонки таблицы
                for it in self._tree_item(lbl):
                    it.setText(1, "—" if d0 is None else f"{d0:.3f}")
                    it.setText(2, "—" if h0 is None else f"{h0:.1f}")
            else:
                self.lblMetrics.setStyleSheet("font-weight: bold")
                self.lblMetrics.setText(
                    f"{name}: {vol0} → {vol1} вокс. (GT нет — метрики недоступны)")
            # цепочка органа
            chain = self.chains.get(lbl)
            text = " → ".join(
                f"{s['method']}({', '.join(f'{k}={v}' for k, v in s.items() if k != 'method')})"
                for s in chain) if chain else "—"
            self.lblChain.setText("цепочка: " + text)

    def _invalidate_metrics_cache(self, lbl):
        """Удаляет кэш метрик для конкретного органа."""
        keys_to_remove = [k for k in self._metrics_cache if k[0] == lbl]
        for k in keys_to_remove:
            self._metrics_cache.pop(k, None)

    def _tree_item(self, lbl):
        for i in range(self.organTree.topLevelItemCount()):
            it = self.organTree.topLevelItem(i)
            if int(it.data(0, Qt.UserRole)) == lbl:
                yield it

    # ---------- применение методов ----------

    def _adjust_spin(self, method):
        """Подстройка спинбокса под параметры выбранного метода."""
        params = core.METHOD_PARAMS.get(method, {})
        if params:
            k = next(iter(params))
            lo, hi, cur = params[k]
            self.spinParam.setRange(lo, hi)
            if hi <= 10:
                self.spinParam.setDecimals(0)
                self.spinParam.setSingleStep(1.0)
            else:
                self.spinParam.setDecimals(1 if k == "sigma" else 0)
                self.spinParam.setSingleStep(0.1 if k == "sigma" else 10.0)
            self.spinParam.setValue(cur)
            self.spinParam.setEnabled(True)
        else:
            self.spinParam.setEnabled(False)

    def _show_method_help(self):
        name = self.cmbMethod.currentText()
        text = METHODS_HELP.get(name, "Нет описания.")
        QMessageBox.information(self, f"Метод: {name}", f"{name}\n\n{text}")

    def _current_step(self):
        """Шаблон шага цепочки из выбранного метода и значения спинбокса."""
        method = self.cmbMethod.currentText()
        step = {"method": method}
        if method in ("closing", "dilate", "erode"):
            step["radius"] = int(self.spinParam.value())
        elif method == "closing_z":
            step["length"] = int(self.spinParam.value())
        elif method == "gaussian":
            step["sigma"] = float(self.spinParam.value())
        elif method == "remove_small":
            step["min_voxels"] = int(self.spinParam.value())
        elif method == "bridge":
            step["components"] = int(self.spinParam.value())
            step["thickness"] = 2
        return step

    def _apply_method(self):
        if self.base_seg is None or self.selected_label is None:
            QMessageBox.information(self, "Постпроцессинг", "Загрузите кейс и выберите орган.")
            return
        method = self.cmbMethod.currentText()
        if method == "fill_all":
            answer = QMessageBox.question(
                self, "fill_all",
                "Заполнит ВСЕ замкнутые дыры органа, включая занятые вложенными\n"
                "органами (воксели линзы/спинного мозга внутри области будут\n"
                "перезаписаны меткой глаза/канала). Продолжить?",
                QMessageBox.Yes | QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        lbl = self.selected_label
        self.busy("Применение метода...")
        try:
            step = self._current_step()
            mask = self.base_seg == lbl
            chain = self.chains.get(lbl, []) + [step]
            processed = core.apply_chain(mask, self.base_seg, lbl, chain)
            self.chains[lbl] = chain
            self.work_seg[self.work_seg == lbl] = 0
            self.work_seg[processed] = lbl
            # инвалидация кэша метрик для этого органа
            self._invalidate_metrics_cache(lbl)
            self.redraw()
            self._update_metrics()
        except Exception as e:
            QMessageBox.critical(self, "Постпроцессинг", f"Ошибка: {e}")
        finally:
            self.busy(None)

    def _reset_organ(self):
        """Отменяет ПОСЛЕДНИЙ метод цепочки выбранного органа."""
        if self.selected_label is None or self.base_seg is None:
            return
        lbl = self.selected_label
        chain = self.chains.get(lbl, [])
        if not chain:
            return
        chain = chain[:-1]
        if chain:
            self.chains[lbl] = chain
            processed = core.apply_chain(self.base_seg == lbl, self.base_seg, lbl, chain)
            self.work_seg[self.work_seg == lbl] = 0
            self.work_seg[processed] = lbl
        else:
            self.chains.pop(lbl, None)
            self.work_seg[self.work_seg == lbl] = 0
            self.work_seg[self.base_seg == lbl] = lbl
        # инвалидация кэша метрик для этого органа
        self._invalidate_metrics_cache(lbl)
        self.redraw()
        self._update_metrics()

    def _reset_all(self):
        if self.base_seg is None:
            return
        self.chains = {}
        self.work_seg = self.base_seg.copy()
        # инвалидация всего кэша метрик
        self._metrics_cache.clear()
        self.redraw()
        self._update_metrics()

    # ---------- шаблоны ----------

    def _refresh_templates(self):
        self.cmbTemplates.blockSignals(True)
        self.cmbTemplates.clear()
        tpl = core.list_templates()
        self.cmbTemplates.addItems(tpl)
        if tpl:
            self.cmbTemplates.setCurrentIndex(0)
        self.cmbTemplates.blockSignals(False)

    def _apply_template(self):
        if self.base_seg is None:
            return
        name = self.cmbTemplates.currentText().strip()
        if not name:
            return
        self.busy("Применение шаблона...")
        try:
            tmpl = core.load_template(core.template_path(name))
            self.chains = {int(v["label"]): v["methods"] for v in tmpl.get("organs", {}).values()
                           if v.get("methods")}
            self.work_seg = core.apply_template(self.base_seg, tmpl)
            # инвалидация кэша метрик после применения шаблона
            self._metrics_cache.clear()
            self.redraw()
            self._update_metrics()
        except Exception as e:
            QMessageBox.critical(self, "Постпроцессинг", f"Ошибка: {e}")
        finally:
            self.busy(None)

    def _batch_apply(self):
        """Применяет выбранный шаблон ко всем валидационным кейсам текущей
        модели: перезаписывает seg_nnunet\<кейс>.nii.gz и пишет таблицу
        Dice до/после (по органам, против GT) в logs\postprocess_batch*.csv.
        Выполняется в рабочем потоке (start_job главного окна)."""
        name = self.cmbTemplates.currentText().strip()
        if not name:
            QMessageBox.information(self, "Постпроцессинг", "Выберите шаблон.")
            return
        try:
            tmpl = core.load_template(core.template_path(name))
        except Exception as e:
            QMessageBox.critical(self, "Постпроцессинг", f"Шаблон: {e}")
            return
        vcases = [(c, img, gt) for c, img, gt in config.validation_cases() if gt]
        if not vcases:
            QMessageBox.information(self, "Постпроцессинг", "Нет валидационных кейсов с GT.")
            return

        def body(log):
            self._batch_body(tmpl, name, log)

        if self.start_job is None:
            self._batch_body(tmpl, name, self.log)
            return
        self.start_job(
            f"Батч-постпроцессинг {name}: {len(vcases)} кейсов",
            in_process=body,
            on_done=lambda ok, t: (
                QMessageBox.information(self, "Батч-постпроцессинг",
                                        "Готово — CSV в logs\\postprocess_batch*.csv, сводка в логе.")
                if ok else QMessageBox.critical(self, "Батч-постпроцессинг",
                                                "Ошибка — см. лог.")))

    def _batch_body(self, tmpl, name, log):
        import csv
        import shutil
        import statistics
        rows = []
        for case, img, gt_p in config.validation_cases():
            if not gt_p:
                continue
            seg_p = os.path.join(config.validation_cases_dir(), case,
                                 "seg_nnunet", f"{case}.nii.gz")
            if not os.path.isfile(seg_p):
                log(f"[batch] {case}: нет сегментации {seg_p} — пропуск\n")
                continue
            nii = sitk.ReadImage(seg_p)
            base = sitk.GetArrayFromImage(nii).astype(np.uint8)
            processed = core.apply_template(base, tmpl)
            # проверка формы перед записью
            if processed.shape != base.shape:
                log(f"[batch] {case}: processed.shape {processed.shape} != base.shape {base.shape} — пропуск записи\n")
                continue
            # бэкап исходного файла
            bak_p = seg_p + ".bak"
            try:
                shutil.copy2(seg_p, bak_p)
            except OSError as e:
                log(f"[batch] {case}: не удалось создать бэкап {bak_p}: {e}\n")
            out = sitk.GetImageFromArray(processed)
            out.CopyInformation(nii)
            sitk.WriteImage(out, seg_p)
            gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_p)).astype(np.uint8)
            for lbl_s, spec in tmpl.get("organs", {}).items():
                lbl = int(lbl_s) if str(lbl_s).isdigit() else int(spec.get("label", 0))
                if lbl == 0:
                    continue
                d0 = core.dice(base == lbl, gt == lbl)
                d1 = core.dice(processed == lbl, gt == lbl)
                rows.append((case, lbl, spec.get("organ", lbl), d0, d1))
            log(f"[batch] {case}: обработан\n")
        csv_path = os.path.join(config.LOG_DIR, f"postprocess_batch_{name}.csv")
        os.makedirs(config.LOG_DIR, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["case", "label", "organ", "dice_before", "dice_after", "delta"])
            for case, lbl, organ, d0, d1 in rows:
                w.writerow([case, lbl, organ,
                            "" if d0 is None else f"{d0:.4f}",
                            "" if d1 is None else f"{d1:.4f}",
                            "" if (d0 is None or d1 is None) else f"{d1 - d0:+.4f}"])
        if rows:
            by_lbl = {}
            for case, lbl, organ, d0, d1 in rows:
                if d0 is None or d1 is None:
                    continue
                by_lbl.setdefault((lbl, organ), []).append((d0, d1))
            log("=== Батч-постпроцессинг: Dice до → после (mean по кейсам) ===\n")
            for (lbl, organ), vals in sorted(by_lbl.items()):
                m0 = statistics.mean(v[0] for v in vals)
                m1 = statistics.mean(v[1] for v in vals)
                log(f"  {organ:<20} {m0:.3f} → {m1:.3f} ({m1 - m0:+.3f})\n")
        log(f"CSV: {csv_path}\n")

    def _template_from_chains(self):
        organs = {}
        for lbl, chain in self.chains.items():
            organs[self._organ_name(lbl)] = {"label": lbl, "methods": chain}
        return {"template_version": 1, "organs": organs}

    def _save_template(self):
        if not self.chains:
            QMessageBox.information(self, "Шаблон", "Цепочки пусты — нечего сохранять.")
            return
        name, ok = QInputDialog.getText(self, "Шаблон", "Имя шаблона (латиницей):")
        if not ok or not name:
            return
        try:
            core.save_template(core.template_path(name.strip()), self._template_from_chains())
            self._refresh_templates()
            self.cmbTemplates.setCurrentText(name.strip())
            self.log(f"[postprocess] шаблон сохранён: {name.strip()}\n")
        except Exception as e:
            QMessageBox.critical(self, "Шаблон", f"Ошибка: {e}")

    # ---------- отрисовка ----------

    def redraw(self):
        if self.ct is None:
            return
        z = self.sliderSlice.value()
        w, l = WINDOW_PRESETS.get(self.cmbWindow.currentText(), (400, 40))
        vmin, vmax = l - w / 2, l + w / 2

        for key, ax, data, title in (
                ("gt", self.mplCanvas.ax_gt,
                 self.gt if self.gt is not None else self.base_seg,
                 "GT" if self.gt is not None else "инференс (база)"),
                ("work", self.mplCanvas.ax_work, self.work_seg,
                 "инференс + постпроцессинг")):
            ax.clear()
            ax.imshow(self.ct[z], cmap="gray", vmin=vmin, vmax=vmax)
            sl = data[z]
            present = [int(v) for v in np.unique(sl) if v > 0]
            if present:
                # ОДИН contour-вызов на панель: уровни = метки-0.5, цвета по меткам
                levels = [v - 0.5 for v in present]
                colors = [_organ_color(v) for v in present]
                widths = [2.4 if v == self.selected_label else 0.9 for v in present]
                ax.contour(sl, levels=levels, colors=colors, linewidths=widths)
            if self.selected_label in present:
                m = (sl == self.selected_label).astype(np.float32)
                ax.contourf(m, levels=[0.5, 1.5],
                            colors=[_organ_color(self.selected_label)], alpha=0.35)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            # ориентация: срез лицом вверх (clear() сбрасывает — применяем явно)
            ax.set_ylim(ax.get_ylim()[::-1])
            # зум (если был) — поверх ориентации
            zl = self.mplCanvas.zoom_lim.get(key)
            if zl:
                ax.set_xlim(*zl[0])
                ax.set_ylim(*zl[1])
        self.mplCanvas.canvas.draw_idle()
