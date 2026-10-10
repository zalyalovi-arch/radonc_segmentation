# ============================================================
# postprocess_viewer.py — вкладка GUI «Постпроцессинг»
# ------------------------------------------------------------
# Две панели (GT | инференс+постпроцессинг) на аксиальных срезах,
# выбор органа, per-структурные методы с мгновенными Dice/HD95,
# шаблоны JSON (см. postprocess_core.py).
# Импортируется nnunet_gui.py: PostprocessTab(notebook).
# ============================================================
import glob
import os

import numpy as np
import SimpleITK as sitk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import hsv_to_rgb

from radonc_segmentation import config
from radonc_segmentation import postprocess_core as core

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
    return hsv_to_rgb(((label - 1) % n / n, 0.85, 0.95))


class PostprocessTab(ttk.Frame):
    def __init__(self, notebook, log=lambda s: None):
        super().__init__(notebook, padding=4)
        self.log = log
        self.ct = None          # (z,y,x) float32, HU
        self.spacing = None     # (z,y,x) мм
        self.base_seg = None    # (z,y,x) uint8
        self.gt = None          # (z,y,x) uint8 или None
        self.work_seg = None
        self.chains = {}        # {label: [steps]}
        self.case = None
        self.slice_idx = tk.IntVar(value=0)
        self.selected_label = None
        self._redraw_after = None
        self.zoom_lim = {"gt": None, "work": None}  # (xlim, ylim) или None

        self._build_case_bar()
        self._build_template_bar()
        body = tk.Frame(self)
        body.pack(fill="both", expand=True)
        self._build_view(body)
        self._build_organ_and_method(body)

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

    def _build_case_bar(self):
        bar = tk.Frame(self)
        bar.pack(fill="x")
        tk.Label(bar, text="Кейс:").pack(side="left")
        self.case_combo = ttk.Combobox(bar, state="readonly", width=40)
        self.case_combo.pack(side="left", padx=4)
        tk.Button(bar, text="Обновить список", command=self._refresh_cases).pack(side="left", padx=2)
        tk.Button(bar, text="Загрузить", command=self._load_case).pack(side="left", padx=2)
        self.lbl_info = tk.Label(bar, text="кейс не загружен", fg="gray")
        self.lbl_info.pack(side="left", padx=10)
        self._cases = self._scan_cases()
        self.case_combo["values"] = [c[0] for c in self._cases]
        if self._cases:
            self.case_combo.current(0)

    def _refresh_cases(self):
        self._cases = self._scan_cases()
        self.case_combo["values"] = [c[0] for c in self._cases]
        if self._cases:
            self.case_combo.current(0)

    def _load_case(self):
        idx = self.case_combo.current()
        if idx < 0:
            return
        name, ct_path, seg_path, gt_path = self._cases[idx]
        try:
            self.busy("Загрузка...")
            self.ct, self.spacing = _read_nifti_zyx(ct_path)
            self.base_seg, _ = _read_nifti_zyx(seg_path)
            self.base_seg = self.base_seg.astype(np.uint8)
            self.gt = None
            if gt_path:
                self.gt, _ = _read_nifti_zyx(gt_path)
                self.gt = self.gt.astype(np.uint8)
        except Exception as e:
            messagebox.showerror("Постпроцессинг", f"Ошибка загрузки: {e}")
            return
        self.case = name
        self.chains = {}
        self.zoom_lim = {"gt": None, "work": None}
        self.work_seg = self.base_seg.copy()
        self.slice_idx.set(self.ct.shape[0] // 2)
        self.slider.configure(to=self.ct.shape[0] - 1)
        self.lbl_info.configure(
            text=f"{name}: {self.ct.shape}, spacing {tuple(round(s, 2) for s in self.spacing)}" +
                 (", GT есть" if self.gt is not None else ", GT нет (метрики недоступны)"),
            fg="black")
        self._fill_organ_list()
        self._reset_zoom()  # сброс зума старого кейса
        self.redraw()

    def busy(self, msg):
        self.configure(cursor="watch" if msg else "")
        if msg:
            self.lbl_info.configure(text=msg, fg="blue")
        self.update_idletasks()

    # ---------- вид ----------

    def _build_view(self, parent):
        frame = tk.LabelFrame(parent, text="КТ + сегментация (слева GT / справа инференс + постпроцессинг)")
        frame.pack(side="left", fill="both", expand=True)
        top = tk.Frame(frame)
        top.pack(fill="x")
        tk.Label(top, text="Срез Z:").pack(side="left")
        self.slider = tk.Scale(top, variable=self.slice_idx, orient="horizontal",
                               from_=0, to=10, showvalue=True, length=400)
        self.slider.pack(side="left", fill="x", expand=True)
        self.slider.bind("<ButtonRelease-1>", lambda e: self.redraw())
        self.slice_idx.trace_add("write", self._on_slice_change)
        tk.Label(top, text="Окно:").pack(side="left", padx=(10, 0))
        self.win_combo = ttk.Combobox(top, state="readonly", width=12,
                                      values=list(WINDOW_PRESETS), )
        self.win_combo.set("мягкая ткань")
        self.win_combo.pack(side="left", padx=4)
        self.win_combo.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        tk.Button(top, text="Сброс зума", command=self._reset_zoom).pack(side="left", padx=4)

        self.fig = Figure(figsize=(6.6, 3.9), dpi=100)
        self.ax_gt = self.fig.add_subplot(121)
        self.ax_work = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_slice_change(self, *args):
        # дебаунс при движении слайдера
        if self._redraw_after:
            self.after_cancel(self._redraw_after)
        self._redraw_after = self.after(120, self.redraw)

    # ---------- колесо мыши и зум ----------

    def _on_scroll(self, event):
        ctrl = bool(getattr(event, "guiEvent", None) is not None
                    and event.guiEvent.state & 0x0004)  # Windows Control mask
        if ctrl and event.inaxes is not None:
            ax = event.inaxes
            f = 1.25 if event.step > 0 else 1 / 1.25
            x, y = event.xdata, event.ydata
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_xlim(x - (x - x0) * f, x + (x1 - x) * f)
            ax.set_ylim(y - (y - y0) * f, y + (y1 - y) * f)
            key = "gt" if ax is self.ax_gt else "work"
            self.zoom_lim[key] = (ax.get_xlim(), ax.get_ylim())
            self.canvas.draw_idle()
        elif self.ct is not None:
            z = int(self.slice_idx.get()) + (1 if event.step > 0 else -1)
            self.slice_idx.set(max(0, min(self.ct.shape[0] - 1, z)))

    def _reset_zoom(self):
        self.zoom_lim = {"gt": None, "work": None}
        self.redraw()

    # ---------- органы и методы ----------

    def _build_organ_and_method(self, parent):
        side = ttk.Frame(parent, width=430)
        side.pack(side="right", fill="y", padx=4)
        side.pack_propagate(False)

        tree_frame = tk.LabelFrame(side, text="Органы (выберите для постпроцессинга)")
        tree_frame.pack(fill="x")
        self.tree = ttk.Treeview(tree_frame, columns=("dice", "hd95"), height=10, show="tree headings")
        self.tree.heading("#0", text="Орган")
        self.tree.heading("dice", text="Dice")
        self.tree.heading("hd95", text="HD95, мм")
        self.tree.column("#0", width=190)
        self.tree.column("dice", width=60, anchor="e")
        self.tree.column("hd95", width=70, anchor="e")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self._on_organ_select)

        method_frame = tk.LabelFrame(side, text="Постпроцессинг выбранного органа")
        method_frame.pack(fill="x", pady=(6, 0))

        row1 = tk.Frame(method_frame)
        row1.pack(fill="x", pady=(2, 0))
        tk.Label(row1, text="Метод:").pack(side="left")
        self.method_combo = ttk.Combobox(row1, state="readonly", width=13,
                                         values=list(core.METHODS))
        self.method_combo.set("closing")
        self.method_combo.pack(side="left", padx=4)
        self.method_combo.bind("<<ComboboxSelected>>", self._on_method_change)
        tk.Button(row1, text="?", width=2, command=self._show_method_help).pack(side="left", padx=2)

        row2 = tk.Frame(method_frame)
        row2.pack(fill="x", pady=2)
        tk.Label(row2, text="Параметр:").pack(side="left")
        self.param_var = tk.DoubleVar(value=3.0)
        self.param_spin = tk.Spinbox(row2, from_=1, to=8, increment=1,
                                     textvariable=self.param_var, width=7)
        self.param_spin.pack(side="left", padx=6)

        row3 = tk.Frame(method_frame)
        row3.pack(fill="x", pady=2)
        tk.Button(row3, text="Применить", command=self._apply_method, width=11).pack(side="left", padx=2)
        tk.Button(row3, text="Отменить", command=self._reset_organ, width=11).pack(side="left", padx=2)
        tk.Button(row3, text="Сброс", command=self._reset_all, width=8).pack(side="left", padx=2)

        self.lbl_metrics = tk.Label(method_frame, text="Метрики: загрузите кейс с GT", justify="left",
                                    anchor="w", font=("TkDefaultFont", 10, "bold"), wraplength=390)
        self.lbl_metrics.pack(fill="x", pady=4)
        self.lbl_chain = tk.Label(method_frame, text="цепочка: —", justify="left", anchor="w",
                                  fg="gray", wraplength=390)
        self.lbl_chain.pack(fill="x")

        hint = tk.Label(side, fg="gray", justify="left", anchor="w", wraplength=400, text=(
            "Колесо мыши — листать срезы, Ctrl+колесо — зум, «Сброс зума» — вернуть.\n"
            "Выберите орган → метод → «Применить». Методы цепочкой: применяйте\n"
            "последовательно; «Отменить» убирает последний метод цепочки,\n"
            "«Сброс» — всю цепочку органа. Шаблон проверяйте на 5-10 внешних\n"
            "случаях до включения в пайплайн."))
        hint.pack(fill="x", pady=(6, 0))

    def _fill_organ_list(self):
        self.tree.delete(*self.tree.get_children())
        if self.base_seg is None:
            return
        for lbl in sorted(config.LABELS, key=config.LABELS.get):
            name = lbl
            self.tree.insert("", "end", iid=str(config.LABELS[lbl]), text=name, values=("—", "—"))

    def _on_organ_select(self, e=None):
        sel = self.tree.selection()
        if not sel:
            return
        self.selected_label = int(sel[0])
        self._update_metrics()

    def _update_metrics(self):
        if self.base_seg is None or self.selected_label is None:
            return
        lbl = self.selected_label
        name = next(n for n, v in config.LABELS.items() if v == lbl)
        base_m = self.base_seg == lbl
        work_m = self.work_seg == lbl
        vol0, vol1 = int(base_m.sum()), int(work_m.sum())
        if self.gt is not None:
            gt_m = self.gt == lbl
            d0 = core.dice(base_m, gt_m)
            h0 = core.hd95(base_m, gt_m, self.spacing)
            d1 = core.dice(work_m, gt_m)
            h1 = core.hd95(work_m, gt_m, self.spacing)
            def fmt(d, h):
                return ("—" if d is None else f"{d:.3f}") + " / " + ("—" if h is None else f"{h:.1f}")
            better = (d1 is not None and d0 is not None and d1 >= d0)
            self.lbl_metrics.configure(
                text=f"{name}\nДО:      {vol0} вокс., Dice {fmt(d0, h0)}\n"
                     f"ПОСЛЕ: {vol1} вокс., Dice {fmt(d1, h1)}",
                fg="black" if better else "red")
        else:
            self.lbl_metrics.configure(text=f"{name}: {vol0} → {vol1} вокс. (GT нет — метрики недоступны)",
                                       fg="black")
        # обновить колонки таблицы
        if self.gt is not None and d0 is not None:
            self.tree.item(str(lbl), values=(f"{d0:.3f}", "—" if h0 is None else f"{h0:.1f}"))
        # цепочка органа
        chain = self.chains.get(lbl)
        self.lbl_chain.configure(text="цепочка: " + (
            " → ".join(f"{s['method']}({', '.join(f'{k}={v}' for k, v in s.items() if k != 'method')})"
                       for s in chain) if chain else "—"))

    # ---------- применение методов ----------

    def _on_method_change(self, e=None):
        """Подстройка спинбокса под параметры выбранного метода."""
        params = core.METHOD_PARAMS.get(self.method_combo.get(), {})
        if params:
            k = next(iter(params))
            lo, hi, cur = params[k]
            self.param_spin.configure(from_=lo, to=hi,
                                      increment=1 if hi <= 10 else 0.1)
            self.param_var.set(cur)
            self.param_spin.configure(state="normal")
        else:
            self.param_spin.configure(state="disabled")

    def _show_method_help(self):
        name = self.method_combo.get()
        text = METHODS_HELP.get(name, "Нет описания.")
        win = tk.Toplevel(self)
        win.title(f"Метод: {name}")
        win.geometry("420x160")
        win.attributes("-topmost", True)
        tk.Label(win, text=name, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 4))
        tk.Label(win, text=text, justify="left", wraplength=390).pack(anchor="w", padx=10)
        tk.Button(win, text="Закрыть", command=win.destroy).pack(pady=8)

    def _apply_method(self):
        if self.base_seg is None or self.selected_label is None:
            messagebox.showinfo("Постпроцессинг", "Загрузите кейс и выберите орган.")
            return
        method = self.method_combo.get()
        step = {"method": method}
        if method in ("closing", "dilate", "erode"):
            step["radius"] = int(self.param_var.get())
        elif method == "closing_z":
            step["length"] = int(self.param_var.get())
        elif method == "gaussian":
            step["sigma"] = float(self.param_var.get())
        elif method == "remove_small":
            step["min_voxels"] = int(self.param_var.get())
        elif method == "bridge":
            step["components"] = int(self.param_var.get())
            step["thickness"] = 2
        if method == "fill_all":
            if not messagebox.askyesno(
                    "fill_all",
                    "Заполнит ВСЕ замкнутые дыры органа, включая занятые вложенными\n"
                    "органами (воксели линзы/спинного мозга внутри области будут\n"
                    "перезаписаны меткой глаза/канала). Продолжить?"):
                return
        lbl = self.selected_label
        self.busy("Применение метода...")
        try:
            mask = self.base_seg == lbl
            chain = self.chains.get(lbl, []) + [step]
            processed = core.apply_chain(mask, self.base_seg, lbl, chain)
            self.chains[lbl] = chain
            self.work_seg[self.work_seg == lbl] = 0
            self.work_seg[processed] = lbl
            self.redraw()
            self._update_metrics()
        except Exception as e:
            messagebox.showerror("Постпроцессинг", f"Ошибка: {e}")
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
        self.redraw()
        self._update_metrics()

    def _reset_all(self):
        if self.base_seg is None:
            return
        self.chains = {}
        self.work_seg = self.base_seg.copy()
        self.redraw()
        self._update_metrics()

    # ---------- шаблоны ----------

    def _build_template_bar(self):
        bar = tk.LabelFrame(self, text="Шаблоны (JSON, применяются целиком к инференсу)")
        bar.pack(fill="x")
        tk.Label(bar, text="Файл:").pack(side="left")
        self.tpl_combo = ttk.Combobox(bar, state="readonly", width=30,
                                      values=core.list_templates())
        self.tpl_combo.pack(side="left", padx=4)
        tk.Button(bar, text="Применить шаблон", command=self._apply_template).pack(side="left", padx=4)
        tk.Button(bar, text="Применить ко всем валид. кейсам", command=self._batch_apply).pack(side="left", padx=4)
        tk.Button(bar, text="Сохранить как...", command=self._save_template).pack(side="left", padx=4)
        tk.Button(bar, text="Обновить список", command=self._refresh_templates).pack(side="left", padx=2)

    def _refresh_templates(self):
        self.tpl_combo["values"] = core.list_templates()
        if core.list_templates():
            self.tpl_combo.current(0)

    def _apply_template(self):
        if self.base_seg is None:
            return
        name = self.tpl_combo.get()
        if not name:
            return
        try:
            self.busy("Применение шаблона...")
            tmpl = core.load_template(core.template_path(name))
            self.chains = {int(v["label"]): v["methods"] for v in tmpl.get("organs", {}).values()
                           if v.get("methods")}
            self.work_seg = core.apply_template(self.base_seg, tmpl)
            self.redraw()
            self._update_metrics()
        except Exception as e:
            messagebox.showerror("Постпроцессинг", f"Ошибка: {e}")
        finally:
            self.busy(None)

    def _batch_apply(self):
        r"""Применяет выбранный шаблон ко всем валидационным кейсам текущей
        модели: перезаписывает seg_nnunet\<кейс>.nii.gz и пишет таблицу
        Dice до/после (по органам, против GT) в logs\postprocess_batch.csv."""
        name = self.tpl_combo.get()
        if not name:
            messagebox.showinfo("Постпроцессинг", "Выберите шаблон.")
            return
        try:
            tmpl = core.load_template(core.template_path(name))
        except Exception as e:
            messagebox.showerror("Постпроцессинг", f"Шаблон: {e}")
            return
        vcases = [(c, img, gt) for c, img, gt in config.validation_cases() if gt]
        if not vcases:
            messagebox.showinfo("Постпроцессинг", "Нет валидационных кейсов с GT.")
            return
        self.busy(f"Батч-постпроцессинг: {len(vcases)} кейсов...")
        try:
            import csv
            rows = []
            for case, img, gt_p in vcases:
                seg_p = os.path.join(config.validation_cases_dir(), case,
                                     "seg_nnunet", f"{case}.nii.gz")
                if not os.path.isfile(seg_p):
                    self.log(f"[batch] {case}: нет сегментации {seg_p} — пропуск\n")
                    continue
                nii = sitk.ReadImage(seg_p)
                base = sitk.GetArrayFromImage(nii).astype(np.uint8)
                processed = core.apply_template(base, tmpl)
                out = sitk.GetImageFromArray(processed)
                out.CopyInformation(nii)
                sitk.WriteImage(out, seg_p)
                if gt_p and os.path.isfile(gt_p):
                    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_p)).astype(np.uint8)
                    for lbl_s, spec in tmpl.get("organs", {}).items():
                        lbl = int(lbl_s) if str(lbl_s).isdigit() else int(spec.get("label", 0))
                        if lbl == 0:
                            continue
                        d0 = core.dice(base == lbl, gt == lbl)
                        d1 = core.dice(processed == lbl, gt == lbl)
                        rows.append((case, lbl, spec.get("organ", lbl), d0, d1))
                self.log(f"[batch] {case}: обработан\n")
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
                import statistics
                by_lbl = {}
                for case, lbl, organ, d0, d1 in rows:
                    if d0 is None or d1 is None:
                        continue
                    by_lbl.setdefault((lbl, organ), []).append((d0, d1))
                self.log("=== Батч-постпроцессинг: Dice до → после (mean по кейсам) ===\n")
                for (lbl, organ), vals in sorted(by_lbl.items()):
                    m0 = statistics.mean(v[0] for v in vals)
                    m1 = statistics.mean(v[1] for v in vals)
                    self.log(f"  {organ:<20} {m0:.3f} → {m1:.3f} ({m1 - m0:+.3f})\n")
            self.log(f"CSV: {csv_path}\n")
        except Exception as e:
            messagebox.showerror("Постпроцессинг", f"Ошибка батч-применения: {e}")
        finally:
            self.busy(None)

    def _template_from_chains(self):
        organs = {}
        for lbl, chain in self.chains.items():
            name = next((n for n, v in config.LABELS.items() if v == lbl), f"label_{lbl}")
            organs[name] = {"label": lbl, "methods": chain}
        return {"template_version": 1, "organs": organs}

    def _save_template(self):
        if not self.chains:
            messagebox.showinfo("Постпроцессинг", "Цепочки пусты — нечего сохранять.")
            return
        name = simpledialog.askstring("Шаблон", "Имя шаблона (латиницей):", parent=self)
        if not name:
            return
        try:
            core.save_template(core.template_path(name), self._template_from_chains())
            self._refresh_templates()
            self.tpl_combo.set(name)
            self.log(f"[postprocess] шаблон сохранён: {name}")
        except Exception as e:
            messagebox.showerror("Постпроцессинг", f"Ошибка: {e}")

    # ---------- отрисовка ----------

    def redraw(self):
        if self.ct is None:
            return
        z = int(self.slice_idx.get())
        w, lev = WINDOW_PRESETS.get(self.win_combo.get(), (400, 40))
        vmin, vmax = lev - w / 2, lev + w / 2

        for key, ax, data, title in (
                ("gt", self.ax_gt,
                 self.gt if self.gt is not None else self.base_seg,
                 "GT" if self.gt is not None else "инференс (база)"),
                ("work", self.ax_work, self.work_seg, "инференс + постпроцессинг")):
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
                ax.contourf(m, levels=[0.5, 1.5], colors=[_organ_color(self.selected_label)],
                            alpha=0.35)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            # ориентация: срез лицом вверх (clear() сбрасывает — применяем явно)
            ax.set_ylim(ax.get_ylim()[::-1])
            # зум (если был) — поверх ориентации
            if self.zoom_lim[key]:
                ax.set_xlim(*self.zoom_lim[key][0])
                ax.set_ylim(*self.zoom_lim[key][1])
        self.canvas.draw_idle()
