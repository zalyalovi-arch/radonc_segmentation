# ============================================================
# mpl_canvas.py — канвас matplotlib для вкладки «Постпроцессинг»
# ------------------------------------------------------------
# В Qt Designer промоутится как класс MplCanvas (header: gui.mpl_canvas).
# Колесо мыши — листать срезы (сигнал sliceRequested),
# Ctrl+колесо — зум вокруг курсора (только та панель, под которой
# курсор; масштаб сохраняется между перерисовками в zoom_lim).
# ============================================================
import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

ZOOM_FACTOR = 1.25


class _InterCanvas(FigureCanvasQTAgg):
    """Перехватывает колесо до matplotlib: без Ctrl — сигнал смены среза,
    с Ctrl — зум. Обычные события мыши работают как раньше (mpl-колбэки)."""

    wheelPlain = Signal(object)   # QWheelEvent без Ctrl
    wheelCtrl = Signal(object)    # QWheelEvent с Ctrl

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            self.wheelCtrl.emit(ev)
        else:
            self.wheelPlain.emit(ev)
        ev.accept()


class MplCanvas(QWidget):
    sliceRequested = Signal(int)  # +1 / -1 — листать срезы

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(6.6, 3.9), dpi=100)
        self.ax_gt = self.fig.add_subplot(121)
        self.ax_work = self.fig.add_subplot(122)
        self.canvas = _InterCanvas(self.fig)
        lay.addWidget(self.canvas)
        # {(xlim, ylim) | None} по панелям; None = весь срез
        self.zoom_lim = {"gt": None, "work": None}
        self.canvas.wheelPlain.connect(self._on_wheel)
        self.canvas.wheelCtrl.connect(self._on_ctrl_wheel)

    # ---------- колесо ----------

    def _display_coords(self, ev):
        """QWheelEvent -> координаты в системе отображения matplotlib."""
        try:
            return self.canvas.mouseEventCoords(ev)
        except Exception:
            x = ev.position().x()
            y = self.canvas.height() - ev.position().y()
            return x, y

    def _on_wheel(self, ev):
        self.sliceRequested.emit(1 if ev.angleDelta().y() > 0 else -1)

    def _on_ctrl_wheel(self, ev):
        x, y = self._display_coords(ev)
        step = 1 if ev.angleDelta().y() > 0 else -1
        for key, ax in (("gt", self.ax_gt), ("work", self.ax_work)):
            if ax.bbox.contains(x, y):
                self.zoom_at(ax, x, y, step)
                self.zoom_lim[key] = (ax.get_xlim(), ax.get_ylim())
                self.canvas.draw_idle()
                break

    # ---------- зум ----------

    @staticmethod
    def zoom_at(ax, x, y, step):
        """Зум вокруг точки (x, y) в координатах отображения."""
        f = ZOOM_FACTOR if step > 0 else 1 / ZOOM_FACTOR
        dx, dy = ax.transData.inverted().transform((x, y))
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(dx - (dx - x0) * f, dx + (x1 - dx) * f)
        ax.set_ylim(dy - (dy - y0) * f, dy + (y1 - dy) * f)

    def reset_zoom(self):
        self.zoom_lim = {"gt": None, "work": None}

    # ---------- вспомогательное ----------

    def pick_axes(self, x, y):
        """Ось под точкой отображения (или None — между панелями)."""
        for ax in (self.ax_gt, self.ax_work):
            if ax.bbox.contains(x, y):
                return ax
        return None

    def mouse_event(self, x, y, step=0):
        """MouseEvent в координатах отображения (для расчётов в логике вкладки)."""
        return MouseEvent("gui", self.canvas, x, y, step=step)
