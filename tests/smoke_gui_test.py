# ============================================================
# smoke_gui_test.py — проверка PySide6-GUI без монитора (offscreen)
# ------------------------------------------------------------
# Собирает MainWindow, ждёт фоновой проверки окружения, грузит
# валидационный кейс на вкладке «Постпроцессинг», снимает скриншоты
# всех вкладок в _screenshots\. Запуск:
#   python -m pytest tests\smoke_gui_test.py  (из корня проекта)
# ============================================================
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# offscreen-платформа не видит шрифты Windows — указываем папку явно,
# иначе весь текст на скриншотах рисуется квадратиками
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

# pytest добавляет корень проекта через conftest.py; при прямом запуске
# cwd и так корень (python -m tests.smoke_gui_test)

from PySide6.QtWidgets import QApplication  # noqa: E402


def main():
    app = QApplication([])
    app.setStyle("Fusion")

    from gui.main_window import MainWindow
    win = MainWindow()
    win.show()
    app.processEvents()

    # КРИТИЧНО: содержимое окна должно быть реально видимым (QUiLoader
    # создаёт формы скрытыми — grab() это маскирует)
    cw = win.centralWidget()
    assert cw is not None, "centralWidget не создан"
    assert cw.isVisible(), (
        "centralWidget скрыт — окно будет ПУСТЫМ "
        "(QUiLoader.load создаёт форму скрытой, нужен setVisible(True))")
    print("видимость окна: OK")

    # ждём завершения фоновой проверки окружения (до 120 с)
    t0 = time.time()
    while win._busy() and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()

    shots = os.path.join(ROOT, "_screenshots")
    os.makedirs(shots, exist_ok=True)

    def shot(name):
        app.processEvents()
        win.grab().save(os.path.join(shots, name + ".png"))
        print("скриншот:", name)

    def idx_of(tab):
        i = win.tabWidget.indexOf(tab)
        return max(i, 0)

    # --- вкладка «Работа» ---
    win.tabWidget.setCurrentIndex(0)
    shot("tab_work")

    # --- вкладка «Квантизация» ---
    win.tabWidget.setCurrentIndex(idx_of(win.tabQuant))
    shot("tab_quant")

    # --- вкладка «Постпроцессинг» (если доступна) ---
    if getattr(win, "post_tab", None) is not None:
        win.tabWidget.setCurrentIndex(idx_of(win.tabPost))
        app.processEvents()
        assert win.post_tab.isVisible(), "вкладка постпроцессинга скрыта"
        pt = win.post_tab
        print("кейсов найдено:", len(pt._cases))
        if pt._cases:
            pt._load_case()
            app.processEvents()
            shot("tab_post_loaded")
            # тест зума: приблизим центр левой панели
            fig = pt.mplCanvas
            bb = fig.ax_gt.bbox
            x, y = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            fig.zoom_at(fig.ax_gt, x, y, 1)
            fig.zoom_lim["gt"] = (fig.ax_gt.get_xlim(), fig.ax_gt.get_ylim())
            pt.redraw()
            app.processEvents()
            shot("tab_post_zoomed")
            # тест цепочки методов: median на первом органе с маской
            if pt.organTree.topLevelItemCount() > 0:
                pt.organTree.setCurrentItem(pt.organTree.topLevelItem(0))
            app.processEvents()
            if pt.selected_label is not None:
                import numpy as np
                if (pt.base_seg == pt.selected_label).any():
                    pt._apply_method()
                    app.processEvents()
                    shot("tab_post_applied")
                    print("метод применён к метке", pt.selected_label,
                          "цепочка:", pt.lblChain.text())
    else:
        print("вкладка «Постпроцессинг» недоступна:", getattr(sys.modules["gui.main_window"], "_POST_ERR", "?"))

    # --- вкладка «Настройки» ---
    win.tabWidget.setCurrentIndex(idx_of(win.tabSettings))
    shot("tab_settings")

    win.close()
    print("SMOKE OK")


def test_smoke_gui():
    main()


if __name__ == "__main__":
    main()
