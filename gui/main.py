# ============================================================
# main.py — точка входа PySide6-версии GUI пайплайна H&N
# ------------------------------------------------------------
# Dev-запуск: run_gui.bat (python окружения nnunet_env).
# exe: nnunet_gui.exe в корне проекта — собирается build.bat
# (spec: nnunet_gui_pyside.spec). Раскладка окон — gui/ui/*.ui
# (Qt Designer), логика — gui/*.
# ============================================================
import os
import sys

if not getattr(sys, "frozen", False):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ROOT)                            # пакет gui
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))   # config и др.

from PySide6.QtWidgets import QApplication  # noqa: E402


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # одинаковый вид на всех машинах
    # НЕ задавать setApplicationDisplayName: Qt дописывает его к windowTitle
    # и заголовок окна задваивается («...H&N - Автоконтуринг H&N»)

    from gui.main_window import MainWindow
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        # в windowed-exe нет консоли: падения пишем в лог РЯДОМ С EXE
        # (не в _MEIPASS — та удаляется при выходе), затем диалог об ошибке
        try:
            main()
        except SystemExit:
            raise
        except Exception:
            import traceback
            log_path = os.path.join(os.path.dirname(os.path.abspath(sys.executable)),
                                    "_gui_error.log")
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            except OSError:
                pass
            raise
    else:
        main()
