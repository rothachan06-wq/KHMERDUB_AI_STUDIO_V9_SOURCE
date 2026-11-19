import os
import sys

def _register_cuda_dll_paths():
    if os.name != "nt":
        return
    candidates = [
        os.path.dirname(sys.executable),
        os.path.join(os.path.dirname(sys.executable), "_internal"),
        r"F:\VoxCPM2\.venv\Lib\site-packages\torch\lib",
    ]
    for folder in candidates:
        if os.path.isdir(folder):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(folder)
                except Exception:
                    pass

_register_cuda_dll_paths()

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication
from studio.ui.main_window import KhmerDubProMainWindow

def main():
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = KhmerDubProMainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
