import sys
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import APP_ICON_PATH
from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
