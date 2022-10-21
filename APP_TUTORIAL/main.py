from PySide2.QtWidgets import QApplication
from application import Home
import sys


def main():
    app = QApplication(sys.argv)
    home = Home()
    home.show()
    app.exit(app.exec_())

if __name__ == '__main__':
    main()