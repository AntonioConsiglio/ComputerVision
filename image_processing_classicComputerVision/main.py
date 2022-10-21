import sys
#3rd party
from PySide2.QtWidgets import QApplication
from mainwindow import MainWindow

def main():

    app = QApplication(sys.argv)
    root = MainWindow()

    root.show()
    app.exec_()

if __name__ == '__main__':
    
    main()
