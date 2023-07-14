import sys 

from PySide2.QtWidgets import QApplication

from application.home import Home

def main():
    app = QApplication(sys.argv)
    root = Home(app)
    root.show()
    
    app.exec_()

if __name__ == "__main__":
    main()