from PySide2.QtWidgets import QMainWindow
from .utils import loadUi


class Home(QMainWindow):
    def __init__(self):
        super(Home,self).__init__()
        loadUi('./ui_files\mainwindow.ui',self)
        self.setFixedSize(740,680)

