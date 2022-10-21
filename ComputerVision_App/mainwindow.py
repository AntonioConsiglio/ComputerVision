import sys
from threading import Thread
from types import MethodType
import cv2
import numpy as np

from utils import loadUi
from PySide2.QtWidgets import QMainWindow
from PySide2.QtGui import QPixmap,QDoubleValidator
from PySide2.QtCore import Signal,Slot,Qt,QLocale

from draw_window import DrawWindow
from pattern_recognition_function import cv_template_matching,convertoToPixmap,nomaximasuppression
from styles import LineEditStyles

#### FUNCTION USED ##

def dragEnterEvent(self, event):
		if event.mimeData().hasImage:
			event.accept()
		else:
			event.ignore()

def dragMoveEvent(self, event):
	if event.mimeData().hasImage:
		event.accept()
	else:
		event.ignore()

def dropEvent(self,event):
	if event.mimeData().hasImage:
		event.setDropAction(Qt.CopyAction)
		filepath = event.mimeData().urls()[0].toLocalFile()
		self.filedropped.emit(filepath)
		self.set_image(filepath)
		event.accept()
	else:
		event.ignore()

def set_image(self,filepath):
	self.image = QPixmap(filepath)
	self.image = self.image.scaled(640,480)
	self.cvimage = cv2.imread(filepath)
	self.cvimagegray = cv2.cvtColor(self.cvimage,cv2.COLOR_BGR2GRAY)
	# cv2.imshow('immagine',self.cvimage)
	# cv2.waitKey(0)
	self.setPixmap(self.image)

qlinestyle = LineEditStyles()

# def setPixmap(self, image):
# 	super().SetPixmap(image)d

class MainWindow(QMainWindow):
	
	def __init__(self):
		super(MainWindow,self).__init__()
		loadUi("./ui_file\main.ui",self)
		self.dialog_execution = False
		self.validator = QDoubleValidator(0.0,1.0,2)
		self.validator.setLocale(QLocale("en"))
		self.threshold_line.setValidator(self.validator)
		self._define_slots()
		self._add_method_to_label_class()

	def _define_slots(self):
		self.draw_window_button.clicked.connect(self._call_draw_window)
		self.execute_button.clicked.connect(self._call_pattern_detection)
		self.threshold_line.editingFinished.connect(self._change_threshold)
		self.threshold_line.textChanged.connect(self._setcolor)
	
	def _change_threshold(self):

		trsh = float(self.threshold_line.text())
		self.threshold = trsh
	
	def _setcolor(self):

		result = self.validator.validate(self.threshold_line.text(),2)
		if result[0] == 2:
			self.threshold_line.setStyleSheet(qlinestyle.goodtext)
		else:
			self.threshold_line.setStyleSheet(qlinestyle.badtext)
		

	def update_train_image(self,image):
	
		self.train_image_label.setAlignment(Qt.AlignCenter)
		self.train_image_label.setPixmap(image)
		w = image.size().width()
		h = image.size().height()
		image = image.toImage()
		image_array = np.array(image.constBits().asarray(h*w*4)).reshape(h,w,4)
		self.cvtrainimage = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
		cv2.imwrite('template.png',self.cvtrainimage)
		cv2.imshow('gray_train_foto',self.cvtrainimage)
		cv2.waitKey(0)

	def _call_draw_window(self):
		if not self.dialog_execution:
			self.dialog_execution = True
			self.dialog = DrawWindow(self.image_label.image)
			self.dialog.updt_train.connect(self.update_train_image)
			self.dialog.is_closed.connect(self._change_dialog_execution)
			self.dialog.show()
			self.dialog.exec_()
	
	def _change_dialog_execution(self):
		self.dialog_execution = False

	def _call_pattern_detection(self):
		#TODO: implement the pattern recognition algorithm 
		h,w = self.cvtrainimage.shape
		risultati = cv_template_matching(self.image_label.cvimagegray,self.cvtrainimage,1)
		print(self.threshold)
		loc = np.where( risultati >= self.threshold)
		image_to_plot = self.image_label.cvimage.copy()
		predictions = []
		for pt in zip(*loc[::-1]):
			value = risultati[pt[1],pt[0]]
			xmin,ymin = pt
			xmax = pt[0] + w
			ymax = pt[1] + h
			predictions.append([value,[xmin,ymin,xmax,ymax]])
		predictions = nomaximasuppression(predictions)
		for score,bbox in predictions:
			cv2.rectangle(image_to_plot, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 2)
		# cv2.imshow('risultati',image_to_plots)
		# cv2.waitKey(0)
		self.image_label.setPixmap(convertoToPixmap(image_to_plot))

	def _add_method_to_label_class(self):
		self.image_label.setAcceptDrops(True)
		self.image_label.image = None
		self.image_label.dragEnterEvent = MethodType(dragEnterEvent,self.image_label)
		self.image_label.dragMoveEvent = MethodType(dragMoveEvent,self.image_label)
		self.image_label.dropEvent = MethodType(dropEvent,self.image_label)
		self.image_label.set_image = MethodType(set_image,self.image_label)
	


	
