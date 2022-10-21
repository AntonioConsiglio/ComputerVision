from types import MethodType
from unittest import result

import numpy as np
import cv2
from PySide2.QtWidgets import QMainWindow,QLabel
from PySide2.QtGui import QImage,QPixmap
from PySide2.QtCore import Qt,Signal,QEvent
from PIL.ImageQt import ImageQt
from pdf2image import convert_from_path

from .utils import loadUi
from .elaboration import ElaborationManager,ElaborationHandler

def dragEnterEvent(self, event):

	if event.mimeData().hasUrls:
		event.accept()
	else:
		event.ignore()

def dragMoveEvent(self, event):

	if event.mimeData().hasUrls:
		event.accept()
	else:
		event.ignore()

def dropEvent(self,event):
	
	if event.mimeData().hasUrls:
		event.setDropAction(Qt.CopyAction)
		filepath = event.mimeData().urls()[0].toLocalFile()
		self.set_image(filepath)
		event.accept()
	else:
		event.ignore()

def set_image(self,filepath):
	self.image = None
	self.image = convert_from_path(filepath)[0]
	self.npimg = np.array(self.image)
	image = ImageQt(self.image)
	qpimage = QPixmap.fromImage(image)
	qpimage = qpimage.scaled(556, 800,Qt.KeepAspectRatio)
	self.setPixmap(qpimage)
	self.setAlignment(Qt.AlignCenter)


class Home(QMainWindow):
	def __init__(self):
		super(Home,self).__init__()
		loadUi("./uiFiles\home.ui",self)
		self.image_label.installEventFilter(self)
		self._add_method_to_label_class(self.image_label)
		self.create_dictionary()
		self.elabManager = ElaborationManager()
		self.elabHandler = ElaborationHandler(self.elabManager.results)
		self.elabHandler.new_results.connect(self.plot_results)
		self.readButton.clicked.connect(self.read_image)
		self.elabHandler.start()
		

	def create_dictionary(self):
		self.image_labels = {}
		self.image_labels['nome'] = self.image_name
		self.image_labels['data'] = self.image_data
		self.image_labels['targa'] = self.image_targa
		self.image_labels['cf'] = self.image_cf
		self.image_labels['via'] = self.image_via 

		self.text_labels = {}
		self.text_labels['nome'] = self.text_name
		self.text_labels['data'] = self.text_data
		self.text_labels['targa'] = self.text_targa
		self.text_labels['cf'] = self.text_cf
		self.text_labels['via'] = self.text_via


	def _add_method_to_label_class(self,label):
		label.setAcceptDrops(True)
		label.image = None
		label.npimg = None
		label.dragEnterEvent = MethodType(dragEnterEvent,label)
		label.dragMoveEvent = MethodType(dragMoveEvent,label)
		label.dropEvent = MethodType(dropEvent,label)
		label.set_image = MethodType(set_image,label)
	
	def plot_results(self,results):

		for key,(image,text) in results.items():
			if key == "nome":
				self.image_labels[key].setPixmap(self.imgtoqpixmap(image))
				text = text.tolist()
				text = " ".join(text)
				self.text_labels[key].setText(text)
			elif key == "via":
				self.image_labels[key].setPixmap(self.imgtoqpixmap(image))
				text = text.tolist()
				text = " ".join(text)
				self.text_labels[key].setText(text.replace("()",""))
			else:
				self.image_labels[key].setPixmap(self.imgtoqpixmap(image))
				self.text_labels[key].setText(text.replace(')',""))
	
	def imgtoqpixmap(self,image):
		h,w,c = image.shape
		bytesPerLine = c*w
		qimage = QImage(image,w,h,bytesPerLine,QImage.Format.Format_RGB888).scaled(250,70)
		qpimage = QPixmap.fromImage(qimage)
		return qpimage

	def read_image(self):
		if self.image_label.npimg is not None and self.elabManager is not None:
			self.elabManager.imagequeue.put(self.image_label.npimg)
	
	def eventFilter(self, o, e):
		if e.type() == QEvent.DragEnter: #remember to accept the enter event
			self.image_label.dragEnterEvent(e)
		if e.type() == QEvent.Drop:
			self.image_label.dropEvent(e)
			# ...
			return True
		return False #remember to return false for other event types
		


