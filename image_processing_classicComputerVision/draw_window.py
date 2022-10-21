import sys
from threading import Thread
from types import MethodType
import math

from utils import loadUi
from PySide2.QtWidgets import QWidget,QDialog,QLabel
from PySide2.QtGui import QImage,QPixmap,QPainter,QBrush,QColor
from PySide2.QtCore import Signal,QRect,QPoint,Qt



def verify_is_correct_point(first,second,point):
	condition = False
	if first.x() > second.x()- point.x(): condition1 = True 
	else: condition1 = False
	if first.y() > second.y()- point.y(): condition2 = True
	else: condition2 = False
	if first.x() < second.x() + point.x(): condition3 = True
	else: condition3 = False
	if first.x() < second.x() + point.x(): condition4 = True
	else: condition4 = False

	condition = condition1*condition2*condition3*condition4
	return condition

def calculate_angle(centro,punto_variabile):
	angle = None
	xc = centro.x()
	yc = centro.y()
	xp = punto_variabile.x()
	yp = punto_variabile.y()
	nom = yp-yc
	den = xp-xc
	angle = math.atan2(nom,den)
	angle = math.degrees(angle)
	return angle

def getRotatedPoint( p, center, angle):

	angle = math.radians(angle)

	x = p.x()
	y = p.y()
	s = math.sin(angle)
	c = math.cos(angle)

	x -= center.x()
	y -= center.y()

	# rotate point
	xnew =int(x * c - y * s)
	ynew = int(x * s + y * c)

	# translate point back:
	x = xnew + center.x()
	y = ynew + center.y()

	return QPoint( x, y )




def paintEvent(self,event):

	image_to = self.image.copy()
	qp = QPainter(image_to)
	qp.drawPixmap(0,0,image_to)
	if self.is_rotating:
		print(f'im rotating about this angle: {self.angle}')
		qp.translate(self.center)
		self.angle = calculate_angle(self.middle_point,self.secondpoint)
		if self.angle is not None:
			qp.rotate(self.angle)
			br = QBrush(QColor(100, 10, 10, 40))  
			qp.setBrush(br)
			qp.drawRect(QRect(self.begin-self.center, self.end-self.center))
			qp.drawEllipse(self.center-self.center,1.0,1.0)
			qp.drawEllipse(self.middle_point-self.center,4.0,4.0)
	elif self.is_traslating:
		qp.translate(self.trasl_point)
		if self.angle is not None:
			qp.rotate(self.angle)
			br = QBrush(QColor(100, 10, 10, 40))  
			qp.setBrush(br)
			qp.drawRect(QRect(self.begin-self.center, self.end-self.center))
			qp.drawEllipse(self.center-self.center,1.0,1.0)
			qp.drawEllipse(self.middle_point-self.center,4.0,4.0)
	else:
		br = QBrush(QColor(100, 10, 10, 40))  
		qp.setBrush(br)
		qp.drawRect(QRect(self.begin, self.end))
		qp.drawEllipse(self.center,1.0,1.0)
		qp.drawEllipse(self.middle_point,4.0,4.0)
	qp = QPainter(self)
	qp.drawPixmap(0,0,image_to)
	#self.update()
	# else:
	# 	if verify_is_correct_point(self.rotate_point,self.middle_point,self.point):
	# 		qp.rotate(60)
	# 		print('im rotating the rectangle')
	# 		qp.drawRect(QRect(self.begin, self.end))
	# 		# qp.drawEllipse(self.center,1.0,1.0)
	# 		# qp.drawEllipse(self.middle_point,4.0,4.0)
	# 		self.update()
				

def mousePressEvent(self, event):

	if not self.rectangledone:
		self.begin = event.pos()
		self.end = event.pos()
		self.update()
	else:
		print(f'envet_pos : {event.pos()}')
		print(f'middle_point: {self.middle_point_new}')
		print(f'center: {self.center}')
		if verify_is_correct_point(event.pos(),self.middle_point_new,self.point):
			print('WE ARE ROTATING NOW')
			self.rotate_point = event.pos()
			self.is_rotating=True
			#self.secondpoint == self.middle_point
			#self.update()
		elif verify_is_correct_point(event.pos(),self.center,self.point2):
			self.is_traslating=True
	

def mouseMoveEvent(self, event):
	
	if not self.rectangledone:
		self.end = event.pos()
		self.center = (self.begin+self.end)/2
		width = (self.begin.x() + self.end.x())/2
		self.middle_point.setX(self.end.x())
		self.middle_point.setY(self.center.y())
		self.middle_point_new.setX(self.end.x())
		self.middle_point_new.setY(self.center.y())
		self.update()
	else:
		if self.is_rotating:
			self.secondpoint = event.pos()
			self.update()
		elif self.is_traslating:
			self.trasl_point = event.pos()
			self.update()
	

def mouseReleaseEvent(self, event):
	if not self.rectangledone:
		# self.begin = event.pos()
		# self.end = event.pos()
		self.rectangledone = True
	if self.rectangledone:
		if self.is_rotating:
			print(f'middle_point_before: {self.middle_point}')
			self.middle_point_new = getRotatedPoint(self.middle_point,self.center,self.angle)
			print(f'middle_point_end: {self.middle_point_new}')
		#self.update()
		if self.is_traslating:
			print(f'center_before: {self.center}')
			print(f'trasl_point: {self.trasl_point}')
			diff = self.trasl_point-self.center
			print(f'differenza = {diff}')
			self.begin += diff
			self.end += diff
			self.middle_point += diff
			self.middle_point_new += diff
			self.center = self.trasl_point			# print(self.center)
			print(f'center_after: {self.center}')
		self.is_rotating=False
		self.is_traslating=False

		self.draw_rectangle()

		

def draw_rectangle(self):
	print(f'begin: {self.begin}')
	rect = QRect(self.begin, self.end)
	
	immagine = QPixmap(640,480)
	qp = QPainter(immagine)
	qp.translate(self.center)
	qp.rotate(-self.angle)
	qp.translate(-self.center)
	qp.drawPixmap(0, 0, self.image)
	qp.end()
	self.new_image = immagine.copy(rect)
	print('rectangle_done')
	#self.update()

class DrawWindow(QDialog):
	
	is_closed = Signal(bool)
	updt_train = Signal(QPixmap)

	def __init__(self,image):
		super(DrawWindow,self).__init__()
		loadUi("draw_rectangle.ui",self)
		self._add_method_to_label_class(image)
		self.ok_button.clicked.connect(self.emit_train_image)
		self.cancel_button.clicked.connect(self.close)
		self.image = image

	def closeEvent(self, event):
		self.is_closed.emit(True)

	def emit_train_image(self):
		self.updt_train.emit(self.draw_label.new_image)
		self.close()
	
	def _add_method_to_label_class(self,image):
		self.draw_label.image=image
		self.draw_label.new_image = None
		self.draw_label.begin = QPoint()
		self.draw_label.end = QPoint()
		self.draw_label.center = QPoint()
		self.draw_label.secondpoint = QPoint()
		self.draw_label.middle_point = QPoint()
		self.draw_label.trasl_point = QPoint()
		self.draw_label.middle_point_new = QPoint()
		self.draw_label.point = QPoint(6,6)
		self.draw_label.point2 = QPoint(6,6)
		self.draw_label.rectangledone = False
		self.draw_label.mPixmap = QPixmap()
		self.draw_label.angle = 0
		self.draw_label.is_rotating = False
		self.draw_label.is_traslating = False
		self.draw_label.setPixmap(image)
		self.draw_label.paintEvent = MethodType(paintEvent,self.draw_label)
		self.draw_label.mousePressEvent = MethodType(mousePressEvent,self.draw_label)
		self.draw_label.mouseMoveEvent = MethodType(mouseMoveEvent,self.draw_label)
		self.draw_label.mouseReleaseEvent = MethodType(mouseReleaseEvent,self.draw_label)
		self.draw_label.draw_rectangle = MethodType(draw_rectangle,self.draw_label)
	
