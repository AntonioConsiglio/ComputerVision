from types import MethodType
import numpy as np
import cv2
from tqdm import tqdm
from PySide2.QtWidgets import QMainWindow,QLabel,QWidget,QVBoxLayout,QProgressBar,QDialog
from PySide2.QtGui import QImage,QPixmap,QPainter,QWheelEvent,QCursor
from PySide2.QtCore import Qt,Signal,QEvent,QSize,QThread,QMetaObject,Slot,QObject
from PIL.ImageQt import ImageQt
from pdf2image import convert_from_path
import fitz
from .utils import loadUi,approximation
# from .elaboration import ElaborationManager,ElaborationHandler
from PIL import ImageDraw,Image
from threading import Thread

class Test(QWidget):

	def __init__(self,image):
		super(Test, self).__init__()
		self.painter = QPainter()
		# placeholder for the real stuff to draw
		self.image = image

	def paintEvent(self, evt):
		rect = evt.rect()
		evt.accept()
		self.painter.begin(self)
		zoomedImage = self.image   # ... calculate this for your images
		sourceRect = rect          # ... caluclate this ...
		# draw it directly
		self.painter.drawImage(rect, self.image, sourceRect)
		self.painter.end()

class CustomSignal(QObject):
	downloadPage = Signal(int)

	def __init__(self):
		super(CustomSignal,self).__init__()

class UploadWorker(QThread):
	progress_update = Signal(int)

	def __init__(self, obj_widget,document,):
		super(UploadWorker, self).__init__()
		self.obj_widget = obj_widget
		self.document = document
		self.obj_widget.image = None

	def run(self):
		
		pages = []
		for i in tqdm(range(len(self.document))):
			pixmap = self.document.get_page_pixmap(i,dpi=300)
			pages.append( Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples))
		self.obj_widget.image = pages

def downloadPagebyClick(self):
	scrollbar = self.verticalScrollBar()
	mainy = self.parent().parent().y()
	scrolly = self.y()
	cursorpos = max(QCursor().pos().y()-mainy-scrolly-42,0)
	print(scrollbar.value()+cursorpos)
	position = min((scrollbar.value()+cursorpos),808*self.npages)
	page2consider = int((position/(808*self.npages))*self.npages)
	self.pageDownloaded.downloadPage.emit(page2consider)
	#self.pageDownloaded.emit(page2consider)	

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
		document = fitz.Document(filepath)
		if not document.is_pdf:
			return event.ignore()
		if document.needs_pass:
			self.pswindow.exec()
			password = str(self.pswindow.get_password())
			document.authenticate(password)
		self.npages = document.page_count
		self.setAcceptDrops(False)
		self.widget = QWidget()
		self.vbox = QVBoxLayout()
		self.t = UploadWorker(self,document)
		self.t.finished.connect(lambda: update_scroll_bar(self))
		self.t.start()
		event.accept()
	else:
		event.ignore()

def update_scroll_bar(scroll):

	result = scroll.app.instance().thread()
	scroll.moveToThread(result)
	scroll.vbox.moveToThread(result)
	scroll.widget.moveToThread(result)
	for number,image_i in enumerate(tqdm(scroll.image),start = 1):
			image_i = image_i.resize((556,800))
			drawer = ImageDraw.Draw(image_i)
			drawer.text((30,30),f"PG: {number}",fill=(255,0,0))
			image = ImageQt(image_i)
			test = Test(image)
			test.setMinimumSize(image.size())
			scroll.vbox.addWidget(test)
			#self.progress_update.emit(number)
	if scroll.widget is not None:
		scroll.widget.setLayout(scroll.vbox)
		scroll.widget.adjustSize()
		widget = scroll.widget
		scroll.setWidget(widget)
	scroll.setAcceptDrops(True)

class PasswordWindow(QDialog):

	def __init__(self,parent):
		super(PasswordWindow,self).__init__(parent=parent)
		loadUi("./uiFiles\password.ui",self)
	
	def get_password(self):
		password = self.password.text()
		return password

class Home(QMainWindow):
	def __init__(self,app):
		super(Home,self).__init__()
		self.app = app
		loadUi("./uiFiles\homeScroll.ui",self)
		self.setFixedSize(self.size())
		self.image_label.installEventFilter(self)
		self._add_method_to_label_class(self.image_label)
		self.extract.clicked.connect(self.extract_page)

	def create_progress_bar(self):
		self.progress_bar = QProgressBar(self)
		self.progress_bar.setGeometry(10, 80, 280, 20)

	def _add_method_to_label_class(self,label):

		label.setAcceptDrops(True)
		label.image = None
		label.npimg = None
		label.app = self.app
		label.dragEnterEvent = MethodType(dragEnterEvent,label)
		label.dragMoveEvent = MethodType(dragMoveEvent,label)
		label.dropEvent = MethodType(dropEvent,label)
		label.downloadPagebyClick = MethodType(downloadPagebyClick,label)
		label.pswindow = PasswordWindow(label)
		label.pageDownloaded = CustomSignal()
		label.pageDownloaded.downloadPage.connect(self.extract_page)
		# label.update_scroll_bar = MethodType(update_scroll_bar,label)
		label.password = "consiglio"
		# label.pageDownloaded.connect(self.extract_page)


	def extract_page(self,page=None):
		if page is None:
			idx = self.extractor.text()
			if "-" in idx:
				s,e = [int(n) for n in idx.split("-")]
				img2save = self.image_label.image[s-1]
				img2save.save(f"./page{s}-{e}.pdf",save_all=True,append_images=self.image_label.image[s:e])
				return
			idx = int(idx)
			img2save = self.image_label.image[idx-1]
			img2save.save(f"./page{idx}.pdf")
			return 
		idx = page-1
		img2save = self.image_label.image[idx]
		img2save.save(f"./page{idx}.pdf")
		


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
		if e.type() == QEvent.MouseButtonDblClick:
			self.image_label.downloadPagebyClick()
		else:
			print(e.type())
			# ...
			return True
		return False #remember to return false for other event types



