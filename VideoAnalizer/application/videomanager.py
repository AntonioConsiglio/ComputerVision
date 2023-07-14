from ast import Break
#from PySide2.QtCore import QThread,Signal
import numpy as np
import torch
import cv2
from vidgear.gears import CamGear

from .functions_utils import write_fps
from .detectormanager import DetectionManager
import config

import time
from multiprocessing import Process,Queue

# class VideoHandler(QThread):

# 	update_image = Signal(np.ndarray)
# 	camera_state = Signal(bool)
# 	newcordinates = Signal(list)

# 	def __init__(self,image_queue,eventstate,cordinatesqueue):
# 		super(VideoHandler,self).__init__()
# 		self.runtime_state = eventstate 
# 		self.image_queue = image_queue
# 		self.cordinatesqueue = cordinatesqueue
# 		self.state = True

# 	def run(self):

# 		while self.isRunning():
			
# 			if self.runtime_state.value == 0: # no calibration - running mode
# 				image = self.image_queue.get()
# 				self.update_image.emit(image)

# 			if self.runtime_state.value == 1:
# 				image = self.image_queue.get()
# 				self.update_image.emit(image)
# 				if not self.cordinatesqueue.empty():
# 					self.newcordinates.emit(self.cordinatesqueue.get())
			
# 			if not self.state or self.runtime_state.value == 2:
# 				break

# 	def stop(self):
# 		self.state = False     
def getframe():
	pass

class VideoCamera():

	def __init__(self,size,fps,filepath,running_mode):
		self.size = size
		self.fps = fps
		self.camera = None
		self.running_mode = running_mode
		self.filepath = filepath
		self.imgqueue = Queue()
		self.stoqueue = Queue()
		self.cordinates_queue = Queue()
		self.calibration_state = Queue()
		self.p = Process(name='VideoManager',target = self.run,args = [self.size,self.fps,
																		self.filepath,
																		self.stoqueue,
																		self.imgqueue])
		self.p.start()
		
	def run(self,size,fps,filepath,stoqueue,imgqueue):
		#try:
		if config.MODE == "video":
			self.camera = cv2.VideoCapture(filepath)
			self.camera.set(cv2.CAP_PROP_FPS, fps)
			self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, size)
			self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size)
		else:
			self.camera = CamGear(source=config.VIDEO_LINK,stream_mode=True,logging=True).start()
		self.detector_manager = DetectionManager()
		self.detector_manager.load_yolor_model()

		# except Exception as e:
		# 	print(e)
		while stoqueue.empty():
			toc = time.time()
			frames: dict(str,np.array) = {}
			if self.running_mode.value == 0: # Read and process Video Frames
				stato = False
				#results = self.camera.read()	
				stato,frames['color_image'] = self.camera.read()	
				if stato:
					frames['prediction'] = frames["color_image"].copy()
					frames['prediction'], detections = self.detector_manager.predict(frames['prediction'])
					cv2.imshow('predictions',frames['prediction'])
					cv2.waitKey(1)
					#imgqueue.put(write_fps(toc,frames))
		
			if self.running_mode.value == 1: #StopMode
				pass

			if not stoqueue.empty():
				break

	def stop(self):
		self.stoqueue.put(False)        

