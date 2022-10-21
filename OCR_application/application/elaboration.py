from multiprocessing import Process,Queue
from tabnanny import verbose
from matplotlib import pyplot as plt
import pytesseract
import cv2
import numpy as np
from PySide2.QtCore import QThread,Signal

import keras_ocr as kocr
import config

class ElaborationHandler(QThread):
    new_results = Signal(dict)
    def __init__(self,queue):
        super(ElaborationHandler,self).__init__()
        self.state = True
        self.imagequeue = queue
    
    def run(self):
        
        while self.state:
            if not self.imagequeue.empty():
                results = self.imagequeue.get()
                self.new_results.emit(results)
    
    def stop(self):
        self.state = False


class ElaborationManager():
    def __init__(self):
        self.stopqueue = Queue()
        self.imagequeue = Queue()
        self.results = Queue()
        self.p = Process(name='elaborationManager',target = self.run,
                        args = [self.stopqueue,self.imagequeue,self.results])
        self.p.start()

    def run(self,stopqueue,imagequeue,resultsqueue):
        self.pipeline = kocr.pipeline.recognition.Recognizer()
        self.completepipeline = kocr.pipeline.Pipeline(scale=1)

        while stopqueue.empty():
           
            image = imagequeue.get()
            images = self.get_images(image)
            results = self.make_prediction(images)
            resultsqueue.put(results)
    
    def get_images(self,image):
        images = {}
        images['data'] = image[config.DATA.y1:config.DATA.y2,
                               config.DATA.x1:config.DATA.x2]
        images['targa'] = image[config.TARGA.y1:config.TARGA.y2,
                               config.TARGA.x1:config.TARGA.x2]
        images['nome'] = image[config.NAME.y1:config.NAME.y2,
                               config.NAME.x1:config.NAME.x2]
        images['cf'] = image[config.CF.y1:config.CF.y2,
                               config.CF.x1:config.CF.x2]
        images['via'] = image[config.VIA.y1:config.VIA.y2,
                               config.VIA.x1:config.VIA.x2]
        for key,image in images.items():
            if key == 'cf':
                pass
            elif key == 'targa':
                images[key] = cv2.medianBlur(image,5)
            else:
                images[key] = cv2.resize(cv2.medianBlur(image,3),(0,0),fx=0.5,fy=0.5)
        return images
    
    def make_prediction(self,images):
        results = {}
        for key,image in images.items():
            kimage = kocr.tools.read(image)
            if key == 'nome':
                pred = self.completepipeline.recognize([kimage])
                pred = np.array(pred[0])[:,0]
            elif key == "via":
                pred = self.completepipeline.recognize([kimage])
                pred = np.array(pred[0])
                sorted = pred[:,1]
                sorted = sorted[::]
                tosort = list()
                for arr in sorted:
                    tosort.append(arr[0,0])
                sorted = np.argsort(np.array(tosort))
                pred = pred[:,0][sorted]
            elif key == "cf":
                pred = pytesseract.image_to_string(image)
            elif key == "targa":
                pred = pytesseract.image_to_string(image)
            else:    
                pred = self.pipeline.recognize(kimage)
            results[key] = [image,pred]
        return results