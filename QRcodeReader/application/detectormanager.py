# from yolor.detect_myself import detect,load_model
from yolov7.detect_myself import detect,load_model
#from yolov7.detect_myself_traking import detect,load_model
from config import opt
import random
import torch
from torch.backends import cudnn
import time
import numpy as np


class DetectionManager():

    def __init__(self):
        self.device = None
        self.model = None
        self.tracker = None
        self.framenumber = 0

    def predict(self,frame):
        with torch.no_grad():
            cudnn.benchmark = True
            start = time.time()
            frame,detections = detect(opt,frame,self.device,self.model,self.names,self.colors,self.framenumber)
            detections = self.postprocess_detection(detections,frame)
        return frame,detections 

    def load_yolor_model(self):
        self.device, self.model, self.tracker = load_model(opt)
        self.model.to(self.device).eval()
        names = opt.names
    # Get names and colors
        self.names = self.load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def postprocess_detection(self,results,frame):
        # contours = np.roll(np.delete(results,-2,1),1,axis=1)
        contours = self.transform_cords(results)
        return contours

    def transform_cords(self,cords):
        cords[:,3] = cords[:,3] - cords[:,1] # w
        cords[:,4] = cords[:,4] - cords[:,2] # h
        return cords[:,0:4].astype(int)

    
    def normilize_cords(self,cords,imgshape):
        cords[:,1] = cords[:,1]/imgshape[1] # xmin
        cords[:,2] = cords[:,2]/imgshape[0] # ymin
        cords[:,3] = cords[:,3]/imgshape[1] - cords[:,1] # w
        cords[:,4] = cords[:,4]/imgshape[0] - cords[:,2] # h
        return cords[:,1:5]
            
    
    def load_classes(self,path):
    # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
            names = list(filter(None, names)) 
            print(names)
        return names  # filter removes empty strings (such as last line)