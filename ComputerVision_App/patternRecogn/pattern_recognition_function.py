## edge base template matching technique

import cv2
import numpy as np
import math
from PySide2.QtGui import QPixmap,QImage

METHODS = [
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED
]

def cv_template_matching(image,template,metod):
    #image_gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(image,template,METHODS[metod])
    return result

def nomaximasuppression(predictions):
 
    objects = []
    for index,prediction in enumerate(predictions):
        xmin,ymin,xmax,ymax = prediction[1]
        w = xmax-xmin
        h = ymax-ymin
        score = prediction[0]
        center = [(xmin+xmax)//2,(ymin+ymax)//2]
        state = -1
        if len(objects) > 0:
            #print('\nnuova verifica')
            for id,finded in enumerate(objects):
                #print(f'{ abs(center[0]-finded[1][0])}, {abs(center[1]-finded[1][1])}')
                if abs(center[0]-finded[1][0]) > w or abs(center[1]-finded[1][1]) > h:
                    pass
                else:
                    state = id
                if state >=0:
                    if score > objects[state][2]:
                        objects[state] = [index,center,score]
                        break
            if state <0:
                objects.append([index,center,score])
                print('ho inserito nuovo punto')
                    
        else:
            objects.append([index,center,score])
    
    pred_idx = np.array(objects,dtype=list)[:,0].tolist()
    predictions = np.array(predictions,dtype = list)[pred_idx]
    return predictions.tolist()

def convertoToPixmap(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    h,w,_ = image.shape
    image = QImage(image,w,h,QImage.Format.Format_RGB888)
    image = image.scaled(640,480)
    image = QPixmap(image)
    return image
    
