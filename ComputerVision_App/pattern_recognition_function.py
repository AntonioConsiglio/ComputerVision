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


def cond_dir1(direction):

    if direction > 0 and direction < 22.5 :
        return True
    elif direction > 157.5 and direction < 202.5:
        return True
    elif direction > 337.5 and direction < 360:
        return True 
    else:
        return False

def cond_dir2(direction):

    if direction > 22.5 and direction < 67.5 :
        return True
    elif direction > 202.5 and direction < 247.5:
        return True
    else:
        return False

def cond_dir3(direction):

    if direction > 67.5 and direction < 112.5 :
        return True
    elif direction > 247.5 and direction < 292.5:
        return True
    else:
        return False

def cond_dir4(direction):

    if direction > 112.5 and direction < 157.5 :
        return True
    elif direction > 292.5 and direction < 337.5:
        return True
    else:
        return False

def template_matching(train_image):

    gradX = cv2.Sobel(train_image,cv2.CV_16S,1,0,3)
    gradY = cv2.Sobel(train_image,cv2.CV_16S,0,1,3)

    h,w = train_image.shape

    magMat = np.zeros((h,w))
    orients = np.zeros((h,w)).flatten()
    count = 0
    maxGrad = -1000
    for i in range(h):
        for j in range(w):
            fdx = gradX[i,j]
            fdy = gradY[i,j]
            magG = math.sqrt(fdx**2 + fdy**2)
            direction = math.degrees(math.atan(fdy/fdx))
            if direction < 0:
                direction = 360-direction
            magMat[i,j] = magG

            if magG > maxGrad:
                maxGrad = magG
            
            if cond_dir1(direction):
                direction = 0
            elif cond_dir2(direction):
                direction = 45
            elif cond_dir3(direction):
                direction = 90
            elif cond_dir4(direction):
                direction = 135
            else:
                direction=0
            
            orients[count] = int(direction)
            count +=1

    ### apply no-maxima suppression

    count = 0
    magMatNew = np.concatenate((np.zeros((magMat.shape[0],1)),magMat),axis=1)
    magMatNew = np.concatenate((np.zeros((1,magMatNew.shape[1])),magMatNew),axis=0)
    magMatNew = np.concatenate((np.zeros((magMat.shape[0],1)),magMat),axis=1)
    magMatNew = np.concatenate((np.zeros((1,magMatNew.shape[1])),magMatNew),axis=0)
    for i in range(h):
        for j in range(w):
            direct = orients[count]
            if direct == 0:
                leftpixel = magMat[i,j-1]
                rightpixel = magMat[i,j+1]
            elif direct == 45:
                leftpixel = magMat[i-1,j+1]
                rightpixel = magMat[i+1,j-1]
            elif direct == 90:
                leftpixel = magMat[i-1,j]
                rightpixel = magMat[i+1,j]
            elif direct == 135:
                leftpixel = magMat[i-1,j-1]
                rightpixel = magMat[i+1,j+1]
            if magMat[i,j] < leftpixel or magMat[i,j] < rightpixel:
                pass
            else:
                pass
            count +=1

## do hysteresis threshold

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
            print('\nnuova verifica')
            for id,finded in enumerate(objects):
                print(f'{ abs(center[0]-finded[1][0])}, {abs(center[1]-finded[1][1])}')
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
    
    pred_idx = np.array(objects)[:,0].tolist()
    predictions = np.array(predictions)[pred_idx]
    return predictions.tolist()

def convertoToPixmap(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = QImage(image,640,480,QImage.Format.Format_RGB888)
    image = QPixmap(image)
    return image
    


if __name__ == '__main__':

    img = cv2.imread('template.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    template_matching(img)