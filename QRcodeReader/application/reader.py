import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pyzbar.pyzbar import decode



class Reader():

    def __init__(self):
        self.type = None
        self.detector = cv2.QRCodeDetector()
        self.detected = []

    def process(self,image,contours):
        self.detected = []
        image_trasformed = self.trasformimage(image.copy())
        try:
            if len(contours) != 0:
                with ThreadPoolExecutor() as executor:
                    future_to_contour = {executor.submit(decode,
                    image_trasformed[y:y+h,x:x+w]):(x, y, w, h) for x, y, w, h in contours}
                    for future in as_completed(future_to_contour):
                        detection = {}
                        x,y,w,h = future_to_contour[future]
                        box:np.array
                        results = future.result()
                        if results:
                            results = results[0]
                            results.data : bytes
                            box = np.array([[p.x,p.y] for p in results.polygon])
                            detection['data'] = results.data.decode('utf-8')
                            detection['box'] = np.array([point + np.array((x,y)) for point in box])
                            detection['area'] = [[x,y],[x+w,y+h]]
                            print(detection)
                            self.detected.append(detection)
        except Exception as e:
            print(e)
    
    def trasformimage(self,image):
        image_trasformed = image
        #image_trasformed = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # _,image_trasformed = cv2.threshold(image,127,255,0)
        return image_trasformed


    def showresults(self,image):
        for detection in self.detected:
            points = detection["box"]
            pt1,pt2 = detection['area']
            cv2.polylines(image,[points],True,(0,255,0),2)
            #cv2.rectangle(image,pt1,pt2,(255,0,0),1)
            cv2.putText(image,detection['data'],(points[0][0],points[0][1]-15),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
        return image
