import cv2
import time

from application.reader import Reader
from application.detectormanager import DetectionManager
IMAGE_TO_READ = "./prova.png"
import numpy as np

from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
# if str(ROOT / 'strong_sort') not in sys.path:
#     sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

PATH = "C:\\Users\\anton\\Downloads\\qrcode.v2-qr-code-v1.yolov7pytorch\\train\\images\\datasets\\train\\images"


def main():
    reader = Reader()
    # cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector_manager = DetectionManager()
    detector_manager.load_yolor_model() 
    while True:
        state = True
        imagelist = [cv2.imread(os.path.join(PATH,image_name)) for image_name in os.listdir(PATH)]
        for image in imagelist:
        # state,image = cap.read()
            if state:
                #cv2.imshow('IMAGE',image)
                start = time.time()
                frame,detections = detector_manager.predict(image.copy())
                cv2.imshow('YOLO',frame)
                reader.process(image,np.array([np.array([0,0,416,416])]))
                #print(f"[ELAPSED]: {(time.time()-start)*1000} ms")
                cv2.imshow('RESULTS',reader.showresults(image))
                key = cv2.waitKey(0)
        if key == ord("s"):
            break 

if __name__ == "__main__":
    main()