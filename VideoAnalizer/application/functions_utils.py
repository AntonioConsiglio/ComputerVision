
import math
import time
import cv2

def write_fps(toc,frames):
    tic =time.time()
    try:
        fps = 1//(tic-toc)
    except ZeroDivisionError:
        fps = 30
    cv2.putText(frames['prediction'],f"FPS: {fps}",(20,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,0),2)
    return frames
