import cv2
from posestimation import PoseEstimation
from utils import ResultsManager,select_color
# import pywhatkit
from heyoo import WhatsApp
import time
from threading import Thread
import math

from cameraManager import DeviceManager
import config

CALL_HELP = 4
ANGLE = 360/CALL_HELP
BLOB_PATH = "./pose_landmark_heavy_sh4.blob"
BLOB_PATH = "./pose_landmark_lite_sh4.blob"

#Connect whatsapp account

wpAccount = WhatsApp(token=config.WP_TOKEN,
                    phone_number_id=cofig.PHONE_NUMBER_ID)

if __name__ == "__main__":

    # camera = DeviceManager(nn_mode=True,
    #                        blob_path= BLOB_PATH)
    # camera.enable_device()

    # while True:
    #     stato,frame,result = camera.pull_for_frames()

    camera = cv2.VideoCapture(0)
    estimator = PoseEstimation()
    resultsManager = ResultsManager(timelimit=2.5,call_help=CALL_HELP)
    message_sent_time = time.time() 
    
    while camera.isOpened():
        state,img = camera.read()
        if state:
            cv2.imshow('immagine',img)
            resultsManager.add(estimator.predict(img,True,[12,11,15]))
            estimator.draw_filtered(img)
            help = resultsManager.calculateDistance()
            angle = ANGLE*help
            color = select_color(angle)
            cv2.ellipse(img,(100,100),(80,80),360,0,angle,color,20)
            if angle == 360:
                cv2.putText(img,"HELP !!!",(50,100),cv2.FONT_HERSHEY_DUPLEX,1,color,3)
            resultsManager.filter_results_bytime()
            cv2.imshow('PoseEstimation',img)
            cv2.waitKey(1)
            if help == CALL_HELP:
                deltatime = time.time() - message_sent_time 
                print(deltatime)
                if deltatime > 10:
                    tempo2send = list(map(int,time.strftime("%H-%M-%S").split('-')))
                    tempo2send[1]+= math.ceil(tempo2send[2]/45)
                    wpAccount.send_message('Hello I am WhatsApp Cloud API', config.EMERGENCY_NUMBER)
                    message_sent_time = time.time()
                



