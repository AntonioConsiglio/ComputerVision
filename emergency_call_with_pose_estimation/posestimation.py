import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np

class ImgSize():

    def __init__(self,h,w):
        self.x = w
        self.y = h
    
    def __repr__(self) -> str:
        print(f"img width: {self.x} -- img height: {self.y}")

class PoseEstimation(mp_pose.Pose):

    def __init__(self,
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.9,
                min_tracking_confidence=0.9):
        super(PoseEstimation,self).__init__(static_image_mode=static_image_mode,
                                            model_complexity=model_complexity,
                                            enable_segmentation=enable_segmentation,
                                            min_detection_confidence=min_detection_confidence,
                                            min_tracking_confidence=min_tracking_confidence)
        self.results = None
        self.img_size = None
        self.collected_results = {}
    
    def predict(self,image,postprocess = False,filter_result:list=None):
        h,w,_ = image.shape
        self.image_size = ImgSize(h,w)
        self.results =  self.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if postprocess:
            self.collected_results = {}
            self.postprocess_results(filter_result)
            return self.collected_results
        return self.results

    def postprocess_results(self,filter_result):
        try:
            landmarks = self.results.pose_landmarks.landmark
            for i,landmark in enumerate(landmarks):
                if i in filter_result:
                    self.collected_results[str(i)] = np.array([landmark.x*self.image_size.x,landmark.y*self.image_size.y],dtype=int)
                #print(self.collected_results)
        except:
            pass

    def draw_image(self,image):
        mp_drawing.draw_landmarks(image,
                                  self.results.pose_landmarks,
                                  mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    def draw_filtered(self,image):
        for number, point in self.collected_results.items():
            point = point.tolist()
            cv2.circle(image,tuple(point),3,(0,0,255),-1)
            cv2.putText(image,number,(point[0],point[1]-5),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
