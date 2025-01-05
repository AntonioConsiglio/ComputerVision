
import time
import cv2
import numpy as np

import easyocr
import ultralytics
from supervision import Detections
from fast_plate_ocr import ONNXPlateRecognizer

class DetectionWithTimer():
    def __init__(self,detection,label):
        self.detection = detection
        self.label = label
        self.time = time.time()
    
    @property
    def elapsed_time(self):
        return time.time() - self.time

class LicencePlateDetections():
    def __init__(self):
        self._detections = []
    
    def append(self,detection,label):
        if isinstance(detection,Detections):
            self._detections.append(DetectionWithTimer(detection,label))
        elif isinstance(detection,list):
            for det,lab in zip(detection,label):
                self._detections.append(DetectionWithTimer(det,lab))
        else:
            raise ValueError("Invalid type for detection")
        
    def remove_expired(self,threshold):
        self._detections = [detection for detection in self._detections if detection.elapsed_time < threshold]
    
    def empty(self):
        return len(self._detections) == 0
    
    @property
    def detections(self):
        try:
            return Detections.merge([detection.detection for detection in self._detections])
        except Exception as e:
            print(f"Error in detections: {str(e)}")
            return None
       
    
    @property
    def labels(self):
        return [detection.label for detection in self._detections]

class EasyOcr():
    def __init__(self,conf_threshold=0.2,num_letters_plate=7):
        self.ocr_model = easyocr.Reader(['en'], gpu=True)
        self.conf_threshold = conf_threshold
        self.num_letters_plate = num_letters_plate
    
    def readtext(self,image):
        # EasyOCR returns list of (bbox, text, confidence) tuples
        ocr_result = self.ocr_model.readtext(image)
        texts = [result[1] for result in ocr_result]
        confidences = [result[2] for result in ocr_result]
        best_idx = np.argmax(confidences)
        if confidences[best_idx] <= self.conf_threshold:
            return None, None
        if len(texts[best_idx]) < self.num_letters_plate:
            return None, None
        return texts[best_idx], confidences[best_idx]
    
class CustomOcr():
    def __init__(self,conf_threshold=0.8,num_letters_plate=7):
        self.ocr_model = ONNXPlateRecognizer("global-plates-mobile-vit-v2-model")
        self.conf_threshold = conf_threshold
        self.num_letters_plate = num_letters_plate
    
    def readtext(self,image):
        text,confidence = self.ocr_model.run(image,return_confidence=True)
        mean_conf = np.mean(confidence,axis=1)
        best_result_id = np.argmax(mean_conf)
        text = text[best_result_id].replace("_","")
        confidence = mean_conf[best_result_id]
        if confidence <= self.conf_threshold:
            return None, None, None
        if len(text) < self.num_letters_plate:
            return None, None, None
        return text, confidence, best_result_id
      
class LicencePlateDetector():
    def __init__(self,lp_model_path = None,
                 ocr_model_path = None,
                 num_letters_plate = 7):
        
        if lp_model_path is None:
            lp_model_path = "licence_plate_yolov10s.pt"
        self.license_plate_model = ultralytics.YOLO(lp_model_path)
        self.license_plate_model.compile()
        # if ocr_model_path is None:
        #     ocr_model_path = "en"
        # self.ocr_model = EasyOcr(conf_threshold=0.2,num_letters_plate=num_letters_plate))
        # self.ocr_model = easyocr.Reader(['en'], gpu=True)
        self.ocr_model = CustomOcr(conf_threshold=0.9,num_letters_plate=num_letters_plate)

    def detect_licence_plate_text(self,sv_det,original_image,):
        # if more than one plate in a single detection select the best restult
        plate_imgs = []
        for bbox in sv_det.xyxy:
            x1, y1, x2, y2 = bbox.astype(int)
            # Extract license plate region
            plate_imgs.append(cv2.cvtColor(original_image[y1:y2, x1:x2],cv2.COLOR_BGR2GRAY))
            # Perform OCR on the license plate region
        try:
            return self.ocr_model.readtext(plate_imgs)
        except Exception as e:
            print(f"OCR failed for plate: {str(e)}")
            return None, None, None

    def get_licence_plate_detection(self,car_detections,original_image) -> list[Detections]:
        # Perform inference
        # extract all the car frame from original image
        images = []
        shifted_image = []
        for car_det in car_detections:
            x1,y1,x2,y2 = car_det[0].astype(int)
            images.append(original_image[y1:y2,x1:x2])
            shifted_image.append([x1,y1])

        detections = self.license_plate_model.predict(images,verbose=False)
        sv_detections = []
        id_plate_detected = []
        labels = []
        for i,detection in enumerate(detections):
            sv_det = Detections.from_ultralytics(detection)
            if len(sv_det) == 0: continue
            sv_det.xyxy[:,0] += shifted_image[i][0]
            sv_det.xyxy[:,1] += shifted_image[i][1]
            sv_det.xyxy[:,2] += shifted_image[i][0]
            sv_det.xyxy[:,3] += shifted_image[i][1]

            text,confidence,best_id= self.detect_licence_plate_text(sv_det,original_image)
            if text:
                # Assign the licence plate detected to the car detection
                id_plate_detected.append(car_detections[i].tracker_id[0])
                if len(sv_det.data.get("class_name")) > 1:
                    best_det = sv_det[best_id]
                    best_det = Detections(xyxy=np.array([best_det.xyxy]),
                                mask=best_det.mask,
                                confidence=np.array([best_det.confidence]),
                                class_id=np.array([best_det.class_id]),
                                tracker_id=np.array([car_detections[i].tracker_id[0]]),
                                data={k:np.array([v]) for k,v in best_det.data.items()},
                                )
                    best_det.data['lincence_plate_text'] = [text]
                    best_det.data['confidence'] = [confidence]
                    sv_detections.append(best_det)
                else:
                    sv_det.tracker_id = np.array([car_detections[i].tracker_id[0]])
                    sv_det.data['lincence_plate_text'] = [text]
                    sv_det.data['confidence'] = [confidence]
                    sv_detections.append(sv_det)
                labels.append(f"{text} - {confidence:.2f}")

        return sv_detections, id_plate_detected, labels if len(labels) > 0 else None