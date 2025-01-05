from supervision import Detections
import numpy as np
import torch

from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, get_sliced_prediction_v2

def from_sahi(sahi_result) -> Detections:
    boxes, scores, labels = [],[],[]
    for detection in sahi_result.object_prediction_list:
        boxes.append([detection.bbox.minx, detection.bbox.miny, detection.bbox.maxx, detection.bbox.maxy])
        scores.append(detection.score.value)
        labels.append(detection.category.id)

    return Detections(
        xyxy=np.array(boxes),
        confidence=np.array(scores),
        class_id=np.array(labels),
    )


class SahiPredictor:
    def __init__(self,model_type,model_path,device="cuda:0" if torch.cuda.is_available() else "cpu"):
        self.detection_model = AutoDetectionModel.from_pretrained(
                                                    model_type=model_type,#'rtdetr',
                                                    model_path=model_path,
                                                    confidence_threshold=0.3,
                                                    device=device
                                                )
    
    def predict(self,frame):
        detections = get_sliced_prediction_v2(frame,self.detection_model,
                                slice_height=640,slice_width=640,
                                overlap_height_ratio=0.1,overlap_width_ratio=0.1,
                                perform_standard_pred=False,verbose=0)
        return from_sahi(detections)
        
        