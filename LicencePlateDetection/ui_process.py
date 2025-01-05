from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import cv2

import supervision
from supervision import VideoSink


class ZoomedOverviewAnnotator:
    def __init__(self):
        self.lp_zoomed_images = {}
  
    def annotate(self,original_image, frame, detections, label_h, zoom_factor=4):
        for detection_idx in range(len(detections)):
            if detections.tracker_id[detection_idx] not in self.lp_zoomed_images:
                xmin, ymin, xmax, ymax = detections.xyxy[detection_idx].astype(int)
                zoomed_detection = original_image[ymin:ymax,xmin:xmax]
                zoomed_detection = cv2.resize(zoomed_detection,(0,0),fx=zoom_factor,fy=zoom_factor)
                h,w,_ = zoomed_detection.shape
                # New shape for licence plate
                ymax = label_h[detection_idx][1]
                ymin = ymax - h
                xmax = xmin + w
                self.lp_zoomed_images[detections.tracker_id[detection_idx]] = [(xmin, ymin, xmax, ymax),zoomed_detection]
            else:
                (xmin, ymin, xmax, ymax), zoomed_detection = self.lp_zoomed_images[detections.tracker_id[detection_idx]]
            
            area = supervision.PolygonZone(np.array([[xmin,ymin], [xmax,ymin], [xmax,ymax], [xmin,ymax]]))
            area_annotator = supervision.PolygonZoneAnnotator(zone=area,thickness=4,display_in_zone_count=False)
            frame[ymin:ymax,xmin:xmax] = zoomed_detection
            frame = area_annotator.annotate(frame)
        
        return frame

class CustomLabelAnnotator(supervision.LabelAnnotator):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.last_label_prop = None
    
    def annotate(self, scene, detections, labels = None, custom_color_lookup = None):
    
        assert isinstance(scene, np.ndarray)
        self._validate_labels(labels, detections)

        labels = self._get_labels_text(detections, labels)
        label_properties = self._get_label_properties(detections, labels)
        self.last_label_prop = label_properties

        self._draw_labels(
            scene=scene,
            labels=labels,
            label_properties=label_properties,
            detections=detections,
            custom_color_lookup=custom_color_lookup,
        )

        return scene

class UIProcess:
    def __init__(self, data_queue, stop_queue, valid_area, video_info):
        self.p = Process(target=self.process, args=(data_queue,stop_queue, valid_area, video_info),daemon=True)

    def start(self):
        self.p.start()

    def process(self, data_queue, stop_queue, valid_area, video_info):

        text_padding = 15
        self.boxcorner_annotator = supervision.BoxCornerAnnotator(thickness=10,corner_length=45)
        self.area_annotator = supervision.PolygonZoneAnnotator(valid_area,display_in_zone_count=False)
        self.annotator = supervision.BoxAnnotator()
        self.label_ann = CustomLabelAnnotator(text_scale=1,text_padding=text_padding,text_thickness=2)
        self.zoomed_annotator = ZoomedOverviewAnnotator()
        
        with tqdm(desc="Video writing: ",total=video_info.total_frames,position=1) as bar:
            with VideoSink("output_vehicles.mp4",video_info) as vsink:
                while stop_queue.empty() or not data_queue.empty():
                    
                    try:
                        datas = data_queue.get(timeout=1)
                    except:
                        continue
                    try:
                        image:np.ndarray = datas["image"]
                        car_boxes = datas.get("car_boxes",[])
                        licence_plate = datas.get("licence_plate",[])
                        licence_plate_labels = datas.get("licence_plate_labels",[])

                        frame = self.area_annotator.annotate(image.copy())
                        if len(car_boxes) > 0:
                            frame = self.boxcorner_annotator.annotate(frame, car_boxes)
                        if len(licence_plate) > 0:
                            frame = self.annotator.annotate(frame, licence_plate)
                            frame = self.label_ann.annotate(frame, licence_plate, licence_plate_labels)
                            frame = self.zoomed_annotator.annotate(image, frame, licence_plate, self.label_ann.last_label_prop)
                    except Exception as e:
                        print(e)

                    vsink.write_frame(frame)
                    bar.update(1)


