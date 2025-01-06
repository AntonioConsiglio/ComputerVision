
from multiprocessing import Queue
from tqdm import tqdm
import numpy as np

import supervision
import ultralytics
from supervision import Detections,VideoInfo, get_video_frames_generator

from licence_plate import LicencePlateDetector,LicencePlateDetections
from ui_process import UIProcess
from sahi_predict import SahiPredictor

MODEL_PATH = "./rtdetr-l.pt"
# VIDEOPATH = "./2103099-uhd_3840_2160_30fps.mp4"
# VIDEOPATH = "./4261446-uhd_3840_2160_25fps.mp4"
VIDEOPATH = "./vehicles.mp4"
WRITE_VIDEO = False
NUMB_LETTERS_PLATE = 7
USE_SAHI = False

def filter_detection_by_id(detections,ids):
    new_detections = []
    for detection in detections:
        if detection[4] not in ids:
            new_detections.append(
                Detections(xyxy=np.array([detection[0]]),
                        mask=detection[1],
                        confidence=np.array([detection[2]]),
                        class_id=np.array([detection[3]]),
                        tracker_id=np.array([detection[4]]),
                        data={k:np.array([v]) for k,v in detection[5].items()},
                        ))
    return Detections.merge(new_detections)

if __name__ == "__main__":
    # Loade car detection model
    if USE_SAHI:
        car_model = SahiPredictor(model_type="rtdetr",model_path=MODEL_PATH)
    else:
        car_model = ultralytics.YOLO("yolov10s.pt")
        car_model.compile()

    tracker = supervision.ByteTrack()
    tracked_ids = set()
    
    # Load the Licence Plate Detector
    lp_detector = LicencePlateDetector(num_letters_plate=NUMB_LETTERS_PLATE)
    # Licence Plate Detections List
    LICENCE_PLATES_DETECTED = LicencePlateDetections()

    # Get video info for supervision video sink
    video_info = VideoInfo.from_video_path(video_path=VIDEOPATH)

    # Create Polygon zone to filtered out detectsion outside the zone
    xpmin,ypmin = 100, video_info.height // 1.5
    xpmax,ypmax = video_info.width-100, video_info.height - 50

    polygon = np.array([[xpmin, ypmin], [xpmax, ypmin], 
                        [xpmax, ypmax], [xpmin, ypmax]])

    valid_area = supervision.PolygonZone(
        polygon=polygon,
        triggering_anchors=[supervision.Position.CENTER]
        )

    # Visualization and VideWriter child process
    datas_queue = Queue()
    stop_queue = Queue()
    uiprocess = UIProcess(datas_queue,stop_queue,valid_area,video_info)
    uiprocess.start()

    # Start the Process

    with tqdm(desc="Video processing: ",total=video_info.total_frames) as bar:
        try:
            for index, frame in enumerate(get_video_frames_generator(source_path=VIDEOPATH)):

                if USE_SAHI:
                    detections = car_model.predict(frame)
                else:
                    detections = car_model.predict(frame,classes=[2,7],verbose=False,conf=0.25,iou=0.7)[0]
                    detections = Detections.from_ultralytics(detections)
                # Update tracker with current detections
                tracked_detections = tracker.update_with_detections(detections)

                # Filter detections in the valid area
                is_in_zone = valid_area.trigger(tracked_detections)
                tracked_detections = tracked_detections[is_in_zone]

                # Extract IDs of currently tracked objects
                new_tracked_ids = tracked_detections.tracker_id

                # Filter for new IDs
                new_tracked_detections = filter_detection_by_id(tracked_detections, tracked_ids)

                datas = {}
                datas["image"] = frame
                # Annotate frame with only new tracked detections
                if new_tracked_detections:
                    lp_detections, id_plate_detected, labels = lp_detector.get_licence_plate_detection(new_tracked_detections,frame) 
                    if labels:
                        LICENCE_PLATES_DETECTED.append(lp_detections,labels)

                    tracked_ids.update(id_plate_detected)     
                    datas["car_boxes"] = new_tracked_detections

                datas["licence_plate"] = LICENCE_PLATES_DETECTED.detections
                if not LICENCE_PLATES_DETECTED.empty():
                    datas["licence_plate_labels"] = LICENCE_PLATES_DETECTED.labels
                
                datas_queue.put(datas)
                # Remove expired detections
                LICENCE_PLATES_DETECTED.remove_expired(3)
                bar.update(1)
            
            stop_queue.put(True)
            uiprocess.p.join()
        except KeyboardInterrupt:
            stop_queue.put(True)
            uiprocess.p.join()
        except Exception as e:
            print("Error: ",e)
            stop_queue.put(True)
            uiprocess.p.join()