from application.videomanager import VideoCamera
import multiprocessing
from time import sleep
from pathlib import Path
import sys
import os
  # creates a packageA entry in sys.modules
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


if __name__ == "__main__":
    FILEPATH = "C:\\Users\\anton\\Downloads\\HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store.mp4"
    process = multiprocessing.Value('i',0)
    video = VideoCamera(640,30,FILEPATH,process)

    for i in range(100):
        sleep(1000)