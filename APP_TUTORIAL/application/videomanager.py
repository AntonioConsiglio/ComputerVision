from audioop import mul
from email.mime import image
from multiprocessing import Process,Queue
from ..cameraLib import DeviceManager
import multiprocessing

class DeviceManagerCustom(DeviceManager):
    def __init__(self,size,fps):
        super(DeviceManagerCustom,self).__init__()

class VideHandler():
    pass

class VideoManager():
    def __init__(self,size,fps):
        self.image_queue = Queue()
        self.inference_status = multiprocessing.Value('i',0) # 0 will be trigger inference while 1 will be continuous
        self.p = Process(name='camera',target=self.run,args=[size,fps,self.image_queue])
        self.p.start()

    
    def run(self,size,fps,imagequeue):
        self.size = size
        self.fps = fps
        self.imgqueue = imagequeue
        self.camera = DeviceManager(size,fps,True)

    