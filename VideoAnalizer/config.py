import os
class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

HOME_PATH = os.getcwd()
opt = {}
opt['weights'] = "C:\\Users\\anton\\Desktop\\PROGETTI\\computer_vision\\VideoAnalizer\\yolov7\\yolov7.pt" #"./applicationLib\\yolor\\best_overall.pt"
opt["data"] = "./yolov7\\data.yaml"
opt['imgsz'] = 640
opt['names'] = './yolor\\coco.names'
opt['conf_thres'] = 0.4
opt['iou_thres'] = 0.5
opt["device"] = '0'
opt["agnostic_nms'"] = False
opt['update'] = False
opt['half'] = True
opt['filter'] = [0,1,2,3,16]
opt = dotdict(opt)

MODE = 'video'
#MODE = 'livestreaming' #'video
VIDEO_LINK = "https://www.youtube.com/watch?v=8oVQjO-0b7k"
#VIDEO_LINK =  "https://www.youtube.com/watch?v=VClJIez-w6Y"
#VIDEO_LINK = "https://www.youtube.com/watch?v=zu6yUYEERwA"