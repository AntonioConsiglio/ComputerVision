import os
class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
HOME_PATH = os.getcwd()
opt = {}
opt['weights'] = "C:\\Users\\anton\\Desktop\\PROGETTI\\computer_vision\\QRcodeReader\\yolov7\\best_410.pt" #"./applicationLib\\yolor\\best_overall.pt"
opt["data"] = "./yolov7\\data.yaml"
opt['imgsz'] = 1280
opt['names'] = './yolov7\\custom.names'
opt['conf_thres'] = 0.3
opt['iou_thres'] = 0.5
opt["device"] = '0'
opt["agnostic_nms'"] = False
opt['update'] = False
opt['half'] = True
opt = dotdict(opt)
