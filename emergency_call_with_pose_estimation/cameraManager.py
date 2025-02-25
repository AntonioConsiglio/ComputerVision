try:
	import depthai as dhai
except:
	print('depthai library is not installed!!')
	print('Install depthai library ... "pip install depthai"')
import cv2
import os
import numpy as np
import json

from camera_funcion_utils import BlobException, \
								  infoprint,\
								  IntrinsicParameters,\
								  create_depthconf_json, \
								  visualise_Axes, \
								  DEPTH_RESOLUTIONS,MEDIAN_KERNEL,COLOR_RESOLUTIONS

class DeviceManager():
	'''
	INPUT:\n
		size: Image size (w,h)\n
		fps: frame per seconds \n
		deviceid: default is None, name of the device\n
		nn_mode: bool, if true the internal NN pipline will be created\n
		calibration_mode: bool, if true the device will be enabled for calibration purpose \n
		blob_path: path to the .blob nn net file\n
		verbose: If True, show all the initialization functions pipeline for debugging.\n
				 If the device is used for calibration purpose, an image with the chessboard corner green coloured will be saved in the home path folder!
	'''

	def __init__(self,size:tuple[int,int]=(640,480),fps:int =30,deviceid: str = None,
				nn_mode:bool=False,nn_model:str = "SSD",calibration_mode:bool= False,
				blob_path:str=None,verbose:bool = False,config_path:str = './'):
		self.pipeline = dhai.Pipeline()
		self.size = size
		self.fps = fps
		self.deviceId = deviceid
		self.nn_active = nn_mode
		assert nn_model in ["SSD","YOLO"], "SELECT THE ONE OF THE TWO AVAILABLE ARCHITECTURE ['SSD' or 'YOLO']"
		self.nn_model = nn_model
		self.calibration = calibration_mode
		self.verbose = verbose
		self.depth_error = False
		self.zmmconversion = 1000
		self.BLOB_PATH = blob_path
		self.names = None
		self.config_path = config_path
		self.depthconfig = self._load_configuration(config_path)
		self._configure_device()
		self.node_list = self.pipeline.getNodeMap()

	def _configure_device(self):

		cam_rgb = self._configure_rgb_sensor(self.verbose)
	
	def _load_configuration(self,jpath):
		configfile = None
		if self.deviceId is None:
			jpath = os.path.join(jpath,"camera_configuration.json")
			if not os.path.isfile(jpath):
				create_depthconf_json(jpath)
				print("#"*70,"\n","[WARNING MESSAGE] : A camera configuration file was created with default setting because",
						"this file is used to configure the stereo camera!\n","#"*70)
			with open(jpath) as jsonfile:
				configfile = json.load(jsonfile)
			return configfile
		jpath = os.path.join(jpath,f"{self.deviceId}_camera_configuration.json")
		if not os.path.isfile(jpath):
			if not "camera_configuration.json" in os.listdir(os.getcwd()):
				create_depthconf_json(jpath)
				print("#"*70,"\n","[WARNING MESSAGE] : A camera configuration file was created with default setting because",
						"this file is used to configure the stereo camera!\n","#"*70)
		with open(jpath) as jsonfile:
				configfile = json.load(jsonfile)
		return configfile

	def enable_device(self,usb2Mode:bool=False):
		self._enable_device(usb2Mode,self.verbose)
	
	def disable_device(self):
		self._disable_device(self.verbose)

	@infoprint
	def _enable_device(self,usb2Mode,verbose):
		if self.deviceId is None:
			self.device_ = dhai.Device(self.pipeline,usb2Mode=usb2Mode)
		else:
			found, device_info = dhai.Device.getDeviceByMxId(self.deviceId)
			if not found:
				raise RuntimeError("Device not found!")
			self.device_ = dhai.Device(self.pipeline,device_info,usb2Mode=usb2Mode)
		self._set_output_queue()
	
	@infoprint
	def _disable_device(self,verbose):
		self.device_.close()
		if self.deviceId is not None:
			print(f"[{self.deviceId}] Device has been disabled!")
		else:
			print("[DEVICE]: Device has been disabled!")

	def _set_output_queue(self):
		self.q_rgb = self.device_.getOutputQueue("rgb",maxSize = 2,blocking = False)
		if self.nn_active:
			self.q_nn = self.device_.getOutputQueue('neural',maxSize=1,blocking=False)
			#self.q_nn_input = self.device_.getOutputQueue('neural_input',maxSize=2,blocking=True)
	
	def pull_for_frames(self,get_pointscloud=True,write_detections=True):
		'''
		- output:\n
			frame_state: bool \n
			frames: dict[color_image,depth,disparity_image,monos_image]
			results: dict['points_cloud_data','detections']
		'''
		frames = {}
		state_frame = False
		frame_count = 0

		while not state_frame:
			rgb_foto = self.q_rgb.tryGet()
			nn_foto = None
			if self.nn_active:
				nn_foto = self.q_nn_input.tryGet()
				nn_detection = self.q_nn.tryGet()
				if nn_detection is not None:
					detections = nn_detection.detections
				else:
					detections = None

			if rgb_foto is not None:

				state_frame = True
				frames['color_image'] = rgb_foto.getCvFrame()
				results = {}
				results['detections'] = None
				if nn_foto is not None:
					frames['nn_input'] = nn_foto.getCvFrame()
					if detections is not None:
						results['detections'] = self._normalize_detections(detections)
				return state_frame,frames,results
			else:
				frame_count += 1
				if frame_count > 10:
					print('empty_frame: ',frame_count)
				return False,None,None

#region CONFIGURATION SENSORS FUNCTION

	############################ CONTROLS CONFIGURATION ############################
	@infoprint
	def _configure_controls(self,cam_rgb,left,right,verbose):

		rgb_control = self.pipeline.create(dhai.node.XLinkIn)
		rgb_control.setStreamName('rgb_control')
		rgb_control.out.link(cam_rgb.inputControl)

		monos_control = self.pipeline.create(dhai.node.XLinkIn)
		monos_control.setStreamName('monos_control')
		monos_control.out.link(left.inputControl)
		monos_control.out.link(right.inputControl)
		
	############################ RGB SENSOR CONFIGURATION FUNCTIONS ############################
	@infoprint
	def _configure_rgb_sensor(self,verbose):

			cam_rgb = self.pipeline.create(dhai.node.ColorCamera)
			cam_rgb.setResolution(COLOR_RESOLUTIONS[self.depthconfig["ColorSensorResolution"]]) #To change the resolution
			#cam_rgb.setPreviewSize(self.size) # to change the output size
			cam_rgb.setBoardSocket(dhai.CameraBoardSocket.RGB)
			cam_rgb.setInterleaved(False)
			cam_rgb.setFps(self.fps)
			xout_rgb = self.pipeline.create(dhai.node.XLinkOut)
			xout_rgb.setStreamName("rgb")
			#cam_rgb.preview.link(xout_rgb.input)
			if self.nn_active:
				manip,_= self._configure_image_manipulator(self.pipeline,verbose)
				cam_rgb.preview.link(manip.inputImage)
				if self.BLOB_PATH is None:
					raise(BlobException(" BLOB PATH NOT SELECTED!! Please select the path to .blob files"))
				self._configure_nn_node(manip,self.pipeline,self.BLOB_PATH,verbose)
			return cam_rgb
	@infoprint
	def _configure_image_manipulator(self,pipeline,verbose):
		size = self.depthconfig["nn_size"]
		manip = pipeline.create(dhai.node.ImageManip)
		manipOut = pipeline.create(dhai.node.XLinkOut)
		manipOut.setStreamName('neural_input')
		manip.initialConfig.setResize(*size)
		manip.initialConfig.setFrameType(dhai.ImgFrame.Type.BGR888p)
		manip.out.link(manipOut.input)
		
		return manip,manipOut

	@infoprint
	def _configure_nn_node(self,manip,pipeline,blob_path,verbose):
			
			nn = pipeline.create(dhai.node.NeuralNetwork)
			nnOut = pipeline.create(dhai.node.XLinkOut)
			nnOut.setStreamName("neural")
			# define nn features
			nn.setBlobPath(blob_path)
			nn.setNumInferenceThreads(2)
			# Linking
			manip.out.link(nn.input)
			nn.out.link(nnOut.input)

	############################ DEPTH CONFIGURATION FUNCTIONS ############################
	@infoprint
	def _configure_depth_sensor(self,verbose):

		monoLeft = self.pipeline.create(dhai.node.MonoCamera)
		monoRight = self.pipeline.create(dhai.node.MonoCamera)
		depth = self.pipeline.create(dhai.node.StereoDepth)
		xout_depth = self.pipeline.create(dhai.node.XLinkOut)
		xout_depth.setStreamName("depth")
		xout_disparity = self.pipeline.create(dhai.node.XLinkOut)
		xout_disparity.setStreamName("disparity")
		self._configure_depth_properties(monoLeft,monoRight,depth,self.calibration,verbose)
		if self.calibration:
			depth.setDepthAlign(dhai.CameraBoardSocket.RGB)
			self.nn_active = False
		monoleft_out = self.pipeline.create(dhai.node.XLinkOut)
		monoleft_out.setStreamName("left")
		monoright_out = self.pipeline.create(dhai.node.XLinkOut)
		monoright_out.setStreamName("right")
		monoLeft.out.link(depth.left)
		monoLeft.out.link(monoleft_out.input)
		monoRight.out.link(depth.right)
		monoRight.out.link(monoright_out.input)
		depth.disparity.link(xout_disparity.input)
		depth.depth.link(xout_depth.input)
		return monoLeft,monoRight

	@infoprint
	def _configure_depth_properties(self,left,right,depth,calibration,verbose):
	
		if not calibration:
			left.setResolution(DEPTH_RESOLUTIONS[self.depthconfig["StereoSensorResolution"]])
			left.setBoardSocket(dhai.CameraBoardSocket.LEFT)
			right.setResolution(DEPTH_RESOLUTIONS[self.depthconfig["StereoSensorResolution"]])
			right.setBoardSocket(dhai.CameraBoardSocket.RIGHT)
		else:
			left.setResolution(DEPTH_RESOLUTIONS[self.depthconfig["StereoSensorResolution_calibration"]])
			left.setBoardSocket(dhai.CameraBoardSocket.LEFT)
			right.setResolution(DEPTH_RESOLUTIONS[self.depthconfig["StereoSensorResolution_calibration"]])
			right.setBoardSocket(dhai.CameraBoardSocket.RIGHT)


		# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
		depth.setDefaultProfilePreset(dhai.node.StereoDepth.PresetMode.HIGH_DENSITY)
		if self.calibration:
			depth.setOutputSize(self.size[0],self.size[1])
		# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
		depth.initialConfig.setMedianFilter(MEDIAN_KERNEL[str(self.depthconfig["MedianFilterKernel"])])
		depth.setLeftRightCheck(self.depthconfig["LeftRightCheck"])
		if self.depthconfig["ExtendedDisparity"] and self.depthconfig["Subpixel"]:
			raise Exception("Is not possible to use Subpixel with ExtendedDisparity !")
		depth.setExtendedDisparity(self.depthconfig["ExtendedDisparity"])
		depth.setSubpixel(self.depthconfig["Subpixel"])

		config = depth.initialConfig.get()
		config.postProcessing.speckleFilter.enable = self.depthconfig["speckleFilter"]
		config.postProcessing.speckleFilter.speckleRange = self.depthconfig["speckleRange"]
		config.postProcessing.temporalFilter.enable = self.depthconfig["temporalFilter"]
		config.postProcessing.spatialFilter.enable = self.depthconfig["spatialFilter"]
		config.postProcessing.spatialFilter.holeFillingRadius = self.depthconfig["holeFillingRadius"]
		config.postProcessing.spatialFilter.numIterations = self.depthconfig["numIterations"]
		config.postProcessing.thresholdFilter.minRange = self.depthconfig["thresholdFilter_minRange"]
		config.postProcessing.thresholdFilter.maxRange = self.depthconfig["thresholdFilter_maxRange"]
		config.postProcessing.decimationFilter.decimationFactor = self.depthconfig["decimationFactor"]
		depth.initialConfig.set(config)
		

#endregion

#region OPERATIONAL FUNCTIONS

	def _normalize_detections(self,detections):
		det_normal = []
		for detection in detections:
			label = detection.label
			if self.names is not None:
				label = self.names[int(label)]
			score = detection.confidence
			xmin,ymin,xmax,ymax = detection.xmin,detection.ymin,detection.xmax,detection.ymax
			xmin,xmax = int(xmin*self.size[0]),int(xmax*self.size[0])
			ymin,ymax = int(ymin*self.size[1]),int(ymax*self.size[1])
			det_normal.append([label,score,[xmin,ymin,xmax,ymax]])

		return det_normal

	def _write_detections_on_image(self,image,detections):
		for detection in detections:
			xmin,ymin,xmax,ymax = detection[2]
			label = detection[0]
			score = detection[1]
			cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,255,255),2)
			cv2.putText(image,f'{label}: {round(score*100,2)} %',(xmin,ymin-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

	def set_conversion_depth(self,conv_factor):
		'''
			default value is 1000 which means the output will be in m as default
		'''
		self.zmmconversion = conv_factor

	def _convert_depth(self,depth):
		h,w = depth.shape
		depth = depth.flatten()/self.zmmconversion
		if not self.depth_error:
			try:
				depth = np.reshape(depth,(self.size[1],self.size[0]))
				return depth
			except Exception as e:
				print(f"[{__name__} - Depth Warning: ]",e)
				return np.reshape(depth,(h,w))
		return np.reshape(depth,(h,w))

	def __determinate_object_location(self,frames,results,offset=10,draw_box=True):
		'''
			FOR INTERNAL SCOPE 		
		'''
		image_to_write = frames["color_image"]
		points_cloud_data = results["points_cloud_data"]
		detections = results["detections"]
		cordinates = []
		xyz_points = points_cloud_data['XYZ_map_valid']
		for detection in detections:
			assert (len(detection[2]) == 4),"bounding box cordinate need to be at the 3rd position of the detection list"
			xmin,ymin,xmax,ymax = detection[2]
			xcenter = (xmin+((xmax-xmin)//2))
			ycenter = (ymin+((ymax-ymin)//2))
			useful_value = xyz_points[ycenter-offset:ycenter+offset,xcenter-offset:xcenter+offset]
			useful_value = useful_value.reshape((useful_value.shape[0]*useful_value.shape[1],3))
			useful_value = useful_value[np.any(useful_value != 0,axis=1)]
			if useful_value.size == 0:
				continue 
			elif useful_value.shape[0] >1:
				avg_pos_obj = np.mean(useful_value,axis=0)*self.zmmconversion
			else:
				avg_pos_obj= useful_value[0]*self.zmmconversion
			avg_pos_obj = avg_pos_obj.astype(int)
			if not np.all(avg_pos_obj == 0):
				cordinates.append(avg_pos_obj.tolist())
				try:
					x,y,z = avg_pos_obj.tolist()
					if draw_box:
						cv2.putText(image_to_write,f"x: {x} mm",(xcenter+8,ycenter-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
						cv2.putText(image_to_write,f'y: {y} mm',(xcenter+8,ycenter-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
						cv2.putText(image_to_write,f'z: {z} mm',(xcenter+8,ycenter),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
						visualise_Axes(image_to_write,self.pointcloud_manager.calibration_info)
				except Exception as e:
					print(f"[CALCULATE OBJECT LOCATION]: {e}")
			
	
	def determinate_object_location(self,image_to_write,points_cloud_data,detections,offset=10,draw_box=True):
		'''
			input:\n
			image_to_write: image where the average position is written
			points_cloud_dat: the points cloud results, stored inside a dictionary as in results \n
							obtained using the pull for frames method\n
			detections: list of detections [[class,probability,[xmin,ymin,xmax,ymax]]]\n
			results: 
			offset: default = 10, the offset from the center of the detection to take the points cloud value\n
					and averaging them to output the position in the space of the object\n
			Output:\n
			cordinates: list of finded cordinates of the obects 		
		'''
		cordinates = []
		xyz_points = points_cloud_data['XYZ_map_valid']
		for detection in detections:
			assert (len(detection[2]) == 4),"bounding box cordinate need to be at the 3rd position of the detection list"
			xmin,ymin,xmax,ymax = detection[2]
			xcenter = (xmin+((xmax-xmin)//2))
			ycenter = (ymin+((ymax-ymin)//2))
			useful_value = xyz_points[ycenter-offset:ycenter+offset,xcenter-offset:xcenter+offset]
			useful_value = useful_value.reshape((useful_value.shape[0]*useful_value.shape[1],3))
			useful_value = useful_value[np.any(useful_value != 0,axis=1)]
			if useful_value.size == 0:
				continue 
			elif useful_value.shape[0] >1:
				avg_pos_obj = np.mean(useful_value,axis=0)*self.zmmconversion
			else:
				avg_pos_obj= useful_value[0]*self.zmmconversion
			avg_pos_obj = avg_pos_obj.astype(int)
			if not np.all(avg_pos_obj == 0):
				cordinates.append(avg_pos_obj.tolist())
				try:
					x,y,z = avg_pos_obj.tolist()
					if draw_box:
						cv2.putText(image_to_write,f"x: {x} mm",(xcenter+8,ycenter-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
						cv2.putText(image_to_write,f'y: {y} mm',(xcenter+8,ycenter-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
						cv2.putText(image_to_write,f'z: {z} mm',(xcenter+8,ycenter),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
						cv2.rectangle(image_to_write,(xmin,ymin),(xmax,ymax),(0,0,255),2)
				except Exception as e:
					print(f"[CALCULATE OBJECT LOCATION]: {e}")
			
		return cordinates
#endregion 

	def _isinsidelimit(self,val,minval,maxval):
		return max(minval,min(val,maxval))

	def set_rgb_exposure(self,exposure:int=None,iso:int=None,region:list[int,int,int,int]=None):
		control = dhai.CameraControl()
		if exposure is None and iso is None:
			control.setAutoExposureEnable()
			if region is not None:
				control.setAutoExposureRegion(*region)
			self.q_rgbcontrol.send(control)
			return
		exposure = int(self._isinsidelimit(exposure,1,33000))
		iso = int(self._isinsidelimit(iso,100,1600))
		control.setManualExposure(exposure,iso)
		self.q_rgbcontrol.send(control)

	def set_focus(self,focus_value:int=None):
		control = dhai.CameraControl()
		if focus_value is None:
			return
		control.setManualFocus(focus_value)
		self.q_rgbcontrol.send(control)
		self.q_monoscontrol.send(control)

	def set_monos_exposure(self,exposure:int=None,iso:int=None,region:list[int,int,int,int]=None):
		control = dhai.CameraControl()
		if exposure is None and iso is None:
			control.setAutoExposureEnable()
			if region is not None:
				control.setAutoExposureRegion(*region)
			self.q_monoscontrol.send(control)
			return
		exposure = int(self._isinsidelimit(exposure,1,33000))
		iso = int(self._isinsidelimit(iso,100,1600))
		control.setManualExposure(exposure,iso)
		self.q_monoscontrol.send(control)

	def set_labels_names(self,names:list = None):
		if names is not None:
			self.names = names

	def get_intrinsic(self):
		self.intrinsic_info = {}
		calibration_info = self.device_.readCalibration()
		intr_info_rgb = calibration_info.getCameraIntrinsics(dhai.CameraBoardSocket.RGB,resizeHeight=self.size[1],resizeWidth=self.size[0])
		intr_info_right = calibration_info.getCameraIntrinsics(dhai.CameraBoardSocket.RIGHT,resizeHeight=self.size[1],resizeWidth=self.size[0])
		self.intrinsic_info['RGB'] = IntrinsicParameters(intr_info_rgb,self.size)
		self.intrinsic_info['RIGHT'] = IntrinsicParameters(intr_info_right,self.size)
		return self.intrinsic_info

	def get_extrinsic(self):
		calibration_info = self.device_.readCalibration()
		extrin_info = np.array(calibration_info.getCameraExtrinsics(dhai.CameraBoardSocket.RIGHT,dhai.CameraBoardSocket.RGB))
		extrin_info[:3,3] = extrin_info[:3,3]/1000 
		self.extrinsic_info = Transformation(trasformation_mat=extrin_info)
		return self.extrinsic_info

