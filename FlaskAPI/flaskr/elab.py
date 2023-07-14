from flask import (
	Blueprint, flash, g, redirect, render_template, request, url_for,jsonify
)
from werkzeug.exceptions import abort
from tqdm import tqdm
from flaskr.auth import login_required
from flaskr.db import get_db
import base64
from PIL import Image
import io
from yolov7_face.detectionclass import YoloV7FaceDetection
import cv2
import numpy as np
import imageio.v3 as iio

bp = Blueprint('elab', __name__)

detector = YoloV7FaceDetection()
MAX_BATCH = 16

@bp.route("/")
def index():
	db = get_db()
	return render_template('elab/cv_interface.html')

@bp.route('/uploade_image', methods=['POST'])
def uploade_image():
	files = request.files.getlist('images')
	images_base64 = []
	for file in files:
		images_base64.append(convert_image_to_base64(file))
	# # Convert the processed image back to bytes
	#img_base64 = convert_image_to_base64(file)

	return render_template('elab/cv_interface.html', uploaded_image=images_base64)


@bp.route('/process_image', methods=['POST'])
@login_required
def process_image():
	files = request.form["images"]
	showface = True if "show_face" in request.form else False
	blur = True if "blur" in request.form else False

	images= convert_base64_to_image(files)
	images_base64 = []
	for file in images:
		images_base64.append(convert_image_to_base64(file,"n"))

	images = np.array([np.array(f) for f in images])
	# Perform image processing operations using Pillow
	if len(images.shape) > 2:
		processed_base64,b64listoffaces = process_equal_frames(images,
						   							showface,
													blur)
	else:
		processed_base64,b64listoffaces = process_notequal_frames(images,
						   							showface,
													blur)

	return render_template('elab/cv_interface.html', 
						   uploaded_image=images_base64,
						   processed_image=processed_base64,
						   list_of_faces = b64listoffaces)

@bp.route('/upload_video', methods=['POST'])
def upload_video():
	file = request.files['video']
	filereaded = file.stream.read()

	# # Convert the processed image back to bytes
	video_base64 = convert_video_to_base64(filereaded)

	return render_template('elab/cv_interface.html', uploaded_video=video_base64)

@bp.route('/process_video', methods=['POST'])
@login_required
def process_video():
	file = request.form['video']
	showface = True if "show_face" in request.form else False
	blur = True if "blur" in request.form else False
	outputVideo = True if "outputVideo" in request.form else False

	video_frames = convert_base64_to_video(file)

	processed_base64_full,b64listoffaces = process_equal_frames(video_frames,
						       							  showface,
														  blur)
	
	if outputVideo:
		processed_base64_full = convert_images_to_b64video(processed_base64_full)

	return render_template('elab/cv_interface.html',
							uploaded_video=file,
						   processed_video=processed_base64_full,
						   list_of_faces = b64listoffaces,
						   showvideo=outputVideo)

def process_equal_frames(frames,showface,blur):
	
	processed_base64 = []
	b64listoffaces = []
	start = 0
	for j in tqdm(range(MAX_BATCH,frames.shape[0],MAX_BATCH)):
		video = frames[start:j,:,:,:]
		start = j
		
		if not showface:
			procecced_imgs,_ = (detector(video,showface,blur))
			for procecced_img in procecced_imgs:
				processed_base64.append(convert_image_to_base64(procecced_img,"a"))
			#processed_base64_full.append(processed_base64)
		else:
			procecced_imgs,listoffaces = detector(video,showface,blur)
			for faces,procecced_img in zip(listoffaces,procecced_imgs):
				processed_base64.append(convert_image_to_base64(procecced_img,"a"))
				faces2save = []
				for face in faces:
					faces2save.append(convert_image_to_base64(face,"a"))
				b64listoffaces.append(faces2save)
	#elaborate last frames
	video = frames[start:,:,:,:]
	if not showface:
		procecced_imgs,_ = (detector(video,showface,blur))
		for procecced_img in procecced_imgs:
			processed_base64.append(convert_image_to_base64(procecced_img,"a"))
	else:
		procecced_imgs,listoffaces = detector(video,showface,blur)
		for faces,procecced_img in zip(listoffaces,procecced_imgs):
			processed_base64.append(convert_image_to_base64(procecced_img,"a"))
			faces2save = []
			for face in faces:
				faces2save.append(convert_image_to_base64(face,"a"))
			b64listoffaces.append(faces2save)

	return processed_base64,b64listoffaces

def process_notequal_frames(frames,showface,blur):
	
	processed_base64 = []
	b64listoffaces = []
	start = 0
	for j in tqdm(range(MAX_BATCH,frames.shape[0],MAX_BATCH)):
		video = frames[start:j]
		start = j
		
		if not showface:
			procecced_imgs,_ = (detector(video,showface,blur))
			for procecced_img in procecced_imgs:
				processed_base64.append(convert_image_to_base64(procecced_img,"a"))
			#processed_base64_full.append(processed_base64)
		else:
			procecced_imgs,listoffaces = detector(video,showface,blur)
			for faces,procecced_img in zip(listoffaces,procecced_imgs):
				processed_base64.append(convert_image_to_base64(procecced_img,"a"))
				faces2save = []
				for face in faces:
					faces2save.append(convert_image_to_base64(face,"a"))
				b64listoffaces.append(faces2save)
	#elaborate last frames
	video = frames[start:]
	if not showface:
		procecced_imgs,_ = (detector(video,showface,blur))
		for procecced_img in procecced_imgs:
			processed_base64.append(convert_image_to_base64(procecced_img,"a"))
	else:
		procecced_imgs,listoffaces = detector(video,showface,blur)
		for faces,procecced_img in zip(listoffaces,procecced_imgs):
			processed_base64.append(convert_image_to_base64(procecced_img,"a"))
			faces2save = []
			for face in faces:
				faces2save.append(convert_image_to_base64(face,"a"))
			b64listoffaces.append(faces2save)

	return processed_base64,b64listoffaces

def convert_images_to_b64video(images):
	imagessplitted = [img.split(',')[1] for img in images]
	pilimglist = convert_base64_to_image(imagessplitted,"l")
	arrimages = [np.array(img) for img in pilimglist]
	tensorimages = np.stack(arrimages,axis=0)
	rawBytes = io.BytesIO()
	iio.imwrite(rawBytes,tensorimages,format_hint=".mp4")
	videoB64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
	videoB64 = add_video_prefix(videoB64)
	return videoB64

def convert_image_to_base64(img,t="p"):
	if t == "a":
		img = Image.fromarray(img)
	elif t == "p":
		img = Image.open(img)
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	imageconverted = base64.b64encode(rawBytes.getvalue()).decode('ascii')
	processed_image_base64 = add_prefix(imageconverted)
	
	return processed_image_base64

def add_prefix(imageconverted):
	mime = "image/jpeg"
	return "data:%s;base64,%s"%(mime, imageconverted)

def add_video_prefix(videoconverted):
	mime = "video/mp4"
	return "data:%s;base64,%s"%(mime, videoconverted)

def convert_video_to_base64(video,t="p"):
	# imgarray = np.frombuffer(video, dtype=np.uint8)
	# im = cv2.imdecode(imgarray, cv2.IMREAD_UNCHANGED)
	rawBytes = io.BytesIO(video)
	rawBytes.seek(0)
	imageconverted = base64.b64encode(rawBytes.getvalue()).decode('ascii')
	mime = "video/mp4"
	processed_video_base64 = "data:%s;base64,%s"%(mime, imageconverted)
	return processed_video_base64

def convert_base64_to_video(b64video,t="p"):
	# imgarray = np.frombuffer(video, dtype=np.uint8)
	# im = cv2.imdecode(imgarray, cv2.IMREAD_UNCHANGED)
	b64video = b64video.split(",")[1]
	b64video = base64.b64decode(b64video)
	rawbytesvideo = io.BytesIO(b64video)
	video_frames = iio.imread(rawbytesvideo,index=None,format_hint=".mp4")

	return video_frames

def convert_base64_to_image(imgbase64,k="t"):
	imgbase64list = imgbase64 
	if not k == "l":
		imgbase64 = imgbase64.split(",")
		imgbase64list = [i for i in imgbase64 if "base64" not in i]
	images = []
	for img64 in imgbase64list:
		images.append(Image.open(io.BytesIO(base64.b64decode(img64))))
	return images