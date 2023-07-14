from patternRecogn.pattern_recognition_function import cv_template_matching,convertoToPixmap,nomaximasuppression
import cv2
import numpy as np

IMAGE = cv2.cvtColor(cv2.imread("./screen.png"),cv2.COLOR_BGR2GRAY)
TEMPLATEstart = cv2.cvtColor(cv2.imread("./link.png"),cv2.COLOR_BGR2GRAY)

# IMAGE = cv2.resize(IMAGE,(0,0),fx=0.5,fy=0.5)
# IMAGE = cv2.Canny(IMAGE,50,150)
if __name__ == "__main__":

	for j in [1,0.8]:
		TEMPLATE = cv2.resize(TEMPLATEstart,(0,0),fx=j,fy=j)
		# TEMPLATEcanny = cv2.Canny(TEMPLATE,50,200)
		cv2.imshow("template",TEMPLATE)
		for i in range(5):
			result = cv_template_matching(IMAGE,TEMPLATE,3)
			h,w= TEMPLATE.shape
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
			image_to_plot = IMAGE.copy()
			predictions = []
			# for pt in zip(*loc[::-1]):
			value = result[maxLoc[1],maxLoc[0]]
			xmin,ymin = maxLoc
			xmax = maxLoc[0] + w
			ymax = maxLoc[1] + h
			predictions.append([value,[xmin,ymin,xmax,ymax]])
			# predictions = nomaximasuppression(predictions)
			print(predictions)
			if predictions and maxVal > 0.93:
				for score,bbox in predictions:
					cv2.rectangle(image_to_plot, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,255,255), 2)
				cv2.imwrite('./risultati.png',image_to_plot)
				cv2.imshow('risultati',image_to_plot)
				cv2.waitKey(0)
