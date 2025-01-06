import cv2
import numpy as np
import math
from tqdm import tqdm
from numba import njit,jit

@jit(cache = True)
def calcolo_score(cordinates,n_cordinates,
					edgedevX,edgedevY,h,w,sbx,sby,
					matGradMag,minScore,normGreediness,
					normMinScore,i,j,edgemagnitude):
	
	partialSum = 0
	for m in range(n_cordinates):
		curX	= int(i + cordinates[m][0])	#// template X coordinate
		curY	= int(j + cordinates[m][1]) # // template Y coordinate
		iTx	= edgedevX[m] #	// template X derivative
		iTy	= edgedevY[m]#   // template Y derivative

		if	(curX<0 or curY<0 or curX > h-1 or curY> w-1):
			pass
		else:
			iSx=sbx[curX,curY] #; // get curresponding  X derivative from source image
			iSy=sby[curX,curY] #;// get curresponding  Y derivative from source image
		
			if (iSx!=0 or iSy!=0) and (iTx!=0 or iTy!=0):
				partialSum = partialSum + ((iSx*iTx)+(iSy*iTy))*(edgemagnitude[m] * matGradMag[curX,curY])

			sumOfCoords = m + 1
			partialScore = partialSum /sumOfCoords
		# // check termination criteria
		# // if partial score score is less than the score than needed to make the required score at that position
		# // break serching at that coordinate.
			if partialScore < min((minScore -1) + normGreediness*sumOfCoords,normMinScore*sumOfCoords):
				break
	return partialScore

class Point():
	def __init__(self,x=0,y=0,score=0.0):
		self.x = x
		self.y = y
		self.score = score

	def get(self,):
		return (int(self.x),int(self.y))

class GeoMatch():
	def __init__(self):
		self.centerOfGravity = Point()
		self.noOfCordinates = 0 #//Number of elements in coordinate array
		self.modelDefined = True
		self.modelHeight = None #//Template height
		self.modelWidth = None #//Template width
		self.edgeMagnitude = None #//gradient magnitude
		self.edgeDerivativeX = None #//gradient in X direction
		self.edgeDerivativeY = None #//radient in Y direction
		self.cordinates = None #//Coordinates array to store model points


	def calculate_sobel_edges(self,img):

		sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the X axis
		sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis 

		return sobelx,sobely

	def no_max_suppression(self,points_list):
		objects = []
		for index,point in enumerate(points_list):
			center = (point.get())
			score = point.score

			state = -1
			if len(objects) > 0:
				# print('\nnuova verifica')
				for id,finded in enumerate(objects):
		
					if abs(center[0]-finded[1][0]) > 30 or abs(center[1]-finded[1][1]) > 30:
						pass
					else:
						state = id
					if state >=0:
						if score > objects[state][2]:
							objects[state] = [index,center,score]
							break
				if state <0:
					objects.append([index,center,score])
					# print('ho inserito nuovo punto')
						
			else:
				objects.append([index,center,score])
	
		pred_idx = np.array(objects)[:,0].tolist()
		points_list = np.array(points_list)[pred_idx]
		return points_list.tolist()

	def FindGeoMatchModel(self,graySearchImg,minScore,greediness):
		resultPoint=[]
		resultScore=0
		partialSum=0
		sumOfCoords=0

		h,w = graySearchImg.shape
		matGradMag = np.zeros((h,w)) # // create image to save gradient magnitude  values
		orients = np.zeros((h,w))
		sbx,sby = self.calculate_sobel_edges(graySearchImg)

		#// stoping criterias to search for model
		normMinScore = minScore /self.noOfCordinates  #precompute minumum score 
		normGreediness = ((1- greediness * minScore)/(1-greediness)) /self.noOfCordinates #precompute greedniness 
		#CvMat *Sdx = 0, *Sdy = 0;
		
		#cv2.cartToPolar(sbx,sby,matGradMag,orients)
		for i in tqdm(range(h)):

			for j in range(w):
		
				iSx=sbx[i,j];  #// X derivative of Source image
				iSy=sby[i,j];  #// Y derivative of Source image

				gradMag=np.sqrt((iSx*iSx)+(iSy*iSy)) # //Magnitude = Sqrt(dx^2 +dy^2)
							
				if gradMag!=0: #// hande divide by zero
					matGradMag[i,j]=1/gradMag; #  // 1/Sqrt(dx^2 +dy^2)
				else:
					matGradMag[i,j]=0

		for i in tqdm(range(h)):
			for j in range(w):
				
				partialSum = 0 #// initilize partialSum measure
				partialScore = calcolo_score(self.cordinates,self.noOfCordinates,self.edgeDerivativeX,self.edgeDerivativeY,
												h,w,sbx,sby,matGradMag,minScore,normGreediness,normMinScore,i,j,self.edgeMagnitude)
				# for m in range(self.noOfCordinates):
				# 	curX	= int(i + self.cordinates[m].x)	#// template X coordinate
				# 	curY	= int(j + self.cordinates[m].y) # // template Y coordinate
				# 	iTx	= self.edgeDerivativeX[m] #	// template X derivative
				# 	iTy	= self.edgeDerivativeY[m]#   // template Y derivative

				# 	if	(curX<0 or curY<0 or curX > h-1 or curY> w-1):
				# 		pass
				# 	else:
				# 		iSx=sbx[curX,curY] #; // get curresponding  X derivative from source image
				# 		iSy=sby[curX,curY] #;// get curresponding  Y derivative from source image
					
				# 		if (iSx!=0 or iSy!=0) and (iTx!=0 or iTy!=0):
				# 			partialSum = partialSum + ((iSx*iTx)+(iSy*iTy))*(self.edgeMagnitude[m] * matGradMag[curX,curY])
		
				# 		sumOfCoords = m + 1
				# 		partialScore = partialSum /sumOfCoords
				# 	# // check termination criteria
				# 	# // if partial score score is less than the score than needed to make the required score at that position
				# 	# // break serching at that coordinate.
				# 		if partialScore < min((minScore -1) + normGreediness*sumOfCoords,normMinScore*sumOfCoords):
				# 			break

				if partialScore >= minScore:
					resultScore = partialScore #//  Match score
					# if resultScore >= minScore:
					resultPoint.append(Point(i,j,resultScore))			#// result coordinate X#// result coordinate Y
		if len(resultPoint)>0:
			resultPoint = self.no_max_suppression(resultPoint)
		else:
			resultPoint = None
		return resultScore,resultPoint

	def CreateGeoMatchModel(self,templateArr,maxContrast,minContrast):

		#// set width and height

		self.modelHeight = templateArr.shape[0]		#Save Template height
		self.modelWidth = templateArr.shape[1]		#Save Template width

		#noOfCordinates=0
													#initialize	
		# cordinates =  new CvPoint[ modelWidth *modelHeight];		//Allocate memory for coorinates of selected points in template image
		#self.cordinates = np.array([Point() for _ in range(self.modelHeight*self.modelWidth)])
		self.cordinates = np.array([])
		self.edgeMagnitude = np.array([])
		self.edgeDerivativeX = np.array([])
		self.edgeDerivativeY = np.array([])
		# edgeMagnitude = new double[ modelWidth *modelHeight];		//Allocate memory for edge magnitude for selected points
		# edgeDerivativeX = new double[modelWidth *modelHeight];			//Allocate memory for edge X derivative for selected points
		# edgeDerivativeY = new double[modelWidth *modelHeight];			////Allocate memory for edge Y derivative for selected points


		# Calculate gradient of Template
		
		gx,gy = self.calculate_sobel_edges(templateArr)
		nmsEdges = np.zeros((self.modelHeight,self.modelWidth))
		#nmsEdges = cvCreateMat( Ssize.height, Ssize.width, CV_32F);		//create Matrix to store Final nmsEdges

		MaxGradient=-99999.99
		# double direction;
		orients = np.zeros((self.modelHeight*self.modelWidth))
		# int count = 0,i,j; // count variable;
		count = 0
		
		# double **magMat;
		magMat = np.zeros((self.modelHeight,self.modelWidth))
		# CreateDoubleMatrix(magMat ,Ssize);
		
		for i in range(1,self.modelHeight):

			for j in range(1,self.modelWidth):
			# _sdx = (short*)(gx->data.ptr + gx->step*i);
			# _sdy = (short*)(gy->data.ptr + gy->step*i);
				fdx = gx[i,j]
				fdy = gy[i,j]       #// read x, y derivatives

				MagG = np.sqrt((fdx*fdx) + (fdy*fdy)); #//Magnitude = Sqrt(gx^2 +gy^2)
				direction = math.degrees(np.arctan2(fdy,fdx));	 #//Direction = invtan (Gy / Gx)
				if direction < 0:
					direction += 360
				magMat[i,j] = MagG
					
				if MagG > MaxGradient :
					MaxGradient=MagG;# // get maximum gradient value for normalizing.

				#get closest angle from 0, 45, 90, 135 set
				if (direction>0 and direction < 22.5) or (direction >157.5 and direction < 202.5) or (direction>337.5 and direction<360):
					direction = 0
				elif (direction>22.5 and direction < 67.5) or (direction >202.5 and direction <247.5):
					direction = 45
				elif (direction >67.5 and direction < 112.5) or (direction>247.5 and direction<292.5):
					direction = 90
				elif (direction >112.5 and direction < 157.5) or (direction>292.5 and direction<337.5):
					direction = 135
				else:
					direction = 0
					
				orients[count] = int(direction)
				count+=1

		count=0 #init count
		#non maximum suppression
		# double leftPixel,rightPixel;
		
		for i in range(1,self.modelHeight-1):
			for j in range(1,self.modelWidth-1):
				case = orients[count]
				if case == 0:
					leftPixel  = magMat[i,j-1]
					rightPixel = magMat[i,j+1]
				elif case == math.pi/4:
					leftPixel  = magMat[i-1,j+1]
					rightPixel = magMat[i+1,j-1]
				elif case == math.pi/2:
					leftPixel  = magMat[i-1,j]
					rightPixel = magMat[i+1,j]
				elif case == 135:
					leftPixel  = magMat[i-1,j-1]
					rightPixel = magMat[i+1,j+1]

					
				#// compare current pixels value with adjacent pixels
				if ( magMat[i,j] < leftPixel ) or (magMat[i,j] < rightPixel):
					nmsEdges[i,j]=0
				else:
					nmsEdges[i,j]= (magMat[i,j]/MaxGradient)*255
			
				count+=1
		
		toplot = nmsEdges.astype(np.uint8)
		# cv2.imshow('prova',toplot)
		# cv2.waitKey(0)
		RSum=0
		CSum=0
		flag = 1
		start = 0
		# int curX,curY;
		#//Hysterisis threshold
		for i in range(1,self.modelHeight-1):

			for j in range(1,self.modelWidth-1):

				fdx = gx[i,j]
				fdy = gy[i,j]       #// read x, y derivatives

				MagG = np.sqrt((fdx*fdx) + (fdy*fdy)); #//Magnitude = Sqrt(gx^2 +gy^2)
				direction = np.arctan2(fdy,fdx);	 #//Direction = invtan (Gy / Gx)

				flag=1
				if nmsEdges[i,j] < maxContrast:

					if nmsEdges[i,j] < minContrast:

						nmsEdges[i,j]=0
						flag=0; #// remove from edge

					else:
					#// if any of 8 neighboring pixel is not greater than max contraxt remove from edge
						if( (nmsEdges[i-1,j-1] < maxContrast)	and \
							(nmsEdges[i-1,j] < maxContrast)	and \
							(nmsEdges[i-1,j+1] < maxContrast)	and \
							(nmsEdges[i,j-1] < maxContrast) and \
							(nmsEdges[i,j+1] < maxContrast) and \
							(nmsEdges[i+1,j-1] < maxContrast) and \
							(nmsEdges[i+1,j] < maxContrast)	and \
							(nmsEdges[i+1,j+1] < maxContrast)):

							nmsEdges[i,j]=0
							flag=0
				
				#// save selected edge information
				curX=i
				curY=j
				if flag!=0:
					if fdx!=0 or fdy!=0:	
						RSum=RSum+curX
						CSum=CSum+curY # // Row sum and column sum for center of gravity
						if start == 0:
							self.cordinates = np.append(self.cordinates,np.array([curX,curY]))
							start = 1
						else:
							self.cordinates = np.vstack((self.cordinates,np.array([curX,curY])))
						# self.cordinates[self.noOfCordinates].x = curX
						# self.cordinates[self.noOfCordinates].y = curY
						self.edgeDerivativeX = np.append(self.edgeDerivativeX,fdx)
						self.edgeDerivativeY = np.append(self.edgeDerivativeY,fdy)
						
						#//handle divide by zero
						if MagG!=0:
							self.edgeMagnitude = np.append(self.edgeMagnitude,1/MagG) # // gradient magnitude 
						else:
							self.edgeMagnitude = np.append(self.edgeMagnitude,0)
																
						self.noOfCordinates+=1


		self.centerOfGravity.x = RSum /self.noOfCordinates #// center of gravity
		self.centerOfGravity.y = CSum/self.noOfCordinates	#// center of gravity
			
		#// change coordinates to reflect center of gravity
		for m in range(self.noOfCordinates):

			self.cordinates[m][0]=self.cordinates[m][0]-self.centerOfGravity.x
			self.cordinates[m][1] =self.cordinates[m][1]-self.centerOfGravity.y

		self.modelDefined=True

	def DrawContours(self,source,COG,color, lineWidth):
		
		if COG is not None:
			point = Point()
			for cog in COG:
				cv2.putText(source,f"{round(cog.score,2)}",(int(cog.y)-10,int(cog.x)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
				for i in range(self.noOfCordinates):
					point.y=self.cordinates[i][0] + cog.x
					point.x=self.cordinates[i][1] + cog.y
					cv2.line(source,point.get(),point.get(),color,lineWidth)
		
		return source
	
	#// draw contour at template image
	def draw_template_contours(self,source,color,lineWidth):

		source = cv2.imread(source)
		point = Point()
		for i in range(self.noOfCordinates):
			point.y=self.cordinates[i][0]+ self.centerOfGravity.x
			point.x=self.cordinates[i][1] + self.centerOfGravity.y
			cv2.line(source,point.get(),point.get(),color,lineWidth)
		
		return source
