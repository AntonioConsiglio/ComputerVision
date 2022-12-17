import cv2
import os

COLOR_LIST = [(0,0,0),(255, 0, 0),(0, 0, 255),(238, 130, 238),(106, 90, 205),(255, 165, 0),(60, 179, 113),(152, 88, 71),(0, 255, 0),(255,255,255)]

def load_images(path):
    images = []
    for image in os.listdir(path):
        if image.rsplit('.',1)[-1] in ["jpg","png","webp"]:
            img = cv2.imread(os.path.join(path,image))
            images.append(img)

    return images

def sift_detector(image1,image2,nfeatures = 200):

    sift  = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)
    
    return [kp1,kp2],[des1,des2]

def flann_matcher(kp,des):

    kp1,kp2 = kp
    des1,des2 = des
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 2)
    search_params = dict(checks=200)
    # Search the matches between the keyopints
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    return pts1,pts2

def feature_matcher(image1,image2,nfeatures = 200):

    kp,des = sift_detector(image1,image2,nfeatures)
    pt = flann_matcher(kp,des)

    return pt

def show_kpoint(images,points):

    for n,image in enumerate(images):
        for number,point in enumerate(points[n],start=1):
            nn = number
            if number > 9:
                nn = 9
            u,v = int(point[0]),int(point[1])
            cv2.circle(image,(u,v),2,COLOR_LIST[nn],-1)
            cv2.putText(image,f"{number}",(u,v),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,COLOR_LIST[nn],1)
            print(number)
        cv2.imshow(f'image{n}',image)
    cv2.waitKey(0)
