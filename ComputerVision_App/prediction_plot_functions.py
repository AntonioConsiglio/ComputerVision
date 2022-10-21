import cv2
import os 

LABEL_PATH = "C:\\Users\\anton\\Desktop\\dataset\\label"
IMAGE_PATH = "C:\\Users\\anton\\Desktop\\dataset"

def read_label_and_transform(label_path,image,filename):

    h,w,c = image.shape
    file_path = os.path.join(label_path,f'{filename.rsplit(".",1)[0]}.txt')
    with open(file_path,'r') as file:
        predictions = file.read()
        file.close()
    cordinates = []
    predictions = predictions.splitlines()
    predictions = [pred.split(' ') for pred in predictions]
    for prediction in predictions:
        label= prediction[4]
        xmin,ymin,xmax,ymax = [float(p) for p in prediction[:-2]]
        xmin = int(xmin*w)
        ymin = int((ymin)*h)
        xmax = xmin + int(xmax*w)
        ymax = ymin + int(ymax*h)
        # ymin = int((1-ymin)*h)
        # ymax = int((1-ymax)*h)
        cordinates.append([label,(xmin,ymin),(xmax,ymax)])

    return cordinates


def plot_images(image_path,filename):

    image = cv2.imread(os.path.join(image_path,filename))
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    cordinates = read_label_and_transform(LABEL_PATH,image,filename)
    for cordinate in cordinates:
        label, point1, point2 = cordinate
        cv2.rectangle(image,point1,point2,(255,255,0),2)
    cv2.imshow(f'immagine',image)
    cv2.waitKey(0)


if __name__ == '__main__':

    for image_name in os.listdir(IMAGE_PATH):
        try:
            plot_images(IMAGE_PATH,image_name)
        except:
            pass