import cv2
import os

input_path = "C:\\Users\\anton\\Desktop\\RetiNeurali\\training\\immagini"
output_path = "C:\\Users\\anton\\Desktop\\RetiNeurali\\training\\immagini_jpg"

os.makedirs(output_path,exist_ok=True)

for image in os.listdir(input_path):
    name = image.rsplit('.',1)[0]
    image_path = os.path.join(input_path,image)
    image_rd = cv2.imread(image_path)
    image_wr = cv2.imwrite(os.path.join(output_path,f'{name}.jpg'),image_rd)