import cv2
import os
from geomatch import GeoMatch,Point
from tqdm import tqdm

TEMPLATE = "C:\\Users\\anton\\Downloads\\GeoMatch_src\\GeoMatch\\Template.jpg"
IMG_TO_FIND = "C:\\Users\\anton\\Downloads\\GeoMatch_src\\GeoMatch\\Search2.jpg"
#IMG_TO_FIND = "C:\\Users\\antonio.consiglio\\OneDrive - E.P.F. Elettrotecnica S.r.l\\Immagini\\Part00.bmp"

def prepare_images_for_detection(template,image):

    template = cv2.imread(template)
    if len(template.shape) >=3:
        template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    image = cv2.imread(image)
    if len(image.shape) >=3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    return template,image


def main():
    template,image = prepare_images_for_detection(TEMPLATE,IMG_TO_FIND)

    gm = GeoMatch()
    gm.CreateGeoMatchModel(template,100,150)
    img_to_plot = gm.draw_template_contours(TEMPLATE,(0,255,0),1)
    cv2.imshow('contours',img_to_plot)
    cv2.waitKey(0)
    result_score,result_points = gm.FindGeoMatchModel(image,0.7,0.99)
    print(result_score)
    result_to_plot = gm.DrawContours(IMG_TO_FIND,result_points,(0,255,0),1)
    cv2.imshow('image_result',result_to_plot)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()