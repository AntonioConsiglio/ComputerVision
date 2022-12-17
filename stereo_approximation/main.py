# import os
# print(os.getcwd())
# os.chdir(__file__.rsplit("\\",1)[0])
# print(os.getcwd())
# from utils import feature_matcher,load_images,show_kpoint

# image1,image2 = load_images("./images")

# pt1,pt2 = feature_matcher(image1,image2,nfeatures=300)

# show_kpoint([image1,image2],[pt1,pt2])


lista1 = [1,3]
lista2 = [1,3]

if any(lista1 == lista2):
    print("OKK")