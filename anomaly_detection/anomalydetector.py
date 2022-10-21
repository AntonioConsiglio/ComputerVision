import os
import cv2
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Dense,Reshape,InputLayer,Flatten
from tensorflow.keras import Sequential
import alibi_detect
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score,plot_feature_outlier_image

IMAGE_DIRECTORY = 'carpet/train/good'
TEST_DIRECTORY = 'carpet/test'

SIZE = 64
dataset = []
testset = []

for image_name in os.listdir(IMAGE_DIRECTORY):
    if image_name.rsplit('.')[1] == "png":
        img = cv2.imread(os.path.join(IMAGE_DIRECTORY,image_name))
        dataset.append(cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(SIZE,SIZE)))
dataset = np.array(dataset)
trainidx = int(len(dataset)*0.7)
train = dataset[0:trainidx]
val = dataset[trainidx:]

for folders in os.listdir(TEST_DIRECTORY):
    for image_name in os.listdir(os.path.join(TEST_DIRECTORY,folders)):
        if image_name.rsplit('.')[1] == "png":
            img = cv2.imread(os.path.join(TEST_DIRECTORY,folders,image_name))
            testset.append(cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(SIZE,SIZE)))
testset = np.array(testset)
print(len(testset))
#normilize
train = train.astype('float32')/255.
val = val.astype('float32')/255.
testset = testset.astype('float32')/255.

#ENCODER
ENCODING_DIM = 1024
FINAL_DIMENSION = [8,8,512]

encoder = Sequential([
    InputLayer(input_shape =train[0].shape),
    Conv2D(64,4,strides=2,padding = 'same',activation = tf.nn.relu),
    Conv2D(128,4,strides=2,padding = 'same',activation = tf.nn.relu),
    Conv2D(256,4,strides=2,padding = 'same',activation = tf.nn.relu),
    Flatten(),
    Dense(ENCODING_DIM)]
)

#print(encoder.summary())

decoder_net = Sequential([
    InputLayer(input_shape =(ENCODING_DIM,)),
    Dense(np.prod(FINAL_DIMENSION)),
    Reshape(target_shape = FINAL_DIMENSION),
    Conv2DTranspose(128,4,strides=2,padding = 'same',activation = tf.nn.relu),   
    Conv2DTranspose(64,4,strides=2,padding = 'same',activation = tf.nn.relu),   
    Conv2DTranspose(3,4,strides=2,padding = 'same',activation = 'sigmoid')]   
)

#print(decoder_net.summary())

od = OutlierVAE(threshold=0.002,
                score_type='mse',
                encoder_net = encoder,
                decoder_net = decoder_net,
                latent_dim = ENCODING_DIM,
                samples = 4)

print(od.threshold)

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

adam = tf.keras.optimizers.Adam(lr=1e-4)
od.fit(train,
       optimizer=adam,
       epochs=2,
       batch_size=4,
       verbose=True)

print('TRAINING FINISHED')

X = testset[:10]
od_preds = od.predict(X,
                      outlier_type='instance',
                      return_feature_score=True,
                      return_instance_score=True)

X_recon = od.vae(X).numpy()

# plot_feature_outlier_image(od_preds,
#                            X,
#                            X_recon=X_recon,
#                            instance_ids=[0,1,2,3,4],
#                            max_instances=5,
#                            outliers_only=False,
#                            figsize=(30,30))

# plot_feature_outlier_image(od_preds,
#                            X,
#                            X_recon=X_recon,
#                            instance_ids=[5,6,7,8,9],
#                            max_instances=5,
#                            outliers_only=False,
#                            figsize=(30,30))

results = od_preds['data']['feature_score']
prova1 = np.array(results[0] > 0.001).astype(int)
import matplotlib.pyplot as plt
prova2 = prova1*255.0
plt.imshow(prova2)
plt.show()
prova2 = prova1*255.0
cv2.imshow('immagine',prova2)
cv2.waitKey(0)

print('prova')