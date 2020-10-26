from keras.applications import VGG16
from keras import models
from keras.models import load_model
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image

image_size = 224
model = load_model('vgg16_tuned_25sty.h5')
model.summary()

# Predict
batch_holder = np.zeros((4, image_size, image_size, 3))
img_dir='E:/ProjektPrzejsciowy/Data_sets/Test3'
for i,img in enumerate(os.listdir(img_dir)):
  img = image.load_img(os.path.join(img_dir,img), target_size=(image_size,image_size))
  batch_holder[i, :] = img

result = model.predict_classes(batch_holder)
for data in result:
  print(data)

for num in range(0, 3000):
  batch_holder = np.zeros((70, image_size, image_size, 3))
  img_dir = f'E:/ProjektPrzejsciowy/Data_sets/Data_set_3000_17.20.31_224x224/{num}'
  for i, img in enumerate(os.listdir(img_dir)):
    img = image.load_img(os.path.join(img_dir, img), target_size=(image_size, image_size))
    batch_holder[i, :] = img
  result = model.predict_classes(batch_holder)
  for data in result:
    if num == data:
      print(num)

