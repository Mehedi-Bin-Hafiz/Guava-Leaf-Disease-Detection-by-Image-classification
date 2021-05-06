import keras
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import time
import os
import cv2

generator = keras.preprocessing.image.ImageDataGenerator(

    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    zoom_range=0.2,

)

Datadir = '../RawData'
save_here = '../Database/training/Whitefly'

################## conversion image to number array ##############

Categories = ['Whitefly']
img_array = 0
path = os.path.join(Datadir)
i = 0
for batch in generator.flow_from_directory(path, save_to_dir=save_here, save_prefix='dis_agu', save_format='jpg',batch_size=1):
    i += 1
    print(i)
    if i == 100:
        break