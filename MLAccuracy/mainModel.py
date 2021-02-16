import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random
dir = '../Database/training'

categories = ['Borer', 'Choanephora']

data = []

for category in categories:
    path = os.path.join(dir,category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        try:
            dis_img = cv2.imread(imgpath)
            dis_img = cv2.resize(dis_img, (50,50))
            image = np.array(dis_img).flatten()
            data.append([image,label])
        except:
            pass

pick_out = open('../Database/pickle/dataset.pickle','wb')
pickle.dump(data,pick_out)
pick_out.close()

pick_in  = open('../Database/pickle/dataset.pickle','rb')
loadData = pickle.load(pick_in)
pick_in.close()
random.shuffle(loadData)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)


