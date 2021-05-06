import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import time
import random
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
dir = '../Database/validation'

categories = ['Leafspot', 'Rust', 'Sound', 'Whitefly']

data = []
# plt.imshow(dis_img)
# plt.show()
for category in categories:
    path = os.path.join(dir,category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        try:
            dis_img = cv2.imread(imgpath)
            dis_img = cv2.resize(dis_img, (250,250))
            # plt.imshow(dis_img)
            # plt.show()
            # print(dis_img.shape)
            image = np.array(dis_img).flatten()
            # print(image)
            data.append([image,label])
            # break
        except:
            pass

pick_out = open('../Database/pickle/Vdataset.pickle','wb')
pickle.dump(data,pick_out)
pick_out.close()