import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
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
import time
startTime = time.time()

pick_in  = open('../Database/pickle/dataset.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()
random.shuffle(data)
x = []
y = []
for feature, label in data:
    x.append(feature)
    y.append(label)

#datauserate
print( 'total training image',len(x))
thirtypercent=0.30  # training size 70%
fourtypercent=0.40   # training size 60%
fiftypercent=0.50    # training size 50%
sixtypercent=0.60    # training size 40%
seventypercent=0.70   # training size 30%


################# Validation dataset ###################
pick_in  = open('../Database/pickle/Vdataset.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()
random.shuffle(data)
vx = []
vy = []
for feature, label in data:
    vx.append(feature)
    vy.append(label)

#datauserate
print('total testing image',len(vx))

print("\n########## SVM algorithm ###########")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)
clf=RandomForestClassifier(n_estimators=100, probability=True)
clf.fit(X_train,y_train)
y_pred=clf.predict_proba(vx)
print(y_pred[0][0]*100)
print(y_pred[0][1]*100)
print(y_pred[0][2]*100)
print(y_pred[0][3]*100)



