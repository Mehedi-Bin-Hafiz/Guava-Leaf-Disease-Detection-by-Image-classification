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
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(vx)
print("test size=30, accuracy = {0:.2f}".format(100*metrics.accuracy_score(vy, y_pred)),"%")


realSound = list()
realRing = list()
predSound = list()
predRing = list()

df = pd.DataFrame({'Real': vy, 'Predicted':y_pred})

print(df)

realSpot = df.loc[(df['Real']==0)]
realRust = df.loc[(df['Real']==1)]
realSound = df.loc[(df['Real']==2)]
realaWhitefly = df.loc[(df['Real']==3)]

predSopt = df.loc[(df['Real'] == 0) & (df['Predicted'] == 0)]
predRust = df.loc[(df['Real'] == 1) & (df['Predicted'] == 1)]
predSound = df.loc[(df['Real'] == 2) & (df['Predicted'] == 2)]
predWhitefly = df.loc[(df['Real'] == 3) & (df['Predicted'] == 3)]

Sound=[len(realSound),len(predSound),]
LeafSpot=[len(realSpot),len(predSopt),]
Rust=[len(realRust),len(predRust),]
Whitefly=[len(realaWhitefly),len(predWhitefly),]
predictdata = [Sound,LeafSpot,Rust,Whitefly]


# Creates pandas DataFrame.
predictdf = pd.DataFrame(predictdata,index=['Sound', ' Leaf Spot ', 'Rust', 'Whitefly'],columns=['Real','Prediction'])
#it create 3 columns
predictdf.plot.bar(rot=0,) #rot write lebel horizontally
plt.xlabel('Price range')
plt.yticks([x for x in range(1,18)])
plt.ylabel('Price type')
plt.grid()
plt.savefig('Real price and predicted.png') # need to call before calling show
plt.show()

#
############################# Confusion Matrix #########################


from sklearn.metrics import confusion_matrix
import seaborn as sns
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
cf_matrix= confusion_matrix(vy,y_pred,)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=True, fmt='',)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("Confusion Matrix.png")
plt.show()

