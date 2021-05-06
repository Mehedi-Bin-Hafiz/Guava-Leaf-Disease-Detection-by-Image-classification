import pickle
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import warnings
from sklearn.metrics import confusion_matrix


warnings.filterwarnings("ignore")


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

thirtypercent=0.30  # training size 70%
fourtypercent=0.40   # training size 60%
fiftypercent=0.50    # training size 50%
sixtypercent=0.60    # training size 40%
seventypercent=0.70   # training size 30%


print("########## KNN algorithm ###########")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=thirtypercent, random_state=0)
knn=KNeighborsClassifier(n_neighbors=3,p=2)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted')
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted')
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")

#naive bayes
print("\n########## Naive Bayes algorithm ###########")
gnb = GaussianNB()
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted' )
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted')
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")



print("\n########## Decision tree algorithm ###########")
dtc = DecisionTreeClassifier()
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=thirtypercent, random_state=0)
clf = dtc.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted' )
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted')
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")



print("\n########## SVM algorithm ###########")
clf = svm.SVC(kernel='linear') # Linear Kernel
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=thirtypercent, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted')
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted' )
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")


print("\n########## Random Forest Algorithm ###########")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")
rscore=recall_score(y_test, y_pred, average='weighted' )
print("test size=30, RScore = {0:.2f}".format(100*rscore),"%")
pscore = precision_score(y_test, y_pred, average='weighted')
print("test size=30, pScore = {0:.2f}".format(100*pscore),"%")
