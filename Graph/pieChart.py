
import pickle
import matplotlib.pyplot as plt

pick_in  = open('../Database/pickle/dataset.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()
x = []
y = []
for feature, label in data:
    x.append(feature)
    y.append(label)

realSound = list()
realRing = list()
for i in y:
    if i == 0:
        realRing.append(i)
    else:
        realSound.append(i)
sizes = len(realSound), len(realRing),
explode = (0.013, 0.013,)
labels = [ 'Sound', 'leafSpot',]
#autopact show percentage inside graph
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',)
plt.axis('equal')
plt.savefig('percentage of Sound and diseased .jpg') # need to call before calling show
plt.show()
