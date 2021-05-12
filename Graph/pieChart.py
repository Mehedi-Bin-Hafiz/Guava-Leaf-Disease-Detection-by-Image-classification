
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
realRust = list()
realWhitefly = list()
realLeafsopt = list()
for i in y:
    if i == 0:
        realLeafsopt.append(i)
    elif i == 1:
        realRust.append(i)
    elif i == 2:
        realSound.append(i)
    else:
        realWhitefly.append(i)
sizes = len(realLeafsopt), len(realRust),len(realSound),len(realWhitefly)
explode = (0.013, 0.013,0.013, 0.013,)
labels = [ 'Leaf spot','Rust','Sound','White fly']
#autopact show percentage inside graph
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',)
plt.axis('equal')
plt.savefig('dataset representation.jpg') # need to call before calling show
plt.show()
