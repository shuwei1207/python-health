import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing

def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def decision_boundry(X,y,a,b):

    X = data.iloc[:,[a,b]]
    X = np.array(X)
    h= 1

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    clf = KNeighborsClassifier(n_neighbors=14)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i, weights = '%s')"
              % (14, 'MLE'))

    plt.show()
######################################################

#case1
data = pd.read_excel("data.xlsx")
del data['時間戳記']
le = preprocessing.LabelEncoder()
for row in data.columns:
    data[row] = le.fit_transform(data[row])

y1 = data.iloc[:,-1]
y2 = data.iloc[:,-2]
X1 = data
del X1['有沒有牙周病']

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

transformer = Normalizer().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

print("case1")
num_of_neighbor = [i for i in range(1,50)]
acc = []
for i in num_of_neighbor:
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    acc.append(neigh.score(X_test,y_test))

plt.plot(num_of_neighbor,acc)
plt.show()
print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)
#plot_confusion_matrix(confusion_matrix(y_test, y_pred),[0,1])
print(classification_report(y_test, y_pred, target_names=['0','1']))


for j in range(1,15):
    pca = PCA(n_components=j)
    pca.fit(X_train)
    pca_x_train = pca.transform(X_train)
    pca_x_test = pca.transform(X_test)
    num_of_neighbor = [i for i in range(1,50)]
    acc = []
    for i in num_of_neighbor:
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(pca_x_train,y_train)
        acc.append(neigh.score(pca_x_test,y_test))

    #plt.plot(num_of_neighbor,acc)
    #plt.show()
    print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))

#case2
data = pd.read_excel("data.xlsx")
del data['時間戳記']
le = preprocessing.LabelEncoder()
for row in data.columns:
    data[row] = le.fit_transform(data[row])

X2 = data
del X2['有沒有敏感性牙齒']

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

transformer = Normalizer().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

print("case2")
num_of_neighbor = [i for i in range(1,50)]
acc = []
for i in num_of_neighbor:
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    acc.append(neigh.score(X_test,y_test))

plt.plot(num_of_neighbor,acc)
plt.show()
print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)
#plot_confusion_matrix(confusion_matrix(y_test, y_pred),[0,1])
print(classification_report(y_test, y_pred, target_names=['0','1']))


for j in range(1,15):
    pca = PCA(n_components=j)
    pca.fit(X_train)
    pca_x_train = pca.transform(X_train)
    pca_x_test = pca.transform(X_test)
    num_of_neighbor = [i for i in range(1,50)]
    acc = []
    for i in num_of_neighbor:
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(pca_x_train,y_train)
        acc.append(neigh.score(pca_x_test,y_test))

    #plt.plot(num_of_neighbor,acc)
    #plt.show()
    print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))
    
#case3
data = pd.read_excel("data.xlsx")
del data['時間戳記']
le = preprocessing.LabelEncoder()
for row in data.columns:
    data[row] = le.fit_transform(data[row])

X3 = data
del X3['有沒有敏感性牙齒']
del X3['有沒有牙周病']

X_train, X_test, y_train, y_test = train_test_split(X3, y1, test_size=0.2, random_state=42)

transformer = Normalizer().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

print("case3")
num_of_neighbor = [i for i in range(1,50)]
acc = []
for i in num_of_neighbor:
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    acc.append(neigh.score(X_test,y_test))

plt.plot(num_of_neighbor,acc)
plt.show()
print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)
#plot_confusion_matrix(confusion_matrix(y_test, y_pred),[0,1])
print(classification_report(y_test, y_pred, target_names=['0','1']))


for j in range(1,15):
    pca = PCA(n_components=j)
    pca.fit(X_train)
    pca_x_train = pca.transform(X_train)
    pca_x_test = pca.transform(X_test)
    num_of_neighbor = [i for i in range(1,50)]
    acc = []
    for i in num_of_neighbor:
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(pca_x_train,y_train)
        acc.append(neigh.score(pca_x_test,y_test))

    #plt.plot(num_of_neighbor,acc)
    #plt.show()
    print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))

#case4

X_train, X_test, y_train, y_test = train_test_split(X3, y2, test_size=0.2, random_state=42)

transformer = Normalizer().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)

print("case4")
num_of_neighbor = [i for i in range(1,50)]
acc = []
for i in num_of_neighbor:
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    acc.append(neigh.score(X_test,y_test))

plt.plot(num_of_neighbor,acc)
plt.show()
print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train,y_train)
y_pred = neigh.predict(X_test)
#plot_confusion_matrix(confusion_matrix(y_test, y_pred),[0,1])
print(classification_report(y_test, y_pred, target_names=['0','1']))


for j in range(1,15):
    pca = PCA(n_components=j)
    pca.fit(X_train)
    pca_x_train = pca.transform(X_train)
    pca_x_test = pca.transform(X_test)
    num_of_neighbor = [i for i in range(1,50)]
    acc = []
    for i in num_of_neighbor:
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(pca_x_train,y_train)
        acc.append(neigh.score(pca_x_test,y_test))

    #plt.plot(num_of_neighbor,acc)
    #plt.show()
    print('k for max acc:',acc.index(max(acc)),'\nMax acc:',max(acc))