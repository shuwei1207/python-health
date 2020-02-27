from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import pandas as pd

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

gnb = GaussianNB()
blf = BernoulliNB()
mlf = MultinomialNB()

gnb.fit(X_train,y_train)
y_gnb_pred = gnb.predict(X_test)
y_gnb_pred = y_gnb_pred.reshape(-1,1)

blf.fit(X_train,y_train)
mlf.fit(X_train,y_train)

print("case1")
print("GaussianNB：", gnb.score(X_test,y_test))
print("BernoulliNB：", blf.score(X_test,y_test))
print("MultinomialNB：", mlf.score(X_test,y_test))
print(confusion_matrix(y_test, mlf.predict(X_test)))

#case2
data = pd.read_excel("data.xlsx")
del data['時間戳記']
le = preprocessing.LabelEncoder()
for row in data.columns:
    data[row] = le.fit_transform(data[row])
X2 = data
del X2['有沒有敏感性牙齒']

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

gnb.fit(X_train,y_train)
y_gnb_pred = gnb.predict(X_test)
y_gnb_pred = y_gnb_pred.reshape(-1,1)

blf.fit(X_train,y_train)
mlf.fit(X_train,y_train)

print("case2")
print("GaussianNB：", gnb.score(X_test,y_test))
print("BernoulliNB：", blf.score(X_test,y_test))
print("MultinomialNB：", mlf.score(X_test,y_test))
print(confusion_matrix(y_test, mlf.predict(X_test)))

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

gnb.fit(X_train,y_train)
y_gnb_pred = gnb.predict(X_test)
y_gnb_pred = y_gnb_pred.reshape(-1,1)

blf.fit(X_train,y_train)
mlf.fit(X_train,y_train)

print("case3")
print("GaussianNB：", gnb.score(X_test,y_test))
print("BernoulliNB：", blf.score(X_test,y_test))
print("MultinomialNB：", mlf.score(X_test,y_test))
print(confusion_matrix(y_test, mlf.predict(X_test)))

#case4
X_train, X_test, y_train, y_test = train_test_split(X3, y2, test_size=0.2, random_state=42)

gnb.fit(X_train,y_train)
y_gnb_pred = gnb.predict(X_test)
y_gnb_pred = y_gnb_pred.reshape(-1,1)

blf.fit(X_train,y_train)
mlf.fit(X_train,y_train)

print("case4")
print("GaussianNB：", gnb.score(X_test,y_test))
print("BernoulliNB：", blf.score(X_test,y_test))
print("MultinomialNB：", mlf.score(X_test,y_test))
print(confusion_matrix(y_test, mlf.predict(X_test)))
