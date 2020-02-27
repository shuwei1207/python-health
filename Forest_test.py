import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

data = pd.read_excel("data.xlsx")
del data['時間戳記']
le = preprocessing.LabelEncoder()
for row in data.columns:
    data[row] = le.fit_transform(data[row])

y1 = data.iloc[:,-1]
y2 = data.iloc[:,-2]

#case1
X1 = data
del X1['有沒有牙周病']

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

print("case1:")
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print('Accuracy:', dt_clf.score(X_test, y_test))

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred2 = rf_clf.predict(X_test)
print('Accuracy:', rf_clf.score(X_test, y_test))

#case2
data = pd.read_excel("data.xlsx")
del data['時間戳記']
le = preprocessing.LabelEncoder()
for row in data.columns:
    data[row] = le.fit_transform(data[row])
    
X2 = data
del X2['有沒有敏感性牙齒']

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

print("case2:")
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print('Accuracy:', dt_clf.score(X_test, y_test))

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred2 = rf_clf.predict(X_test)
print('Accuracy:', rf_clf.score(X_test, y_test))


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

print("case3:")
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print('Accuracy:', dt_clf.score(X_test, y_test))

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred2 = rf_clf.predict(X_test)
print('Accuracy:', rf_clf.score(X_test, y_test))

#case4
data = pd.read_excel("data.xlsx")
del data['時間戳記']
le = preprocessing.LabelEncoder()
for row in data.columns:
    data[row] = le.fit_transform(data[row])
    
X4 = data
del X4['有沒有敏感性牙齒']
del X4['有沒有牙周病']

X_train, X_test, y_train, y_test = train_test_split(X4, y2, test_size=0.2, random_state=42)

print("case4:")
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print('Accuracy:', dt_clf.score(X_test, y_test))

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred2 = rf_clf.predict(X_test)
print('Accuracy:', rf_clf.score(X_test, y_test))