# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:13:13 2018

@author: jz9108
"""

from sklearn import datasets, svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])


import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()

for k in range(3):
     # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test  = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test  = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)

#do the samething via sklearn
from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=3)
[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
   for train, test in k_fold.split(X_digits)]

# even simplify to one line
cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
#n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.

import pandas as pd
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
mydt = pd.DataFrame({'XVal':X[:,0],'YVal':y})

print(type(X))
print(type(y))
#X
#X looks like 4D array ....array([[5.1, 3.5, 1.4, 0.2], [4.9, 3. , 1.4, 0.2],....])
 # change 2.0 to 2.1 will cause error  ValueError: Unknown label type: 'continuous'
 # You are passing floats to a classifier target which expects categorical values (int)  as the target vector.  
y = np.where(y> 1.1, 2.0 , y)
y = y.astype(np.float64)
y
print(y.shape)
#y looks like array([0,....,1,2])
from sklearn.svm import LinearSVC, SVC
clf_1 = LinearSVC().fit(X, y)  # possible to state loss='hinge'
clf_2 = SVC(kernel='linear').fit(X, y)

score_1 = clf_1.score(X, y)
score_2 = clf_2.score(X, y)
print(score_1)
print(score_2)
print('****************LinearSVC*********************')
print(clf_1.coef_)
print(clf_1.coef_.shape)
print(clf_1.intercept_)
print('****************SVC with kernel=linear*********************')
print(clf_2.coef_)
print(clf_2.coef_.shape)
print(clf_2.intercept_)

print(X.shape) # 150 X 4

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=9)
neigh.fit(X, y)
print(neigh.predict(X_test))
print(neigh.kneighbors(X_test)[1])