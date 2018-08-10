# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:52:53 2018

@author: jz9108
"""
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split
# Import supplementary visualizations code visuals.py
#import visuals as vs
 
# Pretty display for notebooks
#%matplotlib inline

# Load the Boston housing dataset
#df1 = pd.read_csv("https://pythonhow.com/data/income_data.csv") 
# df2 = df1.set_index("State", drop = False)   # set State columns as index/key column
# df2.loc[startrow:endrow, startcolumn:endcolumn]  # extract
data = pd.read_csv('housing.csv')
#show me the columns
list(data)
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

#gte column names
lst(features)
list(features.columns.values)


print(data.head(5))
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

#exploring the dataset.....import statistics vs numpy
# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price =  np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))
 
#take a look at price vs room,'LSTAT', 'PTRATIO'
pd.DataFrame({'rm':data['RM'],'price':prices}).plot.scatter(x='rm',y='price',c='DarkBlue')                #increaase
pd.DataFrame({'LSTAT':data['LSTAT'],'price':prices}).plot.scatter(x='LSTAT',y='price',c='Green')          # decrease
pd.DataFrame({'PTRATIO':data['PTRATIO'],'price':prices}).plot.scatter(x='PTRATIO',y='price',c='DarkRed')  # no ttrend

##how good id the prediction
import matplotlib.pyplot as plt
ax = plt.axes()
x = np.linspace(-1, 8, 1000)
ax.plot(x, x, '--c');
plt.scatter([3, -0.5, 2, 7, 4.2],[2.5, 0.0, 2.1, 7.8, 5.3])
plt.show()



np.linspace(1, 10, 10)
list(range(1,11))
np.arange(1, 11, 1)


X = features
y = prices
print( "Shape of X and y are", X.shape, y.shape)


###Random permutation cross-validator
rs = ShuffleSplit(n_splits=y.shape[0], random_state=0, test_size=0.20, train_size=None)
for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

kfold = ShuffleSplit(n_splits=y.shape[0], test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.25, random_state=42)


############try different prediction models
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
svm = LinearSVC()
svm.fit(X_train,y_train)
featrue_neams = features





''''DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

classifier = DecisionTreeClassifier()
model = classifier.fit(X_train,y_train)

# TODO: Make predictions on the test data
y_pred = model.predict(X_test)
# TODO: Calculate the accuracy and assign it to the variable acc on the test data.
from sklearn.metrics import accuracy_score,r2_score,median_absolute_error
acc = accuracy_score(y_test, y_pred)
yP = model.predict(X_test)
score_r2 = r2_score(y_test, yP)
score_MedAE = median_absolute_error(y_test, yP)

 

#do the samething via sklearn
from sklearn.model_selection import KFold, cross_val_score
svc = SVC(C=1, kernel='linear')
svc.fit(X_train,y_train).score(X_train,y_train)

k_fold = KFold(n_splits=3)
[svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
   for train, test in k_fold.split(X_digits)]

# even siplify to one line
cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
#n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.
