# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:09:00 2018

@author: jz9108
"""
# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0 )
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

y_true = y
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.0625)[:, np.newaxis]
print(X_test.shape)
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
def plotme(Xtest, ypredict, clr, lbl, linewd, scale):
    # Plot the results
    plt.figure()
    plt.scatter(X, y*scale, s=20, edgecolor="black", c="darkorange", label="data")
    #plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(Xtest, ypredict, color=clr, label=lbl, linewidth=linewd)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

plotme(X_test,y_1,'blue','max_depth=2',2,1)
plotme(X_test,y_2,'red','max_depth=4',2,1)

#set the column name  np.float   np.int32
#y = y.astype(np.float64)

#prepare data for linearsvc
 # change 2.0 to 2.1 will cause error  ValueError: Unknown label type: 'continuous'
 # You are passing floats to a classifier target which expects categorical values (int)  as the target vector.  
#divide into 20 categories
yi = (y*12).astype(int)

 
mydt = pd.DataFrame({'XVal':X[:,0],'YVal':yi})
XX= mydt.drop('YVal', axis = 1)
yy = mydt["YVal"]
 

####May not use this to predict continuous data...you gor error: ValueError: Unknown label type: 'continuous'
from sklearn.svm import LinearSVC
classifer = LinearSVC()
classifer.fit(X,yi)
#PRINT COEFFICIENTS of model for rm, lstat and ptratio
formaula_coeff = pd.DataFrame(classifer.coef_)
formaula_coeff.describe()
#get intecepts
print(classifer.intercept_)
y_3 = classifer.predict(X_test)
plotme(X_test,y_3,'green','svc',2,12)

#########################
print(y_true.shape)
print(y_1.shape)
print('%%%%%%% LinearSVC is the worst 54% %%%%%%%%%')

from  sklearn.metrics import r2_score
print('r2 Score max_depth = 2 is ',r2_score(y_true, y_1))
print('r2 Score max_depth = 4 is ',r2_score(y_true, y_2))
print('r2 Score linearSVC is ',r2_score(yi, y_3))



from sklearn import svm
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, yi)
y_4 = clf.predict(X_test)
plotme(X_test,y_4,'green','svc',2,12)
print('r2 Score SVC linear  is ',r2_score(yi, y_4))
print('svc linear kernel is better than SVCLinear')



from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree  import DecisionTreeClassifier
model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=4), n_estimators = 4)
model.fit(X, yi)
y_5 = model.predict(X_test)
plotme(X_test,y_5,'orange','adaboost',2,12)
print('r2 Score AdaBoostClassifier  is ',r2_score(yi, y_5))
 
