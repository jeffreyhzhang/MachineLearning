# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 07:20:53 2018

@author: jz9108
"""
#class sklearn.feature_selection.RFE(estimator, n_features_to_select=None, step=1, verbose=0)[source]
#import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.datasets import make_classification
data = pd.read_csv('housing.csv')
y = data['MEDV']
X = data.drop('MEDV', axis = 1)

#Make  classification randomly...outpout of target is 0/1
X_rnd, y_rnd = make_classification(n_samples=100,
                                   n_classes=5, 
                                   n_features=8,
                                   n_informative=4,
                                   n_redundant=0,
                                   class_sep=1,
                                   n_clusters_per_class=2,
                                   flip_y=0.2)
 
# show column header
list(X)
y.shape
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X_rnd, y_rnd)
print(selector.support_ )
print(selector.ranking_)
 