# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:08:13 2018

@author: jz9108@att
"""
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
# Import supplementary visualization code visuals.py
import visuals as vs

# Load the Census dataset
data = pd.read_csv("census.csv")

print(type(data))
# Success - Display the first record
display(data.head(n=1))
# show me sll header/column
print(list(data.columns.values))

#filter on index:   data.set_index('income')

#List unique income values
#print( data.income.unique())
#print( data['native-country'].unique())

print('*******There is no country called [South] ....misspell...need correction!! *******')
display(data[(data['native-country'] == ' South')].head(2))           
data = data.replace({'native-country':' South'},  ' South-Afarica')
display(data[(data['native-country'] ==' South-Afarica')].head(2))
 
#filter on value...show me >50K
display(data[(data.income == '>50K')].head(2))
# what about age and education from statistical point of view
print('----age statistics----')
data.age.describe()
#print('----education----')
#data['education-num'].describe()




# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[(data.income == '>50K')].shape[0]

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data[(data.income == '<=50K')].shape[0]

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (n_greater_50k*100.00)/n_records 

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))


# Log-transform the skewed features
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))

# string datatype
print( type(data['native-country'][1]), data['native-country'].dtype )


# get all string columns
 # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
#prefixes=['workclass','education_level','marital-status','occupation','relationship','race','sex','native-country']
grps = data.columns.to_series().groupby(data.dtypes==data['native-country'].dtype).groups
prefixes = list(grps.values())[1].tolist().remove('income')
print(prefixes)

features_final = pd.get_dummies(features_log_minmax_transform, prefix=prefixes)

# TODO: Encode the 'income_raw' data to numerical values
income = np.where(income_raw=='>50K', 1,0 )

#how many income =1
print(len([elem for elem in income if elem == 1] ) )
# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print(len(encoded))
print(type(income))
               