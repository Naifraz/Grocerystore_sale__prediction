# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:18:20 2019

@author: naif
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('bigData.csv')
X=dataset.iloc[:,[1,2,3,4,5]].values
y=dataset.iloc[:,[6]].values
#print(X)
#print(y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1]= le.fit_transform(X[:,1])
print(X[:,1])
X[:,3]= le.fit_transform(X[:,3])
print(X[:,3])
print(X)
print(y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#print(X_train)
#print(y_train)
from sklearn.ensemble import RandomForestRegressor
reg =  RandomForestRegressor(n_estimators = 1000, random_state = 0)
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test) 
print(y_pred)
from sklearn.metrics import mean_squared_error,r2_score
rms=np.sqrt(mean_squared_error(y_test,y_pred))
print(rms)
r2_score=r2_score(y_test,y_pred)
print(r2_score)
