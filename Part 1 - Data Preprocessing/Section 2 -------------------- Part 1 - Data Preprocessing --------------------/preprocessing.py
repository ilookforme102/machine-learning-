# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 02:31:33 2018

@author: ILOOKFORME102
"""

#Data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import the data set

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values
print(X)
y = dataset.iloc[:, 3].values
print(y)

# take care of missing data
# using sklearn library

from sklearn.preprocessing import Imputer
# strategy is how the NaN values will be replaced, if strategy = mean, it's 
# mean that the NaN will be takeplace by mean of value in column
# instead of using fit and transform function, we can use imputer.fit_transform()

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0 )
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data
# the Cons of LabelEncoder is that the category feature with be encoded into numbers 
# without order
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

# so, we use OneHotEncoder as alternative method

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray() 
print(X)

# Encode for y

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Next is splitting data into test set and training set
# using sklearn cross validation

from sklearn.cross_validation import train_test_split
# random_State the same as set.seed in R programming

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# next step is feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# the StandardScaler object was fitted to the X_train so we dont need to fit it with 
# the X_test anymore, because X train and X test was scaled in the same basis, we only need 1

X_test = sc_X.transform(X_test)
