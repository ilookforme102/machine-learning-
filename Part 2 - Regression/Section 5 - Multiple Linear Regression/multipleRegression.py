# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 02:49:31 2018

@author: ILOOKFORME102
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")


dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4].values


# preprocessing for data
from sklearn.preprocessing  import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 2)


# Fitting the Multiple Linear Regression to the Training set
# *** main function for making linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results:
y_pred = regressor.predict(X_test)

# Building the optimal model using backward elimination
#building linear model by using statmodels library:
import statsmodels.formula.api as sm
 """X = np.append(arr = X, valuess = np.one((50,1) ).astype(int), axis = 1) """
# we need to add column of 1 the beginning of the state: 
X = np.append(arr = np.ones((50,1) ).astype(int), values = X, axis = 1)

X_optimal = X[:, [0,1,2,3,4,5,6]]

# second method for regression: optimizing least square:
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary() 
# remove the feature with highest p-
#repeat the process until there is no feature with p-value > standard
X_optimal = X[:, [0,1,2,3,4,6]]

# second method for regression: optimizing least square:
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary() 

X_optimal = X[:, [0,1,2,3,4]]

# second method for regression: optimizing least square:
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary() 