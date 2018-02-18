# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:44:07 2018

@author: ILOOKFORME102
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Splitting dataset in to train and test set

"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)"""

# Fitting linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


# Compare with fitting polynomial regression
#PolynomialFeature prepare the power of x to fit with y
# the result from Plynomial feature is create the column of coefficient for x^2 

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# visualzing the Linear regression results :

plt.scatter(X,y,color= "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("some")
plt.xlabel("haha")
plt.show()


# and for the polynomial regression:
plt.scatter(X,y,color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color= "blue")
plt.show()

# to make the plot more continual, we can deviced X range in to smaller "pices"


X_grid = np.arange(min(X), max(X), step = 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color= "blue")
plt.show()
