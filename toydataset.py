#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:37:53 2023

@author: djhoannapiolo
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
data = pd.read_csv("toyDataSet.csv")

# Split the data into dependent variable (y) and independent variables (x1, x2)
X = data[['x1', 'x2']]
y = data['y']

# Add a constant term to the independent variables for the intercept in the linear regression
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary of the regression model to evaluate the fit
print(model.summary())

# Get the coefficients of the regression model
intercept, coef_x1, coef_x2 = model.params

# Plot the data points
plt.scatter(data['x1'], data['y'], label='x1 vs y')
plt.scatter(data['x2'], data['y'], label='x2 vs y')

# Plot the regression line
plt.plot(data['x1'], intercept + coef_x1 * data['x1'], 'r', label='Regression line (x1)')
plt.plot(data['x2'], intercept + coef_x2 * data['x2'], 'g', label='Regression line (x2)')

# Add labels and legend
plt.xlabel('x1 and x2')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()
