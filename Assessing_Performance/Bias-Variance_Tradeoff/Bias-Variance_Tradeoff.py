# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:02:33 2017

@author: Jihoon_Kim
"""

# Bias-Variance Tradeoff

# Import Module
import pandas as pd
import matplotlib.pyplot as plt
from assessing_perf_func import polynomial_dataframe
from sklearn import linear_model

# Import Data Type
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':str, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# Load Data from CSV files
sales = pd.read_csv("kc_house_data.csv", dtype = dtype_dict)
train_data = pd.read_csv("kc_house_train_data.csv", dtype = dtype_dict)
test_data = pd.read_csv("kc_house_test_data.csv", dtype = dtype_dict)
valid_data = pd.read_csv("kc_house_valid_data.csv", dtype = dtype_dict)

# Visualizing
sales = sales.sort_values(['sqft_living','price'])

# Making a 1 degree polynomial with sales[‘sqft_living’] as the the feature. 
# Call it ‘poly1_data’.
poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']

# Linear Regression Model
model1 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model1_x = sales[['sqft_living']]
model1_y = sales['price']
model1.fit(model1_x,model1_y)
print("Intercepts: ", model1.intercept_)
print(pd.DataFrame(list(zip(model1_x.columns,model1.coef_)), columns = ['features', 'estimated coefficients']))

# Scatter plot of data
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
poly1_data['power_1'], model1.predict(poly1_data[['power_1']]),'-')
plt.xlabel('Squarefeet')
plt.ylabel('Price')