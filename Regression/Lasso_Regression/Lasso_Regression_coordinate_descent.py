# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:03:07 2017

@author: Jihoon_Kim
"""

# Import Module & Load Data
import pandas as pd
import numpy as np
from lasso_func import normalize_features
from lasso_func import predict_output
from lasso_func import lasso_coordinate_descent_step
from lasso_func import lasso_cyclical_coordinate_descent
from lasso_func import in_l1range

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int)
sales['constant'] = 1
# Coordinate Descent
# Effect of L1 penalty
simple_features = ['constant','sqft_living','bedrooms']
simple_feature_matrix = sales[simple_features].as_matrix()
output = sales['price'].as_matrix()

simple_feature_matrix , norms = normalize_features(simple_feature_matrix)
weights = np.array([1.,4.,1.])
prediction = predict_output(simple_feature_matrix,weights)

rho = [0 for i in range((simple_feature_matrix.shape)[1])]
for j in range((simple_feature_matrix.shape)[1]):
    rho[j] = (simple_feature_matrix[:,j] * (output - prediction + (weights[j] * simple_feature_matrix[:,j]))).sum()
    print("Rho: %.4g" %rho[j])
    
# Single Coordinate Descent Step
# refer: lasso_func.py
# Cyclic Coordinate Descent
# refer: lasso_func.py

simple_features = ['constant','sqft_living', 'bedrooms']
simple_feature_matrix = sales[simple_features].as_matrix()
normalized_simple_feature_matrix, simple_norm = normalize_features(simple_feature_matrix)
output = sales['price'].as_matrix()
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

print()
weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix,output,initial_weights,l1_penalty,tolerance)
print("Weights from cyclical coordinate descent: \n", weights)

prediction =  predict_output(normalized_simple_feature_matrix, weights)
residuals = output - prediction
RSS = (residuals ** 2).sum()
print('RSS for normalized dataset: ', RSS)

# Evaluating Lasso with more features
