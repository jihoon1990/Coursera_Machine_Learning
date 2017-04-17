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
testing = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int)
sales['constant'] = 1
testing['constant'] = 1
training['constant'] = 1
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
# With L1 penalty of 1e7
all_features = ['constant',
                'bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']


feature_matrix = training[all_features].as_matrix()
output = training['price'].as_matrix()
normalized_feature_matrix, norm = normalize_features(feature_matrix)

print("\n=====L1 Penalty: 1e7=====")
initial_weights = np.zeros(len(all_features))
l1_penalty = 1e7
tolerance = 1.0

weights1e7 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output, initial_weights, l1_penalty, tolerance)
print("Weights with L1 of 1e7: \n", weights1e7)

feature_weights1e7 = dict(zip(all_features, weights1e7))
for k,v in feature_weights1e7.items():
    if v != 0.0:
        print(k, v)
        
# With L1 penalty of 1e8
print("\n=====L1 Penalty: 1e8=====")
l1_penalty=1e8
tolerance = 1.0
weights1e8 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output, initial_weights, l1_penalty, tolerance)
print("Weights with L1 of 1e8: \n", weights1e8)

feature_weights1e8 = dict(zip(all_features, weights1e8))
for k,v in feature_weights1e8.items():
    if v != 0.0:
        print(k, v)
        
# With L1 penalty of 1e4
print("\n=====L1 Penalty: 1e4=====")
l1_penalty=1e4
tolerance=5e5
weights1e4 = lasso_cyclical_coordinate_descent(normalized_feature_matrix, output, initial_weights, l1_penalty, tolerance)
print("Weights with L1 of 1e4: \n", weights1e4)

feature_weights1e4 = dict(zip(all_features, weights1e4))
for k,v in feature_weights1e4.items():
    if v != 0.0:
        print(k, v)
        

# Rescaling learned weights
print("\n\n==========Rescaling==========")
feature_matrix = training[all_features].as_matrix()
output = training['price'].as_matrix()
normalized_feature_matrix, norms = normalize_features(feature_matrix)

normalized_weights1e7 = weights1e7 / norms
normalized_weights1e8 = weights1e8 / norms
normalized_weights1e4 = weights1e4 / norms
print("Noramlized weights (L1=1e7): ", normalized_weights1e7)

# Evaluating each of the learned models on the test data
test_feature_matrix = testing[all_features].as_matrix()
test_output = testing['price']

# RSS (L1 = 1e7)
prediction =  predict_output(test_feature_matrix, normalized_weights1e7)
residuals = test_output - prediction
RSS = (residuals **2).sum()
print('RSS for model with weights1e7: %.4g' %RSS)

# RSS (L1 = 1e8)
prediction =  predict_output(test_feature_matrix, normalized_weights1e8)
residuals = test_output - prediction
RSS = (residuals **2).sum()
print('RSS for model with weights1e8: %.4g' %RSS)

# RSS (L1 = 1e4)
prediction =  predict_output(test_feature_matrix, normalized_weights1e4)
residuals = test_output - prediction
RSS = (residuals **2).sum()
print('RSS for model with weights1e4: %.4g' %RSS)