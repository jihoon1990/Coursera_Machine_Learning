# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:09:32 2017

@author: Jihoon_Kim
"""

# Import Module
import pandas as pd
import numpy as np
from Multiple_Linear_Regression_Functions import regression_gradient_descent
from Multiple_Linear_Regression_Functions import predict_output

# Import Data Type
dtype_dict = {'bathrooms':float, 'waterfront':int, 
              'sqft_above':int, 'sqft_living15':float, 
              'grade':int, 'yr_renovated':int, 'price':float, 
              'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 
              'floors':str, 'condition':int, 'lat':float, 
              'date':str, 'sqft_basement':int, 'yr_built':int, 
              'id':str, 'sqft_lot':int, 'view':int}

# Load Data from CSV files
sales = pd.read_csv("kc_house_data.csv", dtype = dtype_dict)
train_data = pd.read_csv("kc_house_train_data.csv", dtype = dtype_dict)
test_data = pd.read_csv("kc_house_test_data.csv", dtype = dtype_dict)

# Prepend 'constant column
sales['constant'] = 1
train_data['constant'] = 1
test_data['constant'] = 1

# Gradient Descent
# now we will run the regression_gradient_descent function on some actual data. In particular we will use the gradient descent to estimate the model from Week 1 using just an intercept and slope. Use the following parameters:
# features: ‘sqft_living’
# output: ‘price’
# initial weights: -47000, 1 (intercept, sqft_living respectively)
# step_size = 7e-12
# tolerance = 2.5e7

simple_feature = train_data[['constant', 'sqft_living']].as_matrix()
output = train_data['price'].as_matrix()
initial_weights = np.array([-47000, 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature,output,initial_weights,step_size,tolerance)
print("Test Weights (Train Data): ", simple_weights)

# Use your newly estimated weights and your predict_output() function to compute the predictions on all the TEST data (you will need to create a numpy array of the test feature_matrix and test output first:
test_simple_feature_matrix = test_data[['constant', 'sqft_living']].as_matrix()
test_predictions = predict_output(test_simple_feature_matrix, simple_weights)
print("Test Predictions: ", test_predictions)

# RSS of test Data
test_residuals = test_data['price']- test_predictions
test_RSS = (test_residuals * test_residuals).sum()
print("RSS of Test Data: %.4g" % test_RSS)

# Multiple Linear Regression
# Now we will use more than one actual feature. Use the following code to produce the weights for a second model with the following parameters:
model_features = ['constant', 'sqft_living', 'sqft_living15']
my_output = 'price'
train_feature_matrix = train_data[model_features].as_matrix()
output = train_data[my_output].as_matrix()
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

print()
# Weights for MLR Model
MLR_weights = regression_gradient_descent(train_feature_matrix,output,initial_weights,step_size,tolerance)
print("Weights: ", MLR_weights)

# Use your newly estimated weights and the predict_output function to compute the predictions on the TEST data.
test_feature_matrix = test_data[model_features].as_matrix()
MLR_predictions = predict_output(test_feature_matrix, MLR_weights)
print("MLR Predictions: ", MLR_predictions)
# RSS of MLR Predictions
MLR_Residuals = test_data[my_output] - MLR_predictions
MLR_RSS = (MLR_Residuals * MLR_Residuals).sum()
print("RSS of MLR: %.4g" % MLR_RSS)