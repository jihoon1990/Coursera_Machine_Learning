# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:59:10 2017

@author: Jihoon_Kim
"""

# Ridge Regression (Gradient Descent)
# Import Module
import pandas as pd
import numpy as np
from ridge_func import predict_output
from ridge_func import ridge_regression_gradient_descent
from matplotlib import pyplot as plt

# Import Data Type
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 
              'sqft_living15':float, 'grade':int, 'yr_renovated':int, 
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
              'sqft_lot15':float, 'sqft_living':float, 'floors':str, 
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

# Load Data from CSV Files
sales = pd.read_csv("kc_house_data.csv", dtype = dtype_dict)
train_data = pd.read_csv("kc_house_train_data.csv", dtype = dtype_dict)
test_data = pd.read_csv("kc_house_test_data.csv", dtype = dtype_dict)

sales['constant'] = 1
train_data['constant'] = 1
test_data['constant'] = 1

# Gradient Descent
# The L2 penalty gets its name because it causes weights to have small L2 norms than otherwise. Let's see how large weights get penalized. Let us consider a simple model with 1 feature:
simple_feature = ['constant','sqft_living']
simple_feature_matrix = train_data[simple_feature].as_matrix()
output = train_data['price'].as_matrix()
simple_test_feature_matrix = test_data[simple_feature].as_matrix()
test_output = test_data['price']

initial_weights = np.array([0.,0.])
step_size = 1e-12
max_iterations = 1000

# Gadient Descent to get weights
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,0.0,max_iterations)
print("Weights (0 penalty): ", simple_weights_0_penalty)

simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,1.0e11,max_iterations)
print("Weights (1e11 penalty): ", simple_weights_high_penalty)

# Visualizing Rigression w/o penalty and w/ penalty
plt.plot(train_data['sqft_living'],output,'k.', label='data')
plt.plot(train_data['sqft_living'],predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',label='w/o penalty')
plt.plot(train_data['sqft_living'],predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-',label='w/ penalty')
plt.legend()
plt.show()

print()
# RSS w/ initial weight
initial_predictions = predict_output(simple_test_feature_matrix, initial_weights)
initial_residuals = test_output - initial_predictions
initial_RSS = (initial_residuals **2).sum()
print("RSS w/ initial weight: %.4g" % initial_RSS)

# RSS w/o penalizing
no_regularization_predictions = predict_output(simple_test_feature_matrix, simple_weights_0_penalty)
no_regularization_residuals = test_output - no_regularization_predictions
no_regularization_RSS = (no_regularization_residuals **2).sum()
print("RSS w/o penalizing: %.4g" % no_regularization_RSS)

# RSS w/ penalizing
regularization_predictions = predict_output(simple_test_feature_matrix, simple_weights_high_penalty)
regularization_residuals = test_output - regularization_predictions
regularization_RSS = (regularization_residuals **2).sum()
print("RSS w/ penalizing: %.4g" % regularization_RSS)

# Running a Multiple Regression with L2 penalty
model_features = ['constant', 'sqft_living', 'sqft_living15']
feature_matrix = train_data[model_features].as_matrix()
output = train_data['price'].as_matrix()
test_feature_matrix = test_data[model_features].as_matrix()
test_output = test_data['price']

initial_weights = np.array([0.,0.,0.])
step_size = 1e-12
max_iterations = 1000

print()
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 0.0, max_iterations)
print("Multiple_Weights (0 penalty): ", multiple_weights_0_penalty)
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)
print("Multiple_Weights (High penalty): ", multiple_weights_high_penalty)

# RSS w/ initial weight
initial_predictions = predict_output(test_feature_matrix, initial_weights)
initial_residuals = test_output - initial_predictions
initial_RSS = (initial_residuals **2).sum()
print("RSS w/ initial weight (Multiple): %.4g" % initial_RSS)

# RSS w/o penalizing
no_regularization_predictions = predict_output(test_feature_matrix, multiple_weights_0_penalty)
no_regularization_residuals = test_output - no_regularization_predictions
no_regularization_RSS = (no_regularization_residuals **2).sum()
print("RSS w/o penalizing (Multiple): %.4g" % no_regularization_RSS)

# RSS w/ penalizing
regularization_predictions = predict_output(test_feature_matrix, multiple_weights_high_penalty)
regularization_residuals = test_output - regularization_predictions
regularization_RSS = (regularization_residuals **2).sum()
print("RSS w/ penalizing (Multiple): %.4g" % regularization_RSS)